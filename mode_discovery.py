"""MOSS macro-mode discovery, complexity-aware allocation and slot scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


@dataclass
class GMMBICResult:
    """Container for one class-wise GMM+BIC fit."""

    n_components: int
    centers: np.ndarray          # [K, D_original]
    weights: np.ndarray          # [K]
    labels: np.ndarray           # [N], labels after weight-based sorting
    bic_scores: Dict[int, float]
    pca_dim: int
    covariance_type: str


def resolve_allocation_temperature(ipc: int, temperature: float = -1.0) -> float:
    """Return the mode-allocation temperature.

    ``temperature >= 0`` uses the user supplied value.

    Default automatic schedule:
      * IPC <= 10: tau = 1.0
      * IPC >= 50: tau = 0.5
      * 10 < IPC < 50: linearly interpolate

    tau affects only the mixture-mass term pi_k ** tau.
    The complexity contribution is independently controlled by
    ``complexity_power`` in ``allocate_ipc_to_modes``.
    """
    if temperature >= 0:
        return float(temperature)
    if ipc <= 10:
        return 1.0
    if ipc >= 50:
        return 0.5
    return 1.0 - 0.5 * ((float(ipc) - 10.0) / 40.0)


def compute_mode_complexities(
    x: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the internal complexity of each macro mode.

    Complexity is defined as the RMS Euclidean dispersion around the macro
    center:

        D_k = sqrt( mean_{i in C_k} ||x_i - mu_k||_2^2 )

    A large D_k means that the macro mode covers a broader/more variable region
    in the VAE latent space and may need more intra-mode K-Means prototypes.

    Parameters
    ----------
    x:
        Original VAE latent features, shape [N, D].
    labels:
        Hard GMM assignments after component reordering, shape [N].
    centers:
        Macro-mode centers in the original VAE latent space, shape [K, D].
    """
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"x must be [N, D], got {x.shape}")
    if labels.ndim != 1 or labels.shape[0] != x.shape[0]:
        raise ValueError("labels must be [N] and match x")
    if centers.ndim != 2 or centers.shape[1] != x.shape[1]:
        raise ValueError("centers must be [K, D] in the same feature space as x")

    complexities = np.zeros(centers.shape[0], dtype=np.float64)

    for k in range(centers.shape[0]):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            # This should not occur for a fitted GMM, but keep a stable fallback.
            complexities[k] = eps
            continue

        diffs = x[idx] - centers[k]
        squared_distance = np.sum(diffs * diffs, axis=1)
        complexities[k] = np.sqrt(np.mean(squared_distance))

    # Avoid a zero mode receiving a numerically zero score.
    complexities = np.clip(complexities, eps, None)
    return complexities


def compute_allocation_scores(
    weights: np.ndarray,
    temperature: float = 1.0,
    complexities: Optional[np.ndarray] = None,
    complexity_power: float = 1.0,
) -> np.ndarray:
    """Return normalized macro-mode allocation scores.

    v2 behavior:
        score_k ∝ pi_k ** tau

    v3 behavior:
        score_k ∝ (pi_k ** tau) * (D_k ** beta)

    For numerical/readability stability, D_k is normalized by its class-wise
    mean before exponentiation. This normalization does not change relative
    complexity ratios.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a non-empty 1-D array")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if complexity_power < 0:
        raise ValueError("complexity_power must be non-negative")

    k = len(weights)

    # Mixture-mass contribution.
    if temperature == 0:
        mass_score = np.ones(k, dtype=np.float64)
    else:
        mass_score = np.power(np.clip(weights, 1e-12, None), float(temperature))

    # Complexity contribution. beta=0 -> all ones -> exact v2 score.
    if complexities is None or complexity_power == 0:
        complexity_score = np.ones(k, dtype=np.float64)
    else:
        complexities = np.asarray(complexities, dtype=np.float64)
        if complexities.shape != weights.shape:
            raise ValueError(
                f"complexities shape {complexities.shape} must match weights "
                f"shape {weights.shape}"
            )
        if np.any(complexities < 0):
            raise ValueError("complexities must be non-negative")

        safe_complexities = np.clip(complexities, 1e-12, None)
        mean_complexity = float(safe_complexities.mean())
        normalized_complexity = safe_complexities / max(mean_complexity, 1e-12)
        complexity_score = np.power(
            normalized_complexity, float(complexity_power)
        )

    scores = mass_score * complexity_score
    if not np.all(np.isfinite(scores)) or scores.sum() <= 0:
        scores = np.ones(k, dtype=np.float64)

    return scores / scores.sum()


def allocate_ipc_to_modes(
    weights: np.ndarray,
    ipc: int,
    temperature: float = 1.0,
    min_per_mode: int = 1,
    complexities: Optional[np.ndarray] = None,
    complexity_power: float = 1.0,
) -> np.ndarray:
    """Allocate an IPC budget to discovered macro modes.

    Every discovered macro mode first gets ``min_per_mode`` slots. The remaining
    budget is assigned with the largest-remainder method according to

        q_k ∝ (pi_k ** temperature) * (D_k ** complexity_power)

    where D_k is optional. When ``complexities=None`` or
    ``complexity_power=0``, this exactly falls back to the previous
    weight-only allocation.

    Keeping ``min_per_mode=1`` preserves the v2 coverage rule so this experiment
    isolates only the effect of complexity-aware reallocation.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a non-empty 1-D array")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if ipc <= 0:
        raise ValueError("ipc must be positive")
    if min_per_mode < 0:
        raise ValueError("min_per_mode must be non-negative")

    k = len(weights)
    required = k * min_per_mode
    if required > ipc:
        raise ValueError(
            f"Cannot allocate at least {min_per_mode} sample(s) to {k} modes "
            f"with IPC={ipc}. Reduce the number of modes or min_per_mode."
        )

    counts = np.full(k, min_per_mode, dtype=np.int64)
    remaining = ipc - required
    if remaining == 0:
        return counts

    probs = compute_allocation_scores(
        weights=weights,
        temperature=temperature,
        complexities=complexities,
        complexity_power=complexity_power,
    )

    raw = probs * remaining
    base = np.floor(raw).astype(np.int64)
    counts += base

    leftover = int(remaining - base.sum())
    if leftover > 0:
        remainder = raw - base
        order = np.argsort(-remainder, kind="mergesort")
        counts[order[:leftover]] += 1

    return counts


def build_mode_schedule(allocations: np.ndarray) -> List[int]:
    """Expand mode counts into an interleaved list of macro-mode IDs.

    Example:
        [4, 3, 2, 1] -> [0,1,2,3,0,1,2,0,1,0]
    """
    allocations = np.asarray(allocations, dtype=np.int64)
    remaining = allocations.copy()
    schedule: List[int] = []

    while remaining.sum() > 0:
        for mode_id in range(len(remaining)):
            if remaining[mode_id] > 0:
                schedule.append(mode_id)
                remaining[mode_id] -= 1

    return schedule


def _fit_optional_pca(
    x: np.ndarray,
    pca_dim: int,
    random_state: int,
) -> Tuple[np.ndarray, Optional[PCA]]:
    """Fit PCA when requested and return transformed features plus transformer."""
    if pca_dim is None or pca_dim <= 0 or pca_dim >= x.shape[1]:
        return x, None

    max_dim = min(x.shape[1], max(1, x.shape[0] - 1))
    n_components = min(int(pca_dim), max_dim)
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(x), pca


def fit_gmm_bic(
    x: np.ndarray,
    min_components: int = 1,
    max_components: int = 20,
    covariance_type: str = "diag",
    pca_dim: int = 64,
    random_state: int = 0,
    n_init: int = 1,
    reg_covar: float = 1e-6,
    max_iter: int = 200,
) -> GMMBICResult:
    """Select class-wise GMM component count using minimum BIC."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x must be [N, D], got shape {x.shape}")

    n_samples = x.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required for GMM+BIC")

    x_fit, pca = _fit_optional_pca(
        x, pca_dim=pca_dim, random_state=random_state
    )

    min_k = max(1, int(min_components))
    max_k = min(int(max_components), n_samples)
    if min_k > max_k:
        raise ValueError(f"Invalid GMM component range [{min_k}, {max_k}]")

    bic_scores: Dict[int, float] = {}
    models: Dict[int, GaussianMixture] = {}

    for k in range(min_k, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
            reg_covar=reg_covar,
            max_iter=max_iter,
            init_params="kmeans",
        )
        gmm.fit(x_fit)
        bic_scores[k] = float(gmm.bic(x_fit))
        models[k] = gmm

    best_k = min(bic_scores, key=bic_scores.get)
    best_gmm = models[best_k]

    centers_fit = best_gmm.means_
    centers = (
        pca.inverse_transform(centers_fit)
        if pca is not None
        else centers_fit
    )
    weights = best_gmm.weights_.copy()
    labels = best_gmm.predict(x_fit)

    # Sort components by mixture mass for deterministic reporting/allocation.
    order = np.argsort(-weights, kind="mergesort")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))

    centers = centers[order]
    weights = weights[order]
    labels = inverse[labels]

    return GMMBICResult(
        n_components=int(best_k),
        centers=np.asarray(centers, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        bic_scores=bic_scores,
        pca_dim=int(x_fit.shape[1]),
        covariance_type=covariance_type,
    )