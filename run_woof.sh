#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SPEC=woof
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/imagewoof2}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/$SPEC}"
STAGE="${STAGE:-all}" # all, generate, select, eval
DRY_RUN="${DRY_RUN:-0}"
CKPT="${CKPT:-$ROOT/pretrained_models/DiT-XL-2-256x256.pt}"
if [[ ! -f "$CKPT" && -f /root/CaO2-main/pretrained_models/DiT-XL-2-256x256.pt ]]; then
  CKPT=/root/CaO2-main/pretrained_models/DiT-XL-2-256x256.pt
fi
VAE_PATH="${VAE_PATH:-stabilityai/sd-vae-ft-mse}"
if [[ "$VAE_PATH" == stabilityai/sd-vae-ft-mse && -d /root/autodl-tmp/DiT-XL-2-256/vae ]]; then
  VAE_PATH=/root/autodl-tmp/DiT-XL-2-256/vae
fi
case "$STAGE" in all|generate|select|eval) ;; *) printf 'Invalid STAGE: %s\n' "$STAGE" >&2; exit 2;; esac
IPCS=("$@")
if (( ${#IPCS[@]} == 0 )); then IPCS=(10 50); fi
for IPC in "${IPCS[@]}"; do
  case "$IPC" in 10|50) ;; *) printf 'IPC must be 10 or 50.\n' >&2; exit 2;; esac
done
if [[ "$DRY_RUN" != 1 ]]; then
  [[ -d "$DATA_ROOT/train" && -d "$DATA_ROOT/val" ]] || { printf 'DATA_ROOT needs train/ and val/: %s\n' "$DATA_ROOT" >&2; exit 1; }
  if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
    [[ -f "$CKPT" ]] || { printf 'Set CKPT to the DiT-XL/2 256 checkpoint.\n' >&2; exit 1; }
  fi
  mkdir -p "$OUTPUT_ROOT/logs"
fi
run() {
  local logfile="$1"; shift
  if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "$@"; printf '\n';
  else "$@" 2>&1 | tee "$logfile"; fi
}
for IPC in "${IPCS[@]}"; do
  CANDIDATES="$OUTPUT_ROOT/ipc$IPC/candidates"
  SELECTED="$OUTPUT_ROOT/ipc$IPC/selected"
  CLUSTERS="$CANDIDATES/modes.pkl"
  if [[ "$IPC" == 10 ]]; then
    MAX_MODES=10; TAU=1.0; SELECTOR=select_candidates_global_coverage.py
  else
    MAX_MODES=20; TAU=0.5; SELECTOR=select_candidates.py
  fi
  if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
    if [[ "$DRY_RUN" != 1 && -d "$CANDIDATES" ]] && [[ -n "$(find "$CANDIDATES" -mindepth 1 -print -quit)" ]]; then
      printf 'Candidate directory is nonempty; choose a fresh OUTPUT_ROOT: %s\n' "$CANDIDATES" >&2; exit 1
    fi
    [[ "$DRY_RUN" == 1 ]] || mkdir -p "$CANDIDATES"
    run "$OUTPUT_ROOT/logs/ipc${IPC}_generate.log" "$PYTHON" sample_mode_guidance.py \
      --model DiT-XL/2 --image-size 256 --ckpt "$CKPT" --vae-path "$VAE_PATH" \
      --imagenet_dir "$DATA_ROOT" --spec "$SPEC" --nclass 10 --num-samples "$IPC" \
      --guidance --stop_t 25 --mode_guidance_scale 0.1 --cfg-scale 4.0 \
      --num-sampling-steps 50 --seed 0 --num-datasets 10 --mode_discovery gmm_bic \
      --gmm_min_components 1 --gmm_max_components "$MAX_MODES" --gmm_pca_dim 64 \
      --gmm_covariance_type diag --gmm_n_init 1 --gmm_random_state 0 \
      --allocation_temperature "$TAU" --complexity_power 1.0 \
      --save-dir "$CANDIDATES" --save_cluster_file "$CLUSTERS"
  fi
  if [[ "$STAGE" == all || "$STAGE" == select ]]; then
    run "$OUTPUT_ROOT/logs/ipc${IPC}_select.log" "$PYTHON" "$SELECTOR" \
      --candidate-root "$CANDIDATES" --real-root "$DATA_ROOT" --cluster-file "$CLUSTERS" \
      --spec "$SPEC" --project-root "$ROOT" --num-datasets 10 --output "$SELECTED" \
      --batch-size 128 --workers 4
  fi
  if [[ "$STAGE" == all || "$STAGE" == eval ]]; then
    # CaO2's evaluator resolves its class lists relative to its own directory.
    cd "$ROOT/evaluation"
    run "$OUTPUT_ROOT/logs/ipc${IPC}_eval.log" "$PYTHON" eval.py -d imagenet \
      --imagenet_dir "$SELECTED" "$DATA_ROOT" -n resnet --depth 18 --nclass 10 \
      --norm_type instance --ipc "$IPC" --slct_type random --spec "$SPEC" \
      --mixup cut --mix_p 1.0 --beta 1.0 --repeat 3 --seed 0 \
      --tag "moss_${SPEC}_ipc${IPC}"
    cd "$ROOT"
  fi
done
