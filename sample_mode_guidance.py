"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from latent_features import get_loader, get_features_per_class
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import pickle
from sklearn.decomposition import PCA
from mode_discovery import fit_gmm_bic, allocate_ipc_to_modes, build_mode_schedule, resolve_allocation_temperature, compute_mode_complexities, compute_allocation_scores
from torchvision import io
import time
import json

def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == 'imagenet100':
        file_list = './misc/class100.txt'
    elif args.spec == 'imagenet1k':
        file_list = './misc/class_indices.txt'
    else:
        raise ValueError('Invalid spec')
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()
    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))
    if args.ckpt is None:
        assert args.model == 'DiT-XL/2', 'Only DiT-XL/2 models are available for auto-download.'
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = args.ckpt or f'DiT-XL-2-{args.image_size}x{args.image_size}.pt'
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    if args.vae_ckpt is not None:
        vae_ckpt = args.vae_ckpt
        ckpt = torch.load(vae_ckpt)
        vae.load_state_dict(ckpt['model'])
    batch_size = 1
    args.dataset = 'imagenet'
    args.net_type = 'resnet_ap'
    args.depth = 10
    args.width = 1.0
    args.norm_type = 'instance'
    args.nch = 3
    args.data_path = os.path.join(args.imagenet_dir, 'train')
    args.size = 224
    args.aug_type = 'color_crop_cutout'
    args.augment = True
    args.dseed = 0
    args.global_batch_size = 1
    args.use_vae = True
    args.finetune_ipc = -1
    mode_metadata = {}
    sampling_centers = {}
    clusters_centers_path = {}
    if args.cluster_file is None:
        original_loader = get_loader(args, return_path=True)
        original_features_per_class, original_paths = get_features_per_class(args, original_loader, vae)
        IPC = args.num_samples
        mode_id_per_class = defaultdict(list)
        clusters_centers = {}
        allocation_temperature = resolve_allocation_temperature(IPC, args.allocation_temperature)
        print(f'[GMM+BIC] IPC={IPC}, allocation temperature tau={allocation_temperature:.3f}')
        for c in tqdm(original_features_per_class.keys(), desc='GMM+BIC mode discovery'):
            start_time = time.time()
            X = np.stack(original_features_per_class[c])
            max_components = min(args.gmm_max_components, IPC, X.shape[0])
            min_components = min(args.gmm_min_components, max_components)
            result = fit_gmm_bic(X, min_components=min_components, max_components=max_components, covariance_type=args.gmm_covariance_type, pca_dim=args.gmm_pca_dim, random_state=args.gmm_random_state, n_init=args.gmm_n_init, reg_covar=args.gmm_reg_covar, max_iter=args.gmm_max_iter)
            mode_complexities = compute_mode_complexities(X, result.labels, result.centers)
            allocation_scores = compute_allocation_scores(result.weights, temperature=allocation_temperature, complexities=mode_complexities, complexity_power=args.complexity_power)
            allocations = allocate_ipc_to_modes(result.weights, IPC, temperature=allocation_temperature, min_per_mode=1, complexities=mode_complexities, complexity_power=args.complexity_power)
            schedule = [int(x) for x in build_mode_schedule(allocations)]
            clusters_centers[c] = result.centers
            mode_id_per_class[c] = result.labels
            sub_centers_by_mode = {}
            sub_paths_by_mode = {}
            for mode_id, n_slots in enumerate(allocations):
                mode_id = int(mode_id)
                n_slots = int(n_slots)
                if n_slots <= 0:
                    sub_centers_by_mode[mode_id] = []
                    sub_paths_by_mode[mode_id] = []
                    continue
                mode_indices = np.where(result.labels == mode_id)[0]
                X_mode = X[mode_indices]
                if X_mode.shape[0] == 0:
                    raise RuntimeError(f'Class {c}, macro mode {mode_id} has no assigned real samples.')
                n_sub = min(n_slots, X_mode.shape[0])
                if n_sub == 1:
                    centers_array = result.centers[mode_id][None, :]
                    sub_labels = np.zeros(X_mode.shape[0], dtype=np.int64)
                else:
                    sub_kmeans = KMeans(n_clusters=n_sub, random_state=args.gmm_random_state, n_init='auto').fit(X_mode)
                    centers_array = sub_kmeans.cluster_centers_
                    sub_labels = sub_kmeans.labels_
                centers_mode = [center.copy() for center in centers_array]
                macro_center = result.centers[mode_id]
                while len(centers_mode) < n_slots:
                    centers_mode.append(macro_center.copy())
                sub_centers_by_mode[mode_id] = centers_mode
                paths_mode = []
                for center in centers_mode:
                    local_idx = int(np.argmin(np.sum((X_mode - center) ** 2, axis=1)))
                    global_idx = int(mode_indices[local_idx])
                    paths_mode.append(original_paths[c][global_idx])
                sub_paths_by_mode[mode_id] = paths_mode
            per_mode_cursor = {int(mode_id): 0 for mode_id in range(result.n_components)}
            expanded_centers = []
            expanded_paths = []
            expanded_macro_ids = []
            for mode_id in schedule:
                cursor = per_mode_cursor[mode_id]
                expanded_centers.append(sub_centers_by_mode[mode_id][cursor])
                expanded_paths.append(sub_paths_by_mode[mode_id][cursor])
                expanded_macro_ids.append(int(mode_id))
                per_mode_cursor[mode_id] += 1
            sampling_centers[c] = np.stack(expanded_centers)
            clusters_centers_path[c] = expanded_paths
            if sampling_centers[c].shape[0] != IPC:
                raise RuntimeError(f'Class {c}: expected {IPC} sampling centers, got {sampling_centers[c].shape[0]}.')
            mode_metadata[c] = {'method': 'gmm_bic_complexity_intra_kmeans', 'selected_k': int(result.n_components), 'weights': result.weights.tolist(), 'mode_complexities': mode_complexities.tolist(), 'normalized_complexities': (mode_complexities / max(float(mode_complexities.mean()), 1e-12)).tolist(), 'allocation_scores': allocation_scores.tolist(), 'complexity_power': float(args.complexity_power), 'allocations': allocations.tolist(), 'schedule': schedule, 'sampling_center_macro_ids': expanded_macro_ids, 'num_sampling_centers': int(sampling_centers[c].shape[0]), 'bic_scores': {str(k): float(v) for k, v in result.bic_scores.items()}, 'pca_dim': int(result.pca_dim), 'covariance_type': result.covariance_type, 'allocation_temperature': float(allocation_temperature)}
            end_time = time.time()
            normalized_complexities = mode_complexities / max(float(mode_complexities.mean()), 1e-12)
            print(f'[class {c}] K*={result.n_components}, allocations={allocations.tolist()}, targets={len(sampling_centers[c])}')
        if args.save_cluster_file is not None:
            cluster_parent = os.path.dirname(args.save_cluster_file)
            if cluster_parent:
                os.makedirs(cluster_parent, exist_ok=True)
            package = {'format_version': 3, 'mode_discovery': args.mode_discovery, 'ipc': int(IPC), 'clusters_centers': clusters_centers, 'sampling_centers': sampling_centers, 'clusters_centers_path': clusters_centers_path, 'mode_metadata': mode_metadata}
            with open(args.save_cluster_file, 'wb') as f:
                pickle.dump(package, f)
            print(f'Saved cluster package to: {args.save_cluster_file}')
        os.makedirs(args.save_dir, exist_ok=True)
        report_path = os.path.join(args.save_dir, 'mode_discovery_report.json')
        report = {str(k): v for k, v in mode_metadata.items()}
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f'Saved mode-discovery report to: {report_path}')
    else:
        with open(args.cluster_file, 'rb') as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict) and 'sampling_centers' in loaded:
            clusters_centers = loaded['clusters_centers']
            sampling_centers = loaded['sampling_centers']
            clusters_centers_path = loaded.get('clusters_centers_path', {})
            mode_metadata = loaded.get('mode_metadata', {})
        else:
            clusters_centers = loaded
            sampling_centers = loaded
    print('Mode counts per class:', {int(k): len(v) for k, v in clusters_centers.items()})
    print('Sampling counts per class:', {int(k): len(v) for k, v in sampling_centers.items()})
    for i in range(args.num_datasets):
        save_dir = os.path.join(args.save_dir, f'dataset_{i}')
        noises = list()
        for class_label, sel_class in zip(class_labels, sel_classes):
            print(class_label, sel_classes)
            index = sel_classes.index(sel_class)
            os.makedirs(os.path.join(save_dir, sel_class), exist_ok=True)
            previous_batch = list()
            start_time = time.time()
            for shift in tqdm(range(args.num_samples // batch_size)):
                z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
                y = torch.tensor([class_label], device=device)
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * batch_size, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                if args.guidance:
                    samples = diffusion.p_sample_loop_with_mode_guidance(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device, mode_features=torch.tensor(sampling_centers[index][shift].reshape(1, 4, 32, 32), device=device, dtype=z.dtype), stop_t=args.stop_t, mode_guidance_scale=args.mode_guidance_scale)
                else:
                    samples = diffusion.p_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
                samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample
                for image_index, image in enumerate(samples):
                    save_image(image, os.path.join(save_dir, sel_class, f'{image_index + shift * batch_size + args.total_shift}.png'), normalize=True, value_range=(-1, 1))
            end_time = time.time()
            print(f'Time taken for class {sel_class}: {end_time - start_time}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(DiT_models.keys()), default='DiT-XL/2')
    parser.add_argument('--vae-path', default='stabilityai/sd-vae-ft-mse', help='Local VAE directory or Hugging Face model ID')
    parser.add_argument('--vae', type=str, choices=['ema', 'mse'], default='mse')
    parser.add_argument('--image-size', type=int, choices=[256, 512], default=256)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--cfg-scale', type=float, default=4.0)
    parser.add_argument('--num-sampling-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default=None, help='Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).')
    parser.add_argument('--spec', type=str, default='none', help='specific subset for generation')
    parser.add_argument('--save-dir', type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument('--num-samples', type=int, default=100, help='the desired IPC for generation')
    parser.add_argument('--total-shift', type=int, default=0, help='index offset for the file name')
    parser.add_argument('--nclass', type=int, default=10, help='the class number for generation')
    parser.add_argument('--phase', type=int, default=0, help='the phase number for generating large datasets')
    parser.add_argument('--guidance', action='store_true', default=False, help='use guidance')
    parser.add_argument('--stop_t', type=int, default=0, help='stop t from 50 to 0')
    parser.add_argument('--mode_guidance_scale', type=float, default=0.1, help='guidance scale')
    parser.add_argument('--imagenet_dir', type=str, default='/ssd_data/imagenet/')
    parser.add_argument('--cluster_file', type=str, default=None, help='Use a cluster file')
    parser.add_argument('--num-datasets', type=int, default=5, help='number of generated datasets')
    parser.add_argument('--vae-ckpt', type=str, default=None, help='vae checkpoint')
    parser.add_argument('--mode_discovery', type=str, default='gmm_bic', choices=['gmm_bic'], help='mode discovery method; gmm_bic uses BIC-selected macro modes, complexity-aware IPC allocation, and intra-mode K-Means sub-centers')
    parser.add_argument('--gmm_min_components', type=int, default=1, help='minimum GMM components considered by BIC')
    parser.add_argument('--gmm_max_components', type=int, default=20, help='maximum GMM components considered by BIC (also capped by IPC)')
    parser.add_argument('--gmm_covariance_type', type=str, default='diag', choices=['diag', 'spherical', 'tied', 'full'], help='GMM covariance parameterization')
    parser.add_argument('--gmm_pca_dim', type=int, default=64, help='PCA dimension used for GMM+BIC; <=0 disables PCA')
    parser.add_argument('--gmm_n_init', type=int, default=1, help='number of GMM initializations per candidate K')
    parser.add_argument('--gmm_random_state', type=int, default=0, help='fixed random state for GMM/PCA, independent of diffusion sampling seed')
    parser.add_argument('--gmm_reg_covar', type=float, default=1e-06, help='non-negative diagonal covariance regularization')
    parser.add_argument('--gmm_max_iter', type=int, default=200, help='maximum EM iterations per GMM fit')
    parser.add_argument('--allocation_temperature', type=float, default=-1.0, help='mode allocation temperature tau. Negative uses IPC-aware auto schedule: IPC<=10 -> 1.0, IPC>=50 -> 0.5')
    parser.add_argument('--complexity_power', type=float, default=1.0, help='strength beta of macro-mode internal complexity in IPC allocation. 1.0 uses pi^tau * D; 0.0 recovers the previous v2 weight-only allocation')
    parser.add_argument('--save_cluster_file', type=str, default=None, help='optional path for saving reusable mode discovery package')
    args = parser.parse_args()
    main(args)
