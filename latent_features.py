"""VAE latent extraction used by MOSS target construction."""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from data import ImageFolder

device = "cuda" if torch.cuda.is_available() else "cpu"

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def get_loader(args, return_path=False, mode_id_file=None):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform, nclass=args.nclass,
                        ipc=args.finetune_ipc, spec=args.spec, phase=args.phase,
                        seed=0, return_origin=True, return_path=return_path, mode_id_file=mode_id_file)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        # sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return loader

def get_features_per_class(args, loader, model):
    features_per_class = defaultdict(list)
    path_per_class = defaultdict(list)
    for img, lb, orig_labels, paths in tqdm(loader):

        if args.use_vae:
            features = model.encode(img.to(device)).latent_dist.mean * 0.18215
            features = features.detach().cpu()
        else:
            features = model.get_feature(img.to(device), 5, 5)[0]
            features = features.detach().cpu()
        # print(paths.shape)
        for feat, label, path in zip(features, lb, paths):
            features_per_class[label.item()].append(feat.flatten().numpy())
            path_per_class[label.item()].append(path)
    return features_per_class, path_per_class
