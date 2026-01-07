from torch.utils.data import DataLoader
from .nuscenes_dataset import NuScenesDataset
from .transforms import Compose

def build_dataset(cfg, split="train"):
    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "data/nuscenes")
    camera_names = data_cfg.get("camera_names")
    tfm = build_transforms(cfg, split=split)
    return NuScenesDataset(root=root, split=split, transform=tfm, camera_names=camera_names)

def build_dataloader(cfg, split="train", shuffle=None):
    train = split == "train"
    runtime = cfg.get("runtime", {})
    bs = cfg.get("training", {}).get("batch_size", 1) if train else 1
    num_workers = int(runtime.get("num_workers", 0))
    pin_memory = bool(runtime.get("pin_memory", False))
    if shuffle is None:
        shuffle = train
    ds = build_dataset(cfg, split=split)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = [b[k] for b in batch]
    return out

def build_transforms(cfg, split="train"):
    return Compose([])
