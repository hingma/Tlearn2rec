import glob
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
from torch_geometric.loader import DataLoader

import config


class HeteroToHomoDataset(Dataset):
    """Wrap HeteroData .pt files and expose homogeneous Data objects using variable nodes.

    - x: variable node features
    - edge_index: variable->constraint edges
    - y: variable labels if available
    """

    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Data:
        hetero = torch.load(self.file_paths[idx], weights_only=False)
        data = Data(
            x=hetero['variable'].x,
            edge_index=hetero['variable', 'in', 'constraint'].edge_index,
            y=hetero['variable'].y if 'y' in hetero['variable'] else None,
        )
        return data


def list_split_files(split_dir: Path, limit: int | None = None) -> List[str]:
    files = sorted(glob.glob(str(split_dir / '*.pt')))
    if limit is not None:
        files = files[:limit]
    return files


def build_loaders(train_limit: int | None = None,
                  valid_limit: int | None = None) -> Tuple[DataLoader, DataLoader]:
    train_files = list_split_files(config.TRAIN_DIR, limit=train_limit)
    valid_files = list_split_files(config.VALID_DIR, limit=valid_limit)

    train_ds = HeteroToHomoDataset(train_files)
    valid_ds = HeteroToHomoDataset(valid_files)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, valid_loader


def load_karate() -> Data:
    dataset = KarateClub()
    return dataset[0]


