import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SudokuDataset(Dataset):
    def __init__(self, h5_path, dataset_type="train"):
        self.h5_path = h5_path
        self.puzzle_key = f"{dataset_type}/features"
        self.solution_key = f"{dataset_type}/labels"

        self.dataset_len = 1000000 if dataset_type == "train" else 10000
        self.file = None  # Lazy worker handle

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # Lazy initialization per PyTorch DataLoader worker thread
        if self.file is None:
            # 64MB chunk cache for faster slice reads
            self.file = h5py.File(self.h5_path, 'r', rdcc_nbytes=1024 * 1024 * 64)

        puzzle = self.file[self.puzzle_key][idx]    # shape: (9, 9), values 0..9
        solution = self.file[self.solution_key][idx]  # shape: (9, 9), values 1..9

        # Return integer tensors suitable for nn.Embedding
        puzzle_tensor = torch.from_numpy(puzzle).long()          # (9, 9), values 0..9
        target_tensor = torch.from_numpy(solution - 1).long()    # (9, 9), values 0..8

        return puzzle_tensor, target_tensor


def worker_init_fn(worker_id):
    """
    Ensures each DataLoader worker gets a fresh HDF5 file handle
    to prevent file lock corruption across process forks.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset.file = None


def get_sudoku_dataloaders(h5_path, batch_size=256, num_workers=4, pin_memory=True):
    """
    Constructs train and validation DataLoaders for the Sudoku Dataset.
    """
    train_dataset = SudokuDataset(h5_path, dataset_type="train")
    val_dataset = SudokuDataset(h5_path, dataset_type="val")
    test_dataset = SudokuDataset(h5_path, dataset_type="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader