from typing import Tuple, Union, Optional

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    valid_sampler: Optional[SubsetRandomSampler]
    sampler: Optional[SubsetRandomSampler]

    def __init__(self, train_dataset, batch_size, shuffle, validation_split: float, num_workers, pin_memory,
                 collate_fn=default_collate, val_dataset=None):
        self.collate_fn = collate_fn
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.val_dataset = val_dataset

        self.batch_idx = 0
        self.n_samples = len(train_dataset) if val_dataset is None else len(train_dataset) + len(val_dataset)
        self.init_kwargs = {
            'dataset': train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }
        if val_dataset is None:
            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
            super().__init__(sampler=self.sampler, **self.init_kwargs)
        else:
            super().__init__(**self.init_kwargs)

    def _split_sampler(self, split) -> Union[Tuple[None, None], Tuple[SubsetRandomSampler, SubsetRandomSampler]]:
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        print(f"Train: {len(train_sampler)} Val: {len(valid_sampler)}")

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self, bs = 1000):
        if self.val_dataset is not None:
            kwargs = {
                'dataset': self.val_dataset,
                'batch_size': bs,
                'shuffle': False,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers
            }
            return DataLoader(**kwargs)
        else:
            print('Using sampler to split!')
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)



