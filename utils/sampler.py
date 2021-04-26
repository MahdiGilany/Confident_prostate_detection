import torch
import numpy as np
from typing import Sequence, Sized, Optional
from torch.utils.data import Sampler


class BalancedBinaryRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, label=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        label = label if isinstance(label, np.ndarray) else np.array(label)
        self.label_set = list(np.unique(label))
        self.label_loc = []
        for l in self.label_set:
            self.label_loc.append(np.where(label == l)[0].T)
        self.min_num_samples = min([len(ll) for ll in self.label_loc])

    @property
    def n_class(self):
        return len(self.label_set)

    def chunk(self):
        """Interleave indices from different classes"""
        sample_idx_mat = np.zeros((self.n_class, self.min_num_samples), dtype='int')
        for i in range(self.n_class):
            sample_idx_mat[i] = self.label_loc[i][torch.randperm(self.min_num_samples,
                                                                     generator=self.generator).numpy()]
        return list(sample_idx_mat.T.flatten())

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:
            raise NotImplementedError('Only replace = False can be used now.')
        else:
            return (i for i in self.chunk())

    def __len__(self):
        return len(self.data_source)
