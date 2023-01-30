import torch
from torch.utils.data import DistributedSampler
import numpy as np
import random
import collections

def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=int)
    return labels_to_indices

def safe_random_choice(input_data, size):
    """
    Randomly samples without replacement from a sequence. It is "safe" because
    if len(input_data) < size, it will randomly sample WITH replacement
    Args:
        input_data is a sequence, like a torch tensor, numpy array,
                        python list, tuple etc
        size is the number of elements to randomly sample from input_data
    Returns:
        An array of size "size", randomly sampled from input_data
    """
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace)

class DistributedMPerClassSampler(DistributedSampler):
    def __init__(self, dataset, labels, m, batch_size, 
                length_before_new_iter, num_replicas=None, rank=None,
                shuffle=False, seed=None, drop_last=False,
                ):
        super().__init__(dataset, num_replicas, rank, seed)
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            self.list_size -= self.list_size % self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        idx_list = indices
        i = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            np.random.shuffle(self.labels)
            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.m_per_class] = safe_random_choice(
                    t, size=self.m_per_class
                )
                i += self.m_per_class
        return iter(idx_list)
    
    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1
