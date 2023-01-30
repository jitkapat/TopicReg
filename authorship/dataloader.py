from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, args, **kwargs):
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         num_workers=4,
                         **kwargs)


class MPerClassDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, args, **kwargs):
        sampler = self.get_sampler(dataset=dataset,
                                   target=args.target,
                                   m=args.m_per_class,
                                   batch_size=batch_size)
        super().__init__(dataset=dataset,
                         sampler=sampler,
                         batch_size=batch_size,
                         **kwargs)

    def get_sampler(self, dataset, target, m, batch_size):
        target_mapping = {"author": dataset.authors,
                          "topic": dataset.topics}
        try:
            sampler = MPerClassSampler(labels=target_mapping[target],
                                       m=m,
                                       batch_size=batch_size,
                                       length_before_new_iter=len(dataset))
        except KeyError:
            raise NotImplementedError
        return sampler
