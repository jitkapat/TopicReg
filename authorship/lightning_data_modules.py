from authorship.dataset import AuthorDataSet, AuthorVerificationDataSet
from authorship.utils import parse_data_folder, loader_from_name
from authorship.sampler import DistributedMPerClassSampler
from torch.utils.data import DistributedSampler
from pytorch_lightning import LightningDataModule
from pytorch_metric_learning.samplers import MPerClassSampler


class AuthorDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_folder_path = args.dataset
        self.batch_size = args.batch_size
        self.loader = loader_from_name(args.loader)
        self.args = args
        self.get_train_dataset()
        if self.args.target == "author":
            self.num_class = self.train_dataset.num_author
        elif self.args.target == "topic":
            self.num_class = self.train_dataset.num_topic

    def get_train_dataset(self):
        train_path, _, _ = parse_data_folder(
            self.data_folder_path)
        self.train_dataset = AuthorDataSet(train_path)
    
    def setup(self, stage=None):
        train_path, val_path, test_path = parse_data_folder(
            self.data_folder_path)
        self.train_dataset = AuthorDataSet(train_path)
        self.val_dataset = AuthorDataSet(val_path)
        self.test_dataset = AuthorDataSet(test_path)

    def train_dataloader(self):
        return self.loader(dataset=self.train_dataset,
                           batch_size=self.batch_size,
                           num_workers=4)

    def val_dataloader(self):
        return self.loader(dataset=self.val_dataset,
                           shuffle=False,
                           batch_size=self.batch_size,
                           num_workers=4)

    def test_dataloader(self):
        return self.loader(dataset=self.test_dataset,
                           shuffle=False,
                           batch_size=self.batch_size,
                           num_workers=4)

class MPerClassDataModule(AuthorDataModule):
    def __init__(self, args):
        super().__init__(args)
        self.m = args.m_per_class
        self.target = args.target
    
    def get_sampler(self, dataset):
        target_mapping = {"author": dataset.authors,
                          "topic": dataset.topics}
        try:
            sampler = DistributedMPerClassSampler(dataset,
                    labels=target_mapping[self.target],
                    m=self.m,
                    batch_size=self.batch_size,
                    length_before_new_iter=len(dataset))
        except KeyError:
            raise NotImplementedError
        return sampler
    
    def setup(self, stage=None):
        train_path, val_path, test_path = parse_data_folder(
            self.data_folder_path)
        self.train_dataset = AuthorDataSet(train_path)
        self.val_dataset = AuthorDataSet(val_path)
        self.test_dataset = AuthorDataSet(test_path)

    def train_dataloader(self):
        sampler = self.get_sampler(self.train_dataset)
        return self.loader(dataset=self.train_dataset,
                           batch_size=self.batch_size,
                           sampler=sampler,
                           num_workers=4)
    
    def val_dataloader(self):
        return self.loader(dataset=self.val_dataset,
                           shuffle=False,
                           batch_size=self.batch_size,
                           num_workers=4)

    def test_dataloader(self):
        return self.loader(dataset=self.test_dataset,
                           shuffle=False,
                           batch_size=self.batch_size,
                           num_workers=4)
        
class AuthorVerificationDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_folder_path = args.dataset
        self.batch_size = args.batch_size
        self.loader = loader_from_name(args.loader)
        self.args = args
        self.get_train_dataset()

    def get_train_dataset(self):
        train_path, _, _ = parse_data_folder(
            self.data_folder_path)
        self.train_dataset = AuthorVerificationDataSet(train_path)
    
    def setup(self, stage=None):
        train_path, val_path, test_path = parse_data_folder(
            self.data_folder_path)
        self.train_dataset = AuthorVerificationDataSet(train_path)
        self.val_dataset = AuthorVerificationDataSet(val_path)
        self.test_dataset = AuthorVerificationDataSet(test_path)

    def train_dataloader(self):
        return self.loader(dataset=self.train_dataset,
                           batch_size=self.batch_size,
                           num_workers=4)

    def val_dataloader(self):
        return self.loader(dataset=self.val_dataset,
                           shuffle=False,
                           batch_size=self.batch_size,
                           num_workers=4)

    def test_dataloader(self):
        return self.loader(dataset=self.test_dataset,
                           shuffle=False,
                           batch_size=self.batch_size,
                           num_workers=4)
        
    def teardown(self, stage="validate"):
        train_path, val_path, test_path = parse_data_folder(
            self.data_folder_path)
        self.train_dataset = AuthorVerificationDataSet(train_path)
