import torch
from authorship.dataloader import MPerClassDataLoader
from authorship.losses import SupConLoss
from pytorch_lightning import Trainer, callbacks


def optimizer_from_name(optimizer_name):
    optimizer_name_mapping = {"adam": torch.optim.Adam,
                              "sgd": torch.optim.SGD}
    try:
        return optimizer_name_mapping[optimizer_name.lower()]
    except KeyError:
        raise NotImplementedError


def criterion_from_name(criterion_name):
    criterion_name_mapping = {"nll": torch.nn.NLLLoss,
                              "crossentropy": torch.nn.CrossEntropyLoss,
                              "supcon": SupConLoss}
    try:
        return criterion_name_mapping[criterion_name.lower()]
    except KeyError:
        raise NotImplementedError


def loader_from_name(loader_name):
    loader_name_mapping = {"default": torch.utils.data.DataLoader,
                           "m_per_class": MPerClassDataLoader}
    try:
        return loader_name_mapping[loader_name.lower()]
    except KeyError:
        raise NotImplementedError


def parse_data_folder(folder_path, biased=True):
    train_path = f"{folder_path}/train.csv"
    val_path = f"{folder_path}/val.csv"
    test_path = f"{folder_path}/test1.csv"
    return train_path, val_path, test_path

    
def get_callbacks(moniter_metric):
    early =  callbacks.EarlyStopping(monitor=moniter_metric, patience=3, verbose=True, mode="max")
    modelcheckpoint = callbacks.ModelCheckpoint(monitor=moniter_metric,
                                                auto_insert_metric_name=False,
                                                filename='best_checkpoint',
                                                mode="max",
                                                verbose=True)
    callbacks_list = [early, modelcheckpoint]
    return callbacks_list

def get_trainer(gpus, save_path, num_epoch, replace_sampler_ddp=False, moniter_metric='val_accuracy@k=8'):
    trainer = Trainer(gpus=gpus,
                      default_root_dir=save_path,
                      max_epochs=num_epoch,
                      strategy='ddp',
                      replace_sampler_ddp=replace_sampler_ddp,
                      callbacks = get_callbacks(moniter_metric),
                      num_sanity_val_steps=0
                      #deterministic=True,
                      )
    return trainer
