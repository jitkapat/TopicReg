from authorship.dataset import AuthorDataSet
from torch.utils.data import DataLoader
from authorship.lightning_modules import SentenceEncoder
from authorship.get_arg_parser import get_train_parser
from authorship.utils import get_trainer
from pytorch_lightning import seed_everything
import json

def get_dataloader(data_path, batch_size=64):
    dataset = AuthorDataSet(data_path)
    loader = DataLoader(dataset=dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=4)
    return loader

def get_test_loader(data_path, mode):
    if mode == "OODAT":
        return get_dataloader(f"{data_path}/test1.csv")
    elif mode == "OODA":
        return get_dataloader(f"{data_path}/OOD_A.csv")
    else:
        raise NotImplementedError
    
parser = get_train_parser()
parser.add_argument("--model_checkpoint", type=str)
parser.add_argument("--test_mode", type=str, default="OODAT")
args = parser.parse_args()
seed_everything(args.seed, workers=True)
checkpoint_path = args.model_checkpoint
test_loader = get_test_loader(args.dataset, args.test_mode)

model = SentenceEncoder.load_from_checkpoint(checkpoint_path)
model_name = args.model.split("/")[-1]
dataset_name = args.dataset.split("/")[-1]
hyperparam_name = f"{model_name}_bsz_{args.batch_size}_lr_{args.learning_rate}_temp_{args.temperature}_seed{args.seed}"
save_path = f"{args.save_path}"


trainer = get_trainer(1, "", 1, replace_sampler_ddp=False)
test_result = trainer.test(model, test_loader)

with open(f"{save_path}/test_result_{args.test_mode}.json", 'w') as outfile:
    json.dump(test_result, outfile)