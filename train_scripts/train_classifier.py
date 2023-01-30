from authorship.lightning_modules import SentenceClassifier
from authorship.lightning_data_modules import AuthorDataModule
from authorship.get_arg_parser import get_train_parser
from authorship.utils import get_trainer
from pytorch_lightning import Trainer, seed_everything
import json


if __name__ == "__main__":
    parser = get_train_parser()
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    author_datamodule = AuthorDataModule(args)
    model = SentenceClassifier(args,
                               num_class=author_datamodule.num_class)
    model_name = args.model.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    hyperparam_name = f"{model_name}_bsz_{args.batch_size}_lr_{args.learning_rate}_temp_{args.temperature}_seed{args.seed}"

    save_path = f"{args.save_path}/{dataset_name}/{hyperparam_name}"
    
    trainer = get_trainer(args.gpus, save_path, args.num_epoch)

    trainer.fit(model=model,
                datamodule=author_datamodule)
    
    validate_result = trainer.validate(model=model,
                        datamodule=author_datamodule)
    
    with open(f"{save_path}/validate_result.json", 'w') as outfile:
        json.dump(validate_result, outfile)
    
    test_result = trainer.test(model=model,
                        datamodule=author_datamodule)

    with open(f"{save_path}/test_result.json", 'w') as outfile:
        json.dump(test_result, outfile)

