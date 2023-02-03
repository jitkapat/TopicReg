# TopicReg
The implementation of [Topic-Regularized Authorship Representation Learning](https://aclanthology.org/2022.emnlp-main.70.pdf) (Proceedings of EMNLP2022) in Pytorch Lightning.
The code is primarily meant for performing experiments for research purposes, focusing on quick training and evaluation. It is not designed with deployment for downstream applications in mind.

## Installation
```
git clone https://github.com/jitkapat/TopicReg.git
cd TopicReg
pip install -e .
```
## Datasets
The source datasets of the ones used in our experiments can be obtained as follows:

- [Amazon](https://nijianmo.github.io/amazon/index.html)
- [Reddit](https://zenodo.org/record/3608135)
- [Fanfiction](https://pan.webis.de/clef21/pan21-web/author-identification.html)

Our code assume that the path containing the dataset are comprised of following files:
- train.csv
- val.csv
- test1.csv (cross-topic test set)
- OOD_A.csv (in-distribution topic test set, optional)
  
Our training scripts also assume that each file in each dataset have been preprocessed into .csv format with [text, author, topic] columns.

## Usage
You can train contrastive, classifier (mll), and then  authorship representation regularization (ARR) with the scripts in "/train_scripts." These scripts will perform both train, validation and test in one script.

For example, we first train a contrastive model (using supervised contrastive loss without data augmentation).

```
python train_scripts/train_contrastive.py   --dataset $data_path \
                                            --model $model_path \
                                            --save_path $save_path \
                                            --criterion supcon \
                                            --batch_size 32 \
                                            --num_epoch 20 \
                                            --gpus 1 \
                                            --learning_rate $LR \
                                            --temperature $TEMP
```

Then we perform another step of ARR with the above model as the base model.
```
python train_scripts/train_conf_reg.py  --dataset $data_path \
                                        --model $model_path \
                                        --model_checkpoint $save_path
                                        --save_path $save_path2 \
                                        --criterion crossentropy \
                                        --batch_size 64 \
                                        --num_epoch 1 \
                                        --gpus 1 \
                                        --temperature $TEMP \
                                        --learning_rate $LR \
                                        
```
(Note that model argument means the pre-trained model (in Huggingface format) for the target model, but the model_checkpoint argument is the one trained in previous step (In Pytorch lightning checkpoint format).)


The above scripts evaluate the models directly after training, but you can evaluate the training models checkpoints in external test data using scripts in "/evaluation_scripts." The parameters are the same as training parameters for that model variant.

## Citation
```
@inproceedings{sawatphol-etal-2022-topicreg,
    title = "Topic-Regularized Authorship Representation Learning",
    author = "Sawatphol, Jitkapat  and
      Chaiwong, Nonthakit and
      Udomcharoenchaikit, Can  and
      Nutanong, Sarana",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

