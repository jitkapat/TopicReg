import torch
import torch.nn.functional as F
from authorship.utils import optimizer_from_name, criterion_from_name
from authorship.losses import SupConLoss
from authorship.evaluate import accuracy_error_rate_at_k, mrr, knn_search, split_query_target
from transformers import (AutoModel,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification,
                          AutoModelForMaskedLM,
                          AutoTokenizer
                          )
from pytorch_lightning import LightningModule
from collections import Counter


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded,
                     1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BaseModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.learning_rate = args.learning_rate
        self.optimizer = optimizer_from_name(args.optimizer)
        self.criterion = criterion_from_name(args.criterion)()
        self.target = args.target
        self.pooling = mean_pooling
        self.ks = [1, 8]

    def encode(self, x):
        batch_inputs = self.tokenizer(list(x),
                                      padding=True,
                                      truncation=True,
                                      return_tensors='pt').to(self.device)

        attn_mask = batch_inputs['attention_mask']
        batch_output = self.model(**batch_inputs)
        token_embeddings = batch_output.hidden_states[-1]
        features = self.pooling(token_embeddings, attn_mask)
        return features        
    
    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(),
                              self.learning_rate)

    def step(self, batch, batch_idx):
        if self.target == "author":
            x, y, _ = batch
        elif self.target == "topic":
            x, _, y = batch
        else:
            raise NotImplementedError
        x = self(x)
        y = y.long()
        loss = self.criterion(x, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss,
                 sync_dist=True,
                 batch_size=len(batch[0]))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss,
                 sync_dist=True,
                 batch_size=len(batch[0]))
        return batch

    def validation_epoch_end(self, outputs):
        self.evaluation_report(outputs, "val")


    def test_step(self, batch, batch_idx):
        # loss = self.step(batch, batch_idx)
        # self.log("test_loss", loss,
        #          sync_dist=True,
        #          batch_size=len(batch[0]))
        return batch

    def test_epoch_end(self, outputs):
        self.evaluation_report(outputs, "test")

    def evaluation_report(self, outputs, report_prefix):
        texts = []
        authors = []
        topics = []
        for text, author, topic in outputs:
            texts.append(text)
            authors.append(author)
            topics.append(topic)
        features = [self.encode(batch) for batch in texts]
        features = torch.cat(features, 0)
        features = F.normalize(features)
        authors = torch.cat(authors, 0)
        topics = torch.cat(topics, 0)

        (queries, targets,
        query_authors, target_authors,
        query_topics, target_topics) = split_query_target(features, authors, topics)

        _, ranks = knn_search(queries, targets, 10)

        for k in self.ks:
            (accuracy,
            precision,
            ste,
            dte,) = accuracy_error_rate_at_k(query_authors, target_authors,
                                            query_topics, target_topics,
                                            ranks,
                                            k)
            if k==1:
                self.log(f"{report_prefix}_st_error@k={k}", ste,
                    sync_dist=True,
                    batch_size=len(texts))

                self.log(f"{report_prefix}_dt_error@k={k}", dte,
                        sync_dist=True,
                        batch_size=len(texts))
            else:
                self.log(f"{report_prefix}_precision@k={k}", precision,
                        sync_dist=True,
                        batch_size=len(texts))

            self.log(f"{report_prefix}_accuracy@k={k}", accuracy,
                    sync_dist=True,
                    batch_size=len(texts))

        mrr_value = mrr(query_authors, target_authors,
                        ranks)
        self.log(f"{report_prefix}_mrr", mrr_value,
                sync_dist=True,
                batch_size=len(texts))

        

class SentenceEncoder(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        model_path = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.temperature = args.temperature
        self.criterion = SupConLoss(temperature=self.temperature)
        self.save_hyperparameters()

    def forward(self, x):
        return self.encode(x)

    def step(self, batch, batch_idx):
        if self.target == "author":
            x, y, _ = batch
        elif self.target == "topic":
            x, _, y = batch
        else:
            raise NotImplementedError
        x = self(x)
        x = F.normalize(x)
        x = x.unsqueeze(1)
        y = y.long()
        loss = self.criterion(x, y)
        return loss
class SentenceClassifier(BaseModel):
    def __init__(self,
                 args,
                 num_class):
        super().__init__(args)
        model_path = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_class)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(),
                              self.learning_rate)
    
    def forward(self, x, y):
        batch_inputs = self.tokenizer(list(x),
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt').to(self.device)
        batch_output = self.model(**batch_inputs,
                                  labels=y)
        logits = batch_output.logits
        return logits

    def step(self, batch, batch_idx):
        if self.target == "author":
            x, y, _ = batch
        elif self.target == "topic":
            x, _, y = batch
        else:
            raise NotImplementedError
        y = y.long()
        logits = self(x, y)
        loss = self.criterion(logits, y)
        return loss

