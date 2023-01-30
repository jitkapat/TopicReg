import torch
import torch.nn.functional as F
from authorship.utils import optimizer_from_name, criterion_from_name
from authorship.lightning_modules import mean_pooling, BaseModel, SentenceEncoder, SentenceClassifier
from authorship.traditional_modules import TFIDFEncoder
from authorship.evaluate import knn_search
from transformers import (AutoModel,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification,
                          AutoTokenizer)
from pytorch_lightning import LightningModule

from collections import Counter
    
class DistanceRegularizedEncoderCL(BaseModel):
    def __init__(self, args, dataset):
        super().__init__(args)
        model_path = args.model
        teacher_type = self.get_teacher_type()
        self.teacher = teacher_type.load_from_checkpoint(args.teacher_checkpoint)
        self.teacher.freeze()
        self.bias_model = TFIDFEncoder(dataset).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.temperature = args.temperature
        self.sigmoid = torch.nn.Sigmoid()
        self.eps = 1e-12

    def get_teacher_type(self):
        return SentenceEncoder     

    def forward(self, x):
        return self.encode(x)
    
    def numerical_stability(self, distances):
        logits_max, _ = torch.max(distances, dim=1, keepdim=True)
        distances = distances - logits_max.detach()
        return distances
    
    def compute_teacher_score(self, x):
        features = F.normalize(self.teacher.encode(x))
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T),
                                        self.temperature)
        anchor_dot_contrast = self.numerical_stability(anchor_dot_contrast)
        return anchor_dot_contrast
        
    def compute_score(self, x):
        features = F.normalize(self(x))
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T),
                                        self.temperature)
        anchor_dot_contrast = self.numerical_stability(anchor_dot_contrast)
        return anchor_dot_contrast
    

    def distances_to_prob(self, distances, logits_mask):
        exp_distances = torch.exp(distances) 
        prob = torch.div(exp_distances, (torch.sum((exp_distances * logits_mask), 1).unsqueeze(1)))
        return prob
    
    def step(self, batch, batch_idx):
        if self.target == "author":
            x, y, _ = batch
        elif self.target == "topic":
            x, _, y = batch
        else:
            raise NotImplementedError

        # create mask from label
        labels = y.contiguous().view(-1, 1)
        batch_size = labels.shape[0]
        mask = torch.eq(labels, labels.T).float().to(self.device)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
                    torch.ones_like(mask),
                    1, torch.arange(batch_size).view(-1, 1).to(self.device)
                    , 0)
        mask = mask * logits_mask
        
        # get teacher log prob
        teacher_score = self.compute_teacher_score(x)
        teacher_score = self.distances_to_prob(teacher_score, logits_mask)
        
        # get overconfidence scaling value (B)
        bias_distances = self.bias_model.compute_score(x).to(self.device)
        bias_distances = torch.div(bias_distances, self.temperature)
        bias_distances = self.numerical_stability(bias_distances)
        bias_distances = self.distances_to_prob(bias_distances, logits_mask)
        
        # mask non-positive value
        B = bias_distances * mask
        
        # scale the whole batch with mean of all biased ground truth pair distance
        B = B.sum() / (mask.sum() + self.eps)
        
        # scaling function
        S = torch.div(teacher_score** (1-B), torch.sum(teacher_score, 1).unsqueeze(1)** (1-B))
        
        # obtain student score
        student_score = self.compute_score(x)
        student_score = self.distances_to_prob(student_score, logits_mask)
        
        # change student score to log scale
        P = torch.log(student_score)
        
        # prevent loss to calculate from self-contrast cases
        S = logits_mask * S
        P = logits_mask * P
        teacher_score = logits_mask * teacher_score
        
        # crossentropy between student and teacher
        loss = -torch.sum(S * P, 1)
        loss = torch.mean(loss)
        return loss
    
class DistanceRegularizedEncoderNLL(DistanceRegularizedEncoderCL):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

    def get_teacher_type(self):
        return SentenceClassifier