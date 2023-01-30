import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TFIDFEncoder(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        corpus =  dataset.texts
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(corpus)
        
    def forward(self,x):
        return self.vectorizer.transform(x)
    
    def compute_score(self, texts):
        features = self(texts)
        anchor_dot_contrast = torch.tensor(
            features.dot(features.transpose()).toarray()
            )
        return anchor_dot_contrast
    
class BOWEncoder(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        corpus =  dataset.texts
        self.vectorizer = CountVectorizer(binary=True)
        self.vectorizer.fit(corpus)
        
    def forward(self,x):
        return self.vectorizer.transform(x)
    
    def compute_score(self, texts):
        features = self(texts)
        anchor_dot_contrast = torch.tensor(
            features.dot(features.transpose()).toarray()
            )
        return anchor_dot_contrast