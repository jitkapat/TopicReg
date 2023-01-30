import pandas as pd
from torch.utils.data import Dataset
import random

class AuthorDataSet(Dataset):
    def __init__(self,
                 data_path,
                 xname="text",
                 yname="author"):
        df = pd.read_csv(data_path)
        self.texts, self.authors, self.topics = self.process_df(df)
        self.num_text = len(self.texts)
        self.num_author = len(set(self.authors))
        self.num_topic = len(set(self.topics))

    def process_df(self, df):
        df['text'] = df.text.apply(str)
        texts = list(df.text)
        authors = list(pd.Categorical(df.author).codes)
        topics = list(pd.Categorical(df.topic).codes)
        return texts, authors, topics

    def __len__(self):
        return self.num_text

    def __getitem__(self, idx):
        text = self.texts[idx]
        author = self.authors[idx]
        topic = self.topics[idx]
        return text, author, topic
class AuthorVerificationDataSet(Dataset):
    def __init__(self,
                 data_path,
                 xname="text",
                 yname="author"):
        df = pd.read_csv(data_path)
        texts, authors, topics = self.process_df(df)
        self.pairs, self.labels = self.sample_pairs(texts,
                                                    authors)
        self.num_pairs = len(self.pairs)

    def process_df(self, df):
        df['text'] = df.text.apply(str)
        texts = list(df.text)
        authors = list(pd.Categorical(df.author).codes)
        topics = list(pd.Categorical(df.topic).codes)
        return texts, authors, topics
    
    def sample_pairs(self, texts, authors):
        num_pairs = len(texts)
        author2sampleidx = {}
        for idx, author in enumerate(authors):
            if author not in author2sampleidx:
                author2sampleidx[author] = []
            author2sampleidx[author].append(idx)
        
        pos_pairs = []
        while len(pos_pairs) < int(num_pairs/2):
            #print(len(pos_pairs))
            for author, idxs in author2sampleidx.items():
                if len(idxs) >= 2:
                    shuffled_idxs = idxs[:]
                    random.shuffle(shuffled_idxs)
                    pos_pairs.append((shuffled_idxs[0], shuffled_idxs[1]))
        
        neg_pairs = []
        
        idxs = list(range(len(texts)))
        while len(neg_pairs) < int(num_pairs/2):
            #print(len(neg_pairs))
            sampled_idxs = random.sample(idxs, 2)
            idx1, idx2 = sampled_idxs
            if authors[idx1] != authors[idx2]:
                neg_pairs.append((idx1, idx2))
        
        text_pairs = []
        labels = []
        for idx1, idx2 in pos_pairs:
            text1 = texts[idx1]
            text2 = texts[idx2]
            text_pairs.append((text1, text2))
            labels.append(1)
        
        for idx1, idx2 in neg_pairs:
            text1 = texts[idx1]
            text2 = texts[idx2]
            text_pairs.append((text1, text2))
            labels.append(0)

        data = list(zip(text_pairs, labels))
        random.shuffle(data)
        text_pairs, labels = zip(*data)
        
        return text_pairs, labels

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        return pair, label
    
