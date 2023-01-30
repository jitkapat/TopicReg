import random
import faiss
import torch
import torch.nn.functional as F
from collections import Counter

def knn_search(queries, targets, k):
    dim = queries.shape[-1]
    index = faiss.IndexFlatIP(dim)
    index.add(targets.cpu().numpy())
    distances, ranks = index.search(queries.cpu().numpy(), k)
    return distances, ranks

def sample_queries(classes, ratio=0.5):
    sample_idxs = []
    class_samples = {}
    for idx, one_class in enumerate(classes):
        if one_class not in class_samples:
            class_samples[one_class] = []
        class_samples[one_class].append(idx)
    for one_class, class_sample in class_samples.items():
        num_sample = int(len(class_sample)*ratio)
        sample = class_sample[:num_sample]
        sample_idxs += sample        
    return sample_idxs
    
def split_query_target(features, authors, topics):
    if len(features) == len(authors) == len(topics):
        features = features.detach()
        features = F.normalize(features)
        authors = authors.detach().tolist()
        topics = topics.detach().tolist() 
        idx_list = list(range(len(features)))
        author_count = Counter(authors)
        query_idxs = sample_queries(authors)
        query_idxs = [idx for idx in query_idxs if author_count[authors[idx]] > 1]
        target_idxs = [idx for idx in idx_list if idx not in query_idxs]
        
        query_authors = [authors[idx]for idx in query_idxs]
        query_topics = [topics[idx ]for idx in query_idxs]
        
        target_authors = [authors[idx ]for idx in target_idxs]
        target_topics = [topics[idx ]for idx in target_idxs]
        
        queries = [features[idx].unsqueeze(0) for idx in query_idxs]
        targets = [features[idx].unsqueeze(0) for idx in target_idxs]
        queries = torch.cat(queries, 0)
        targets = torch.cat(targets, 0)    
    else:
        raise Exception("sample number of text, author, topics are not the same")
    return (queries, targets,
            query_authors, target_authors,
            query_topics, target_topics)

def accuracy_error_rate_at_k(query_authors, target_authors, query_topics, target_topics, ranks, k):
    correct = 0
    diff_topic_pred = 0
    same_topic_pred = 0
    error_count = 0
    precision = 0

    for idx, rank in enumerate(ranks):
        gold_author = query_authors[idx]
        pred_authors = [target_authors[i] for i in rank[:k]]
        sample_correct = False
        correct_count = 0
        for pred_author in pred_authors:
            if pred_author == gold_author:
                sample_correct = True
                correct_count += 1
        precision += correct_count/k
        if sample_correct:
            correct += 1        
        else:
            pred_topics = [target_topics[i] for i in rank[:k]]
            sample_topic = query_topics[idx]
            error_count += 1
            same_topic_count = 0
            diff_topic_count = 0
            for topic in pred_topics:
                if topic == sample_topic:
                    same_topic_count+=1
                else:
                    diff_topic_count+=1
            same_topic_pred += same_topic_count/k
            diff_topic_pred += diff_topic_count/k
    accuracy = correct/len(ranks)
    precision = precision/len(ranks)
    if error_count > 0:
        same_topic_error = same_topic_pred/error_count
        diff_topic_error = diff_topic_pred/error_count
    else:
        same_topic_error = 0
        diff_topic_error = 0
    
    return accuracy, precision, same_topic_error, diff_topic_error

def mrr(query_authors, target_authors, ranks):
    mrr_value = 0.0
    for idx, rank in enumerate(ranks):
        gold_author = query_authors[idx]
        pred_authors = [target_authors[i] for i in rank]
        for rank_idx, pred_author in enumerate(pred_authors):
            if pred_author == gold_author:
                mrr_value += 1.0 / (rank_idx + 1)
                break
    mrr_value /= len(ranks)
    
    return mrr_value