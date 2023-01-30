from authorship.dataset import AuthorDataSet
from authorship.traditional_modules import BOWEncoder
from authorship.evaluate import *
from authorship.get_arg_parser import get_train_parser
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_lightning import seed_everything
from collections import Counter
import json
import os
import numpy as np

def evaluate(dataset_path, vectorizer, k=8):
    # load dataset
    dataset = AuthorDataSet(dataset_path)
    authors = list(dataset.authors)
    topics = list(dataset.topics)
    texts = list(dataset.texts)
    idx_list = list(range(len(authors)))
    author_count = Counter(authors)
    
    #split query and target
    query_idxs = sample_queries(authors)
    query_idxs = [idx for idx in query_idxs if author_count[authors[idx]] > 1]
    target_idxs = [idx for idx in idx_list if idx not in query_idxs]
    query_authors = [authors[idx]for idx in query_idxs]
    query_topics = [topics[idx ]for idx in query_idxs]
    target_authors = [authors[idx] for idx in target_idxs]
    target_topics = [topics[idx] for idx in target_idxs]
    
    queries = vectorizer([texts[idx] for idx in query_idxs])
    targets = vectorizer([texts[idx] for idx in target_idxs])
    
    # cosine similarity search
    logits = cosine_similarity(queries, targets)
    
    #ranking
    ranks = []
    distances = []
    for result in logits:
        sorted_result = np.flip(result.argsort())
        ranks.append(sorted_result[:k])
        distances.append([result[idx] for idx in sorted_result[:k]])
    
    #evaluate
    accuracy, same_topic_error, diff_topic_error = accuracy_error_rate_at_k(query_authors,
                                                                            target_authors,
                                                                            query_topics,
                                                                            target_topics,
                                                                            ranks,
                                                                            k)
    mrr_score = mrr(query_authors, target_authors, ranks)
    result = {f"accuracy@{k}":accuracy,
              "STE@{k}": same_topic_error,
              "DTE@{k}": diff_topic_error,
              "MRR": mrr_score}
    return result

    
parser = get_train_parser()
parser.add_argument("--test_mode", type=str, default="OODAT")
args = parser.parse_args()
seed_everything(args.seed, workers=True)

train_path = f"{args.dataset}/train.csv"
train_dataset = AuthorDataSet(train_path)
vectorizer =  BOWEncoder(train_dataset)

if args.test_mode=="OODAT":
    test_path = f"{args.dataset}/test1.csv"
elif args.test_mode=="OODA":
    test_path = f"{args.dataset}/OOD_A.csv"
else:
    raise NotImplementedError

save_path = f"{args.save_path}"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
test_result = evaluate(test_path, vectorizer, k=8)
with open(f"{save_path}/test_result_{args.test_mode}.json", 'w') as outfile:
    json.dump(test_result, outfile)
