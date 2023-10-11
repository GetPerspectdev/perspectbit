import modelbit, sys
from typing import *
from datasets.load import load_dataset
from shutil import rmtree
from io import open
from sentence_transformers.SentenceTransformer import SentenceTransformer
from collections import Counter
import subprocess
import os
import numpy as np

embedder = modelbit.load_value("data/embedder.pkl") # SentenceTransformer( (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False,...

# main function
def designpatterns_local_score(repo_url: str = "", verbose: bool = True):
    vectorDB = load_dataset('csv', data_files="2_design_patterns_embedded_dataset.csv", split='train')
    vectorDB.load_faiss_index('embedding', '2_designpattern_index.faiss')
    df_pattern = pd.read_csv("./data/Design_Patterns.csv")

    repo_path = "/tmp/curr_repo"
    rmtree(repo_path, ignore_errors=True)
    result = subprocess.run(["git", "clone", "--depth=1", repo_url, repo_path])
    repo_files = [os.path.join(repo_path, f) for f in os.listdir(repo_path)]
    contents = [f for f in repo_files if os.path.isfile(f)]
    files = {}
    for file in contents:
        try:
            with open(file, 'r') as f:
                content = f.read()
                embed = embedder.encode(content)
                query = np.array(embed, dtype=np.float32)
                score, samples = vectorDB.get_nearest_examples('embedding', query, k=1)
                files[file] = {
                    'score': score, 
                    'samples': samples, 
                    'content': content, 
                    }
        except UnicodeDecodeError:
            pass # binary or ascii file

    pp = []
    scores = []
    patterns = []
    resources = []
    avgs = {}
    for k, v in files.items():
        scores.append(v['score'])
        patterns.append(v['samples']['Design Pattern'])
        resources.append(v['samples']['Unnamed: 4'])
        if v['samples']['Design Pattern'][0] not in avgs:
            avgs[v['samples']['Design Pattern'][0]] = [v['score']]
        else:
            avgs[v['samples']['Design Pattern'][0]] += [v['score']]
        # try: 
        #     pp.append(f"Name: {k.split('/')[-1]} | Score: {v['score']} | Closest: {v['samples']['Language']} {v['samples']['Design Pattern']} | Model: {v['model_out']} |")
        # except KeyError:
        if v['score'] > 1.0:
            pp.append(
                {
                    "name": k.split('/')[-1], 
                    "score": float(v['score']), 
                    "language": v['samples']['Language'][0], 
                    "pattern": v['samples']['Design Pattern'][0],
                    "resource": v['samples']['Unnamed: 4'][0],
                }
            )
        else:
            pp.append(
                {
                    "name": k.split('/')[-1], 
                    "score": float(v['score']), 
                    "language": v['samples']['Language'][0], 
                    "pattern": v['samples']['Design Pattern'][0],
                }
            )

    if verbose:
        print("Getting Average Score and Highest Pattern Likelihood")
    score = float(0)
    if len(scores) > 0:
        score = np.mean(scores)
    eval = score>0.75
    top_pattern = "nothing"
    bot_pattern = "nothing"
    resource = "No resource"
    resource_name = "no resource"
    if len(patterns) > 0:
        occurence = Counter()
        for i in patterns:
            occurence.update(i)
        top_pattern = occurence.most_common(3)
        bot_pattern = occurence.most_common()[-3:]
    if len(resources) > 0:
        resource = max(resources, key=resources.count)
        resource_name = max(patterns, key=patterns.count)
    for key in avgs.keys():
        avgs[key] = float(sum(avgs[key])/len(avgs[key]))
    
    rmtree("/tmp/curr_repo", ignore_errors=True)

    if verbose:
        print({
        "design_pattern": bool(eval), 
        "repo_url": repo_url, 
        "overall_score": str(score), 
        "top_3_patterns": top_pattern,
        "bot_3_patterns": bot_pattern, 
        "resource": resource, 
        "resource_name": resource_name,
        "files": np.asarray(pp).tolist(),
        "occurance": dict(occurence),
        "averages": avgs,
    })
    return {
        "design_pattern": bool(eval), 
        "repo_url": repo_url, 
        "overall_score": str(score), 
        "top_3_patterns": top_pattern,
        "bot_3_patterns": bot_pattern, 
        "resource": resource, 
        "resource_name": resource_name,
        "files": np.asarray(pp).tolist(),
        "occurance": dict(occurence),
        "averages": avgs,
    }

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(designpatterns_local_score(*(modelbit.parseArg(v) for v in sys.argv[1:])))