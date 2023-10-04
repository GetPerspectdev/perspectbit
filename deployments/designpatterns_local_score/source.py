import modelbit, sys
from typing import *
from datasets.load import load_dataset
from sentence_transformers.SentenceTransformer import SentenceTransformer
from collections import Counter
import numpy as np

embedder = modelbit.load_value("data/embedder.pkl") # SentenceTransformer( (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False,...

# main function
def designpatterns_local_score(user_token: str = "", verbose: bool = True):
    vectorDB = load_dataset('csv', data_files="2_design_patterns_embedded_dataset.csv", split='train')
    vectorDB.load_faiss_index('embedding', '2_designpattern_index.faiss')

    from github import Github, Auth
    import base64

    if verbose:
        print("Authenticating...")
    auth = Auth.Token(user_token)
    g = Github(auth=auth)
    user_login = g.get_user().login

    
    repos = {}
    for repo in g.get_user().get_repos():
        if verbose:
            print(f"Analyzing {repo.full_name}")

        skips = (
            ".md",
            ".yml",
            ".txt",
            ".toml",
            ".in",
            ".rst"
        )

        files = {}
        pp = []
        scores = []
        patterns = []
        resources = []
        avgs = {}
        repo = g.get_repo("PyGithub/PyGithub")
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            elif file_content.name.startswith(".") or file_content.name.endswith(skips):
                continue
            else:
                if verbose:
                    print(f"File: {file_content.name}")
                readable = base64.b64decode(file_content.content)
                embed = embedder.encode(str(readable))
                query = np.array(embed, dtype=np.float32)
                score, samples = vectorDB.get_nearest_examples('embedding', query, k=1)
                files[file_content.name] = {
                    'score': score, 
                    'samples': samples, 
                    'content': readable, 
                }
        for k, v in files.items():
            scores.append(v['score'])
            patterns.append(v['samples']['Design Pattern'])
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
        if len(patterns) > 0:
            occurence = Counter()
            for i in patterns:
                occurence.update(i)
            top_pattern = occurence.most_common(3)
            bot_pattern = occurence.most_common()[-3:]
        if len(resources) > 0:
            resource = max(resources, key=resources.count)
        else:
            resource = "No resource"
        for key in avgs.keys():
            avgs[key] = float(sum(avgs[key])/len(avgs[key]))

        if verbose:
            print({
            "design_pattern": bool(eval), 
            "repo_url": repo.full_name, 
            "overall_score": str(score), 
            "top_3_patterns": top_pattern,
            "bot_3_patterns": bot_pattern, 
            "resource": resource, 
            "files": np.asarray(pp).tolist(),
            "occurance": dict(occurence),
            "averages": avgs,
        })
        repos[repo.full_name] = ({
            "design_pattern": bool(eval), 
            "repo_url": repo.full_name, 
            "overall_score": str(score), 
            "top_3_patterns": top_pattern,
            "bot_3_patterns": bot_pattern, 
            "resource": resource, 
            "files": np.asarray(pp).tolist(),
            "occurance": dict(occurence),
            "averages": avgs,
        })
    return repos

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(designpatterns_local_score(*(modelbit.parseArg(v) for v in sys.argv[1:])))