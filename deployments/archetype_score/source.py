import modelbit, sys
from typing import *
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms.llamacpp import LlamaCpp
import json

llm = modelbit.load_value("data/llm.pkl") # [1mLlamaCpp[0m Params: {'model_path': './test_model/mistral-7b-instruct-v0.1.Q4_0.gguf', 'suffix': None, 'max_tokens': 100, 'temperature': 0.75, 'top_p': 1.0, 'logprobs': None, 'echo': False, 'stop_...

# main function
def archetype_score(user_token: str, repo_name: str, verbose: bool = True):
        from github import Github, Auth

        template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ###Instruction:
        You are an expert programming assistant who is also an expert in the Star Wars universe and all the main heroes and villains from all the movies and shows. You will tell the truth, even if the truth is that you don't know.
        Given a person's GitHub activity data in JSON format, you must determine what Jedi class does the user fit based on their GitHub activity data (the four classes in order from best to worst are Jedi Master, Jedi Knight, Jedi Apprentice, and Jedi Padawan).
        The data includes the following fields:
        -repo_name: this has the name of the GitHub repository that the user has worked on.
        -branch_name: this has the name of each branch in the GitHub repository that the user has worked on.
        -commit_count: this shows the number of commits the user has made in the respective branch.
        -pull_count: this shows the number of pulls the user has made in the respective branch.
        -pull_file_count: this shows the total number of files affected by the user's pulls made in the respective branch.

        ###Input:
        GitHub Activity Data:
        {data}

        ###Example Response:
        Jedi Knight, Anakin Skywalker, because you get a lot done as evidenced by the number of your commits.
        Jedi Master, Jocasta Nu, because you have large quantity of pull requests.
        Jedi Padawan, Qui Gon Jinn, because you have very few repositories in your activity history, but you have great potential to grow.

        ###Response:
        The Star Wars class and character that personifies this person is"""

        if verbose:
            print("Authenticating...")
        auth = Auth.Token(user_token)
        g = Github(auth=auth)
        user_login = g.get_user().login

        data = []
        repo = ''
        try:
            repo = g.get_repo(repo_name)
        except:
            if verbose:
                print(f"{repo_name} not found")
            # obj = json.dumps({"repo": "not found"}, indent=4)
            return f"{repo_name} not found"
        
        if(repo):
            # repo name
            if verbose:
                print('Looking at data from repo ', repo.name)
            repo_name = {"repo_name": repo.name}
            # Date of last push
            # print('Pushed at:', repo.pushed_at)
            # pushed_at = repo.pushed_at
            has_branch = False
            if verbose:
                print(f'Retrieving data from {repo.name}')
            for branch in repo.get_branches():
                # goes through each branch
                if len(branch.name) > 0:
                    has_branch = True
                branch_name = branch.name
            commit_count = 0
            if has_branch == True:
                for commit in repo.get_commits():
                    if verbose:
                        print('Retrieving your commits...')
                    author = str(commit.author)
                    if (user_login in author) == True:
                        # number of commits by user
                        commit_count += 1
            pull_count = 0
            pull_file_count = 0
            for pull in repo.get_pulls():
                #number of pulls and num files changed in each pull
                pull_count =+ 1
                pull_count =+ pull.changed_files
            item = {"repo_name": repo_name,
                    # "pushed_at": pushed_at,
                    "branch_name": branch.name,
                    "commit_count": commit_count,
                    "pull_count": pull_count,
                    "pull_file_count": pull_file_count}
            data.append(item)

        gitData = json.dumps(data)
        arch_prompt_template = PromptTemplate(input_variables=['data'], template=template)
        
        if verbose:
            print("Running chain")
        arch_chain = LLMChain(llm=llm, prompt=arch_prompt_template)
        archetype = arch_chain.run(gitData)

        # obj = json.dumps({"archetype": archetype})
        # return obj
        return archetype

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(archetype_score(*(modelbit.parseArg(v) for v in sys.argv[1:])))