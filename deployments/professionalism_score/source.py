import modelbit, sys
from typing import *
from datasets.load import load_dataset
from sentence_transformers.SentenceTransformer import SentenceTransformer
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms.llamacpp import LlamaCpp
import os
import numpy as np

embedder = modelbit.load_value("data/embedder.pkl") # SentenceTransformer( (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False,...
llm = modelbit.load_value("data/llm.pkl") # [1mLlamaCpp[0m Params: {'model_path': './test_model/mistral-7b-instruct-v0.1.Q4_0.gguf', 'suffix': None, 'max_tokens': 100, 'temperature': 0.75, 'top_p': 1.0, 'logprobs': None, 'echo': False, 'stop_...

def _get_slack_diff(user_id: str = None, slack_token: str = None):
        import slack_sdk
        import pandas as pd

        client=slack_sdk.WebClient(token=slack_token)
        dm_channels_response = client.conversations_list(types="im")
        
        all_messages = {}

        for channel in dm_channels_response["channels"]:
            # Get conversation history
            history_response = client.conversations_history(channel=channel["id"])

            # Store messages
            all_messages[channel["id"]] = history_response["messages"]

        txts = []

        for channel_id, messages in all_messages.items():
            for message in messages:
                try:
                    text = message["text"]
                    user = message["user"]
                    timestamp = message["ts"]
                    txts.append([timestamp,user,text])
                except:
                    pass
        new_df = pd.DataFrame(txts)
        new_df.columns =  ['timestamp','user','text']
        self_user = new_df['user'].value_counts().idxmax()
        new_df = new_df[new_df.user == self_user]

        try:
            files = os.listdir("./scores/user_slack_data/")
            file = [i for i in files if user_id in i]
            old_df = pd.read_csv(f"./scores/user_slack_data/{file[0]}")
            old_df = old_df['text'].values.tolist()
            
            messages = pd.concat([new_df, old_df], ignore_index=True).drop_duplicates('timestamp')
        except Exception as e:
            print(e)
            messages = new_df
        return messages


# main function
def professionalism_score(slack_token: str = "", user_id: str = None, verbose: bool = True):
    if user_id == None:
        import slack_sdk
    import pandas as pd
    import tiktoken
    tiktoker = tiktoken.encoding_for_model('gpt-3.5-turbo')

    vectorDB = load_dataset('csv', data_files='3_professionalism_embedded_dataset.csv', split='train')
    vectorDB.load_faiss_index('embedding', '3_professionalism_index.faiss')

    number_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ###Instruction:
    You are an expert witness specializing in empathy, toxicity, and professionalism.
    Given a person's message history, some already-rated examples as context, and a current message, rate the messages on a scale of 1-100 for how professional they are.
    Please respond with only an integer between 1 and 100 where 1 is super toxic, 100 is super professional, and 50 is completely neutral.
    Then give a short explanation of how the person could be more professional.

    ###Input:

    #Examples: {examples}
    
    #Message History: {message_history}
    
    Current Message:
    {current_message}


    ###Response:
    Your Professionalism rating from 1-100 is """

    if user_id == None:
        client=slack_sdk.WebClient(token=slack_token)
        dm_channels_response = client.conversations_list(types="im")
        
        all_messages = {}

        for channel in dm_channels_response["channels"]:
            # Get conversation history
            history_response = client.conversations_history(channel=channel["id"])

            # Store messages
            all_messages[channel["id"]] = history_response["messages"]

        txts = []

        for channel_id, messages in all_messages.items():
            for message in messages:
                try:
                    text = message["text"]
                    user = message["user"]
                    timestamp = message["ts"]
                    txts.append([timestamp,user,text])
                except:
                    pass

        df = pd.DataFrame(txts)
        df.columns =  ['timestamp','user','text']
        self_user = df['user'].value_counts().idxmax()
        df = df[df.user == self_user]
        df.to_csv(f"./scores/user_slack_data/{self_user}_messages.csv")

        messages = df['text'].values.tolist()
    elif user_id and slack_token:
        messages = _get_slack_diff(user_id=user_id, slack_token=slack_token)
    else:
        files = os.listdir("./scores/user_slack_data/")
        file = [i for i in files if user_id in i]
        df = pd.read_csv(f"./scores/user_slack_data/{file[0]}")
        messages = df['text'].values.tolist()

    embeddings_list = []
    for message in messages:
        message = str(message)
        if len(message)>0:
            embed = embedder.encode(message)
            embeddings_list.append(embed)
        else:
            embed = embedder.encode("Likely an emoji")
            embeddings_list.append(embed)
    df['embedding'] = embeddings_list

    message_history = []
    scores = []
    i = 1
    for message in messages:
        message = str(message)
        if verbose:
            print(f"Searching VectorDB for {message[:10]}...")
        if len(message)>0:
            db_query = embedder.encode(message)
        else:
            db_query = embedder.encode("Emoji")
        db_query = np.array(db_query, dtype=np.float32)
        _, context_list = vectorDB.get_nearest_examples("embedding", db_query, k=3)

        if verbose:
            print("Gathering Context...")
        score_string = ""
        for similar_message, rating, comment in zip(
            context_list['text'], 
            context_list['rating'], 
            context_list['comment']
            ):
            score_string += f"Example: {similar_message}, Rating: {rating}, Reasoning: {comment}\n"
        # if verbose:
        #     print(f"Similar Messages from DB: {score_string}")
        
        formatted_prompt_template = PromptTemplate(
            input_variables=['examples', 'message_history', 'current_message'],
            template=number_template
        )
        chain = LLMChain(llm=llm, prompt=formatted_prompt_template)
        if verbose:
            print("Running Chain...")

        dumb_message = "No Message History"
        if len(message_history) == 1:
            dumb_message = message_history[0]
        
        num_tokens = len(tiktoker.encode(f"{score_string},\n" + ",\n".join(message_history) + message))
        while num_tokens > 3800:
            message_history.pop(0)
            num_tokens = len(tiktoker.encode(f"{score_string},\n" + ",\n".join(message_history) + message))
        
        if verbose:
            print(f"Message token count: {num_tokens}")
            print(f"examples_from_db: {score_string}\ncurr_message: {message}")
        obj = chain.run({
            "examples": score_string,
            "message_history": ",\n".join(message_history) if len(message_history) > 1 else dumb_message,
            "current_message": message,
        })
        message_history.append(message)
        scores.append(obj)
        if verbose:
            print(f"Finished Message {i} of {len(messages)}")
        i += 1

    df['scores'] = scores
    df.to_csv(f"./scores/user_slack_data/{user_id if user_id else self_user}_messages.csv")
    return df.to_json()

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(professionalism_score(*(modelbit.parseArg(v) for v in sys.argv[1:])))