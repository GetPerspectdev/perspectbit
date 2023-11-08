import time
from openai import OpenAI

if __name__ == '__main__':
  import os
  OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
else:
  import modelbit
  mb = modelbit.login()
  OPENAI_API_KEY = mb.get_secret("OPENAI_API_KEY")


client = OpenAI(api_key = OPENAI_API_KEY)

# assistant = client.beta.assistants.create(
#     name="Agent Arthur Data Sci",
#     instructions='''You are an assistant designed to assess critical thinking skills through interviews. Make this specific for data scientists. Please ask 3 questions, waiting for a response after each questions and asking a minimum of one but max of 2 follow up questions, that help to assess someone's ability to think critically about a problem. 

# Once you have asked all of the questions, return your assessment of their critical thinking skills, along with ways that they can approve. Use a 1-10 scale.''',
#     model="gpt-4-1106-preview"
# )

#thread = client.beta.threads.create()

def assistantCriticalThinking(user_input_message, thread_id):

    user_message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input_message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id='asst_rGJqixsQSr3KG42MmU1yEhXx'
    )

    runstatus = run.status
    
    while runstatus != 'completed':
        print('waiting for gpt4-turb0')
        time.sleep(0.3)
        runcheck = client.beta.threads.runs.retrieve(
                                thread_id=thread_id,
                                run_id=run.id
                                )
        runstatus = runcheck.status



    messages = client.beta.threads.messages.list(
        thread_id=thread_id,
        before = user_message.id
    )

    for i in range(len(messages.data)):
        value = messages.data[i].content[0].text.value
        role = messages.data[i].role
        last_message_id = messages.data[i].id
        yield f"{value}\n"