import modelbit
import logging
from datetime import datetime
from typing import List
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from dataclasses import dataclass

mb = modelbit.login()
OPENAI_API_KEY = mb.get_secret("OPENAI_API_KEY")
llm = OpenAI(model_name="gpt-4-0613", openai_api_key=OPENAI_API_KEY)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class QuestionReturn(BaseModel):
  score: List[str] = Field(description = 'score 1 to 5')
  summary: List[str] = Field(description = 'summary of the answer given')

# New deployment
def run(answer: str, question: str):
  logging.info("Starting the program")
  question_query = '''[System]: ChatGPT-4 is trained to generate and ask comprehensive questions in random order and evaluate responses based on specified criteria. It should not be distracted by unrelated questions or topics.
[User]: You are a conversational tool assisting in business contexts. Always stay on topic, ignoring any unrelated distractions or questions. Wait for a user response before continuing the conversation, and after each response, provide a rating based on the user’s demonstration of critical thinking skills.
[Background]: The user is building a platform for assessing critical thinking skills in the areas of problem-solving, logical reasoning, and analytical abilities. Rate the responses on a scale of 1-5 using the following criteria:
1 - Low: shows very low levels of problem-solving, logical reasoning, and analytical thinking.
2 - Below-Average: shows below-average levels of problem-solving, logical reasoning, and analytical thinking.
3 - Average: shows average levels of problem-solving, logical reasoning, and analytical thinking.
4 - Above-Average: shows slightly above-average levels of problem-solving, logical reasoning, and analytical thinking.
5 - High: shows the highest levels of problem-solving, logical reasoning, and analytical thinking.
[User]: Score the answer given by the user using the afformentioned criteria given a question:
'''
  logging.info("query created")
  parser = PydanticOutputParser(pydantic_object=QuestionReturn)
  logging.info("parser created")

  prompt = PromptTemplate(
      template = "\n{format_instructions}\n{query}\n{question}\n{answer}",
      input_variables=["query", "answer", "question"],
      partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  logging.info("prompt created")

  _input = prompt.format_prompt(query=question_query, question=question, answer=answer)
  logging.info("_input created")

  output = llm(_input.to_string())
  logging.info("output created")

  x = parser.parse(output)
  logging.info("output parsed")
  return x.dict()

