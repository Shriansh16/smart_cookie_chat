from utils import *
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
import os
import streamlit as st
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone
load_dotenv()
#KEY = os.getenv("OPENAI_API_KEY")
KEY=st.secrets["OPENAI_API_KEY"]

pc = PineconeClient(
    api_key=st.secrets["PINECONE_API_KEY"]
)
index_name='smart-cookie-chatbot'
index=pc.Index(index_name)
model=OpenAIEmbeddings(api_key=KEY)

def find_match(input):
    vectorstore = Pinecone(
    index, model.embed_query,"text"
                       )
    result=vectorstore.similarity_search(
    input,  # our search query
    k=5  # return 5 most relevant docs
      )
    return result


def query_refiner(conversation, query):
  client = OpenAI(api_key=KEY)  
  chat_service = client.chat  
  response = chat_service.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You are a helpdesk chatbot on a website and your task is to assist visitors."},
          {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
      ],
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
  )
  return response.choices[0].message.content

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string