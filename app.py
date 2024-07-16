import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from utils1 import *

# Load environment variables
load_dotenv()
#KEY = os.getenv("OPENAI_API_KEY")
KEY=st.secrets["OPENAI_API_KEY"]

# Streamlit setup
st.subheader("HELPDESK CHAT")

# Initialize session state variables
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=KEY)

# Initialize conversation memory
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Define prompt templates
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context,
and if the answer is not contained within the text below, say 'I don't know' and while answering please don't use the sentences like 'Based on the provided context'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Create conversation chain
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Container for chat history
response_container = st.container()
# Container for text box
text_container = st.container()

with text_container:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            #st.subheader("Refined Query:")
            #st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
        
        # Append the new query and response to the session state
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Display chat history
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
