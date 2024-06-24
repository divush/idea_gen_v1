import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain_core.prompts import MessagesPlaceholder
import openai
from langchain_openai import ChatOpenAI

import streamlit as st

st.title("Idea Generator (v1)")

# welcome_message = """Enter your idea below"""
# st.write(st.secrets)
if "model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"

def get_download_stream():
    download_text = ""
    for message in st.session_state.chat_history:
        download_text += f"{message['role']}: {message['content']}\n\n"
    return download_text


with st.sidebar:
    option = st.selectbox("Select Model to use:", ("GPT 3.5", "GPT 4", "GPT 4o"))
    mapping_dict = {
        "GPT 4o": "gpt-4o",
        "GPT 4": "gpt-4",
        "GPT 3.5": "gpt-3.5-turbo",
    }
    st.success(f"Model selected: {option}")
    if "chat_history" not in st.session_state or st.session_state.model!=mapping_dict[option]:
        st.session_state.model=mapping_dict[option]
        st.session_state.chat_history = [{'role':'assistant', 'content':f"Model selected is {st.session_state.model}"}]
    
    st.download_button("Download chat", data=get_download_stream())


# openai.api_key = st.secrets["openai_api_key"]

DEBUG=False
MODEL = st.session_state.model

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [{'role':'assistant', 'content':f"Model selected is {MODEL}"}]

if "memory" not in st.session_state:
    memory = ConversationBufferMemory()
else:
    memory = st.session_state.memory

system_message = st.secrets['system_message']
system_message+="""
The current conversation history is: {history}

User input is : {input}
"""
llm = ChatOpenAI(model=MODEL, api_key=st.secrets["openai_api_key"])

template = ChatPromptTemplate.from_template(system_message)

parser = StrOutputParser()

chain = ConversationChain(llm=llm,
    memory=memory,
    verbose=DEBUG,
    prompt=template,
)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your idea here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.chat_history.append({'role':'user', 'content':prompt})
    
    with st.chat_message("assistant"):
        response = chain.predict(input=prompt, history=memory.chat_memory)
        st.session_state.chat_history.append({'role':'ai', 'content':response})
        st.markdown(response)
        
    st.session_state.memory = memory