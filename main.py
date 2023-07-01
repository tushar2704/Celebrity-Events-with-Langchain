#Author Tushar Aggarwal(https://www.tushar-aggarwal.com/)
# Celebrity & Events with LangChain
# Impoting the required libraries

import os
from langchain.llms import OpenAI
from jsonschema.exceptions import ValidationError
#custom
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
# Page Configration
# Application title and body
st.set_page_config(page_title="🤖Celebrity & Events with LangChain🤖",
                   page_icon="🤖",
                   layout='wide')
# Title of application
st.title("🤖Celebrity & Events with LangChain")
st.markdown("### By [Tushar Aggarwal](https://www.tushar-aggarwal.com/)")

# App body
apikey=st.text_input("Your API KEY, its wont store it, so it is safe")

if not api_key:
    st.write("Please enter your OpenAI API key first. Don't worry this will not be stored.")
    st.stop()
  
input_text = st.text_input("Enter Celebrity name & please wait some time after entering")
# OpneAI config
os.environ["OPENAI_API_KEY"] =apikey

#Prompt Templates variables
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name} celebrity")


second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="When was {person} born")
third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around the world near date {dob} ")

#Memory avriables
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')


#LLM chaining
llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm, prompt=first_input_prompt, verbose=True,
               output_key='person', memory=person_memory)
chain2=LLMChain(llm=llm, prompt=second_input_prompt, verbose=True,
               output_key='dob',memory=dob_memory)
chain3=LLMChain(llm=llm, prompt=third_input_prompt, verbose=True,
               output_key='description',memory=descr_memory)
parent_chain =SequentialChain(chains=[chain,chain2,chain3], verbose=True,
                              input_variables=['name'],
                              output_variables=['person','dob','description'])

if input_text:
    st.write(parent_chain({'name':input_text}))
    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
































