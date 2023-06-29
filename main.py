# Celebrity Search with LangChain

#integrating OpenAI API
import os
from langchain.llms import OpenAI
#custom
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
st.title("Celebrity & Events")
apikey=st.text_input("Your API KEY")

    
if apikey:
    input_text = st.text_input("Your query")

# OpneAI
os.environ["OPENAI_API_KEY"] =apikey

#Prompt Templates
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name} celebrity")


second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="When was{person} born")
third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happed around{dob} around the world")

#Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')


#LLM chain
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


































