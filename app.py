#UI for the chatbot (streamlit ui)
import streamlit as st
from rag_chain import agent

st.title("Anas's ChatBot")

#chat history init
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):

    with st.chat_message("user"):
        st.markdown(prompt)
    

    response = agent.invoke({"messages": st.session_state.messages + [{"role": "user", "content": prompt}]})
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer = response["messages"][-1].content

    with st.chat_message("assistant"):
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

