import streamlit as st
from langraph_backend import chatbot

st.title("LangGraph Chatbot with Streamlit UI")

CONFIG = {"configurable": {"thread_id": "1"}}


if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message here...")
if user_input:

    st.session_state["messages_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    response = chatbot.invoke({"messages": [user_input]}, config=CONFIG)
    ai_message = response["messages"][-1].content
    st.session_state["messages_history"].append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)