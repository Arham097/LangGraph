import streamlit as st
from langraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid




# --------------------------Utility Function----------------------------------
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    t_id = generate_thread_id()
    st.session_state["thread_id"] = t_id
    st.session_state["messages_history"] = []
    add_threads(st.session_state["thread_id"])

def add_threads(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_messages(thread_id):
    return chatbot.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]

# --------------------------Session Setup -------------------------------------
if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

add_threads(st.session_state["thread_id"])

# -------------------------- Sidebar -------------------------------------

st.sidebar.title('LangGraph Chatbot')
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header('My Conversation history')

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        messages = load_messages(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role": role, "content": message.content})
        st.session_state["messages_history"] = temp_messages

# -------------------------- MAIN UI -------------------------------------

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message here...")
if user_input:

    st.session_state["messages_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config= CONFIG,
                stream_mode="messages"
            )
        )

    st.session_state["messages_history"].append({"role": "assistant", "content": ai_message})