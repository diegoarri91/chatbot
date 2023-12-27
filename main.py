from uuid import uuid4 as uuid

from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
import streamlit as st

from config import OPENAI_API_KEY, SYSTEM_MSG
from file_chat_history import FileChatMessageHistory

# USER = None
USER = "test_user"
# st.session_state.user = str(uuid())


def get_message(messages):
    # chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.7)
    # msg = chat_llm.invoke(messages)
    msg = AIMessage(content="Initial AI message") if len(messages) == 1 else AIMessage(content=messages[-2].content)
    return msg


def write_and_append_message(msg):
    st.session_state.messages.append(msg)
    st.chat_message(msg.type).write(msg.content)
    st.session_state.storage.add_message(msg)


def write_all_session_messages():
    for msg in st.session_state.messages:
        st.chat_message(msg.type).write(msg.content)


def initialize_conversation():
    st.session_state.storage.clear()
    msg = get_message([SystemMessage(content=SYSTEM_MSG)])
    st.session_state.messages = [msg]
    st.session_state.storage.add_message(msg)


st.title("💬 Chatbot")

# TODO. add voice recording

if "messages" in st.session_state:
    write_all_session_messages()
else:
    st.session_state.storage = FileChatMessageHistory(USER)
    if st.session_state.storage.has_history:
        st.session_state.messages = st.session_state.storage.messages
    else:
        initialize_conversation()
    write_all_session_messages()


if prompt := st.chat_input():
    msg = HumanMessage(content=prompt)
    write_and_append_message(msg)

    msg = get_message(st.session_state.messages + [SystemMessage(content=SYSTEM_MSG)])
    write_and_append_message(msg)
