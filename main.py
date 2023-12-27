from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import streamlit as st

from config import OPENAI_API_KEY, SYSTEM_MSG


def get_ai_message(messages):
    chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.7)
    msg = chat_llm.invoke(messages)
    return msg


def write_and_append_message(msg):
    st.chat_message(msg.type).write(msg.content)
    st.session_state.messages.append(msg)


def write_all_session_messages():
    for msg in st.session_state.messages:
        st.chat_message(msg.type).write(msg.content)


def initialize_conversation():
    msg = get_ai_message([SystemMessage(content=SYSTEM_MSG)])
    st.session_state.messages = [msg]


st.title("💬 Chatbot")

if "messages" in st.session_state:
    write_all_session_messages()
else:
    initialize_conversation()
    write_all_session_messages()


if prompt := st.chat_input():
    human_msg = HumanMessage(content=prompt)
    write_and_append_message(human_msg)

    ai_msg = get_ai_message(st.session_state.messages + [SystemMessage(content=SYSTEM_MSG)])
    write_and_append_message(ai_msg)
