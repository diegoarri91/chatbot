import os
import tempfile

from aiortc.contrib.media import MediaRecorder
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import openai
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.title("💬 Chatbot")

OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')
SYSTEM_MSG = "You are a helpful assistant."


def get_ai_message(messages):
    chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.7)
    msg = chat_llm.invoke(messages)
    return msg


def initialize_conversation():
    msg = get_ai_message([SystemMessage(content=SYSTEM_MSG)])
    st.session_state.messages = [msg]


def write_and_append_message(msg):
    print("writing")
    st.chat_message(msg.type).write(msg.content)
    print("done writing")
    st.session_state.messages.append(msg)


def write_all_session_messages():
    for msg in st.session_state.messages:
        st.chat_message(msg.type).write(msg.content)


if "messages" not in st.session_state:
    initialize_conversation()
    write_all_session_messages()
else:
    write_all_session_messages()

with st.sidebar:
    tmp_dir = tempfile.gettempdir()
    audio_file_path = f"{tmp_dir}/tmp_bfec39bf-952a-4a02-8cbb-f1120ab3879e.wav"
    audio_prompt = None

    def audio_recorder() -> MediaRecorder:
        return MediaRecorder(audio_file_path, format="wav")

    webrtc_ctx = webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": False,
            "audio": True,
        },
        in_recorder_factory=audio_recorder,
        sendback_audio=False,
    )
    if webrtc_ctx.state.playing:
        st.write("Recording...")

    if os.path.exists(audio_file_path):
        audio_prompt = openai.Audio.transcribe(
            "whisper-1",
            open(audio_file_path, "rb"),
            api_key=OPENAI_API_KEY
        )["text"]
        os.remove(audio_file_path)

prompt = st.chat_input()

if audio_prompt is not None:
    prompt = audio_prompt

if prompt:
    human_msg = HumanMessage(content=prompt)
    write_and_append_message(human_msg)

    ai_msg = get_ai_message(st.session_state.messages + [SystemMessage(content=SYSTEM_MSG)])
    write_and_append_message(ai_msg)
