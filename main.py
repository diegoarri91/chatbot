import logging
import queue

from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from config import OPENAI_API_KEY, SYSTEM_MSG


logger = logging.getLogger(__name__)

st.title("💬 Chatbot")


def record(webrtc_ctx):
    print('just outside of while loop')
    sound_chunk = pydub.AudioSegment.empty()
    while True:
        if webrtc_ctx.audio_receiver:
            print("audio receiver set")
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound
                print("inside web", len(sound_chunk))

        else:
            logger.warning("AudioReceiver is not set. Abort.")
            break
    return sound_chunk


def get_ai_message(messages):
    # chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.7)
    # msg = chat_llm.invoke(messages)
    msg = AIMessage(content="gpt-3.5-turbo")
    return msg


def initialize_conversation():
    msg = get_ai_message([SystemMessage(content=SYSTEM_MSG)])
    st.session_state.messages = [msg]


def write_and_append_message(msg):
    st.chat_message(msg.type).write(msg.content)
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
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )
    sound_chunk = record(webrtc_ctx)

    print("end of loop", len(sound_chunk))

    if len(sound_chunk) > 0:
        filename = "./mashup.wav"
        sound_chunk.export(filename, format="mp3", tags={'user': 'diegote'})

if prompt := st.chat_input():
    human_msg = HumanMessage(content=prompt)
    write_and_append_message(human_msg)

    ai_msg = get_ai_message(st.session_state.messages + [SystemMessage(content=SYSTEM_MSG)])
    write_and_append_message(ai_msg)
