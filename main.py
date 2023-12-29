import logging
import os
import queue
from pathlib import Path

import threading

from aiortc.contrib.media import MediaRecorder
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
import openai
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from config import OPENAI_API_KEY, SYSTEM_MSG

RECORD_DIR = Path("./")
RECORD_DIR.mkdir(exist_ok=True)

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

from pydub import AudioSegment
def transcribe(audio_segment: AudioSegment) -> str:
    """
    Transcribe an audio segment using OpenAI's Whisper ASR system.

    Args:
        audio_segment (AudioSegment): The audio segment to transcribe.

    Returns:
        str: The transcribed text.
    """
    import os
    import openai
    import tempfile
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", dir="./", delete=False)
    audio_segment.export(tmpfile.name, format="wav")
    print("open ai call")
    answer = openai.Audio.transcribe(
        "whisper-1",
        tmpfile,
        api_key=OPENAI_API_KEY
    )["text"]
    print("done open ai call")
    tmpfile.close()
    os.remove(tmpfile.name)
    # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
    #     audio_segment.export(tmpfile.name, format="wav")
    #     answer = openai.Audio.transcribe(
    #         "whisper-1",
    #         tmpfile,
    #         api_key=OPENAI_API_KEY
    #     )["text"]
    #     tmpfile.close()
    #     os.remove(tmpfile.name)
    return answer


def get_ai_message(messages):
    # chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.7)
    # msg = chat_llm.invoke(messages)
    msg = AIMessage(content="gpt-3.5-turbo")
    return msg


def initialize_conversation():
    msg = get_ai_message([SystemMessage(content=SYSTEM_MSG)])
    st.session_state.messages = [msg]


def write_and_append_message(msg):
    print("writing")
    st.chat_message(msg.type).write(msg.content)
    print("done writing")
    st.session_state.messages.append(msg)

text_prompt = None

def write_all_session_messages():
    for msg in st.session_state.messages:
        st.chat_message(msg.type).write(msg.content)

if "messages" not in st.session_state:
    initialize_conversation()
    write_all_session_messages()
else:
    write_all_session_messages()

with st.sidebar:
    # webrtc_ctx = webrtc_streamer(
    #     key="speech-to-text",
    #     mode=WebRtcMode.SENDONLY,
    #     audio_receiver_size=1024,
    #     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    #     media_stream_constraints={"video": False, "audio": True},
    # )
    # sound_chunk = record(webrtc_ctx)
    #
    # print("end of loop", len(sound_chunk))
    #
    # if len(sound_chunk) > 0:
    #     print("transcribing")
    #     text_prompt = transcribe(sound_chunk)
    #     print("done transcribing")
    #     print(text_prompt, type(text_prompt))
    # if st.button("Button"):
    #     text_prompt = "transcript"

    prefix = "test"
    audio_file_path = RECORD_DIR / f"{prefix}_input.wav"

    def in_recorder_factory() -> MediaRecorder:
        return MediaRecorder(str(audio_file_path), format="wav")

    webrtc_ctx = webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": False,
            "audio": True,
        },
        in_recorder_factory=in_recorder_factory,
    )
    if webrtc_ctx.state.playing:
        st.write("Recording...")

    if audio_file_path.exists():
        audio_file = audio_file_path.open("rb")
        text_prompt = openai.Audio.transcribe(
            "whisper-1",
            audio_file,
            api_key=OPENAI_API_KEY
        )["text"]
        os.remove(audio_file_path)

# import time
# time.sleep(5)
print("out of sidebar")
print(threading.active_count())
print(threading.current_thread())
print(threading.get_ident())
print(text_prompt)

if text_prompt is not None:
    print("entering prompt if")
    human_msg = HumanMessage(content=str(text_prompt))
    print("human write and append")
    write_and_append_message(human_msg)
    print("done human")

    ai_msg = get_ai_message(st.session_state.messages + [SystemMessage(content=SYSTEM_MSG)])
    write_and_append_message(ai_msg)
    print("done ai")
else:
    if prompt := st.chat_input():
        print("chat input")
        human_msg = HumanMessage(content=prompt)
        write_and_append_message(human_msg)

        ai_msg = get_ai_message(st.session_state.messages + [SystemMessage(content=SYSTEM_MSG)])
        write_and_append_message(ai_msg)
print("END")