import streamlit as st
import chatbot_backend as chatbot

st.title('Hi, this is Chatbot Bob :sunglasses:')

if 'memory' not in st.session_state:
    st.session_state.memory = chatbot.memory()


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['text'])


input_text = st.chat_input('Chat with Bob here')
if input_text:
    with st.chat_message('user'):
        st.markdown(input_text)
    st.session_state.chat_history.append({'role': 'user', 'text': input_text})
    response = chatbot.converse(input_text=input_text, memory=st.session_state.memory)
    with st.chat_message('assistant'):
        st.markdown(response)
    st.session_state.chat_history.append({'role': 'assistant', 'text': response})