import streamlit as st
from AgenticRag import return_output
import uuid

# Set Streamlit page configuration
st.set_page_config(page_title="Conversational Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Canada Immegration Chatbot")

# Initialize session state for thread ID and messages
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
if prompt := st.chat_input("Ask me anything..."):
    # Display user's question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = return_output(prompt, st.session_state.thread_id)
            st.markdown(response)

    # Append chatbot response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
