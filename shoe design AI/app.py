#app.py


import streamlit as st
from agents import (
    agent1_respond_stream, agent1_respond,
    agent2_generate_prompt_stream, agent2_generate_prompt,
    generate_trend_summary_stream, generate_trend_summary
)
from generate import generate_image
from retriever import search_duckduckgo

st.set_page_config(page_title="AI Shoe Designer", layout="wide")

st.title("👟 AI Shoe Designer")
st.write("Talk to an AI to design your custom shoe.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "prompt_generated" not in st.session_state:
    st.session_state.prompt_generated = False

if "final_prompt" not in st.session_state:
    st.session_state.final_prompt = None

if "image_path" not in st.session_state:
    st.session_state.image_path = None

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Trend keyword matching
trend_keywords = ["trend", "trendy", "trending", "latest", "what's hot", "popular", "hot right now"]


def is_trend_query(text):
    return any(keyword in text.lower() for keyword in trend_keywords)


# User input
user_input = st.chat_input("Describe your shoe idea or ask about trends...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if is_trend_query(user_input):
        with st.spinner("Searching for trends..."):
            retrieved_data = search_duckduckgo(user_input, max_results=5)

        chat_history_text = ""
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "AI"
            chat_history_text += f"{role}: {msg['content']}\n"

        # Create container for streaming response
        with st.chat_message("assistant"):
            response_container = st.empty()

        # Stream the trend summary
        trend_summary = generate_trend_summary_stream(
            chat_history_text,
            retrieved_data,
            response_container
        )

        st.session_state.messages.append({"role": "assistant", "content": trend_summary})

    else:
        # Create container for streaming response
        with st.chat_message("assistant"):
            response_container = st.empty()

        # Stream the AI response
        ai_response = agent1_respond_stream(user_input, response_container)

        if "ready_to_generate" in ai_response.lower():
            st.session_state.prompt_generated = True
            display_response = "🧵 Thanks for your creativity and patience — your dream shoe is ready to step into reality!"
            response_container.markdown(display_response)
            st.session_state.messages.append({"role": "assistant", "content": display_response})
        else:
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Generate prompt and image
if st.session_state.prompt_generated and not st.session_state.final_prompt:
    st.divider()
    st.subheader("🎨 Generating your AI shoe design...")

    with st.spinner("Generating prompt..."):
        # You can also stream the prompt generation if desired
        final_prompt = agent2_generate_prompt()
        st.session_state.final_prompt = final_prompt
        st.write(f"*Generated Prompt:* {final_prompt}")

    with st.spinner("Generating image..."):
        image_path = generate_image(final_prompt)
        st.session_state.image_path = image_path

# Display final image
if st.session_state.image_path:
    st.image(st.session_state.image_path, caption="Your AI-designed shoe")