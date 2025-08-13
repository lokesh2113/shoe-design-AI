#agents.py


from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming tokens to Streamlit"""

    def _init_(self, container):
        self.container = container
        self.text = ""
        self.word_buffer = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.word_buffer += token

        # Check if we have a complete word (ends with space or punctuation)
        if token.endswith((' ', '\n', '.', ',', '!', '?', ':', ';')):
            self.text += self.word_buffer
            self.container.markdown(self.text)
            self.word_buffer = ""

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        if self.word_buffer:  # Add any remaining buffer
            self.text += self.word_buffer
            self.container.markdown(self.text)


# Load your local LLaMA 3 model using Ollama
llm = Ollama(model="llama3.2:3b")

# Memory for both agents
designer_memory = ConversationBufferMemory(memory_key="chat_history")

# Agent 1: Conversational Shoe Designer
designer_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""
You are an expert AI shoe designer. Your goal is to ask the user insightful questions and help design the perfect shoe.
Ask detailed questions about:
  - shoe type (e.g. sneakers, boots, sandals)
  - color and finish (e.g. matte black, glossy red)
  - material (e.g. leather, mesh, rubber)
  - texture (e.g. ribbed, smooth, perforated)
  - accents or details (e.g. stitching, logos, patterns)
  - style or era (e.g. futuristic, retro)
  - camera or artistic style (e.g. 3D render, photorealistic)
Use the chat history to continue the conversation.
When you're done collecting all details, output this exact phrase on a new line: READY_TO_GENERATE
Dont show the chat history on the output.

Chat history:
{chat_history}
User: {input}
AI:
"""
)

# Agent 2: Prompt Generator for Stable Diffusion
prompt_generator_prompt = PromptTemplate(
    input_variables=["chat_history"],
    template="""
You are an assistant helping a shoe designer turn their conversation into an image generation prompt for ControlNet + Stable Diffusion.

Use the full chat history to describe:
  - shoe type (e.g. sneakers, boots, sandals)
  - color and finish (e.g. matte black, glossy red)
  - material (e.g. leather, mesh, rubber)
  - texture (e.g. ribbed, smooth, perforated)
  - accents or details (e.g. stitching, logos, patterns)
  - style or era (e.g. futuristic, retro)
  - camera or artistic style (e.g. 3D render, photorealistic)
- mention the word shoe in the prompt like start with (generate a shoe in)
- camera or artistic style should be the first word of prompt (e.g. 3D render.... , photorealistic.....)
- Keep the prompt concise: maximum 60 words.
- The description must be suitable for direct input into an image generation model.

Strict rules:
- ONLY output the image prompt. Do not add extra text, headers, commentary, instructions or bullet points.
- Use a single descriptive paragraph.

Chat History:
{chat_history}
"""
)

# Trend Agent: Compare trend data and conversation
trend_prompt = PromptTemplate(
    input_variables=["chat_history", "retrieved_data"],
    template="""
You are a trend analyst AI helping users learn about current shoe trends.

Compare the user's interest and questions (chat history) with the retrieved online trend data. Provide a concise, helpful summary highlighting how the trends relate to the user's design goals or preferences.

Chat History:
{chat_history}

Retrieved Trend Data:
{retrieved_data}

Answer:
"""
)

# Chains
designer_chain = LLMChain(llm=llm, prompt=designer_prompt, memory=designer_memory)
agent2_chain = LLMChain(llm=llm, prompt=prompt_generator_prompt)
trend_agent = LLMChain(llm=llm, prompt=trend_prompt)


# Agent 1 with streaming
def agent1_respond_stream(user_input, container):
    """Stream the response from agent 1 word by word"""
    callback_handler = StreamlitCallbackHandler(container)
    response = designer_chain.run(input=user_input, callbacks=[callback_handler])
    return callback_handler.text


# Agent 1 (non-streaming for compatibility)
def agent1_respond(user_input):
    response = designer_chain.run(input=user_input)
    return response


# Agent 2 with streaming
def agent2_generate_prompt_stream(container):
    """Stream the prompt generation"""
    chat_history_text = ""
    for msg in designer_memory.chat_memory.messages:
        role = "User" if msg.type == "human" else "AI"
        chat_history_text += f"{role}: {msg.content}\n"

    callback_handler = StreamlitCallbackHandler(container)
    response = agent2_chain.run(chat_history=chat_history_text, callbacks=[callback_handler])
    return callback_handler.text


# Agent 2 (non-streaming for compatibility)
def agent2_generate_prompt():
    chat_history_text = ""
    for msg in designer_memory.chat_memory.messages:
        role = "User" if msg.type == "human" else "AI"
        chat_history_text += f"{role}: {msg.content}\n"
    return agent2_chain.run(chat_history=chat_history_text)


# Trend Agent with streaming
def generate_trend_summary_stream(chat_history, retrieved_data, container):
    """Stream the trend summary generation"""
    callback_handler = StreamlitCallbackHandler(container)
    response = trend_agent.run(
        chat_history=chat_history,
        retrieved_data=retrieved_data,
        callbacks=[callback_handler]
    )
    return callback_handler.text


# Trend Agent (non-streaming for compatibility)
def generate_trend_summary(chat_history, retrieved_data):
    return trend_agent.run(chat_history=chat_history, retrieved_data=retrieved_data)