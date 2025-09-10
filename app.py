from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
# When creating an intial prompt, it will follow the prompt template
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os 
from dotenv import load_dotenv

# Load environment variables from .env (for HUGGINGFACEHUB_API_TOKEN, etc.)
load_dotenv()



## Prompt template

chat_prompt=ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are a concise symptom checker. Based on the user's described symptoms, return only: "
            "- Top 3 likely conditions (most to least likely), each 1 short line. "
            "- Optional: one line starting with 'Urgent:' if symptoms suggest emergency care. "
            "Keep it under 5 lines total. Do not add extra explanations, disclaimers, or Q&A formatting."
        )),
        ("user", "Symptoms: {question}")
    ]
)

# Follow-up mode: give tailored guidance when users ask further questions
chat_prompt_followup=ChatPromptTemplate.from_messages(
    [
        ("system", (
            "You are a careful medical triage assistant. Provide a personalized, practical reply: "
            "- 1–2 brief lines on the most likely cause, referencing the user's details if present. "
            "- 2–4 specific self-care steps tailored to the situation. "
            "- 'See a clinician if:' with 1–3 red flags. "
            "Stay under 8 lines total. Avoid definitive diagnosis and medical jargon."
        )),
        ("user", "Follow-up: {question}")
    ]
)


## streamlit app

st.set_page_config(page_title="BeMyDoc • Symptom Checker", page_icon="🩺", layout="wide")
st.title("🩺 BeMyDoc")

# Minimal UI polish
st.markdown(
    """
    <style>
      .stChatMessage .stMarkdown p { margin-bottom: 0.25rem; }
      .small-note { color: #9aa0a6; font-size: 0.9rem; }
      .block-container { max-width: 900px; margin: 0 auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar helpers
with st.sidebar:
    st.subheader("How to use")
    st.write("Describe your symptoms. You'll get top 3 likely conditions and an urgent-care note if needed.")
    st.divider()
    if st.button("Clear chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Not a medical diagnosis. For emergencies, call local services.")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🩺"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])  

# Chat input
prefill = st.session_state.pop("prefill", None) if "prefill" in st.session_state else None
user_input = st.chat_input(
    placeholder=prefill or "Describe your symptoms (e.g., 'fever, cough, sore throat 2 days')",
    key="chat_input",
)

# Hugging Face via LangChain
endpoint = HuggingFaceEndpoint(
    repo_id=os.getenv("HUGGINGFACE_MODEL", "HuggingFaceH4/zephyr-7b-beta"),
    temperature=0.2,
    max_new_tokens=64,
    do_sample=False,
    timeout=300,
)
llm = ChatHuggingFace(llm=endpoint)
output_parser = StrOutputParser()

if user_input:
    st.chat_message("user", avatar="🧑").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Choose concise list for first turn, more personalized guidance for follow-ups
    has_prior_assistant = any(m["role"] == "assistant" for m in st.session_state.messages[:-1])
    prompt_to_use = chat_prompt_followup if has_prior_assistant else chat_prompt

    chain = prompt_to_use | llm | output_parser
    response = chain.invoke({"question": user_input})
    st.chat_message("assistant", avatar="🩺").markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
## Langchain
