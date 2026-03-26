##THIS IS Addition to APP_UNIVERSAL_1.PY ALONG WITH OTHER LLM SELECTORS (OLLAMA, CLAUDE, OPENAI AND GORQ), ##
## OPTION TO SELECT 3 OTHER TYPES OF OLLAMA##

import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

# Import the provider brains
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

st.set_page_config(page_title="Universal AI Agent", layout="wide")
st.title("🌐 Universal Analytics Agent (2026)")

# --- SIDEBAR: Model Selection ---
st.sidebar.header("Configuration")
provider = st.sidebar.selectbox(
    "Choose Provider", 
    ["Ollama (Local)", "OpenAI", "Anthropic", "Groq"]
)

llm = None

# 1. Logic for Ollama (Local)
if provider == "Ollama (Local)":
    # Specific dropdown for your requested local models
    local_model = st.sidebar.selectbox(
        "Select Local Model", 
        ["llama3.2:3b", "qwen3:4b", "phi4-mini:latest"]
    )
    st.sidebar.info(f"Ensure you have run 'ollama pull {local_model}' in your terminal.")
    llm = ChatOllama(model=local_model, temperature=0)

# 2. Logic for Cloud Providers
else:
    api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")
    
    if provider == "OpenAI":
        model_choice = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
        if api_key:
            llm = ChatOpenAI(model=model_choice, api_key=api_key, temperature=0)
            
    elif provider == "Anthropic":
        model_choice = st.sidebar.selectbox("Model", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"])
        if api_key:
            llm = ChatAnthropic(model=model_choice, anthropic_api_key=api_key, temperature=0)
            
    elif provider == "Groq":
        model_choice = st.sidebar.selectbox("Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
        if api_key:
            llm = ChatGroq(model=model_choice, groq_api_key=api_key, temperature=0)

# --- AGENT EXECUTION ---
if llm:
    # Setup Tools (DuckDuckGo for web search)
    tools = [DuckDuckGoSearchRun()]

    # Create the Agent using LangGraph
    agent = create_react_agent(llm, tools)

    # Main Chat Interface
    user_input = st.text_input("Ask your agent a question (e.g., 'What is the current price of Gold?'):")

    if user_input:
        with st.spinner(f"Agent is processing using {provider}..."):
            try:
                # The LangGraph 'invoke' pattern
                inputs = {"messages": [("user", user_input)]}
                result = agent.invoke(inputs)
                
                # Extract and display the final response
                final_answer = result["messages"][-1].content
                st.markdown("---")
                st.markdown("### Agent Output")
                st.write(final_answer)
                
            except Exception as e:
                # Handle common 'Tool Support' errors gracefully
                if "does not support tools" in str(e):
                    st.error(f"The model '{local_model}' does not support tool-calling. Try llama3.2:3b or qwen3:4b.")
                else:
                    st.error(f"An error occurred: {e}")
else:
    st.warning("Please provide an API Key or select a Local Model to begin.")