##THIS IS APP_ollama.PY ALONG WITH OTHER LLM SELECTOR (OLLAMA, CLAUDE, OPENAI AND GORQ)##
import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

# Import the different provider brains
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

# Initialize variables
llm = None
api_key = None

# Logic to handle different providers
if provider == "Ollama (Local)":
    model_name = st.sidebar.text_input("Local Model Name", value="llama3.2:3b")
    st.sidebar.info("Make sure Ollama is running on your PC!")
    llm = ChatOllama(model=model_name, temperature=0)

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

# --- AGENT LOGIC ---
if llm:
    # 1. Setup Tools
    tools = [DuckDuckGoSearchRun()]

    # 2. Create the Agent using LangGraph
    # (This works the same regardless of which LLM we picked above)
    agent = create_react_agent(llm, tools)

    # 3. User Interface
    user_input = st.text_input("What analytics or search task can I help with?")

    if user_input:
        with st.spinner(f"Agent is thinking using {provider}..."):
            try:
                # LangGraph follows the 'messages' format
                inputs = {"messages": [("user", user_input)]}
                result = agent.invoke(inputs)
                
                # Show the output
                final_answer = result["messages"][-1].content
                st.markdown("### Agent Response:")
                st.write(final_answer)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Please provide the necessary API Key or ensure Ollama is configured to start.")