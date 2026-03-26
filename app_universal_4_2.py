##THIS IS Addition to APP_UNIVERSAL_3.PY (FILE UPLOADER && DATA ANALYSIS USING NO TOOL PATH - Data-in-context approadch)
## If your CSV is small (under 100 rows), you don't need a tool at all. You can convert the dataframe to a string and inject it directly into the System Message
## This app allows you to upload a CSV file, which the agent can then analyze using Python code execution. You can also ask it to perform web searches. The agent will use the appropriate tool based on your input.
## Dangerous Code Execution: The PythonAstREPLTool allows the AI to run code on your computer. This is perfectly safe for your own local CSVs, but never use this in a public-facing website without a "Sandbox," as someone could theoretically ask the agent to delete files on your PC!
## To move away from PythonAstREPLTool, you essentially have two paths: The "No-Code" Path (passing data directly in the prompt) or the "Custom Tool" Path (writing specific Python functions for the agent to call).

import streamlit as st
import pandas as pd
import os
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
##from langchain_experimental.tools import PythonAstREPLTool

# Import the provider brains
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("📊 AI Data Analyst Agent (2026)")

# --- SIDEBAR: Configuration ---
st.sidebar.header("1. Choose Your Brain")
provider = st.sidebar.selectbox("Provider", ["Ollama (Local)", "OpenAI", "Anthropic", "Groq"])

llm = None
if provider == "Ollama (Local)":
    local_model = st.sidebar.selectbox("Model", ["llama3.2:3b", "qwen3:4b", "phi4-mini:latest"])
    llm = ChatOllama(model=local_model, temperature=0)
else:
    api_key = st.sidebar.text_input(f"{provider} API Key", type="password")
    if provider == "OpenAI" and api_key:
        llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
    elif provider == "Anthropic" and api_key:
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", anthropic_api_key=api_key, temperature=0)
    elif provider == "Groq" and api_key:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)

st.sidebar.header("2. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# --- DATA PROCESSING ---
tools = [DuckDuckGoSearchRun()] # Default tool

if uploaded_file is not None:
    # Read the file so we can show a preview and give it to the agent
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    # Create the Python Tool and 'inject' the dataframe into it
    # We name the variable 'df' so the agent knows how to refer to it
    ##python_tool = PythonAstREPLTool(locals={"df": df})
    ##python_tool.description = "A Python shell. Use this to execute python commands for data analysis on a dataframe named 'df'. Use 'df.head()' to see columns."
    ##tools.append(python_tool)
    st.sidebar.success("CSV Loaded! Agent can now analyze 'df'.")

# Create a summary of the data for the agent's memory
    data_summary = f"""
    You have access to a dataset with the following columns: {list(df.columns)}.
    Here is a sample of the data:
    {df.head(10).to_string()}
    """
    
    # Update the prompt/agent to include this context
    system_message = f"You are a data analyst. {data_summary}"
    agent = create_react_agent(llm, tools, prompt=system_message)

# --- AGENT EXECUTION ---
if llm:
    agent = create_react_agent(llm, tools)
    user_input = st.text_input("Ask about your data or search the web:")

    if user_input:
        with st.spinner("Analyzing..."):
            try:
                inputs = {"messages": [("user", user_input)]}
                result = agent.invoke(inputs)
                
                final_answer = result["messages"][-1].content
                st.markdown("### Agent Output")
                st.write(final_answer)
                
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please configure the model in the sidebar to start.")