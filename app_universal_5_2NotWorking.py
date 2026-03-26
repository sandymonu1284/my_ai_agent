import streamlit as st
import pandas as pd
import plotly.express as px
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

# Import the provider brains
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# --- 1. TOOL DEFINITIONS ---

@tool
def smart_table_query(query_str: str):
    """
    Use this tool ONLY for the uploaded CSV data. 
    It handles filtering, sorting, or math using pandas query syntax.
    Example: 'Age > 25' or 'df.groupby("Category")["Sales"].sum()'
    """
    if 'df' not in st.session_state:
        return "Error: No CSV uploaded. Tell the user to upload a file first."
    
    df = st.session_state.df
    try:
        if "df." in query_str:
            result = eval(query_str)
        else:
            result = df.query(query_str)
        return str(result.head(20))
    except Exception as e:
        return f"Pandas Error: {e}. Ensure column names match exactly."

@tool
def generate_viz(chart_type: str, x: str, y: str, title: str = "Data Chart"):
    """
    Use this to create visual charts from the CSV.
    chart_type: 'bar', 'line', or 'scatter'
    """
    if 'df' not in st.session_state:
        return "Error: No data to plot."
    
    df = st.session_state.df
    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, title=title)
        else:
            fig = px.scatter(df, x=x, y=y, title=title)
        
        st.plotly_chart(fig, use_container_width=True)
        return f"Successfully displayed {chart_type} chart."
    except Exception as e:
        return f"Plotting Error: {e}"

# --- 2. SETUP & SIDEBAR ---

st.set_page_config(page_title="Hybrid Data & Web Agent", layout="wide")
st.title("📊 AI Analyst + 🌐 Web Search (2026)")

st.sidebar.header("1. Brain Configuration")
provider = st.sidebar.selectbox("Provider", ["Ollama (Local)", "OpenAI", "Anthropic", "Groq"])

llm = None
if provider == "Ollama (Local)":
    # Ensure you have the model pulled locally
    local_model = st.sidebar.selectbox("Model", ["llama3.2:3b", "qwen2.5:7b"])
    llm = ChatOllama(model=local_model, temperature=0)
else:
    api_key = st.sidebar.text_input(f"{provider} API Key", type="password")
    if provider == "OpenAI" and api_key:
        llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
    elif provider == "Anthropic" and api_key:
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=api_key, temperature=0)
    elif provider == "Groq" and api_key:
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)

st.sidebar.header("2. Local Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- 3. AGENT CORE ---

# FIX: Explicitly name the search tool so it matches the prompt
search_tool = DuckDuckGoSearchRun(name="duckduckgo_search")
tools = [search_tool, smart_table_query, generate_viz]

if uploaded_file is not None:
    # Use a check to prevent re-reading the file unnecessarily
    st.session_state.df = pd.read_csv(uploaded_file)
    with st.expander("Data Preview"):
        st.write(st.session_state.df.head(3))

if llm:
    column_list = list(st.session_state.df.columns) if 'df' in st.session_state else "No file uploaded"
    
    # Improved Prompt for better tool-calling reliability
    system_prompt = f"""You are a versatile AI Research Assistant.
    - If the user asks about the uploaded data, you MUST use 'smart_table_query' or 'generate_viz'.
    - If the user asks for real-time info, news, or facts not in the file, you MUST use 'duckduckgo_search'.
    - Current Data Columns: {column_list}
    - Always provide a final answer based on the tool outputs.
    """
    
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    
    user_input = st.chat_input("Ask about your data OR the web...")

    if user_input:
        st.chat_message("user").write(user_input)
        with st.spinner("Searching and Analyzing..."):
            try:
                # Running the agent
                response = agent.invoke({"messages": [("user", user_input)]})
                
                # Extracting the last AI message
                final_text = response["messages"][-1].content
                st.chat_message("assistant").write(final_text)
                
            except Exception as e:
                st.error(f"Agent Error: {e}")
else:
    st.info("Please provide an API Key or select Ollama to begin.")