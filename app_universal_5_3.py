import uuid  # <--- MUST BE AT THE TOP
import streamlit as st
import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field

# Core LangChain/LangGraph Imports
from langchain_core.tools import Tool
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

# Provider Brains
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# --- 1. TOOL DEFINITIONS ---
# 1. Define the schema explicitly
class SearchInput(BaseModel):
    query: str = Field(description="The search query for DuckDuckGo")

# Use a clean wrapper for the search tool
def run_ddg_search(query: str) -> str:
    search = DuckDuckGoSearchRun()
    return search.run(query)

search_tool = Tool(
    name="duckduckgo_search",
    func=run_ddg_search,
    description="Search the web for real-time news, stock prices, or general facts.",
    args_schema=SearchInput
)



@tool
def smart_table_query(query_str: str):
    """
    Query the uploaded CSV data using pandas query syntax.
    """
    if 'df' not in st.session_state:
        return "Error: No CSV uploaded."
    df = st.session_state.df
    try:
        # Check for direct df access or query
        result = eval(query_str) if ("df." in query_str or "df[" in query_str) else df.query(query_str)
        return str(result.head(10))
    except Exception as e:
        return f"Pandas Error: {e}"

@tool
def generate_viz(chart_type: str, x: str, y: str, title: str = "Data Chart"):
    """
    Create 'bar', 'line', or 'scatter' charts from the CSV.
    """
    if 'df' not in st.session_state:
        return "Error: No data."
    df = st.session_state.df
    try:
        if chart_type == "bar": fig = px.bar(df, x=x, y=y, title=title)
        elif chart_type == "line": fig = px.line(df, x=x, y=y, title=title)
        else: fig = px.scatter(df, x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
        return f"Displayed {chart_type} chart."
    except Exception as e:
        return f"Plotting Error: {e}"

# --- 2. STREAMLIT UI & STATE ---

st.set_page_config(page_title="AI Analyst 2026", layout="wide")
st.title("📊 AI Analyst + 🌐 Web Search")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.header("Settings")
provider = st.sidebar.selectbox("Provider", ["Ollama (Local)", "OpenAI", "Anthropic", "Groq"])

llm = None
if provider == "Ollama (Local)":
    m = st.sidebar.selectbox("Model", ["llama3.2:3b", "qwen3:4b", "phi4-mini:latest"])
    llm = ChatOllama(model=m, temperature=0)
else:
    key = st.sidebar.text_input(f"{provider} API Key", type="password")
    if provider == "OpenAI" and key: llm = ChatOpenAI(model="gpt-4o", api_key=key)
    elif provider == "Anthropic" and key: llm = ChatAnthropic(model="claude-3-5-sonnet-latest", api_key=key)
    elif provider == "Groq" and key: llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=key)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)

# --- 3. EXECUTION ---

for msg_type, content in st.session_state.messages:
    st.chat_message(msg_type).write(content)

if llm:
    col_info = list(st.session_state.df.columns) if 'df' in st.session_state else "None"
    prompt = f"Assistant. Tools: 'smart_table_query' (CSV), 'duckduckgo_search' (Web), 'generate_viz' (Charts). Cols: {col_info}"
    
    agent = create_react_agent(llm, [search_tool, smart_table_query, generate_viz], prompt=prompt)
    
    user_input = st.chat_input("Ask me anything...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append(("user", user_input))
        
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent.invoke({"messages": [("user", user_input)]}, config=config)
            ans = result["messages"][-1].content
            st.chat_message("assistant").write(ans)
            st.session_state.messages.append(("assistant", ans))