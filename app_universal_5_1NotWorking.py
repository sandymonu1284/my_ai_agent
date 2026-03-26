# THIS IS Addition to APP_UNIVERSAL_3.PY
# Added FILE UPLOADER && 
# Added DATA ANALYSIS USING CUSTOM "SMART TABLE" TOOL
# Added Graph Plotting Capabilities using Plotly
## This app allows you to upload a CSV file, which the agent can then analyze using Smart Table. You can also ask it to perform web searches. The agent will use the appropriate tool based on your input.
##Removed PythonAstREPLTool and replaced it with a custom tool 'analyze_csv' that provides specific data analysis functions. Also added a 'plot_data' tool for visualization and an 'export_results' tool to download filtered data.
## Dangerous Code Execution: The PythonAstREPLTool allows the AI to run code on your 
## PythonAstREPLTool is perfectly safe for your own local CSVs, but never use this in a public-facing website without a "Sandbox," as someone could theoretically ask the agent to delete files on your PC!
## To move away from PythonAstREPLTool, you essentially have two paths: The "No-Code" Path (passing data directly in the prompt) or the "Custom Tool" Path (writing specific Python functions for the agent to call).

import streamlit as st
import pandas as pd
import io
import plotly.express as px
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

# --- PROVIDER BRAINS ---
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

st.set_page_config(page_title="Advanced Data Analyst", layout="wide")
st.title("📊 Smart Data Analyst Agent")

# --- TOOL DEFINITIONS ---

@tool
def smart_table_tool(operation: str, column: str = None, value: str = None):
    """
    Handles data manipulation without raw code.
    Operations: 
    - 'summary': Get statistics (mean, max, etc.) for all columns.
    - 'filter': Show rows where 'column' contains 'value'.
    - 'sort': Sort data by 'column' in descending order.
    - 'unique': List unique values in a 'column'.
    """
    if 'df' not in st.session_state:
        return "No data loaded. Please upload a CSV first."
    
    df = st.session_state.df
    try:
        if operation == "summary":
            return df.describe(include='all').to_string()
        
        elif operation == "filter" and column and value:
            # Flexible filtering (works for strings and numbers)
            mask = df[column].astype(str).str.contains(value, case=False, na=False)
            filtered_df = df[mask]
            st.session_state.last_filtered_df = filtered_df # Save for export
            return f"Filtered to {len(filtered_df)} rows. Preview:\n{filtered_df.head(10).to_string()}"
        
        elif operation == "sort" and column:
            sorted_df = df.sort_values(by=column, ascending=False)
            return sorted_df.head(10).to_string()
        
        elif operation == "unique" and column:
            return f"Unique values in {column}: {df[column].unique().tolist()[:20]}"
            
        return "Unknown operation. Use 'summary', 'filter', 'sort', or 'unique'."
    except Exception as e:
        return f"Error processing table: {str(e)}"

@tool
def plotting_tool(chart_type: str, x_axis: str, y_axis: str, title: str = "Agent Chart"):
    """
    Creates visual charts. 
    chart_type: 'bar', 'line', or 'scatter'.
    x_axis: The column name for the horizontal axis.
    y_axis: The column name for the vertical axis (numeric).
    """
    if 'df' not in st.session_state:
        return "No data loaded."
    
    df = st.session_state.df
    try:
        st.write(f"### {title}")
        if chart_type == "bar":
            fig = px.bar(df, x=x_axis, y=y_axis, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x_axis, y=y_axis, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=title)
        else:
            return "Unsupported chart_type. Use 'bar', 'line', or 'scatter'."
        
        st.plotly_chart(fig, use_container_width=True)
        return f"Successfully displayed a {chart_type} chart of {y_axis} vs {x_axis}."
    except Exception as e:
        return f"Error creating plot: {str(e)}"

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

# --- DATA PROCESSING & AGENT ---
tools = [DuckDuckGoSearchRun(), smart_table_tool, plotting_tool]

if uploaded_file is not None:
    # Use session_state so tools can access the DF
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)
    
    df = st.session_state.df
    st.write("### Data Preview", df.head())
    st.sidebar.success(f"CSV Loaded! Columns: {len(df.columns)}")

    if llm:
        # Give the agent context about the data in the system message
        prompt_context = f"""You are a helpful Data Analyst. 
        The user has uploaded a dataset with these columns: {list(df.columns)}.
        Use 'smart_table_tool' for filtering/sorting and 'plotting_tool' for charts.
        Always verify column names from the provided list before using tools.
        """
        
        agent = create_react_agent(llm, tools, state_modifier=prompt_context)
        
        user_input = st.text_input("Ask about your data (e.g., 'Show me a bar chart of Sales by Date' or 'Filter for Category: Electronics'):")

        if user_input:
            with st.spinner("Agent is working..."):
                try:
                    result = agent.invoke({"messages": [("user", user_input)]})
                    st.markdown("### Agent Response")
                    st.write(result["messages"][-1].content)
                except Exception as e:
                    st.error(f"Execution Error: {e}")
else:
    st.info("Please upload a CSV file and configure your model to start.")