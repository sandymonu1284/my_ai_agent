import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent # The modern replacement

st.title("🦙 Modern Local Agent (2026)")

# 1. Setup the Brain
##llm = ChatOllama(model="llama3.2:3b", temperature=0) ##working
##llm = ChatOllama(model="phi4-mini:latest", temperature=0) ##Deployed but Not Working
llm = ChatOllama(model="qwen3:4b", temperature=0) ## working
##llm = ChatOllama(model="gemma3:4b", temperature=0) ##NOT COMPATIBLE
##llm = ChatOllama(model="phi4:14b", temperature=0) ##NOT COMPATIBLE

# 2. Setup Tools
tools = [DuckDuckGoSearchRun()]

# 3. Create the Agent (No Executor needed!)
# This creates a 'graph' that knows how to use tools
agent = create_react_agent(llm, tools)

# 4. Interface
user_input = st.text_input("Ask your local agent a question:")

if user_input:
    with st.spinner("Analyzing..."):
        # LangGraph uses 'invoke' and tracks 'messages'
        inputs = {"messages": [("user", user_input)]}
        result = agent.invoke(inputs)
        
        # The last message in the list is the AI's final answer
        final_answer = result["messages"][-1].content
        st.write(final_answer)