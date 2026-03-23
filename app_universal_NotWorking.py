import streamlit as st
##from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun

# Import the different "Brains"
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

st.title("🔎 Universal AI Agent")

# Sidebar settings
provider = st.sidebar.selectbox("Choose Provider", ["OpenAI", "Anthropic", "Groq"])
api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")


### 1. Setup the Web Interface
###st.title("🔎 My First Analytics Agent")
###api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:

# Logic to pick the right brain based on your choice
    if provider == "OpenAI":
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
    elif provider == "Anthropic":
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", anthropic_api_key=api_key)
    elif provider == "Groq":
        llm = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key)

    # ... (Rest of the search tool and agent code remains exactly the same!)

    ## 2. Initialize the "Brain" (LLM)
    ##llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

    # 3. Give the Agent a "Tool" (Web Search)
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]

    # 4. Create the Agent
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )

    # 5. The User Interaction
    user_input = st.text_input("What analytics task should I perform?")
    if st.button("Run Agent"):
        with st.spinner("Thinking and searching..."):
            response = agent.run(user_input)
            st.write(response)
else:
    st.info("Please enter your OpenAI API key in the sidebar to begin.")