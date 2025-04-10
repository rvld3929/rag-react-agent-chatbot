import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from typing import Annotated
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain.agents import AgentExecutor, create_react_agent


_ = load_dotenv(find_dotenv())


st.set_page_config(
    page_title="Chat with Document",
    page_icon="💬",
)

if "retriever" not in st.session_state:
    st.warning("Please upload a document first!", icon="⚠️")
elif st.session_state.retriever is None:
    st.warning("Please upload a document first!", icon="⚠️")

st.title("Chat with Document")

@tool
def retriever_tool(query: Annotated[str, "A query string to retrieve relevant documents based on its content."]) -> str:
    """
    Retrieve relevant documents based on the provided query string.

    This tool leverages the retriever to search for and return the most relevant
    documents that match the content of the query. It is particularly useful for
    extracting information from indexed documents or datasets.

    Args:
        query (str): A query string to retrieve relevant documents based on its content.
    
    Returns:
        str: The retrieved documents as a string.
    """
    return st.session_state.retriever.invoke(query)

if "conversation" not in st.session_state:
    prompt = hub.pull("kwangvatar/react-chat")
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    tools = [retriever_tool]
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) 
    st.session_state.conversation = agent_executor

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if input := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.markdown(input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use the conversation object to invoke the agent with the input and chat history
            chat_output = st.session_state.conversation.invoke(
                {
                    "input": input,
                    "chat_history": st.session_state.memory.load_memory_variables({})["history"],
                }
            )
            response = st.markdown(chat_output["output"])
    st.session_state.memory.save_context({"input": input}, {"output": chat_output["output"]})
    st.session_state.messages.append({"role": "assistant", "content": chat_output["output"]})

