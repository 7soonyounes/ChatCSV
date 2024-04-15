# Import necessary libraries.
import openai
import pandas as pd
import os 
import streamlit as st

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import AzureOpenAI
from dotenv import find_dotenv, load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI

api_key = os.getenv("AZURE_OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set")
    st.stop()

client = AzureChatOpenAI(
    azure_endpoint="https://openai-if-test.openai.azure.com/",
    api_key=api_key,
    api_version="2024-02-15-preview",
    model_name="gpt-4"
)

# Define a function to create Pandas DataFrame agent from a CSV file.
def create_pd_agent(filename: str):
    # Initiate a connection to the LLM from Azure OpenAI Service via LangChain.
    llm = AzureChatOpenAI(
        temperature=0,
        api_version="2024-02-15-preview", 
        azure_endpoint="https://openai-if-test.openai.azure.com/",
        model_name="gpt-4"
    )

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)

    # Create a Pandas DataFrame agent from the CSV file.
    return create_pandas_dataframe_agent(llm, df, verbose=False)

# Define a function to query the agent.
def query_pd_agent(agent, query):
    prompt = (
        """
        You must need to use matplotlib library if required to create a any chart.

        If the query requires creating a chart, please save the chart as "./chart_image/chart.png" and "Here is the chart:" when reply as follows:
        {"chart": "Here is the chart:"}

        If the query requires creating a table, reply as follows:
        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
        
        If the query is just not asking for a chart, but requires a response, reply as follows:
        {"answer": "answer"}
        Example:
        {"answer": "The product with the highest sales is 'Minions'."}
        
        Lets think step by step.

        Here is the query: 
        """
        + query
    )

    # Run the agent with the prompt.
    response = agent.run(prompt)

    # Return the response in string format.
    return response.__str__()
