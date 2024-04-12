from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.agents.agent_types import AgentType

import pandas as pd
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import json

def main():
    load_dotenv(find_dotenv())

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Ask Ask")
    st.header("Ask 📈")

    file = st.file_uploader("Upload a CSV or excel file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)

        llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
        agent = create_pandas_dataframe_agent(llm, df, agent_type="openai-tools", verbose=True)

        
        # llm = AzureChatOpenAI(
        #     temperature=0,
        #     deployment_name="chat-endpoint",
        #     model_name="gpt-35-turbo"
        # )
        # agent = agent = create_pandas_dataframe_agent(
        #             llm,
        #             df,
        #             verbose=True,
        #             agent_type=AgentType.OPENAI_FUNCTIONS,
        #         )

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                resp = agent.invoke(user_question)["output"]
                st.write(resp)

if __name__ == "__main__":
    main()