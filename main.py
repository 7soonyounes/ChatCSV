from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import os
import streamlit as st
import matplotlib.pyplot as plt
import io
import contextlib

def main():
    def extract_code(response):
                if '```python' in response:
                    start = response.find('```python') + len('```python')
                    end = response.find('```', start)
                    return response[start:end].strip()
                return None

    def execute_code(code, df):
        local_vars = {'df': df, 'plt': plt, 'pd': pd}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(code, globals(), local_vars)
        output = buffer.getvalue()
        return output.strip()

    load_dotenv(find_dotenv())

    if not os.getenv("AZURE_OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set")
        st.stop()

    st.set_page_config(page_title="ChatCSV")
    st.header("Ask CSV 📈")

    file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if file is not None:
            
            st.info("CSV Uploaded Successfully")
            df = pd.read_csv(file)
            st.dataframe(df, use_container_width=True)
            
            llm = AzureChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', api_version="2023-03-15-preview", azure_endpoint="https://imagepdfprocessing.openai.azure.com/", azure_deployment="chat-endpoint")
            agent = create_pandas_dataframe_agent(llm, df, agent_type="openai-tools", verbose=True)

            user_question = st.text_input("Ask a question about your CSV: ")
            if user_question:
                instruction = "Please say 'Hello!' before reply to the query. You must need to use matplotlib library to create any graph. Here is the query: "
                prompt = instruction + user_question
                with st.spinner(text="In progress..."):
                    response = agent.invoke(prompt, handle_parsing_errors=True)
                    code = extract_code(response.get("output", ""))  

                    if code:
                        result = execute_code(code, df)
                        if 'plt.show' in code:
                            st.pyplot(plt)
                        elif result is not None:
                            st.write(result)
                    else:
                        st.write("Received non-executable response: ", response.get("output", "No response or incorrect format received."))

if __name__ == "__main__":
    main()
