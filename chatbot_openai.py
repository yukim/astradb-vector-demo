import streamlit as st
import dotenv
import langchain
import json
import os

from cassandra.cluster import Session
from cassandra.query import PreparedStatement

from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import BaseRetriever, Document, SystemMessage

from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider

# Enable langchain debug mode
langchain.debug = True

dotenv.load_dotenv(dotenv.find_dotenv())

# OpenAI model to use
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
# Template to use to generate hyperlink on product
PRODUCT_LINK_TEMPLATE = os.getenv("PRODUCT_LINK_TEMPLATE") or "/product/{product id}"


class AstraProductRetriever(BaseRetriever):
    session: Session
    embedding: OpenAIEmbeddings
    search_statement: PreparedStatement = None

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query):
        docs = []
        embeddingvector = self.embedding.embed_query(query)
        if self.search_statement is None:
            self.search_statement = self.session.prepare("""
                SELECT
                    product_id,
                    brand,
                    saleprice,
                    product_categories,
                    product_name_en,
                    short_description_en,
                    long_description_en
                FROM hybridretail.products_cg_hybrid
                ORDER BY openai_description_embedding_en ANN OF ?
                LIMIT ?
                """)
        query = self.search_statement
        results = self.session.execute(query, [embeddingvector, 5])
        top_products = results._current_rows
        for r in top_products:
            docs.append(Document(
                id=r.product_id,
                page_content=r.short_description_en,
                metadata={"product id": r.product_id,
                          "brand": r.brand,
                          "product category": r.product_categories,
                          "product name": r.product_name_en,
                          "description": r.short_description_en,
                          "price": r.saleprice
                          }
            ))

        return docs


def get_session(scb: str, secrets: str) -> Session:
    """
    Connect to Astra DB using secure connect bundle and credentials.

    Parameters
    ----------
    scb : str
        Path to secure connect bundle.
    secrets : str
        Path to credentials.
    """

    cloud_config = {
        'secure_connect_bundle': scb
    }

    with open(secrets) as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    return cluster.connect()


@st.cache_resource
def create_chatbot(model="gpt-3.5-turbo"):
    session = get_session(scb='./config/secure-connect-multilingual.zip',
                          secrets='./config/multilingual-token.json')
    llm = ChatOpenAI(model=model, temperature=0, streaming=True)
    embedding = OpenAIEmbeddings()
    # Define tool to query products from Astra DB
    # Instruct OpenAI to use the tool when searching for products
    # and call the tool with English translation of the query
    retriever = AstraProductRetriever(session=session, embedding=embedding)
    retriever_tool = create_retriever_tool(
        retriever,
        "products_retriever",
        "Useful when searching for products from a product description. \
        Prices are in THB. When calling this tool, include as much detail as possible, \
        and translate arguments to English.")
    # Define system message
    # Here, we instruct the agent to create a hyperlink on product id, display product description also,
    # and respond in the same language as the user used
    system_message = f"""You are a customer service of a home improvement store and you are asked to pick products for a customer.
    When generating product hyperlink, use the following format: '{PRODUCT_LINK_TEMPLATE}'.
    Include the product description when responding with the list of product recommendation.
    All the responses should be the same language as the user used.
    """
    message = SystemMessage(content=system_message)
    agent_executor = create_conversational_retrieval_agent(
        llm=llm, tools=[retriever_tool], system_message=message, verbose=True)
    return agent_executor


if 'history' not in st.session_state:
    st.session_state['history'] = []

st.set_page_config(layout="wide")

chatbot = create_chatbot(OPENAI_MODEL)

if st.button("Clear chat history"):
    st.session_state['history'] = []
    chatbot.memory.clear()

# Display chat messages from history on app rerun
for (query, answer) in st.session_state['history']:
    with st.chat_message("User"):
        st.markdown(query)
    with st.chat_message("Bot"):
        st.markdown(answer)

prompt = st.chat_input(placeholder="Ask chatbot")
if prompt:
    # Display user message in chat message container
    with st.chat_message("User"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("Bot"):
        st_callback = StreamlitCallbackHandler(st.container())
        result = result = chatbot.invoke({
            "input": prompt,
            "chat_history": st.session_state['history']
        }, config={"callbacks": [st_callback]})
        st.session_state['history'].append((prompt, result["output"]))
        st.markdown(result["output"])
