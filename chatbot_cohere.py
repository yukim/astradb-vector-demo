import os
import streamlit as st
import dotenv

from langchain.llms import Cohere
from langchain.vectorstores import Cassandra
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import json
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider


dotenv.load_dotenv(dotenv.find_dotenv())

st.set_page_config(layout="wide")


def get_astra_session(scb: str, secrets: str) -> Session:
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
def get_session():
    print('Connecting to Astra DB...')
    return get_astra_session(scb='./config/secure-connect-benchmark.zip',
                             secrets='./config/benchmark-token.json')


session = get_session()

cohere_embedding = CohereEmbeddings(model='embed-multilingual-v2.0',)
product_store = Cassandra(embedding=cohere_embedding,
                          session=session,
                          keyspace='bench',
                          table_name='bench_cohere_en')
chat_history_store = Cassandra(embedding=cohere_embedding,
                               session=session,
                               keyspace='bench',
                               table_name='chat_history')

cohere = Cohere(cohere_api_key=os.environ['COHERE_API_KEY'])

template = """You are a customer service of a home improvement store and you are asked to pick products for a customer.
Question: {question}
"""
# Answer: Return the list of items only containing descriptions in JSON String Array format.

product_list_prompt = PromptTemplate.from_template(template)

template = """You are a friendly, conversational home improvement store shopping assistant. Use the following context including product names, descriptions, and keywords to show the shopper whats available, help find what they want, and answer any questions.
 
It's ok if you don't know the answer.
Context:\"""
 
{context}
\"""
Question:\"
\"""
 
Helpful Answer:
"""

qa_prompt = PromptTemplate.from_template(template)

question_generator = LLMChain(
    llm=cohere,
    prompt=product_list_prompt
)

doc_chain = load_qa_chain(
    llm=cohere,
    chain_type="stuff",
    prompt=qa_prompt
)

# chain = LLMChain(prompt=prompt, llm=cohere)
# chain = ConversationalRetrievalChain.from_llm(
#     cohere, retriever=product_store.as_retriever())
chatbot = ConversationalRetrievalChain(
    retriever=product_store.as_retriever(),
    combine_docs_chain=doc_chain,
    question_generator=question_generator
)

if 'history' not in st.session_state:
    st.session_state['history'] = []


def conversational_chat(query):
    result = result = chatbot(
        {"question": query, "chat_history": st.session_state['history']}
    )
    st.session_state['history'].append((result["question"], result["answer"]))

    return result["answer"]


for (query, answer) in st.session_state['history']:
    with st.chat_message("User"):
        st.write(query)
    with st.chat_message("Bot"):
        st.write(answer)

# I'd like to build a small fish tank. Could you suggest the products I need to purchase?
prompt = st.chat_input(placeholder="What do you want to build?")
if prompt:
    answer = conversational_chat(prompt)
