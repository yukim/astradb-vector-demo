import streamlit as st
import pandas as pd
import dotenv
import os
from cassandra.cluster import Session
from demo.model import Product
from demo.test_patterns import TestPattern, CohereTestPattern
from demo import db

# Set up environment variables from .env file
dotenv.load_dotenv(dotenv.find_dotenv())

st.set_page_config(layout="wide")


@st.cache_resource
def get_session() -> Session:
    print('Connecting to Astra DB...')
    return db.get_session(scb='./config/secure-connect-benchmark.zip',
                                 secrets='./config/benchmark-token.json')


# Set up Astra DB connection
session = get_session()

test_patterns = st.session_state.get('test_patterns', None)
if not test_patterns:
    test_patterns = [
        CohereTestPattern(session=session, model_name='embed-multilingual-v2.0',
                          api_key=os.environ['COHERE_API_KEY'],
                          keyspace='bench',
                          table_name='bench_cohere'),
    ]

    st.session_state.test_patterns = test_patterns

with st.sidebar:
    selected = st.selectbox("Test pattern", range(0, len(test_patterns)),
                            format_func=lambda idx: test_patterns[idx].name())
    test_pattern: TestPattern = test_patterns[selected]
    with st.expander("Test pattern description"):
        st.markdown(test_pattern.__doc__)

st.title("Product search evaluation in Thai")
query = st.text_input('Query in Thai', '')


@st.cache_data(persist="disk")
def search(query) -> list[Product]:
    return test_pattern.search(query, k=100)


def recall_at_k(data, k: int) -> float:
    return data[0:k].mean()


query = query.strip()
if query and len(query) > 0:
    products = search(query)
    evaluation = pd.Series([False for _ in range(len(products))])
    st.write(f"Found {len(products)} results")
    checked = []
    for rank, prod in enumerate(products):
        id = prod.product_id
        with st.container():
            col1, col2 = st.columns([0.15, 0.85])
            with col1:
                checked = st.checkbox(id)
                if checked:
                    evaluation[rank] = True
                if prod.image:
                    st.image(prod.image)
            with col2:
                st.subheader(prod.product_name)
                st.caption(f"Brand: {prod.brand}")
                st.write(
                    f"<span style=\"word-wrap:break-word;\">{prod.short_description}</span>", unsafe_allow_html=True)
                st.write(
                    f"<span style=\"word-wrap:break-word;\">{prod.long_description}</span>", unsafe_allow_html=True)
                if len(prod.specs) > 0:
                    st.table(prod.specs)
            st.divider()
    st.toast(
        f"Rank@5: {recall_at_k(evaluation, 5)}, Rank@10: {recall_at_k(evaluation, 10)}, Rank@100: {recall_at_k(evaluation, 100)}")
