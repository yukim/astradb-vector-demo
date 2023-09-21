from cassandra.cluster import Session
from cassandra.query import BatchStatement
from langchain.embeddings.base import Embeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.cassandra import Cassandra
from abc import ABC, abstractmethod
from typing import List
from .model import Product
import json

RESULT_STORE_INSERT = "INSERT INTO %s.%s (query, pattern, rank, id, relevant) VALUES (?, ?, ?, ?, ?);"
RESULT_STORE_TABLE = 'query_results'


class TestPattern(ABC):
    """
    Abstrct class for test pattern
    """

    def __init__(self, session: Session, keyspace: str):
        self.session = session
        self.result_store_insert = session.prepare(
            RESULT_STORE_INSERT % (keyspace, RESULT_STORE_TABLE))

    def search(self, query: str, k=100) -> List[Product]:
        """
        Query
        """
        result = self.vectore_store().similarity_search(query, k)
        products: List[Product] = []
        for prod in result:
            metadata = prod.metadata
            try:
                image_link_text = metadata['image_link'].replace(
                    "'", '"').replace("\n", ",")
                images = json.loads(image_link_text)
                image = images[0]
            except:
                image = None
            try:
                specs_text = metadata['spec'].replace(
                    "'", '"').replace("\n", ",")
                spec = json.loads(specs_text)
                specs = spec
            except:
                specs = []
            product = Product(product_id=metadata['product_id'],
                              product_name=metadata['product_name'],
                              brand=metadata['brand'],
                              short_description=metadata['short_description'],
                              long_description=metadata['long_description'],
                              image=image,
                              specs=specs)
            products.append(product)

        # Insert results to result store
        batch = BatchStatement()
        for i, prod in enumerate(products):
            batch.add(self.result_store_insert,
                      (query, self.name(), i + 1, prod.product_id, False))
        try:
            self.session.execute(batch)
        except Exception as e:
            print("Storing result failed", e)

        return products

    def get_previous_results(self) -> List[Product]:
        pass

    @abstractmethod
    def name(self) -> str:
        """
        This name will be used as an identifier for the test pattern
        """
        pass

    @abstractmethod
    def vectore_store(self) -> VectorStore:
        pass

    @abstractmethod
    def embeddings(self) -> Embeddings:
        pass


class CohereTestPattern(TestPattern):
    """
    Test pattern for Cohere embedding model `embed-multilingual-v2.0`.
    For text embeddings, this pattern constructs a text in the following format:
    ```
    ชื่อผลิตภัณฑ์ {['product_name']}
    ยี่ห้อสินค้า {['brand']}
    หมวดหมู่สินค้า {['product_categories']}
    {['long_description']}
    ```
    """

    def __init__(self, session: Session, model_name: str, api_key: str, keyspace: str, table_name: str):
        super().__init__(session=session, keyspace=keyspace)

        self.model_name = model_name
        self.embeddings = CohereEmbeddings(model=self.model_name,
                                           cohere_api_key=api_key)
        self._vstore = Cassandra(embedding=self.embeddings,
                                 session=session,
                                 keyspace=keyspace,
                                 table_name=table_name)

    def vectore_store(self) -> VectorStore:
        return self._vstore

    def name(self) -> str:
        return f"cohere_{self.model_name}_v1"

    def embeddings(self) -> Embeddings:
        return self.embeddings()
