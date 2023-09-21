import time
import dotenv
import os
from datasets import load_dataset

from demo import db
from demo.test_patterns import CohereTestPattern

# Set up environment variables from .env file
dotenv.load_dotenv(dotenv.find_dotenv())

session = db.get_session(scb='./config/secure-connect-benchmark.zip',
                         secrets='./config/benchmark-token.json')

data = load_dataset(path='./data')['train'].to_pandas()
# Text to embed
# texts = data.apply(lambda r:
#                   f"ชื่อผลิตภัณฑ์ {r['product_name']}"
#                   f" ยี่ห้อสินค้า {r['brand']}"
#                   f" หมวดหมู่สินค้า {r['product_categories']}"
#                   f" {r['long_description']}", axis=1)
texts = data.apply(lambda r:
                   f"Product name: {r['product_name_en']}, "
                   f"Brand: {r['brand']}, "
                   f"Product category: {r['product_categories']}, "
                   f"Description: {r['long_description_en']}", axis=1)

test_pattern = CohereTestPattern(session=session, model_name='embed-multilingual-v2.0',
                                 api_key=os.environ['COHERE_API_KEY'],
                                 keyspace='bench',
                                 table_name='bench_cohere_en')
# To make Product ID as document ID, use `adds_texts` instead of `from_documents`
vstore = test_pattern.vectore_store()

# Add texts to vector store.
# In order to workaround cohere's trial key limit, 100 API calls per minutes,
# devide the texts into 100 texts per call, and wait for 1 minute.
start = 27000
batch_size = 500
if start == 0:
    vstore.clear()
for i in range(start, len(data), batch_size):
    print(f"Adding {i} to {i+batch_size} texts")
    batch = data[i:i+batch_size]
    vstore.add_texts(texts=texts[i:i+batch_size].to_list(),
                     metadatas=batch.to_dict(orient='records'),
                     ids=batch['product_id'].to_list())
    time.sleep(60)
