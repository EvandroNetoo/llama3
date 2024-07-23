from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import TokenTextSplitter
import os
from langchain_community.vectorstores.redis import Redis
from langchain_community.embeddings import OllamaEmbeddings


loader = DirectoryLoader(
    path=os.path.join(os.path.dirname(__file__), 'data'),
    glob='*.txt',
    loader_cls=TextLoader,
)

def load():
    docs = loader.load()
    
    splitter = TokenTextSplitter(
        encoding_name='cl100k_base',
        chunk_size=100,
        chunk_overlap=0,
    )

    splitted_documents = splitter.split_documents(docs)
    Redis.from_documents(
        documents=splitted_documents,
        embedding=OllamaEmbeddings(model='llama3'),
        redis_url='redis://localhost:8002',
        index_name='llama3_contents',
    )

load()
