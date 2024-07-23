from langchain_community.vectorstores.redis import Redis
from langchain_community.embeddings import OllamaEmbeddings
from pprint import pprint

def search():

    redis = Redis(
        redis_url='redis://localhost:8002',
        index_name='llama3_contents',
        embedding=OllamaEmbeddings(model='llama3'),
    )
    response = Redis.similarity_search_with_score(redis, 'django', k=5)
    pprint(response)

search()
