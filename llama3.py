from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.redis import Redis
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

llama3 = Ollama(model='llama3', temperature=0.3)
base_prompt = PromptTemplate.from_template(
    '''Você é um modelo de linguagem treinado para responder com base apenas nas informações fornecidas no prompt. Por favor, use apenas os dados fornecidos a seguir para responder a pergunta.

Dados fornecidos:
{context}

Pergunta:
{question}'''
)


redis = Redis(
    redis_url='redis://localhost:8002',
    index_name='llama3_contents',
    embedding=OllamaEmbeddings(model='llama3'),
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retiever = redis.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': .5})

rag_chain = (
    {"context":  retiever| format_docs, "question": RunnablePassthrough()}
    | base_prompt
    | llama3
    | StrOutputParser()
)

print(rag_chain.invoke('Oque é o framework django?'))
