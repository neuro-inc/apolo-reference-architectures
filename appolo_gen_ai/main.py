from openai import OpenAI
from pgvector.psycopg import register_vector
import psycopg
import glob
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector


EMB_URL = 'https://embedding-inference--9771360698.jobs.scottdc.org.neu.ro'
GEN_URL = 'https://generation-inference--9771360698.jobs.scottdc.org.neu.ro'
DB_URL = 'https://pgvector--9771360698.jobs.scottdc.org.neu.ro'


def example_run_embedding():
    client = OpenAI(base_url=f'{EMB_URL}/v1/', api_key='token')
    response = client.embeddings.create(input="Your text string goes here", model="tgi")
    print(response.data[0].embedding)

def example_run_generation():

    client = OpenAI(
        base_url=f"{GEN_URL}/v1/",
        api_key="token",
    )
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Why is open-source software important?"},
        ],
    )
    print(chat_completion.choices[0].message.content)

def example_vector_db():
    DB_HOST = '0.0.0.0'
    DB_NAME = 'postgres'
    DB_USER = 'postgres'
    DB_PASSWORD = 'postgres'
    DB_PORT = '5432'

    connection_string = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"
    
    conn = psycopg.connect(connection_string, autocommit=True)
    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)

    conn.execute('DROP TABLE IF EXISTS documents')
    conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))')

    input = [
        'The dog is barking',
        'The cat is purring',
        'The bear is growling'
    ]

    client = OpenAI(base_url=f'{EMB_URL}/v1/', api_key='token')
    response = client.embeddings.create(input=input, model='tgi')
    embeddings = [v.embedding for v in response.data]

    for content, embedding in zip(input, embeddings):
        conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

    document_id = 1
    neighbors = conn.execute('SELECT content FROM documents WHERE id != %(id)s ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = %(id)s) LIMIT 5', {'id': document_id}).fetchall()
    for neighbor in neighbors:
        print(neighbor[0])



def build_apolo_docs_rag():
    path_to_dir = "platform-docs/"
    markdown_files = glob.glob(os.path.join(path_to_dir, "**/*.md"), recursive=True)
    docs = [UnstructuredMarkdownLoader(f).load()[0] for f in markdown_files]
    chunks = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100).split_documents(docs)
    os.environ["OPENAI_API_KEY"] = 'token'
    embeddings = OpenAIEmbeddings(model="tgi", base_url=f'{EMB_URL}/v1/', chunk_size=1, show_progress_bar=True, skip_empty=True)
    connection = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"  # Uses psycopg3!
    collection_name = "neuro_docs_test"
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    vector_store.add_documents([x for x in chunks if len(x.page_content) > 100])
    min([len(chunks[i].page_content) for i in range(len(chunks))])


def query_apolo_docs_rag():
    connection = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    collection_name = "neuro_docs"
    os.environ["OPENAI_API_KEY"] = 'token'
    embeddings = OpenAIEmbeddings(model="tgi", base_url=f'{EMB_URL}/v1/', chunk_size=1, show_progress_bar=True, skip_empty=True)
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    question = "How to run create storage?"
    results = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in results])
    prompt = f"""
    You are an assistant for question-answering tasks about Apolo/Neuro platform. 
    Use the following pieces of retrieved documentation to answer the question. 
    If you don't know the answer, just say that you don't know. 

    Question: 
    ###
    {question} 
    ###

    Apolo/Neuro documentation: 
    ###
    {context} 
    ###

    Answer:

    """
    
    gen_client = OpenAI(
        base_url=f"{GEN_URL}/v1/",
        api_key="token",
    )
    chat_completion = gen_client.chat.completions.create(
        model="tgi",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    print(chat_completion.choices[0].message.content)           