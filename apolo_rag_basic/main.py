import glob
import itertools
import os
import subprocess
from itertools import chain
from pathlib import Path
from typing import List, Optional

import argilla as rg
import psycopg
import requests
import typer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from openai import OpenAI
from pgvector.psycopg import register_vector
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "token"

EMB_URL = "https://embedding-inference--9771360698.jobs.scottdc.org.neu.ro"
RERANKER_URL = "https://reranker-inference--9771360698.jobs.scottdc.org.neu.ro"
GEN_URL = "https://generation-inference--9771360698.jobs.scottdc.org.neu.ro"
DB_URL = "https://pgvector--9771360698.jobs.scottdc.org.neu.ro"
ARGILLA_URL = "https://argilla--9771360698.jobs.scottdc.org.neu.ro"


def get_embedding_client() -> OpenAI:
    client = OpenAI(base_url=f"{EMB_URL}/v1/", api_key="token")
    return client


def get_generation_client() -> OpenAI:
    client = OpenAI(base_url=f"{GEN_URL}/v1/", api_key="token")
    return client


def get_db_connection() -> psycopg.Connection:
    DB_HOST = "0.0.0.0"
    DB_NAME = "postgres"
    DB_USER = "postgres"
    DB_PASSWORD = "postgres"
    DB_PORT = "5432"

    connection_string = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"
    conn = psycopg.connect(connection_string, autocommit=True)
    return conn


def create_schema(table_name: str, dimensions: int):
    conn = get_db_connection()

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    register_vector(conn)

    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(
        f"CREATE TABLE {table_name} (id bigserial PRIMARY KEY, content text, embedding vector({dimensions}))"
    )
    conn.execute(
        f"CREATE INDEX ON {table_name} USING GIN (to_tsvector('english', content))"
    )


def insert_data(
    table_name: str,
    embeddings: List[List[float]],
    sentences: List[List[str]],
    batch_size: int = 256,
):
    conn = get_db_connection()
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        sql = f"INSERT INTO {table_name} (content, embedding) VALUES " + ", ".join(
            ["(%s, %s)" for _ in batch_embeddings]
        )
        params = list(itertools.chain(*zip(batch_sentences, batch_embeddings)))
        conn.execute(sql, params)


def generate_with_context(query: str, context: str) -> str:
    prompt = f"""

    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.

    Question: {query} 
    Context: {context} 
    Answer:
    """

    client = get_generation_client()
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    response = chat_completion.choices[0].message.content
    return response


def log_sample(
    query: str,
    context: str,
    response: str,
    semantic_search: str,
    keyword_search: str,
    dataset_name: str,
):
    client = rg.Argilla(api_url=ARGILLA_URL, api_key="admin.apikey")

    dataset = client.datasets(name=dataset_name)
    if not dataset.exists():
        settings = rg.Settings(
            fields=[
                rg.TextField(name="query"),
                rg.TextField(name="response"),
                rg.TextField(name="context"),
                rg.TextField(name="semantic_search"),
                rg.TextField(name="keyword_search"),
                
            ],
            questions=[
                rg.RatingQuestion(
                    name="context_relevance",
                    title="Relevance of the context",
                    values=[1, 2, 3, 4, 5],
                ),
                rg.RatingQuestion(
                    name="response_relevance",
                    title="Relevance of the response",
                    values=[1, 2, 3, 4, 5],
                ),
            ],
        )
        dataset = rg.Dataset(
            name=dataset_name,
            settings=settings,
            workspace="admin",
            client=client,
        )
        dataset.create()

    record = rg.Record(
        fields={
            "query": query,
            "context": context,
            "response": response,
            "semantic_search": semantic_search,
            "keyword_search": keyword_search,
        }
    )

    dataset.records.log([record])


def test_embedding():
    client = get_embedding_client()
    response = client.embeddings.create(input="Your text string goes here", model="tgi")
    print(response.data[0].embedding)


def test_generation():
    client = get_generation_client()
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Why is open-source software important?"},
        ],
    )
    print(chat_completion.choices[0].message.content)


def test_reranker():
    query = "What is Deep Learning?"
    sentences = ["Deep Learning is ..", "Art is ..", "Machine Learning is ..."]
    sentences_result = rerank(query=query, sentences=sentences, top_n=5)
    print(sentences_result)


def test_db_connection():
    table_name = "test_db"
    create_schema(table_name=table_name, dimensions=19)
    sentences = ["The dog is barking", "The cat is purring", "The bear is growling"]
    embeddings = [
        [x * 0.5 for x in range(19)],
        [x * 0.5 for x in range(19)],
        [x * 0.5 for x in range(19)],
    ]
    insert_data(table_name=table_name, embeddings=embeddings, sentences=sentences)


def semantic_search(
    table_name: str, query_embedding: List[float], top_n: int = 5
) -> List[str]:
    conn = get_db_connection()

    embedding_str = ",".join(map(str, query_embedding))
    sql = f"""
        SELECT content 
        FROM {table_name} 
        ORDER BY embedding <=> '[{embedding_str}]'::vector 
        LIMIT {top_n}
    """
    resutls = conn.execute(sql).fetchall()
    return [x[0] for x in resutls]


def keyword_search(table_name: str, query: str, top_n: int = 5) -> List[str]:
    conn = get_db_connection()

    # Create the SQL query using trigram similarity
    sql = f"""
        SELECT content 
        FROM {table_name} 
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s) 
        OR content ILIKE %s
        ORDER BY GREATEST(
            ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)),
            similarity(content, %s)
        ) DESC 
        LIMIT {top_n}
    """

    search_query = f"%{query}%"
    results = conn.execute(sql, (query, search_query, query, query)).fetchall()
    return [x[0] for x in results]


def get_embeddings(sentences: List[str], batch_size: int = 4) -> List[List[float]]:
    embeddings = []
    embedding_client = get_embedding_client()
    for i in tqdm(range(0, len(sentences), batch_size)):
        sentences_batch = sentences[i : i + batch_size]
        response_batch = embedding_client.embeddings.create(
            input=sentences_batch, model="tgi"
        )
        embeddings.extend([x.embedding for x in response_batch.data])
    return embeddings


def rerank(query: str, sentences: List[str], top_n: int = 5) -> List[str]:
    unique_sentences = list(set(sentences))
    payload = {"query": query, "texts": unique_sentences, "return_text": True}
    response = requests.post(f"{RERANKER_URL}/rerank", json=payload)
    reranked_data = response.json()
    reranked_sentences = [
        item["text"]
        for item in sorted(reranked_data, key=lambda x: x["score"], reverse=True)
    ]
    return reranked_sentences[:top_n]


def clone_repo_to_tmp(repo_url: str, repo_name: Optional[str] = None) -> str:
    tmp_dir = Path("/tmp")
    repo_name = repo_name or Path(repo_url).stem
    clone_path = tmp_dir / repo_name
    if clone_path.exists():
        print(f"{clone_path} exists")
        return str(clone_path)
    subprocess.run(["git", "clone", repo_url, str(clone_path)], check=True)
    return str(clone_path)


def build_apolo_docs_rag():
    table_name = "apolo_docs"
    chunk_size = 1024
    chunk_overlap = 100

    print("Processing data")
    apolo_docs_path = clone_repo_to_tmp(
        repo_url="https://github.com/neuro-inc/platform-docs.git"
    )
    markdown_files = glob.glob(os.path.join(apolo_docs_path, "**/*.md"), recursive=True)
    docs = list(
        chain.from_iterable(
            [UnstructuredMarkdownLoader(f).load() for f in markdown_files]
        )
    )
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)

    print("Get ebeddings")
    sentences = [x.page_content for x in chunks]
    embeddings = get_embeddings(sentences=sentences)

    print("Ingest data")
    create_schema(table_name=table_name, dimensions=len(embeddings[0]))
    insert_data(
        table_name=table_name, embeddings=embeddings, sentences=sentences, batch_size=64
    )


def query_apolo_docs_rag(query: str = "How to run mlflow?"):
    table_name = "apolo_docs"
    query_embedding = get_embeddings(sentences=[query])[0]

    semantic_search_result = semantic_search(
        table_name=table_name, query_embedding=query_embedding, top_n=20
    )
    keyword_search_result = keyword_search(table_name=table_name, query=query, top_n=20)
    search_result = rerank(
        query=query, sentences=semantic_search_result + keyword_search_result, top_n=5
    )

    context = "\n".join(search_result)
    response = generate_with_context(query=query, context=context)
    print(response)
    log_sample(
        query=query,
        context=context,
        response=response,
        dataset_name=table_name,
        keyword_search="\n".join(keyword_search_result),
        semantic_search="\n".join(semantic_search_result),
    )


def build_canada_budget_rag():
    table_name = "canada_budget"
    chunk_size = 1024
    chunk_overlap = 100
    data_path = "./data/canada/"

    pdf_files = list(Path(data_path).iterdir())

    print("Processing data")
    list_of_pages = [
        PyPDFLoader(pdf_files[idx]).load() for idx in range(len(pdf_files))
    ]
    docs = list(chain.from_iterable(list_of_pages))
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)

    print("Get ebeddings")
    sentences = [x.page_content for x in chunks]
    embeddings = get_embeddings(sentences=sentences, batch_size=32)

    print("Ingest data")
    create_schema(table_name=table_name, dimensions=len(embeddings[0]))
    insert_data(
        table_name=table_name, embeddings=embeddings, sentences=sentences, batch_size=64
    )


def query_canada_budget_rag(query="How to run mlflow?"):
    table_name = "canada_budget"

    query_embedding = get_embeddings(sentences=[query])[0]

    semantic_search_result = semantic_search(
        table_name=table_name, query_embedding=query_embedding, top_n=20
    )
    keyword_search_result = keyword_search(table_name=table_name, query=query, top_n=20)
    search_result = rerank(
        query=query, sentences=semantic_search_result + keyword_search_result, top_n=5
    )

    context = "\n".join([doc for doc in search_result])
    response = generate_with_context(query=query, context=context)
    print(response)

    log_sample(
        query=query,
        context=context,
        response=response,
        dataset_name=table_name,
        keyword_search="\n".join(keyword_search_result),
        semantic_search="\n".join(semantic_search_result),
    )


def main():
    app = typer.Typer()
    app.command()(build_apolo_docs_rag)
    app.command()(query_apolo_docs_rag)
    app.command()(build_canada_budget_rag)
    app.command()(query_canada_budget_rag)
    app()


if __name__ == "__main__":
    main()
