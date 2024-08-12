# apolo-rag-reference-architecture



```
pip install apolo-all

apolo login
apolo config show
pip install unstructured
```

## Setup

```bash
apolo run --detach --no-http-auth --preset cpu-8-20 --name embedding-inference --http-port 80 ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 -- --model-id jinaai/jina-embeddings-v2-base-en
apolo run --detach --no-http-auth --preset H100x1 --name generation-inference --http-port 80 -e HF_TOKEN=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 ghcr.io/huggingface/text-generation-inference:2.2.0 -- --model-id meta-llama/Meta-Llama-3.1-8B-Instruct
apolo run --detach --no-http-auth --preset cpu-medium --name pgvector --http-port 5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16



apolo run --detach --no-http-auth --preset cpu-medium --name open-webui --http-port 8080 -e OPENAI_API_BASE_URLS="https://generation-inference--9771360698.jobs.scottdc.org.neu.ro/v1" -e OPENAI_API_KEYS="tgi-key" ghcr.io/open-webui/open-webui:main

https://generation-inference--9771360698.jobs.scottdc.org.neu.ro/docs/#/

docker run -d -p 3000:8080 \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URLS="https://api.openai.com/v1;https://api.mistral.ai/v1" \
  -e OPENAI_API_KEYS="<OPENAI_API_KEY_1>;<OPENAI_API_KEY_2>" \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main


```


```
apolo job status pgvector
apolo job port-forward pgvector 5432:5432

psql -h 0.0.0.0 -U postgres -d postgres

psql -h https://pgvector--9771360698.jobs.scottdc.org.neu.ro/ -p 5432 -U postgres -d postgres

```

## PGVector

```bash
docker run -it -p 5432:5432 -e POSTGRES_PASSWORD=password pgvector/pgvector:pg16

psql -h pgvector--9771360698.jobs.scottdc.org.neu.ro -U postgres -d postgres

psql -h https://pgvector--9771360698.jobs.scottdc.org.neu.ro -p 80 -U postgres -d postgres
psql -h https://pgvector2--9771360698.jobs.scottdc.org.neu.ro -p 80 -U postgres -d postgres


apolo job ls 
apolo job status job-f87af2c1-0148-4b78-9e20-4bce05847d38
apolo job port-forward job-f87af2c1-0148-4b78-9e20-4bce05847d38 5432:5432
psql -h 0.0.0.0 -p 5432 -U postgres -d postgres


psql -h https://pgvector--9771360698.jobs.scottdc.org.neu.ro -p 5432 -U postgres -d postgres
psql -h https://pgvector--9771360698.jobs.scottdc.org.neu.ro -p 80 -U postgres -d postgres
psql -h pgvector--9771360698.jobs.scottdc.org.neu.ro -p 5432 -U postgres -d postgres
psql -h pgvector--9771360698.jobs.scottdc.org.neu.ro -p 80 -U postgres -d postgres

 Http URL                 https://pgvector--9771360698.jobs.scottdc.org.neu.ro
 Http port                5432



https://pgvector--9771360698.jobs.scottdc.org.neu.ro
https://pgvector2--9771360698.jobs.scottdc.org.neu.ro
https://pgvector3--9771360698.jobs.scottdc.org.neu.ro

import psycopg2

conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="https://pgvector--9771360698.jobs.scottdc.org.neu.ro", port="80")
conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="https://pgvector--9771360698.jobs.scottdc.org.neu.ro", port="5432")
conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="pgvector--9771360698.jobs.scottdc.org.neu.ro", port="80")
conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="pgvector--9771360698.jobs.scottdc.org.neu.ro", port="5432")

conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="https://pgvector2--9771360698.jobs.scottdc.org.neu.ro", port="80")
conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="https://pgvector2--9771360698.jobs.scottdc.org.neu.ro", port="5432")

conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="https://pgvector3--9771360698.jobs.scottdc.org.neu.ro", port="80")
conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="https://pgvector3--9771360698.jobs.scottdc.org.neu.ro", port="5432")
conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="pgvector3--9771360698.jobs.scottdc.org.neu.ro", port="80")
conn = psycopg2.connect(database="postgres", user="postgres", password="password", host="pgvector3--9771360698.jobs.scottdc.org.neu.ro", port="5432")


apolo cp my-postgres.conf storage:my-postgres
apolo cp my-postgres.conf storage:my-postgres.conf
apolo cp my-postgres.conf storage:my-postgres.conf

apolo run --detach --no-http-auth --preset cpu-medium --name pgvector6 --volume storage:my-postgres.conf:/etc/postgresql/postgresql.conf --http-port 80 -e POSTGRES_PASSWORD=password pgvector/pgvector:pg16 -- postgres -c 'config_file=/etc/postgresql/postgresql.conf'

```


## References:

https://huggingface.co/learn/cookbook/en/rag_with_unstructured_data 
https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/

https://huggingface.co/datasets/lamini/earnings-calls-qa
https://huggingface.co/datasets/eloukas/edgar-corpus

https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83
https://pathway.com/developers/templates/unstructured-to-structured
