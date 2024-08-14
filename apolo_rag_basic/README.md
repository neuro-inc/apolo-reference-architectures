# Apolo Rag Basic

## Architecture setup

Setup apolo CLI:

```bash
pip install apolo-all
apolo login
apolo config show
```

Create storage:

```bash

apolo mkdir -p storage:generative-models
apolo mkdir -p storage:embedding-models
apolo mkdir -p storage:database
apolo ls storage
```

Create Postgres:

```bash
apolo run --detach \
          --no-http-auth \
          --preset cpu-medium \
          --name pgvector \
          --http-port 5432 \
          --volume storage:database/pgvector:/var/lib/postgresql/data:rw \
          -e POSTGRES_PASSWORD=postgres \
          pgvector/pgvector:pg16
```

 Create Argilla:

```bash
apolo run --detach \
          --no-http-auth \
          --preset cpu-medium \
          --name argilla \
          --http-port 6900 \
          argilla/argilla-quickstart:v2.0.0rc2

```

Create models:

Generative LLM:

```bash
apolo run --detach \
          --no-http-auth \
          --preset H100x1 \
          --name generation-inference \
          --http-port 80 \
          --volume storage:generative-models:/data:rw \
          -e HF_TOKEN=hf_ \
          ghcr.io/huggingface/text-generation-inference:2.2.0 -- --model-id meta-llama/Meta-Llama-3.1-70B-Instruct --quantize bitsandbytes-nf4
```

```bash
apolo run --detach \
          --no-http-auth \
          --preset H100x1 \
          --name generation-inference \
          --http-port 80 \
          --volume storage:generative-models:/data:rw \
          -e HF_TOKEN=hf_ \
          ghcr.io/huggingface/text-generation-inference:2.2.0 -- --model-id meta-llama/Meta-Llama-3.1-8B
```

Embedding LLM:

```bash
apolo run \
  --detach \
  --no-http-auth \
  --preset H100x1 \
  --name embedding-inference \
  --http-port 80 \
  --volume storage:embedding-models:/data:rw \
  ghcr.io/huggingface/text-embeddings-inference:hopper-1.5 -- --model-id BAAI/bge-m3
```

Reference: <https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3>

Reranker LLM:

```bash
apolo run --detach \
          --no-http-auth \
          --preset H100x1 \
          --name reranker-inference \
          --http-port 80 \
          --volume storage:embedding-models:/data:rw \
          ghcr.io/huggingface/text-embeddings-inference:hopper-1.5 -- \
          --model-id BAAI/bge-reranker-v2-m3
```

Reference: <https://huggingface.co/BAAI/bge-reranker-v2-m3>

Connect to DB

```bash
apolo job port-forward pgvector 5432:5432
psql -h 0.0.0.0 -U postgres -d postgres
```


## Apolo RAG

```bash
python main.py build-apolo-docs-rag
```

```bash
python main.py query-apolo-docs-rag --query 'How to run mlflow?'
python main.py query-apolo-docs-rag --query 'How to run training?'
python main.py query-apolo-docs-rag --query 'How to run custom job? Be specific'
```

## Canada 2024 budget RAG

```bash
python main.py build-canada-budget-rag
```

```bash
python main.py query-canada-budget-rag --query 'What is Canada’s main spending?'
python main.py query-canada-budget-rag --query 'What is the housing situation?'
python main.py query-canada-budget-rag --query 'What actions is the government taking to increase the new housing supply?'
```

## References

- https://huggingface.co/learn/cookbook/en/rag_with_unstructured_data 
- https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/
- https://huggingface.co/datasets/lamini/earnings-calls-qa
- https://huggingface.co/datasets/eloukas/edgar-corpus
- https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83
- https://www.anyscale.com/blog/a-comprehensive-guidfe-for-building-rag-based-llm-applications-part-1
- https://huggingface.co/spaces/mteb/leaderboard
