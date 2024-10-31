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
apolo mkdir -p storage:visual_rag
apolo ls storage:visual_rag
```

Docker image build & share: 

```bash

docker build -t ghcr.io/kyryl-opens-ml/apolo_visual_rag:latest .
docker push ghcr.io/kyryl-opens-ml/apolo_visual_rag:latest
```


Upload data sample: 

```bash

apolo cp -r ./sample_data/ storage:visual_rag/raw-data/
```

Ingest data to search later:


```bash
apolo run --detach \
          --no-http-auth \
          --preset H100x1 \
          --name ingest-data \
          --http-port 80 \
          --volume storage:visual_rag/cache:/root/.cache/huggingface:rw \
          --volume storage:visual_rag/raw-data/:/raw-data:rw \
          --volume storage:visual_rag/lancedb-data/:/lancedb-data:rw \
          -e HF_TOKEN=$HF_TOKEN \
          ghcr.io/kyryl-opens-ml/apolo_visual_rag:latest -- python main.py ingest-data /raw-data --table-name=demo --db-path=/lancedb-data/datastore
```

Generative LLM:

```bash
apolo run --detach \
          --no-http-auth \
          --preset H100x1 \
          --name generation-inference \
          --http-port 80 \
          --volume storage:visual_rag:/models:rw \
          -e HF_TOKEN=$HF_TOKEN \
          ghcr.io/huggingface/text-generation-inference:2.4.0 -- --model-id meta-llama/Llama-3.2-11B-Vision-Instruct
```

Ask data and vLLM:

```bash
apolo run --detach \
          --no-http-auth \
          --preset H100x1 \
          --name ask-data \
          --http-port 80 \
          --volume storage:visual_rag/cache:/root/.cache/huggingface:rw \
          --volume storage:visual_rag/raw-data/:/raw-data:rw \
          --volume storage:visual_rag/lancedb-data/:/lancedb-data:rw \
          -e HF_TOKEN=$HF_TOKEN \
          ghcr.io/kyryl-opens-ml/apolo_visual_rag:latest -- python main.py ask-data --user-query="Market share by region?" --table-name=demo --db-path=/lancedb-data/datastore
```

Bring together in UI:

```bash
apolo run --detach \
          --no-http-auth \
          --preset H100x1 \
          --name ask-data-ui \
          --http-port 8080 \
          --volume storage:visual_rag/cache:/root/.cache/huggingface:rw \
          --volume storage:visual_rag/raw-data/:/raw-data:rw \
          --volume storage:visual_rag/lancedb-data/:/lancedb-data:rw \
          -e HF_TOKEN=$HF_TOKEN \
          ghcr.io/kyryl-opens-ml/apolo_visual_rag:latest -- streamlit run --server.address 0.0.0.0 --server.port 8080 app.py
```



## References

- [ColPali: Efficient Document Retrieval with Vision Language Models](https://github.com/illuin-tech/colpali)
- [Remove Complexity from Your RAG Applications](https://kyrylai.com/2024/09/09/remove-complexity-from-your-rag-applications/)

