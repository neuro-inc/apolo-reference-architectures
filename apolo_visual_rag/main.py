import base64
import io
import os
from pathlib import Path
from typing import List, Tuple, cast

import lancedb
import numpy as np
import PIL
import PIL.Image
import requests
import torch
import typer
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from openai import OpenAI
from pdf2image import convert_from_path
from pypdf import PdfReader
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_base64_image(img: str | PIL.Image.Image, add_url_prefix: bool = True) -> str:
    """
    Convert an image (from a filepath or a PIL.Image object) to a JPEG-base64 string.
    """
    if isinstance(img, str):
        img = PIL.Image.open(img)
    elif isinstance(img, PIL.Image.Image):
        pass
    else:
        raise ValueError("`img` must be a path to an image or a PIL Image object.")

    buffered = io.BytesIO()
    img.save(buffered, format="jpeg")
    b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64_data}" if add_url_prefix else b64_data


def base64_to_pil(base64_str: str) -> PIL.Image.Image:
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    image = PIL.Image.open(io.BytesIO(image_data))
    return image


def download_pdf(url: str, save_directory: str = "."):
    response = requests.get(url)
    if response.status_code == 200:
        # Check for Content-Disposition header to get the filename
        if "Content-Disposition" in response.headers:
            # Extract filename from header if available
            filename = response.headers.get("Content-Disposition").split("filename=")[-1].strip('"')
        else:
            # Fallback: Use the last part of the URL as filename
            filename = os.path.basename(url)
            # Ensure the file has a .pdf extension
            if not filename.endswith(".pdf"):
                filename += ".pdf"

        # Save the file to the specified directory
        file_path = os.path.join(save_directory, filename)
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"PDF downloaded and saved as {file_path}")
        return file_path
    else:
        raise Exception(f"Failed to download PDF: Status code {response.status_code}")


def get_pdf_images(pdf_path: str | Path) -> Tuple[List[PIL.Image.Image], List[str]]:
    reader = PdfReader(pdf_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)

    page_images = convert_from_path(pdf_path)
    assert len(page_images) == len(page_texts)
    return page_images, page_texts


def get_model_colpali() -> Tuple[ColPali, ColPaliProcessor]:
    device = get_torch_device("auto")
    print(f"Device used: {device}")

    # Model name
    model_name = "vidore/colpali-v1.2"

    # Load model
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    # Load processor
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

    if not isinstance(processor, BaseVisualRetrieverProcessor):
        raise ValueError("Processor should be a BaseVisualRetrieverProcessor")
    return model, processor



def get_images_embedding(images: List[PIL.Image.Image], model: ColPali, processor: ColPaliProcessor) -> List[torch.Tensor]:
    # Run inference - docs
    dataloader = DataLoader(
        dataset=images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    ds: List[torch.Tensor] = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad() and torch.autocast(device_type="cuda"):
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return ds

def get_query_embedding(query: str, model: ColPali, processor: ColPaliProcessor):
    batch_queries = processor.process_queries([query]).to(model.device)

    with torch.no_grad() and torch.autocast(device_type="cuda"):
        query_embeddings = model(**batch_queries)

    return query_embeddings


def add_to_db(pdf_path: str | Path, page_images, page_texts, page_embeddings, table_name: str = "demo", db_path: str = "lancedb"):
    assert len(page_images) == len(page_texts) == len(page_embeddings)

    db = lancedb.connect(db_path)


    data = []
    for page_idx in range(len(page_images)):

        record = {
            "name": str(pdf_path),
            "page_texts": page_texts[page_idx],
            "image": get_base64_image(page_images[page_idx]),
            "page_idx": page_idx,

            "page_embedding_flatten": page_embeddings[page_idx].float().numpy().flatten(),
            "page_embedding_shape": page_embeddings[page_idx].float().numpy().shape
        }
        data.append(record)
    if table_name not in db.table_names():
        table = db.create_table(table_name, data)
    else:
        table = db.open_table(table_name)
        table.add(data)
    return table





def search_db(query_embeddings: str, processor, db_path: str = "lancedb", table_name: str = "demo", top_k: int = 3):

    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    r = table.search().limit(None).to_polars()
    
    def process_patch_embeddings(x):
        patches = np.reshape(x[4], x[5])
        return torch.from_numpy(patches).to(torch.float)
    
    image_embeddings = [process_patch_embeddings(r.row(idx)) for idx in range(len(r))]
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)

    top_k_indices = torch.topk(scores, k=top_k, dim=1).indices

    results = []
    for idx in top_k_indices[0]:
        name, _, image, page_idx, _, _ = r.row(idx)
        pil_image = base64_to_pil(image)
        result = {"name": name, "page_idx": page_idx, "pil_image": pil_image}
        results.append(result)
    return results

def run_vision_inference(input_images: List[PIL.Image.Image], prompt: str, base_url: str):
    client = OpenAI(base_url=base_url, api_key="-")

    content = [
                {"type": "text", "text": prompt},
                ]

    for idx in range(len(input_images)):
        content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": get_base64_image(input_images[idx].resize((512, 512)))
                        },
                    })
    print(f"content = {len(content)}")
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
        stream=False, 
    )

    text_response = chat_completion.choices[0].message.content
    return text_response

def ingest_data(folder_with_pdfs: str, table_name: str = "demo", db_path: str = "lancedb"):
    model, processor = get_model_colpali()

    pdfs = [x for x in Path(folder_with_pdfs).iterdir() if x.name.endswith('.pdf')]
    print(f"Input PDFs {pdfs}")

    for pdf_path in tqdm(pdfs):
        print(f"Getting images and text from {pdf_path}")
        page_images, page_texts = get_pdf_images(pdf_path=pdf_path)
        print(f"Getting embeddings from {pdf_path}")
        page_embeddings = get_images_embedding(images=page_images, model=model, processor=processor)
        print(f"Adding to db {pdf_path}")
        table = add_to_db(pdf_path=pdf_path, page_images=page_images, page_texts=page_texts, page_embeddings=page_embeddings, table_name=table_name, db_path=db_path)
        print(f"Done! {pdf_path} should be in {table} table.")
    print("All files are processed")
    
def ask_data(user_query = "What is market share by region?", table_name: str = "demo", db_path: str = "lancedb", base_url: str = "http://generation-inference--9771360698.jobs.scottdc.org.neu.ro/v1", top_k: int = 5):
    model, processor = get_model_colpali()
    print(f"Asking {user_query} query.")

    print("1. Search relevant images")
    query_embeddings = get_query_embedding(query=user_query, model=model, processor=processor)
    results = search_db(query_embeddings=query_embeddings, processor=processor, db_path=db_path, table_name=table_name, top_k=top_k)
    print(f"result most relevant {results}")

    print("2. Build prompt")
    # https://cookbook.openai.com/examples/custom_image_embedding_search#user-querying-the-most-similar-image
    prompt = f"""
    Below is a user query, I want you to answer the query using images provided.
    user query:
    {user_query}
    """    
    print(f"Prompt = {prompt}")
    print("3. Query LLM with prompt and relavent images")
    input_images = [results[idx]['pil_image'] for idx in range(top_k)]
    llm_response = run_vision_inference(input_images=input_images, prompt=prompt, base_url=base_url)
    print(f"llm_response {llm_response}")


def cli():
    app = typer.Typer()
    app.command()(ingest_data)
    app.command()(ask_data)
    app()

if __name__ == '__main__':
    cli()
