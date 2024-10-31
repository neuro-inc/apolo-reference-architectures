from pathlib import Path

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

# Import your custom functions from your codebase
from main import (
    get_model_colpali,
    get_query_embedding,
    run_vision_inference,
    search_db,
)

st.set_page_config(layout="wide")

# Cache the model and processor to improve performance
@st.cache_resource
def load_model_and_processor():
    model, processor = get_model_colpali()
    return model, processor

def main():
    st.title("PDF Query App")

    # Sidebar inputs for configuration
    st.sidebar.header("Configuration")
    pdf_folder = st.sidebar.text_input("PDF Folder Path", value="./pdfs")
    db_path = st.sidebar.text_input("LanceDB Path", value="lancedb")
    table_name = st.sidebar.text_input("Table Name", value="demo")
    base_url = st.sidebar.text_input("VLLM Base URL", value="http://generation-inference--9771360698.jobs.scottdc.org.neu.ro/v1")

    # Verify if the PDF folder exists
    pdf_folder_path = Path(pdf_folder)
    if not pdf_folder_path.exists():
        st.error(f"The folder '{pdf_folder}' does not exist.")
        return

    # Retrieve all PDF files in the folder
    pdf_files = list(pdf_folder_path.glob("*.pdf"))
    if not pdf_files:
        st.warning("No PDF files found in the specified folder.")
        return

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # First Column: Display PDFs using streamlit_pdf_viewer
    with col1:
        st.header("PDF Files")
        for pdf_file in pdf_files:
            st.subheader(pdf_file.name)
            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()
            pdf_viewer(pdf_bytes)

    # Second Column: Search functionality
    with col2:
        st.header("Search")
        user_query = st.text_input("Enter your question:")
        search_button = st.button("Search")

        if search_button and user_query:
            st.write("Searching for relevant images...")
            model, processor = load_model_and_processor()
            query_embeddings = get_query_embedding(query=user_query, model=model, processor=processor)
            results = search_db(
                query_embeddings=query_embeddings,
                processor=processor,
                db_path=db_path,
                table_name=table_name,
                top_k=3
            )
            if results:
                st.write("Top relevant images:")
                for idx, result in enumerate(results):
                    pil_image = result['pil_image']
                    name = result['name']
                    page = result['page_idx'] + 1

                    st.image(pil_image, caption=f"Result {idx+1}")
                    st.write(f"document name {name} page {page}")
            else:
                st.write("No relevant images found.")
        else:
            st.write("Please enter a question and click 'Search'.")

    # Third Column: Display VLLM response
    with col3:
        st.header("vLLM Response")
        if 'results' in locals() and results and user_query:
            st.write("Generating response...")
            input_images = [x['pil_image'] for x in results]

            prompt = f"""
            Below is a user query, I want you to answer the query using images provided.
            user query:
            {user_query}
            """    
                        
            response = run_vision_inference(
                input_images=input_images,
                prompt=prompt,
                base_url=base_url
            )
            st.write(response)
        else:
            st.write("Please perform a search to see the VLLM response.")

if __name__ == "__main__":
    main()