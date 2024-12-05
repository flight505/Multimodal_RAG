import os

# Set environment variable for tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CLASSES"] = "1"  # Enable torch classes before import

import base64
from io import BytesIO

import streamlit as st
import torch
from byaldi import RAGMultiModalModel
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


# Device configuration
def get_device():
    if torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()

load_dotenv()
client = OpenAI()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Directory setup
upload_dir = "./doc"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Set page layout to wide
st.set_page_config(layout="wide")

st.title("Multimodal RAG App")

# Create sidebar for configuration options
with st.sidebar:
    st.header("Configuration Options")
    st.write(f"Using device: {device}")

    # Dropdown for selecting Colpali model
    colpali_model = st.selectbox(
        "Select Colpali Model", options=["vidore/colpali", "vidore/colpali-v1.2"]
    )

    # Dropdown for selecting Multi-Model LLM
    multi_model_llm = st.selectbox(
        "Select Multi-Model LLM", options=["gpt-4o", "Qwin", "Llama3.2"]
    )

    # File upload button
    uploaded_file = st.file_uploader("Choose a Document", type=["pdf"])


@st.cache_resource
def load_models(colpali_model):
    try:
        RAG = RAGMultiModalModel.from_pretrained(colpali_model, verbose=10)
        # Move model components to device and ensure float32
        if hasattr(RAG.model, "model"):
            RAG.model.model = RAG.model.model.to(device)
            RAG.model.model = RAG.model.model.float()
        if hasattr(RAG.model, "vision_tower"):
            RAG.model.vision_tower = RAG.model.vision_tower.to(device)
            RAG.model.vision_tower = RAG.model.vision_tower.float()
        return RAG
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Load the model
RAG = load_models(colpali_model)

if RAG is None:
    st.error("Failed to load the model. Please check the model configuration.")
    st.stop()

# Main content layout with columns
col1, col2 = st.columns([1, 2])

with col1:
    if uploaded_file is not None:
        st.write("### Uploaded Document")
        save_path = os.path.join(upload_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved: {uploaded_file.name}")

        try:
            RAG.index(
                input_path=save_path,
                index_name="image_index",
                store_collection_with_index=True,
                overwrite=True,
            )
            st.success("Document indexed successfully")
        except RuntimeError as e:
            if "Input type" in str(e):
                st.error("Device compatibility issue detected. Falling back to CPU...")
                # Move model back to CPU if MPS fails
                if hasattr(RAG.model, "model"):
                    RAG.model.model = RAG.model.model.cpu()
                if hasattr(RAG.model, "vision_tower"):
                    RAG.model.vision_tower = RAG.model.vision_tower.cpu()
                try:
                    RAG.index(
                        input_path=save_path,
                        index_name="image_index",
                        store_collection_with_index=True,
                        overwrite=True,
                    )
                    st.success("Successfully indexed document using CPU")
                except Exception as e:
                    st.error(f"Failed to index document: {str(e)}")
            else:
                st.error(f"Error indexing document: {str(e)}")

with col2:
    # Text input for the user query
    text_query = st.text_input("Enter your text query")

    # Search and Extract Text button
    if st.button("Search and Extract Text"):
        if text_query:
            try:
                # Add special tokens to the query
                query_with_tokens = f"<image><bos>{text_query}"
                results = RAG.search(query_with_tokens, k=1, return_base64_results=True)

                if results and len(results) > 0:
                    image_data = base64.b64decode(results[0].base64)
                    image = Image.open(BytesIO(image_data))
                    st.image(image, caption="Result Image", use_container_width=True)

                    response = client.chat.completions.create(
                        model=multi_model_llm,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": query_with_tokens},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{results[0].base64}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=300,
                    )
                    output = response.choices[0].message.content
                    st.subheader("Query with LLM Model")
                    st.markdown(output, unsafe_allow_html=True)
                else:
                    st.warning("No results found for your query.")
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
        else:
            st.warning("Please enter a query.")
