# Multimodal RAG Pipeline

A robust Retrieval-Augmented Generation (RAG) system that processes and queries complex documents containing text, images, tables, and plots. This pipeline leverages LangChain and the Unstructured library to create an intelligent document processing and querying system.

## Features

- **Multimodal Content Extraction**
  - Text extraction from PDFs and text files
  - Image extraction and processing
  - Table detection and parsing
  - Plot recognition and analysis

- **Advanced Processing**
  - Automatic content summarization using LLMs
  - Embedding generation for efficient retrieval
  - Intelligent query processing
  - Context-aware response generation

## Directory Structure
.
├── images/ # Extracted images from documents
├── tables/ # Extracted tables in structured format
├── texts/ # Extracted text content
├── summaries/ # Generated content summaries
├── vector_db/ # Vector embeddings storage
└── query_results/ # Query processing results

## Requirements

- Python 3.10+
- LangChain
- Unstructured
- Additional dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

bash
git clone https://github.com/yourusername/multimodal-rag-pipeline.git
cd multimodal-rag-pipeline

2. Install dependencies:

bash
pip install -r requirements.txt

## Usage

1. Document Processing:
2. Query the System:
python
from pipeline import QueryEngine
engine = QueryEngine()
results = engine.query("Your question about the document")

## Configuration

Configure the system by modifying `config.yaml`:
- Set up model parameters
- Configure extraction settings
- Adjust processing pipelines
- Customize storage locations

## Development

- Follow PEP 8 style guidelines
- Run tests using pytest
- Submit pull requests with clear descriptions
- Update documentation as needed

## Testing

Run the test suite:
bash
pytest tests/

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

If you have any questions or suggestions, please contact me at [jesper_vang (at) hotmail.com](mailto:jesper_vang (at) hotmail.com).