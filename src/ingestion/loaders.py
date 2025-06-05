# intelligent_rag_system/src/ingestion/loaders.py
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import PDFReader, CSVReader, MarkdownReader
from llama_index.readers.json import JSONReader

from bs4 import BeautifulSoup
import json

class DocumentLoader:
    """
    Handles loading documents from various file types into LlamaIndex Document objects.
    Adheres to SRP by focusing solely on data loading.
    """
    def __init__(self):
        pass
    
    def load_markdown_docs(self, directory_path: Path) -> List[Document]:
        """Loads markdown files from a directory."""
        print(f"Loading markdown documents from: {directory_path}")
        reader = SimpleDirectoryReader(
            input_dir=str(directory_path),
            file_extractor={".md": MarkdownReader()},
            required_exts=[".md"],
        )
        return reader.load_data()
    
    def load_text_docs(self, directory_path: Path) -> List[Document]:
        """Loads plain text files from a directory."""
        print(f"Loading text documents from: {directory_path}")
        reader = SimpleDirectoryReader(
            input_dir=str(directory_path),
            required_exts=[".txt"],
        )
        return reader.load_data()
    
    def load_pdf_docs(self, directory_path: Path) -> List[Document]:
        """Loads PDF files from a directory."""
        print(f"Loading PDF documents from: {directory_path}")
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            file_extractor={".pdf": PDFReader()},
            required_exts=[".pdf"],
        )
        return reader.load_data()
    
    def load_csv_docs(self, directory_path: Path) -> List[Document]:
        """Loads CSV files from a directory."""
        print(f"Loading CSV documents from: {directory_path}")
        reader = SimpleDirectoryReader(
            input_dir=str(directory_path),
            file_extractor={".csv": CSVReader(concat_rows=False)},
            required_exts=[".csv"]
        )
        return reader.load_data()
    
    def load_html_docs(self, directory_path: Path) -> List[Document]:
        """
        Loads HTML files and extracts text content.
        Uses BeautifulSoup for robust HTML parsing.
        """
        print(f"Loading HTML documents from: {directory_path}")
        html_docs = []
        for file_path in directory_path.glob("*.html"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                    
                    # Extract text from common content tags
                    text_content = ' '.join(p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']))
                    
                    # Clean up extra whitespace
                    text_content = ' '.join(text_content.split()).strip()
                    
                    html_docs.append(Document(text=text_content))
            except Exception as e:
                print(f"Error loading HTML file {file_path}: {e}")
        return html_docs
    
    def load_json_docs(self, directory_path: Path) -> List[Document]:
        """Loads JSON files from a directory."""
        print(f"Loading JSON documents from: {directory_path}")
        reader = SimpleDirectoryReader(
            input_dir=str(directory_path),
            file_extractor={".json": JSONReader(levels_back=0, is_jsonl=True)},
            required_exts=[".json"]
        )
        return reader.load_data()
        
if __name__ == "__main__":
    from config.settings import settings

    loader = DocumentLoader()

    # Test Markdown
    md_docs = loader.load_markdown_docs(settings.RAW_DOCS_DIR)
    print(f"\nLoaded {len(md_docs)} Markdown documents.")
    if md_docs:
        print(f"First Markdown doc content snippet: {md_docs[0].text[:200]}...")
        print(f"First Markdown doc metadata: {md_docs[0].metadata}")

    # Test Text
    txt_docs = loader.load_text_docs(settings.RAW_DOCS_DIR)
    print(f"\nLoaded {len(txt_docs)} Text documents.")
    if txt_docs:
        print(f"First Text doc content snippet: {txt_docs[0].text[:200]}...")

    # Test PDF (ensure q2_report.pdf exists in data/raw/docs)
    pdf_docs = loader.load_pdf_docs(settings.RAW_DOCS_DIR)
    print(f"\nLoaded {len(pdf_docs)} PDF documents.")
    if pdf_docs:
        print(f"First PDF doc content snippet: {pdf_docs[0].text[:200]}...")

    # Test HTML (ensure landing_page.html exists in data/raw/web_content)
    html_docs = loader.load_html_docs(settings.RAW_WEB_CONTENT_DIR)
    print(f"\nLoaded {len(html_docs)} HTML documents.")
    if html_docs:
        print(f"First HTML doc content snippet: {html_docs[0].text[:200]}...")

    # Test CSV
    employees_data = loader.load_csv_docs(settings.RAW_STRUCTURED_DIR)
    print(f"\nLoaded {len(employees_data)} employees from CSV.")
    if employees_data:
        print(f"First employee record: {employees_data[0]}")

    # Test JSON
    products_data = loader.load_json_docs(settings.RAW_STRUCTURED_DIR)
    print(f"\nLoaded {len(products_data)} products from JSON.")
    if products_data:
        print(f"First product record: {products_data[0]}")