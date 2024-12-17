import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import fitz  # PyMuPDF for fast text extraction
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

# Directories
SOURCE_DIR = "./source_pdfs"
PDF_DIR = "./pdfs"
CHROMA_PATH = "./chroma"

# Reusable ChromaDB initialization
def initialize_chroma_db():
    if not hasattr(initialize_chroma_db, "db"):
        embedding_function = MistralAIEmbeddings()
        initialize_chroma_db.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return initialize_chroma_db.db

# Combined upload and list function
def upload_and_list_pdfs(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    pdf_files = [f for f in os.listdir(source_dir) if f.endswith('.pdf')]
    if not pdf_files:
        logging.warning(f"No PDF files found in {source_dir}.")
        return []

    for filename in pdf_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
        logging.info(f"Uploaded: {filename}")

    logging.info("\nUploaded PDFs:")
    for pdf in pdf_files:
        logging.info(f"- {pdf}")
    return pdf_files

# Text extraction using PyMuPDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return "".join([page.get_text() for page in doc])
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Process PDFs in parallel
def load_and_split_pdfs(pdf_directory, chunk_size=500, chunk_overlap=200):
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    chunks = []

    with ThreadPoolExecutor() as executor:
        extracted_texts = list(executor.map(extract_text_from_pdf, pdf_files))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    for i, text in enumerate(extracted_texts):
        if text:
            filename = os.path.basename(pdf_files[i])
            document_chunks = text_splitter.split_text(text)
            logging.info(f"Split {filename} into {len(document_chunks)} chunks.")
            chunks.extend((filename, chunk) for chunk in document_chunks)

    return chunks

# Create structured chunks with metadata
def create_structured_chunks(chunks):
    return [
        Document(page_content=chunk, metadata={"filename": filename, "chunk_num": i + 1})
        for i, (filename, chunk) in enumerate(chunks)
    ]

# Save chunks to ChromaDB
def save_to_chroma(structured_chunks):
    db = initialize_chroma_db()
    try:
        db.add_documents(structured_chunks)
        db.persist()
        logging.info(f"Added {len(structured_chunks)} chunks to ChromaDB.")
    except Exception as e:
        logging.error(f"Error saving to ChromaDB: {e}")

# Main function
def main():
    # Step 1: Upload and list PDFs
    uploaded_pdfs = upload_and_list_pdfs(SOURCE_DIR, PDF_DIR)
    if not uploaded_pdfs:
        return

    # Step 2: Load and split PDFs
    chunks = load_and_split_pdfs(PDF_DIR)
    if not chunks:
        logging.warning("No chunks generated.")
        return

    # Step 3: Create structured chunks
    structured_chunks = create_structured_chunks(chunks)

    # Step 4: Save to ChromaDB
    save_to_chroma(structured_chunks)

    # Step 5: Query and generate response
    query_text = "Act as a Hiring Manager and give this cover letter marks based on getting this job."
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """)

    # Using only the first chunk for demonstration
    context_text = structured_chunks[0].page_content
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Model prediction
    model = ChatMistralAI()
    try:
        response_text = model.predict(prompt)
        logging.info("Generated Response:")
        logging.info(response_text)
    except Exception as e:
        logging.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
