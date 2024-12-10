from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
import argparse
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.api.types import Metadata
from dotenv import load_dotenv
import PyPDF2
from PyPDF2 import PdfReader
import shutil
import os
import uuid
import getpass
import os

os.environ["MISTRAL_API_KEY"] = "API_KEY"

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model="mistral-large-latest")
PDF_DIR = "./pdfs"

#Directory containing PDFs to upload 
SOURCE_DIR = ""

def uploads():
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
    else:
        print(f"{PDF_DIR} already exists.")

    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory {SOURCE_DIR} does not exists")
        return 

    pdf_files = [f for f in os.listdir(SOURCE_DIR) if f.endswitch('.pdf')]
    if not pdf_files:
        print(f"NO PDF files found in {SOURCE_DIR}")
        return

    for filename in pdf_files:
        source_path = os.path.join(SOURCE_DIR, filename)
        target_path = os.path.join(PDF_DIR, filename)
        shutil.copy(source_path, target_path)
        print(f"Uploaded {filename} to {PDF_DIR}.")
    print(f"Uploaded {len(pdf_files)} files.")

def list_uploaded_pdfs():
    print("\n------Uploaded PDFs------")
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswitch('.pdf')]
    if pdf_files:
        for pdf_files in pdf_files:
            print(pdf_file)
    else:
        print("no PDF files uploaded yet.")

uploads()
list_uploaded_pdfs()

load_dotenv()

CHROMA_PATH = "/chroma"
PDF_DIRECTORY = PDF_DIR


def load_and_split_pdfs(PDF_DIRECTORY):
    chunks = []

    # Iterate over all PDFs in the directory
    for pdf_file in os.listdir(PDF_DIRECTORY):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
            print(f"Processing {pdf_file}...")

            # Load the PDF and extract text (PyPDFLoader)
            reader = PdfReader(pdf_path)
            document_text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                document_text += page.extract_text()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=200,
                length_function=len,
                add_start_index=True,
            )
            document_chunks = text_splitter.split_text(document_text)
            chunks.extend(document_chunks)  # Add the chunks to the list
            print(f"Split {pdf_file} into {len(document_chunks)} chunks.")

    return chunks

def load_chroma_db(embedding_function=None):
    if os.path.exists(CHROMA_PATH):
        # Load existing DB with the specified embedding function
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        print(f"Loaded existing Chroma database from {CHROMA_PATH}")
    else:
        # Create a new DB if it doesn't exist
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        print(f"Created new Chroma database at {CHROMA_PATH}")
    return db

def create_structured_chunks(chunks, metadata=None):
    structured_chunks = []
    for i, chunk in enumerate(chunks):
        # Create the structured Document object with content and optional metadata
        document = Document(
            page_content=chunk,  # The actual chunk text
            metadata=metadata[i] if metadata else {}  # Metadata for the chunk
        )
        structured_chunks.append(document)

    return structured_chunks

def save_to_chroma(chunks, metadata=None):
    # Create an instance of the embeddings model
    embedding_function = MistralAIEmbeddings()

    db = load_chroma_db(embedding_function=embedding_function)  # Load existing DB or create a new one if none exists

    structured_chunks = create_structured_chunks(chunks, metadata)

    try:
        # Save the structured chunks to the Chroma database
        db.add_documents(structured_chunks)  # No need to pass an id; Chroma handles it
        db.persist()  # Persist the database with new documents
        print(f"Added {len(structured_chunks)} new chunks to Chroma at {CHROMA_PATH}.")
    except Exception as e:
        print(f"Error while saving to Chroma: {e}")

    return db

def delete_from_chroma(filename):
    db = load_chroma_db()  # Load existing DB

    try:
        # Assuming each chunk has metadata that contains the filename
        db.delete_documents(where={"filename": filename})  # Delete based on filename
        db.persist()  # Save changes to the DB
        print(f"Deleted documents related to {filename} from Chroma.")
    except Exception as e:
        print(f"Error while deleting from Chroma: {e}")

    return db
def create_metadata_for_chunks(pdf_filename, num_chunks):
    metadata = [{"filename": pdf_filename, "chunk_num": i + 1} for i in range(num_chunks)]
    return metadata





PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():

      # Assume you have chunks and the filename for the PDF
    pdf_filename = "data.csv"
    chunks = load_and_split_pdfs(PDF_DIRECTORY)

    # Create metadata for the chunks
    metadata = create_metadata_for_chunks(pdf_filename, len(chunks))

    # Save chunks to Chroma
    save_to_chroma(chunks, metadata=metadata)

    # Step 3: Define the query and create the prompt
    query_text = "Act as a Hiring Manager and give this cover letter marks based on getting this job"

    # Retrieve the first chunk for demonstration purposes (In real use, you'd search for relevant chunks)
    context_text = chunks[0]
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("Generated Prompt:")
    print(prompt)

    # Step 4: Use the model to answer the question based on the prompt
    model = ChatMistralAI()
    response_text = model.predict(prompt)

    formatted_response = f"Response: {response_text}"
    print(formatted_response)


if __name__ == "__main__":
    main()



