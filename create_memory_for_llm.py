from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step1: Load the PDF files
DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader,
    )

    documents = loader.load()
    return documents
documents = load_pdf_files(DATA_PATH)
print("Length of PDF pages:",len(documents))

# Step2: Split the documents into smaller chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks=create_chunks(documents)
print("Length of text chunks:",len(text_chunks))

# Step3: Create embeddings for the text chunks

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model = get_embedding_model()
print("Embedding model loaded successfully")


# Step4: Store the embeddings in a vector store FAISS
DB_FAISS = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS)
print("FAISS vector store created and saved successfully")
