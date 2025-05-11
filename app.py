import streamlit as st
import os
from PyPDF2 import PdfReader
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Please check your .env file.")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Set up Streamlit app
st.set_page_config(page_title="Search Bot", layout="wide")
st.title("🔍 Search Bot")

# Create index name for Pinecone
INDEX_NAME = "geo-research-papers"

def create_pinecone_index_if_not_exists():
    """Create Pinecone index if it doesn't exist"""
    if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # Dimension for Gemini Embeddings exp-03-07
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION
            )
        )
        st.success(f"Created new Pinecone index: {INDEX_NAME}")
    return pc.Index(INDEX_NAME)

def process_pdf(pdf_file):
    """Extract and process text from PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Clean the text - remove extra whitespace, fix chemical formulas, etc.
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    return text

def split_text(text):
    """Split text into chunks for embedding"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# def get_custom_embedding(text):
#     model = genai.EmbeddingModel(model_name="gemini-embedding-exp-03-07")
#     result = model.embed_content(content=text)
#     return result["embedding"]


def get_embeddings_model():
    """Initialize the Google Generative AI Embeddings model"""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query"
    )

def get_llm():
    """Initialize the Google Generative AI LLM model"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        convert_system_message_to_human=True
    )

def get_vector_store(chunks, embeddings):
    """Create vector store from document chunks"""
    # Create vector store without pinecone_index parameter
    vector_store = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    return vector_store

def create_qa_chain():
    """Create question-answering chain with Gemini"""
    prompt_template = """
    You are a helpful expert in geological sciences and chemistry. 
    Answer the question based solely on the provided context from research papers.
    If you don't know the answer or can't find it in the context, say so - don't make up information.
    For chemical reactions, format them properly with correct subscripts and superscripts noted in plain text.
    
    Context: {context}
    
    Question: {question}
    
    Your detailed answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(
        llm=get_llm(),
        chain_type="stuff",
        prompt=prompt
    )
    
    return chain

def main():
    # Initialize Pinecone index
    pinecone_index = create_pinecone_index_if_not_exists()
    
    # Create sidebar for uploading documents
    with st.sidebar:
        st.header("Upload Research Papers")
        uploaded_files = st.file_uploader(
            "Upload geological research papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        process_button = st.button("Process Papers")
    
    # Process uploaded files when the button is clicked
    if uploaded_files and process_button:
        with st.spinner("Processing research papers..."):
            embeddings = get_embeddings_model()
            
            for pdf_file in uploaded_files:
                # Extract text from PDF
                st.write(f"Processing: {pdf_file.name}")
                
                # Process PDF text
                raw_text = process_pdf(pdf_file)
                
                # Split text into chunks
                text_chunks = split_text(raw_text)
                st.write(f"Created {len(text_chunks)} text chunks")
                
                # Create vector store
                vector_store = get_vector_store(text_chunks, embeddings)
                
                st.success(f"Successfully processed: {pdf_file.name}")
    
    # Create the main query section
    st.header("Query Research Papers")
    
    # Get user question
    user_question = st.text_input("Ask a question about the geological research papers:")
    
    if user_question:
        with st.spinner("Searching for answers..."):
            try:
                # Initialize embeddings model
                embeddings = get_embeddings_model()
                
                # Create vector store without pinecone_index parameter
                vector_store = PineconeVectorStore(
                    index_name=INDEX_NAME,
                    embedding=embeddings
                )
                
                # Search for similar documents
                docs = vector_store.similarity_search(user_question, k=4)
                
                if docs:
                    # Create QA chain
                    chain = create_qa_chain()
                    
                    # Get answer
                    response = chain(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                    
                    # Display answer
                    st.header("Answer")
                    st.write(response["output_text"])
                    
                    # Show source documents
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(docs):
                            st.write(f"Source {i+1}")
                            st.write(doc.page_content)
                            st.divider()
                else:
                    st.warning("No relevant information found in the documents.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Make sure you've uploaded and processed documents before asking questions.")

if __name__ == "__main__":
    main()



#  import streamlit as st
