from dotenv import load_dotenv
from langchain_community.document_loaders import  PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate
import os

def doc_preprocessing():
    load_dotenv()

    wcs_cluster_url = os.getenv("WEAVIATE_URL")
    wcs_api_key = os.getenv("WEAVIATE_API_KEY")

    pdf_loader = PyPDFLoader("tip.pdf")
    docs = pdf_loader.load()


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlapkayde=200,
        separators=["\n", "\n\n", ".", "?", "!", " "]
    )
    chunks = text_splitter.split_documents(docs)

    alldocs = list[Document]()
    for i in range(len(chunks)):
        document_for_embedding = chunks[i]
        parts_for_full_context = []

        if i > 0:
            parts_for_full_context.append(chunks[i - 1])

        parts_for_full_context.append(chunks[i])

        if i < len(chunks) - 1:
            parts_for_full_context.append(chunks[i + 1])

        full_context_for_llm_retrieval = " ".join(parts_for_full_context)

        doc = Document(
            page_content=document_for_embedding,
            metadata={
                "full_retrieval_context": full_context_for_llm_retrieval,
                "chunk_index": i
            }
        )
        alldocs.append(doc)


    embedding_s = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcs_cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key=wcs_api_key)
    )

    db=WeaviateVectorStore.from_documents(chunks, embedding_s,client=weaviate_client, index_name="SliChunk")

    retriever =db.as_retriever()

    return retriever


