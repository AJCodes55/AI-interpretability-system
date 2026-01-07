from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import uuid

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "olmo-3:7b"
TEMPERATURE = 0.2

embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = Ollama(
        model= LLM_MODEL,
        temperature= TEMPERATURE
    )

def read_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    knowledge_chunks = []
    for page_index, page in enumerate(reader.pages):
        
        page_content = page.extract_text()
        if len(page_content) < 20:
            continue
        
        chunks = splitter.split_text(page_content)
        for chunk_index in chunks:
            chunk_id = str(uuid.uuid4())
            embedding = embeddings_model.embed_query(chunk_index)

            knowledge_prompt = f""" you are knowledge extractor for a rental agreement.
            I want you to understand the following text and extract factual things about it

            TEXT: {chunk_index} """

            knowledge = llm.invoke(knowledge_prompt)
            knowledge_chunks.append({chunk_id: "chunk_id" ,
            embedding: "embedding" ,
            knowledge: "knowledge" , 
            chunk_index: "chunk_index",
            page_index+1: "page_content"})

    
    return knowledge_chunks







