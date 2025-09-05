from dotenv import load_dotenv
import os

load_dotenv()
from bs4 import BeautifulSoup
import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ollama
from langchain_ollama import OllamaEmbeddings, ChatOllama
# vector store
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser


llm = ChatOllama(model="llama3.1")

api_key = os.getenv("PINECONE_API_KEY")

pinecone = Pinecone(api_key=api_key)

index_name = "langchain-web-crawler"

if not pinecone.has_index(index_name):
    pinecone.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec= ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
index = pinecone.Index(index_name)
st.title("Web Crawler")

def site_page(url):
    res = requests.get(url)
    
    soup = BeautifulSoup(res.text, 'html.parser')
    
    paragrap = [p.get_text() for p in soup.find_all('p')]
    
    return " ".join(paragrap)


def crawl():
    try:
       

        url = st.text_input("Enter URL to crawl")
        # site = site_page("https://techstudioacademy.com/")

        crawl_btn = st.button("Click to Crawl")

        if crawl_btn:
            if url:
                site = site_page(url) 
                return site
                # st.write(site) 
            else:
                st.error("Please enter a valid URL") 
    except Exception as e:
        st.error(f"Error: {e}")
            
            
def vector():
    try:
        text = crawl()
    
        if text:
            # chunk the text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            chunked_text = text_splitter.split_text(text)
            
            
            # st.write(chunked_text)
            # embened the chunks into a vector representation
            
            embedding = OllamaEmbeddings(model="llama3.1")
            
            # print(len(embedding.embed_documents(chunked_text)))
            
            # store in vector store
            vector_store = PineconeVectorStore(index=index, embedding=embedding)
            
            vector_store.add_texts(chunked_text)
            st.success("Vector store updated with text.")
        else:
            st.error("No text to process. Please provide a valid URL and crawl the site.")
    except Exception as e:
        st.error(f"Error: {e}")
        
        
def search():
       embedding = OllamaEmbeddings(model="llama3.1")
       vector_store = PineconeVectorStore(index=index, embedding=embedding)
       query = st.text_input("Enter query to search")
       
       btn = st.button("Search")
       
       if btn:
           if query:
               result = vector_store.similarity_search(query, k=5)
               
               message = [
                   (
                       "system", f"""
                       You are an assistant that help people find informations based on the documents provided. If the information you are looking for is not available, respond with "I don't have the information".
                       
                       here is the document: {[res.page_content for res in result]}
                       """
                   ),
                   
                   (
                       "human", "{query}"
                   )
               ]
               
               chat = ChatPromptTemplate.from_messages(message)
               
               chain = chat | llm | StrOutputParser()
               
               res = chain.invoke({"query": query})
               
               st.write(res)
        
if __name__ == "__main__":
    # vector()
    search()
    