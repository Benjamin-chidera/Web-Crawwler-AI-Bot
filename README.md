# Web Crawler with Vector Search

This project is a web crawler application built with Python and Streamlit. It allows users to crawl a website, process the content into vector embeddings, and perform similarity-based searches on the crawled data. The application leverages LangChain, Pinecone, and Ollama for text processing, vector storage, and querying.

## Features

- **Web Crawling**: Extracts text content from a given URL.
- **Text Chunking**: Splits the crawled text into smaller chunks for efficient processing.
- **Vector Embedding**: Converts text chunks into vector representations using Ollama embeddings.
- **Vector Storage**: Stores vector embeddings in Pinecone for fast similarity searches.
- **Search Functionality**: Allows users to query the stored data and retrieve the most relevant results.

## Prerequisites

- Python 3.13 or higher
- A Pinecone API key (stored in `.env` file)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Benjamin-chidera/Web-Crawwler-AI-Bot.git
   ```
