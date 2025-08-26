# Land Cruiser 80 Series AI Q\&A Assistant

This project is a Retrieval-Augmented Generation (RAG) system designed to answer questions about the Toyota Land Cruiser 80 Series. It uses a knowledge base created from thousands of real-world conversations scraped from the `IH8MUD.com` online forum.

The system is deployed as an interactive web application using Streamlit, where users can ask technical questions and receive relevant answers sourced directly from experienced owners.

## Live Demo

**You can access the live, interactive demo of this application here:**

[**https://your-streamlit-app-url.streamlit.app**](https://www.google.com/search?q=https://your-streamlit-app-url.streamlit.app)  *(\<-- Replace this with your actual Streamlit Cloud URL after deploying)*

## Features

  * **Web Scraping**: Gathers thousands of posts from the IH8MUD 80 Series technical forum.
  * **Advanced Text Cleaning**: Implements a multi-stage cleaning pipeline that removes HTML, normalizes text, and uses a vocabulary-based approach to filter out infrequent, misspelled words.
  * **Data Delineation**: Processes data at the individual comment level, preserving the author and timestamp for each entry.
  * **Vector Embeddings**: Converts cleaned text data into high-dimensional vectors using a `sentence-transformers` model (`all-MiniLM-L6-v2`).
  * **Efficient Similarity Search**: Indexes all comment vectors into a FAISS (Facebook AI Similarity Search) database for fast and accurate retrieval.
  * **Interactive UI**: A simple and intuitive web interface built with Streamlit that allows users to ask questions and view the most relevant comments from the knowledge base.

## Technology Stack

  * **Data Scraping**: `requests`, `BeautifulSoup`
  * **Data Processing**: `pandas`, `nltk`, `pyspellchecker`
  * **LLM & Vector DB Framework**: `langchain`
  * **Vector Embeddings**: `sentence-transformers` (Hugging Face)
  * **Vector Database**: `faiss-cpu`
  * **Web Application**: `streamlit`
  * **Deployment**: Streamlit Community Cloud

## Project Architecture

The project follows a standard RAG pipeline to prepare the data and serve the application:

1.  **Scrape**: The `my_web_scraper.py` script crawls the IH8MUD forum and saves raw conversations to a CSV file.
2.  **Clean & Preprocess**: The `prepare_forum_data.py` script loads the raw data. It performs extensive cleaning, including HTML tag removal, text normalization, and the removal of infrequent, misspelled words identified by checking against a standard English dictionary.
3.  **Chunk**: Each cleaned comment is treated as a `Document`. A text splitter is used to break down any exceptionally long comments into smaller, manageable chunks.
4.  **Embed**: The cleaned text chunks are fed into the `all-MiniLM-L6-v2` model to generate semantic vector embeddings.
5.  **Index**: The embeddings and their associated metadata (author, timestamp, source URL, etc.) are stored in a FAISS vector index, which is saved locally.
6.  **Serve**: The `app.py` script loads the pre-built FAISS index and the embedding model. It provides a simple UI where a user's query is embedded and used to perform a similarity search against the indexed comments. The top results are then displayed.

## Local Setup and Installation

To run this project on your local machine, follow these steps.

#### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### 2\. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\\venv\\Scripts\\activate
```

#### 3\. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

#### 4\. Download NLTK Data

The script for removing misspelled words uses the NLTK `words` corpus. The first time you run the processing script, it will automatically download this for you.

## Usage

The project is divided into two main parts: data preparation and the interactive application.

#### 1\. Prepare the Data

First, you need to run the processing script to generate the vector index from the source CSV file.

**Note:** This script processes 10,000 comments by default to create a demo-sized index that is small enough for free hosting platforms.

```bash
python prepare_forum_data.py
```

This will create a `faiss_index_forum_10k_sample` folder in your project directory containing the `index.faiss` and `index.pkl` files.

#### 2\. Run the Streamlit Application

Once the index is built, you can launch the web app.

```bash
streamlit run app.py
```

This will open a new tab in your browser where you can interact with the Q\&A system.
