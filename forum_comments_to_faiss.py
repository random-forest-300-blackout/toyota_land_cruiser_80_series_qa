import pandas as pd
import ast
import re
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from collections import Counter
from spellchecker import SpellChecker
import nltk
import random # Import the random module

# --- Required LangChain imports ---
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 0. Setup ---
SAMPLE_SIZE = 50000 # 50k comments
SAMPLE_SIZE_STR = f"{SAMPLE_SIZE//1000}k" if SAMPLE_SIZE % 1000 == 0 else str(SAMPLE_SIZE)
INDEX_NAME = f"faiss_index_forum_{SAMPLE_SIZE_STR}_sample"

# --- 1. Setup and Initial Download for NLTK ---
try:
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading the NLTK 'words' corpus (this only needs to happen once)...")
    nltk.download('words')
from nltk.corpus import words

# --- 2. Enhanced Data Cleaning and Preprocessing ---
print("Setting up the cleaning and processing functions...")

def build_vocabulary(all_texts):
    """Creates a frequency count of all words in the provided text."""
    all_words = []
    for text in tqdm(all_texts, desc="Building vocabulary"):
        all_words.extend(text.split())
    return Counter(all_words)

def identify_misspelled_words(word_counts, english_vocab):
    """Identifies words that are infrequent and not in a standard dictionary."""
    spell = SpellChecker()
    infrequent_words = {word for word, count in word_counts.items() if count < 2}
    unknown_to_pyspellchecker = spell.unknown(infrequent_words)
    misspelled_set = {
        word for word in unknown_to_pyspellchecker
        if word.lower() not in english_vocab and not word.isdigit()
    }
    print(f"Identified {len(misspelled_set)} infrequent/misspelled words to remove.")
    return misspelled_set

def create_cleaning_function(words_to_remove):
    """Creates a function to clean text and remove a specific set of words."""
    def clean_and_normalize_text(text):
        if not isinstance(text, str):
            return ""
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        tokens = [word for word in tokens if word not in words_to_remove]
        return " ".join(tokens)
    return clean_and_normalize_text

def get_thread_title_from_url(thread_url):
    """Derives a title from the thread's URL slug."""
    try:
        title_slug = thread_url.split('/')[-2].split('.')[-2]
        return title_slug.replace('-', ' ').capitalize()
    except IndexError:
        return "Untitled Thread"

def safe_parse_conversation(conversation_str):
    """Safely parses the string representation of a list."""
    if not isinstance(conversation_str, str):
        return []
    try:
        return ast.literal_eval(conversation_str)
    except (ValueError, SyntaxError, MemoryError):
        return []

# --- 3. Main Processing ---
print("Loading data...")
try:
    df = pd.read_csv("forum_conversations_with_details_20240928.csv")
except FileNotFoundError:
    print("Error: The CSV file was not found.")
    exit()

df['conversation_list'] = df['conversation'].apply(safe_parse_conversation)

# --- Collect all comments from the dataframe first ---
print("Collecting all comments from the CSV...")
all_comments_raw = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Reading threads"):
    thread_url = row['thread_url']
    conversation_list = row['conversation_list']
    thread_title = get_thread_title_from_url(thread_url)
    source_url = f"https://forum.ih8mud.com{thread_url}"

    for comment_data in conversation_list:
        if isinstance(comment_data, list) and len(comment_data) == 3:
            comment_text, username, timestamp = comment_data
            if isinstance(comment_text, str) and comment_text.strip():
                all_comments_raw.append({
                    "text": comment_text,
                    "metadata": {
                        "source": source_url,
                        "thread_title": thread_title,
                        "author": username,
                        "timestamp": timestamp
                    }
                })

print(f"Found a total of {len(all_comments_raw)} comments.")

# --- Downsize the dataset by taking a random sample ---
if len(all_comments_raw) > SAMPLE_SIZE:
    print(f"Downsizing dataset to a random sample of {SAMPLE_SIZE} comments...")
    documents_to_process = random.sample(all_comments_raw, SAMPLE_SIZE)
else:
    print("Dataset is smaller than the target sample size. Using all comments.")
    documents_to_process = all_comments_raw

print("-" * 20)

# --- First Pass on the SAMPLE: Extract and clean text to build vocabulary ---
sample_texts = [doc['text'] for doc in documents_to_process]
word_counts = build_vocabulary(sample_texts)
english_vocab = set(words.words())
misspelled_to_remove = identify_misspelled_words(word_counts, english_vocab)
clean_and_normalize_text = create_cleaning_function(misspelled_to_remove)

# --- Final Pass: Create LangChain documents from the sample ---
print("\nProcessing sampled comments with the final cleaning function...")
documents = []
for doc_data in tqdm(documents_to_process, desc="Cleaning sampled comments"):
    cleaned_comment = clean_and_normalize_text(doc_data['text'])
    if cleaned_comment:
        documents.append(Document(
            page_content=cleaned_comment,
            metadata=doc_data['metadata']
        ))

print(f"\nSuccessfully created {len(documents)} cleaned comment documents for indexing.")
print("-" * 20)

# --- 4. Chunking, Embedding, and Indexing ---
print("Chunking any long comments...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks to be indexed: {len(chunks)}.")
print("-" * 20)

print("Loading sentence-transformer embedding model...")
model_name = "all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
print("Model loaded.")
print("-" * 20)

print("Embedding and indexing comment chunks into FAISS...")
start_time = time.time()
vector_store = FAISS.from_documents(tqdm(chunks, desc="Embedding"), embedding_model)
end_time = time.time()

# Use a new name for the downsized index
vector_store.save_local(INDEX_NAME)
print(f"FAISS index created and saved in {end_time - start_time:.2f} seconds.")
print("-" * 20)

# --- Example Usage ---
print("Performing a similarity search on the sampled index...")
loaded_vector_store = FAISS.load_local(
    INDEX_NAME, 
    embedding_model
)
query = "What did mudgudgeon say about buying a rusty vehicle?"
results = loaded_vector_store.similarity_search_with_score(query, k=3)

print(f"Query: '{query}'")
print("\nTop 3 relevant comments found:")
for i, (doc, score) in enumerate(results):
    print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
    print(f"Author: {doc.metadata.get('author', 'N/A')}")
    print(f"Timestamp: {doc.metadata.get('timestamp', 'N/A')}")
    print(f"Thread: '{doc.metadata.get('thread_title', 'N/A')}'")
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Comment: \n{doc.page_content}")