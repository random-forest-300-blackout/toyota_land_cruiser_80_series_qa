import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Page Configuration ---
st.set_page_config(
    page_title="Toyota Land Cruiser Forum Q&A",
    page_icon="🚙",
    layout="wide"
)

# --- Caching ---
# Cache the model and vector store loading to avoid reloading on every interaction
@st.cache_resource
def load_resources():
    """Loads the embedding model and the FAISS vector store."""
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        
        vector_store = FAISS.load_local(
            "faiss_index_forum_10k_sample", 
            embedding_model,
            # This is needed for newer langchain versions
            allow_dangerous_deserialization=True 
        )
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None
        
    return embedding_model, vector_store

# --- Main Application ---

st.title("🚙 Toyota Land Cruiser 80 Series Forum Q&A")
st.markdown("""
Welcome! This app is a demonstration of a Retrieval-Augmented Generation (RAG) system. 
Ask a question about the Toyota Land Cruiser 80 Series, and the model will search a database of real forum conversations to find the most relevant answers.

**How it works:**
1.  Over 10,000 forum comments were scraped, cleaned, and processed.
2.  Each comment was converted into a numerical vector (embedding).
3.  When you ask a question, it's also converted into a vector.
4.  The app performs a similarity search to find the most relevant comments from the forum database.

Try asking questions like: *'How often should I change the oil?'* or *'What is the best way to deal with rust on the frame?'*
""")

# Load the model and vector store
embedding_model, vector_store = load_resources()

if embedding_model and vector_store:
    # --- User Input ---
    query = st.text_input("Enter your question about the 80 Series Land Cruiser:", "")

    if query:
        with st.spinner("Searching for the best answers..."):
            # Perform the similarity search
            results = vector_store.similarity_search_with_score(query, k=3)

            st.subheader("Top 3 Most Relevant Comments:")

            if not results:
                st.warning("No relevant comments found. Try rephrasing your question.")
            else:
                # Display the results
                for i, (doc, score) in enumerate(results):
                    with st.container(border=True):
                        st.markdown(f"**Result {i+1}** (Relevance Score: {score:.4f})")
                        st.markdown(f"**Author:** `{doc.metadata.get('author', 'N/A')}`")
                        st.markdown(f"**Timestamp:** `{doc.metadata.get('timestamp', 'N/A')}`")
                        st.markdown(f"**Source Thread:** [{doc.metadata.get('thread_title', 'N/A')}]({doc.metadata.get('source', 'N/A')})")
                        st.info(f"**Comment:**\n\n>{doc.page_content.replace('\n', ' ')}")
else:
    st.error("The application could not be loaded. Please check the logs.")