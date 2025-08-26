Toyota Land Cruiser 80 Series Forum Q&A
This project is a Retrieval-Augmented Generation (RAG) system designed to answer questions about the Toyota Land Cruiser 80 Series. It uses a web scraper to gather data from the IH8MUD online forum, processes the text data, and creates a searchable vector database. A Streamlit web application then allows users to ask questions and receive relevant answers extracted from the forum conversations.
🚙 How It Works
The project follows a three-step process:
Scrape Data: The web_scraper.py script fetches thousands of forum posts from the IH8MUD 80-series technical discussion board. The raw conversations are saved to a CSV file.
Create Vector Index: The forum_comments_to_faiss.py script cleans the scraped text, processes a sample of 10,000 comments, and converts them into numerical vectors using a sentence-transformer model. These vectors are then stored in a FAISS index for efficient similarity searching.
Question & Answer: The app.py script launches a Streamlit web application. When a user asks a question, the app converts the query into a vector and searches the FAISS index to find the most similar (and therefore most relevant) forum comments. The top results are then displayed to the user.
🚀 Getting Started
Follow these instructions to get a local copy up and running.
Prerequisites
Python 3.8+
pip
Installation
Clone the repository (or download the files):
Bash
git clone https://your-repository-url.com/project.git
cd project
Create and activate a virtual environment:
macOS/Linux:
Bash
python3 -m venv venv
source venv/bin/activate
Code
* Windows:
Bash
python -m venv venv
.\venv\Scripts\activate
Install the required dependencies:
Bash
pip install -r requirements.txt
⚙️ Usage
To run the application, you must execute the scripts in the following order:
Run the Web Scraper: This script will scrape the forum and create the forum_conversations_with_details_YYYYMMDD.csv file.
Bash
python web_scraper.py
Code
* Note: This process can take a very long time as it scrapes over 3,500 pages.
Generate the FAISS Index: After the scraper finishes, you need to update the filename in forum_comments_to_faiss.py to match the output from the scraper.
Find this line:
Python
df = pd.read_csv("forum_conversations_with_details_20240928.csv")
Code
* And change `20240928` to the date the scraper generated. Then, run the script:
Bash
python forum_comments_to_faiss.py
Code
* This will create a folder named `faiss_index_forum_10k_sample` in your directory.
Launch the Streamlit App: Once the index is created, you can start the web application.
Bash
streamlit run app.py
Code
* Open your web browser and navigate to the local URL provided by Streamlit to start asking questions.
📄 File Descriptions
web_scraper.py: A Python script that uses requests and BeautifulSoup to scrape conversation data from the IH8MUD forum.
forum_comments_to_faiss.py: Processes the raw CSV data. It cleans the text, embeds the comments using a Hugging Face model, and saves them into a FAISS vector index.
app.py: A Streamlit application that provides a user interface for the Q&A system. It loads the FAISS index and allows users to perform similarity searches with their questions.
requirements.txt: A list of all the Python libraries required to run the project.
