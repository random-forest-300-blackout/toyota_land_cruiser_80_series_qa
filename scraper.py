import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

from datetime import date
today = date.today().strftime("%Y%m%d")

# Define the base URL
base_url = "https://forum.ih8mud.com/forums/80-series-tech.9/"

# Set headers to mimic a real browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

proxies = {
    'http': 'http://proxy_ip:proxy_port',
    'https': 'https://proxy_ip:proxy_port'
}

# Function to extract thread links from a page
def get_thread_links(soup):
    threads = []
    for thread in soup.find_all('div', class_='structItem-title'):
        thread_link = thread.find('a')['href']
        threads.append(thread_link)
    return threads

# Function to scrape individual thread content including message, author, and date
def scrape_thread(thread_url):
    full_thread_url = f"https://forum.ih8mud.com{thread_url}"

    from stem import Signal
    from stem.control import Controller
    thread_response = requests.get(full_thread_url, headers=headers)
    thread_soup = BeautifulSoup(thread_response.content, 'html.parser')

    # Find all posts (messages) in the thread
    messages = thread_soup.find_all('article', {'class': 'message-body'})
    authors = thread_soup.find_all('a', {'class': 'username'})
    dates = thread_soup.find_all('time', {'class': 'u-dt'})

    thread_data = []

    # Loop through each message, author, and date
    for i in range(len(messages)):
        try:
            message_content = messages[i].get_text(strip=True)
            author = authors[i].get_text(strip=True)
            date = dates[i]['datetime']  # Date is usually in the 'datetime' attribute
            thread_data.append([message_content, author, date])
        except IndexError:
            continue  # In case of any mismatch in length, skip that entry

    return thread_data

    """

    with Controller.from_port(port=9051) as controller:
    
        controller.authenticate()
        controller.signal(Signal.NEWNYM)

        proxies = {
            'http': 'socks5h://127.0.0.1:9050',
            'https': 'socks5h://127.0.0.1:9050'
        }
    
        thread_response = requests.get(full_thread_url, headers=headers, proxies=proxies)
        thread_soup = BeautifulSoup(thread_response.content, 'html.parser')

        # Find all posts (messages) in the thread
        messages = thread_soup.find_all('article', {'class': 'message-body'})
        authors = thread_soup.find_all('a', {'class': 'username'})
        dates = thread_soup.find_all('time', {'class': 'u-dt'})

        thread_data = []

        # Loop through each message, author, and date
        for i in range(len(messages)):
            try:
                message_content = messages[i].get_text(strip=True)
                author = authors[i].get_text(strip=True)
                date = dates[i]['datetime']  # Date is usually in the 'datetime' attribute
                thread_data.append([message_content, author, date])
            except IndexError:
                continue  # In case of any mismatch in length, skip that entry

        return thread_data
        """

# Function to scrape all threads across multiple pages
def scrape_forum_with_pagination(base_url, max_pages=3511):
    all_threads = []
    
    # Loop through each page
    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}")
        
        # Build the URL for each paginated page
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}page-{page}"

        # Get the page content
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}")
            continue
        
        soup = BeautifulSoup(response.content, 'html.parser')

        # Get the thread links from the current page
        threads = get_thread_links(soup)
        if not threads:
            print(f"No threads found on page {page}")
            break

        all_threads.extend(threads)
        #time.sleep(1)  # Be respectful by adding a delay between requests

    return all_threads

# Initialize a list to store conversations from each thread
conversations = []

# Get all threads from all pages
max_pages_to_scrape = 3511  # Define how many pages to scrape
threads = scrape_forum_with_pagination(base_url, max_pages=max_pages_to_scrape)

# Loop through each thread and scrape conversations
for thread in threads:
    try:
        print(f"Scraping thread: {thread}")
        thread_data = scrape_thread(thread)
        conversations.append({
            'thread_url': thread,
            'conversation': thread_data
        })
        #time.sleep(2)  # Be respectful by adding a delay between requests
    except Exception as e:
        print(e)
        pass

# Save the scraped data to a CSV file
df = pd.DataFrame(conversations)
df.to_csv(f'/Users/loganpiersall/Documents/data_science/generative_ai/toyota_land_cruiser_80_series/forum_conversations_with_details_{today}.csv', index=False)

print("Scraping complete. Data saved to forum_conversations_with_details.csv.")

