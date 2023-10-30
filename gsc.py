import pandas as pd
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from tqdm.asyncio import tqdm_asyncio
from cachetools import TTLCache

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for storing webpage content
cache = TTLCache(maxsize=1000, ttl=86400)


# Function to compute semantic similarity
def bert_similarity(query_embeddings, text_chunks):
    max_similarities = {query: 0 for query in query_embeddings}
    for chunk in text_chunks:
        chunk_embedding = model.encode(chunk, convert_to_tensor=True)
        for query, query_embedding in query_embeddings.items():
            cos_sim = util.pytorch_cos_sim(query_embedding, chunk_embedding)
            max_similarities[query] = max(max_similarities[query], cos_sim.item())
    return max_similarities


# Asynchronous function to fetch webpage content
async def fetch(session, page):
    if page in cache:
        return cache[page]
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with session.get(page, timeout=20, headers=headers) as response:
            if response.status == 200:
                text = await response.text()
                cache[page] = text
                return text
            else:
                print(f"Error {response.status} fetching {page}")
                return ""
    except asyncio.TimeoutError:
        print(f"Timeout error fetching {page}")
        return ""
    except Exception as e:
        print(f"General Error fetching {page}: {e}")
        return ""


# Function to process each page
async def process_page(page, queries, content, query_embeddings):
    print(f"Starting to process: {page}")
    if content:
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        text_chunks = text.split('\n')[:100]  # Limiting the text processed
        similarities = bert_similarity(query_embeddings, text_chunks)
        results = {query: similarities[query] > 0.7 for query in queries}  # Adjust the threshold as needed
        print(f"\n Finished processing: {page}")
        return results
    print(f" No content for: {page}")
    return {query: False for query in queries}


# Main asynchronous function
async def main(df):
    unique_queries = set(df['Query'])
    query_embeddings = {query: model.encode(query, convert_to_tensor=True) for query in unique_queries}
    grouped = df.groupby('Page')
    connector = aiohttp.TCPConnector(limit_per_host=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        all_results = {}
        for page, group in tqdm_asyncio(grouped):
            content = await fetch(session, page)
            results = await process_page(page, group['Query'].tolist(), content, query_embeddings)
            all_results.update({(page, query): result for query, result in results.items()})
    return all_results


# Load the CSV file
df = pd.read_csv('data.csv')

# Run the main function
page_query_results = asyncio.run(main(df))
df['Query Covered'] = df.apply(lambda row: page_query_results.get((row['Page'], row['Query']), False), axis=1)

# Save the results
df.to_csv('output_file.csv', index=False)
