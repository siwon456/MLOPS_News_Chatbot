import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import chromadb
import sys
import urllib.parse
import logging
from chromadb.utils import embedding_functions 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_news(keyword, max_crawl_items=100): # <-- ⭐ 여기를 100으로 변경 ⭐
    logging.info(f"Starting detailed news collection for keyword: '{keyword}' (max {max_crawl_items} items)")
    base_url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query="
    encoded_keyword = urllib.parse.quote(keyword)
    news_list = []
    page = 1
    crawled_count = 0

    while crawled_count < max_crawl_items: # <-- ⭐ 여전히 max_crawl_items 기준으로 루프를 돌고 ⭐
        url = f"{base_url}{encoded_keyword}&sort=1&ds=&de=&docid=&nso=so:r,p:all,a:all&start={((page-1)*10) + 1}"
        logging.info(f"Crawling news for '{keyword}' from page {page} (URL: {url})...")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() 
            soup = BeautifulSoup(response.text, 'html.parser')

            news_links = soup.select('div.news_area > div.news_info > div.info_group > a.info')
            
            if not news_links:
                logging.info(f"No more news links found for '{keyword}' on page {page}. Stopping crawl.")
                break

            for link_tag in news_links:
                if 'naver.com/article/' in link_tag['href']: 
                    article_link = link_tag['href']
                    try:
                        article_response = requests.get(article_link, headers=headers, timeout=10)
                        article_response.raise_for_status()
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        
                        title = article_soup.select_one('h2#title_area > span').text.strip() if article_soup.select_one('h2#title_area > span') else '제목 없음'
                        description_tag = article_soup.select_one('meta[property="og:description"]')
                        description = description_tag['content'].strip() if description_tag else '설명 없음'
                        content_tags = article_soup.select('div#newsct_article')
                        content = ''
                        if content_tags:
                            for tag in content_tags:
                                for script_or_ad in tag(['script', 'a', 'strong', 'em', 'span']): 
                                    script_or_ad.extract()
                                content += tag.get_text(separator='\n', strip=True)
                            
                            content = re.sub(r'\s+', ' ', content).strip()
                            content = re.sub(r'\[.*?\]', '', content).strip()
                            content = re.sub(r'\(.*?\)', '', content).strip()
                            
                            content = re.split(r'저작권자 ⓒ 한경닷컴|▶ 네이버에서 서울경제', content)[0].strip()
                            content = re.split(r'기자 =.+?|작가 =.+?|사진 =.+?', content)[0].strip()
                        else:
                            content = '내용 없음'

                        if content != '내용 없음' and len(content) > 100:
                            news_list.append({
                                'title': title,
                                'description': description,
                                'link': article_link,
                                'content': content
                            })
                            crawled_count += 1
                            if crawled_count >= max_crawl_items: # <-- ⭐ 여기에 도달하면 루프 종료 ⭐
                                break
                    except requests.exceptions.RequestException as e:
                        logging.warning(f"Failed to fetch article {article_link}: {e}")
                    except Exception as e:
                        logging.warning(f"Error parsing article {article_link}: {e}")
                time.sleep(0.5) 
            
            page += 1
            time.sleep(1) 
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to crawl news for '{keyword}' from page {page}: {e}")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred during crawling for '{keyword}' on page {page}: {e}")
            break

    logging.info(f"Finished crawling for '{keyword}'. Total items collected: {len(news_list)}")
    return pd.DataFrame(news_list)

def create_and_store_embeddings(df, db_name="my_chroma_db"):
    logging.info(f"Building and embedding the database for '{db_name}'...")
    
    client = chromadb.PersistentClient(path=f"./chroma_db_{db_name}")
    
    chroma_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jhgan/ko-sroberta-multitask"
    )
    logging.info("DEBUG: Initialized ChromaDB compatible SentenceTransformer EmbeddingFunction for data_pipeline.")

    try:
        collection = client.get_or_create_collection(name=db_name, embedding_function=chroma_embedding_function)
        logging.info(f"Accessing ChromaDB collection: '{db_name}'. Current total articles: {collection.count()} before adding.")
    except Exception as e:
        logging.error(f"CRITICAL ERROR: Error accessing/creating ChromaDB collection '{db_name}': {e}")
        import traceback
        traceback.print_exc()
        return False

    df_to_add = []
    existing_links_in_db = set()
    if collection.count() > 0:
        try:
            # collection.get()은 기본적으로 limit가 있으므로 모든 메타데이터를 가져오려면 반복문이 필요
            # 여기서는 편의상 collection.count()를 limit으로 사용.
            all_db_items = collection.get(ids=collection.get(limit=collection.count())['ids'], include=['metadatas'])
            existing_links_in_db = set(item['link'] for item in all_db_items['metadatas'] if 'link' in item)
            logging.info(f"DEBUG: Found {len(existing_links_in_db)} existing links in '{db_name}' collection.")
        except Exception as e:
            logging.warning(f"WARNING: Could not retrieve all existing links from '{db_name}' for duplicate check: {e}. "
                            "New articles might be added even if their links already exist.")
            existing_links_in_db = set() 


    for index, row in df.iterrows():
        article_id = row['link']
        if article_id in existing_links_in_db:
            logging.info(f"Skipping already existing article (link: {article_id}) in '{db_name}'.")
            continue
        
        if not row['content'] or len(row['content']) < 50:
             logging.info(f"Skipping article '{row.get('title', 'No Title')}' due to empty/short content.")
             continue

        df_to_add.append(row) 

    logging.info(f"DEBUG: {len(df_to_add)} unique articles identified for addition to '{db_name}'.") 

    if not df_to_add:
        logging.info(f"No new unique articles to add to '{db_name}' database for this run.")
        return True 

    documents = []
    metadatas = []
    ids = []
    
    for i, row in enumerate(df_to_add):
        documents.append(row['content'])
        metadatas.append({'title': row['title'], 'description': row['description'], 'link': row['link']})
        ids.append(row['link']) 

    try:
        logging.info(f"DEBUG: Attempting to add {len(documents)} documents to ChromaDB collection '{db_name}'.") 
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Successfully added {len(df_to_add)} new articles to '{db_name}' database. Total articles: {collection.count()}")
        return True
    except Exception as e:
        logging.error(f"CRITICAL ERROR: Error adding documents to ChromaDB for '{db_name}': {e}")
        import traceback
        traceback.print_exc() 
        return False

if __name__ == "__main__":
    keywords = ["경제", "IT", "정치", "사회", "세계", "스포츠"] 

    if not os.path.exists('./data'):
        os.makedirs('./data')

    for i, keyword in enumerate(keywords):
        db_name = f"DB{i+1}" 
        logging.info(f"\n--- Processing keyword: {keyword} (DB: {db_name}) ---")
        
        csv_file_path = f'./data/{db_name}_naver_news_with_content.csv'
        if os.path.exists(csv_file_path):
            logging.info(f"Loading data from existing CSV: {csv_file_path}")
            try:
                # 기존 CSV에서 로드 후, 크롤링 개수를 제한
                news_df_existing = pd.read_csv(csv_file_path)
                news_df = news_df_existing.head(100) # CSV에서도 상위 100개만 사용
                logging.info(f"Loaded {len(news_df)} items from existing CSV for {keyword}.")
            except Exception as e:
                logging.error(f"Error loading CSV {csv_file_path}: {e}. Attempting full crawl instead (max 100 items).")
                news_df = get_all_news(keyword, max_crawl_items=100) 
        else:
            logging.info(f"CSV file not found. Starting full crawl for {keyword} (max 100 items)...")
            news_df = get_all_news(keyword, max_crawl_items=100) 

        if news_df.empty:
            logging.warning(f"No news data collected or loaded for {keyword}. Skipping embedding.")
            continue
        
        # 크롤링 또는 로드된 데이터를 CSV로 저장 (항상 최신 상태 유지)
        news_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        logging.info(f"Saved crawled/loaded data to {csv_file_path}")

        success = create_and_store_embeddings(news_df, db_name=db_name)
        if not success:
            logging.error(f"Failed to create embeddings for keyword: {keyword}")
            sys.exit(1) 

    logging.info("\n--- All data pipelines finished successfully ---")
