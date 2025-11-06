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
# --- 새로운 임포트 추가 ---
from chromadb.utils import embedding_functions 

# 로깅 설정 (GitHub Actions 로그에서 더 잘 보이도록)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_news(keyword, max_crawl_items=300):
    logging.info(f"Starting detailed news collection for keyword: '{keyword}'")
    base_url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query="
    encoded_keyword = urllib.parse.quote(keyword)
    news_list = []
    page = 1
    crawled_count = 0

    while crawled_count < max_crawl_items:
        url = f"{base_url}{encoded_keyword}&sort=1&ds=&de=&docid=&nso=so:r,p:all,a:all&start={((page-1)*10) + 1}"
        logging.info(f"Crawling news for '{keyword}' from page {page} (URL: {url})...")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            soup = BeautifulSoup(response.text, 'html.parser')

            # 뉴스 기사 링크들 찾기
            news_links = soup.select('div.news_area > div.news_info > div.info_group > a.info')
            
            if not news_links:
                logging.info(f"No more news links found for '{keyword}' on page {page}. Stopping crawl.")
                break

            for link_tag in news_links:
                if 'naver.com/article/' in link_tag['href']: # 네이버 뉴스 원문 링크만
                    article_link = link_tag['href']
                    try:
                        article_response = requests.get(article_link, headers=headers, timeout=10)
                        article_response.raise_for_status()
                        article_soup = BeautifulSoup(article_response.text, 'html.parser')
                        
                        title = article_soup.select_one('h2#title_area > span').text.strip() if article_soup.select_one('h2#title_area > span') else '제목 없음'
                        description_tag = article_soup.select_one('meta[property="og:description"]')
                        description = description_tag['content'].strip() if description_tag else '설명 없음'
                        content_tags = article_soup.select('div#newsct_article') # 기사 본문 영역
                        content = ''
                        if content_tags:
                            for tag in content_tags:
                                # 광고, 기자 이름, 이메일, 저작권 등 불필요한 정보 제거
                                for script_or_ad in tag(['script', 'a', 'strong', 'em', 'span']): # 필요한 태그만 남기기
                                    script_or_ad.extract()
                                content += tag.get_text(separator='\n', strip=True)
                            
                            # 불필요한 공백, 줄바꿈, 특수문자 정리
                            content = re.sub(r'\s+', ' ', content).strip()
                            content = re.sub(r'\[.*?\]', '', content).strip() # [사진], [영상] 등 제거
                            content = re.sub(r'\(.*?\)', '', content).strip() # (서울=연합뉴스) 등 제거
                            
                            # 기사 끝부분 흔한 불필요 텍스트 제거
                            content = re.split(r'저작권자 ⓒ 한경닷컴|▶ 네이버에서 서울경제', content)[0].strip()
                            content = re.split(r'기자 =.+?|작가 =.+?|사진 =.+?', content)[0].strip() # 기자/작가 정보 제거
                        else:
                            content = '내용 없음'

                        if content != '내용 없음' and len(content) > 100: # 내용이 충분히 길 때만 추가
                            news_list.append({
                                'title': title,
                                'description': description,
                                'link': article_link,
                                'content': content
                            })
                            crawled_count += 1
                            if crawled_count >= max_crawl_items:
                                break
                    except requests.exceptions.RequestException as e:
                        logging.warning(f"Failed to fetch article {article_link}: {e}")
                    except Exception as e:
                        logging.warning(f"Error parsing article {article_link}: {e}")
                time.sleep(0.5) # 과도한 요청 방지
            
            page += 1
            time.sleep(1) # 페이지 전환 대기
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
    
    # ChromaDB 클라이언트 설정
    client = chromadb.PersistentClient(path=f"./chroma_db_{db_name}")
    
    # --- ⭐ 핵심 수정: ChromaDB 호환 임베딩 함수 초기화 및 전달 ⭐ ---
    chroma_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jhgan/ko-sroberta-multitask"
    )
    logging.info("DEBUG: Initialized ChromaDB compatible SentenceTransformer EmbeddingFunction for data_pipeline.")

    try:
        # get_or_create_collection 호출 시 embedding_function을 명시적으로 전달해야 합니다.
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
            # 대규모 DB에서는 비효율적일 수 있으나, 현재는 작동 여부 확인이 우선
            # collection.get()은 기본적으로 limit가 있으므로 모든 메타데이터를 가져오려면 반복문이 필요
            # 여기서는 편의상 collection.count()를 limit으로 사용.
            all_db_items = collection.get(ids=collection.get(limit=collection.count())['ids'], include=['metadatas'])
            existing_links_in_db = set(item['link'] for item in all_db_items['metadatas'] if 'link' in item)
            logging.info(f"DEBUG: Found {len(existing_links_in_db)} existing links in '{db_name}' collection.")
        except Exception as e:
            logging.warning(f"WARNING: Could not retrieve all existing links from '{db_name}' for duplicate check: {e}. "
                            "New articles might be added even if their links already exist.")
            # 오류 발생 시 existing_links_in_db를 비워 새로 추가하도록 유도 (중복 발생 가능성 있음)
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
        # collection.add()는 collection 생성 시 지정된 embedding_function을 사용합니다.
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
    # --- ⭐ 키워드 리스트를 실제 사용하시는 키워드로 변경 ⭐ ---
    # app.py의 db_names_to_load와 일치하는 개수 및 순서가 중요합니다.
    keywords = ["경제", "IT", "정치", "사회", "세계", "스포츠"] # 예시 (6개 키워드)
    # keywords = ["DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7", "DB8"] # 이전에 사용했던 DB 이름과 동일한 키워드라면 그대로 사용

    # data 폴더 생성 확인
    if not os.path.exists('./data'):
        os.makedirs('./data')

    for i, keyword in enumerate(keywords):
        db_name = f"DB{i+1}" # DB1, DB2, ...
        logging.info(f"\n--- Processing keyword: {keyword} (DB: {db_name}) ---")
        
        # --- ⭐ CSV 파일에서 데이터 로드하는 임시 디버깅 모드 (선택 사항) ⭐ ---
        # 기존 CSV 파일이 있다면 크롤링 대신 해당 파일을 로드하여 임베딩만 시도
        # 이전에 크롤링이 성공적으로 되었다면 시간을 절약할 수 있습니다.
        csv_file_path = f'./data/{db_name}_naver_news_with_content.csv'
        if os.path.exists(csv_file_path):
            logging.info(f"Loading data from existing CSV: {csv_file_path}")
            try:
                news_df = pd.read_csv(csv_file_path)
            except Exception as e:
                logging.error(f"Error loading CSV {csv_file_path}: {e}. Attempting full crawl instead.")
                news_df = get_all_news(keyword, max_crawl_items=100) # 디버깅용으로 item 수 줄임
        else:
            logging.info(f"CSV file not found. Starting full crawl for {keyword}...")
            news_df = get_all_news(keyword, max_crawl_items=100) # 디버깅용으로 item 수 줄임

        if news_df.empty:
            logging.warning(f"No news data collected or loaded for {keyword}. Skipping embedding.")
            continue
        
        # 크롤링 또는 로드된 데이터를 CSV로 저장 (항상 최신 상태 유지)
        news_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        logging.info(f"Saved crawled/loaded data to {csv_file_path}")

        success = create_and_store_embeddings(news_df, db_name=db_name)
        if not success:
            logging.error(f"Failed to create embeddings for keyword: {keyword}")
            # 이 경우 전체 워크플로우를 실패로 표시
            sys.exit(1) 

    logging.info("\n--- All data pipelines finished successfully ---")
