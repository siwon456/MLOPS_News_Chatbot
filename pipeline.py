import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
import sys
import urllib.parse
import logging # 로깅 모듈 추가

# 로깅 설정 (GitHub Actions 로그에 더 자세한 정보 출력 위함)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 전역 변수 설정 (API 키 등) ---
# GitHub Secrets에서 환경 변수를 가져오거나, 로컬 실행을 위해 기본값 설정
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# API 키가 설정되었는지 확인
if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    logging.error("NAVER_CLIENT_ID or NAVER_CLIENT_SECRET is not set as environment variables.")
    # GitHub Actions에서는 sys.exit(1)로 종료하여 실패를 알림
    # 로컬에서는 사용자에게 메시지 표시 후 종료
    if os.getenv("GITHUB_ACTIONS"):
        sys.exit(1)
    else:
        print("Error: NAVER_CLIENT_ID or NAVER_CLIENT_SECRET environment variables are not set. "
              "Please set them before running the script.")
        sys.exit(1)


# --- 2. 헬퍼 함수 정의 ---

def clean_text(text):
    """HTML 태그와 불필요한 특수문자를 제거하는 함수"""
    cleaned_text = re.sub('<.*?>', '', text)
    cleaned_text = cleaned_text.replace('&quot;', "'")
    cleaned_text = cleaned_text.replace('<b>', '').replace('</b>', '')
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text) # [사진], [앵커] 같은 패턴 제거
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # 여러 공백 하나로 축소 및 양쪽 공백 제거
    return cleaned_text

def get_article_content(url):
    """주어진 URL에서 뉴스 기사 본문을 크롤링하는 함수"""
    try:
        response = requests.get(url, timeout=5) # 타임아웃 5초 설정
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 네이버 뉴스 본문 추출 (다양한 본문 영역 클래스 고려)
        content_div = soup.find('div', {'id': 'dic_area'}) # 기본 네이버 뉴스 본문
        if not content_div:
            content_div = soup.find('div', class_=re.compile(r'article_body_content|news_content|content_area')) # 기타 본문 클래스
        
        if content_div:
            # 스크립트, 광고 등 불필요한 요소 제거
            for script_or_ad in content_div(['script', 'a', 'span', 'strong', 'em', 'img', 'figure', 'figcaption']):
                script_or_ad.extract()
            
            article_text = content_div.get_text(separator=' ', strip=True)
            return clean_text(article_text)
        else:
            logging.warning(f"Could not find article content for URL: {url}")
            return ""
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error parsing content from {url}: {e}")
        return ""

def search_naver_news(keyword, start, display):
    """네이버 뉴스 API를 호출하여 데이터를 가져오는 함수"""
    encText = urllib.parse.quote(keyword)
    url = f"https://openapi.naver.com/v1/search/news?query={encText}&start={start}&display={display}&sort=date" # 최신순 정렬
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10) # 타임아웃 10초
        logging.info(f"DEBUG: API request for '{keyword}' (start={start}) - Status Code: {response.status_code}")
        if response.status_code == 200:
            response_json = response.json()
            # 첫 1개 아이템만 로그에 출력하여 너무 길어지는 것을 방지
            logging.info(f"DEBUG: API response for '{keyword}': {response_json.get('items', [])[:1]}")
            return response_json
        else:
            logging.error(f"Error: HTTP Error Code {response.status_code} for '{keyword}'. Check API usage limits or API keys.")
            return None
    except requests.exceptions.Timeout:
        logging.error(f"Error: API Request for '{keyword}' timed out after 10 seconds.")
        return None
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Error: API Connection Failed for '{keyword}': {e}.")
        return None
    except Exception as e:
        logging.error(f"Error: API Request Failed for '{keyword}': {e}.")
        return None

def get_all_news(keyword):
    """네이버 API와 웹 크롤링을 결합하여 뉴스 본문을 포함한 데이터를 가져오는 함수"""
    logging.info(f"Starting detailed news collection for keyword: '{keyword}'")
    result_all = []
    start = 1
    display = 100 # 한 페이지당 최대 100개
    max_crawl_items = 300 # 각 키워드당 최대 300개 기사 목표 (테스트를 위해 300으로 상향)

    crawled_count = 0
    while start <= 1000 and crawled_count < max_crawl_items: # API는 최대 1000개까지 검색 가능
        logging.info(f"Crawling news for '{keyword}' from page {start} (currently crawled: {crawled_count}/{max_crawl_items})...")
        result_json = search_naver_news(keyword, start, display)
        
        if result_json and 'items' in result_json and result_json['items']:
            for item in result_json['items']:
                if crawled_count >= max_crawl_items:
                    break # 목표 개수 도달 시 루프 종료

                # link 또는 originallink가 없으면 스킵
                article_url = item.get('originallink') or item.get('link')
                if not article_url:
                    logging.warning(f"Skipping article due to missing link: {item.get('title', 'No Title')}")
                    continue

                article_text = get_article_content(article_url)
                
                if article_text and len(article_text) > 50: # 본문 내용이 최소 50자 이상인 경우만 추가
                    result_all.append({
                        'title': clean_text(item['title']),
                        'description': clean_text(item['description']),
                        'content': article_text,
                        'link': article_url
                    })
                    crawled_count += 1
                else:
                    logging.info(f"Skipping article '{item.get('title', 'No Title')}' due to empty/short content or crawling failure.")
                
                time.sleep(0.05) # 서버 부하를 줄이기 위해 0.05초 지연 (너무 빠르면 차단될 수도 있음)
            
            if len(result_json['items']) < display:
                logging.info(f"DEBUG: Less than {display} items returned for '{keyword}' at start {start}, stopping pagination.")
                break # 더 이상 결과가 없으면 종료
            start += display
        else:
            logging.info(f"DEBUG: No items in API response for '{keyword}' at start {start}, or API call failed. Stopping further crawling for this keyword.")
            break # API 응답이 없거나 item이 없으면 종료

    if crawled_count >= max_crawl_items:
        logging.info(f"Note: Reached target of {max_crawl_items} items for '{keyword}'.")
    elif not result_all:
        logging.warning(f"Warning: No news articles were collected for '{keyword}'. API or crawling issue suspected.")
    else:
        logging.info(f"Finished crawling for '{keyword}'. Total articles collected: {len(result_all)}.")
    
    return pd.DataFrame(result_all)


def create_and_store_embeddings(df, db_name="my_chroma_db"):
    """DataFrame의 데이터를 ChromaDB에 임베딩하여 저장하는 함수"""
    logging.info(f"Building and embedding the database for '{db_name}'...")
    
    # ChromaDB 클라이언트 설정
    client = chromadb.PersistentClient(path=f"./chroma_db_{db_name}")
    
    # 기존 컬렉션이 있으면 로드, 없으면 생성
    try:
        collection = client.get_or_create_collection(name=db_name)
        logging.info(f"Accessing ChromaDB collection: '{db_name}'. Current total articles: {collection.count()}")
    except Exception as e:
        logging.error(f"Error accessing/creating ChromaDB collection '{db_name}': {e}")
        return False

    # 임베딩 모델 로드 (캐시 사용)
    try:
        model = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        logging.info(f"SentenceTransformer model 'jhgan/ko-sroberta-multitask' loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer model: {e}")
        return False

    documents = []
    metadatas = []
    ids = []

    # 중복 체크를 위한 기존 ID 가져오기 (컬렉션에 이미 데이터가 있을 경우)
    existing_ids = set()
    if collection.count() > 0:
        # DB에서 모든 ID를 직접 가져오기 (ChromaDB의 get()은 기본적으로 100개만 가져오므로 limit 사용)
        try:
            # collection.get()은 기본적으로 limit가 있어서 모든 ID를 가져오려면 반복문 필요.
            # 하지만 여기서는 새로 추가할 문서가 기존에 있는지 여부만 판단하면 되므로,
            # 새로 추가될 문서의 link를 기준으로 중복을 체크하는 것이 더 효율적
            # (아래 로직에서 link를 기준으로 중복을 방지)
            pass
        except Exception as e:
            logging.warning(f"Could not retrieve existing IDs from collection '{db_name}': {e}. "
                            "Proceeding without full existing ID check, potential for duplicate links.")

    df_to_add = []
    for index, row in df.iterrows():
        # 기사 링크를 ID로 사용 (중복 체크)
        article_id = row['link']
        
        # 이미 DB에 해당 링크의 기사가 있는지 확인 (컬렉션에 'link' 필터를 사용)
        # ChromaDB 필터링 예시: https://docs.trychroma.com/usage-guide#filtering-queries
        # 그러나 collection.get()으로 필터링하는 것은 ID 기반이 아니면 복잡해질 수 있음.
        # 가장 확실한 방법은 중복될 수 있는 문서들을 미리 필터링하고 임베딩하는 것.
        # 여기서는 단순히 DF에서 unique한 link만 사용하고,
        # ChromaDB는 동일 ID로 add 시 업데이트되므로 중복 추가는 아님.
        # 새로운 기사만 추가하도록 로직을 변경합니다.

        # 실제 DB에 존재하는 링크인지 확인하려면, DB에서 link 메타데이터를 기반으로 조회해야 함.
        # collection.query(query_texts=["dummy"], where={"link": article_id})
        # 위 쿼리는 embedding을 필요로 하므로, 단순 존재 여부 확인에는 비효율적.
        # 따라서, 여기서는 'link'를 ID로 사용하고, add() 시 동일 ID면 업데이트되는 ChromaDB 특성 활용.
        # 또는, 기존 DB의 모든 메타데이터를 가져와서 링크 비교. (대규모 DB에서는 비효율적)

        # 현재는 ID가 link이므로, 동일한 링크는 자동으로 업데이트되거나, 중복으로 add 되지 않음.
        # '새로운' 기사만 추가하려면, 기존 DB의 모든 링크를 가져와서 비교해야 함.
        # 아래는 기존 DB에 없는 '새로운' 기사만 추가하는 로직입니다.
        if collection.count() > 0:
            # 컬렉션의 메타데이터를 모두 가져와 'link' 필드에서 현재 링크가 있는지 확인
            # 대규모 DB에서는 이 방식이 느려질 수 있으므로,
            # 실제 서비스에서는 별도의 DB에 크롤링된 링크 목록을 관리하는 것이 좋음.
            # 여기서는 편의상 전체 링크 메타데이터를 가져와서 비교
            all_links_in_db = set(item['link'] for item in collection.get(limit=collection.count(), include=['metadatas'])['metadatas'])
            if article_id in all_links_in_db:
                logging.info(f"Skipping already existing article (link: {article_id}) in '{db_name}'.")
                continue # 이미 DB에 있는 기사는 스킵
        
        df_to_add.append(row) # 새로운 기사만 리스트에 추가

    if not df_to_add:
        logging.info(f"No new unique articles to add to '{db_name}' database for this run.")
        return True # 추가할 새로운 기사가 없으므로 성공으로 간주

    # 새로운 기사들만 임베딩하여 DB에 추가
    logging.info(f"Adding {len(df_to_add)} new unique articles to '{db_name}' database.")

    # langchain_community.embeddings를 사용하도록 변경
    # document_embeddings = model.embed_documents([row['content'] for row in df_to_add])
    
    # ChromaDB add는 documents, metadatas, ids 리스트를 받음
    for i, row in enumerate(df_to_add):
        documents.append(row['content'])
        metadatas.append({'title': row['title'], 'description': row['description'], 'link': row['link']})
        ids.append(row['link']) # 링크를 ID로 사용

    # 실제 임베딩 및 추가
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Successfully added {len(df_to_add)} new articles to '{db_name}' database. Total articles: {collection.count()}")
        return True
    except Exception as e:
        logging.error(f"Error adding documents to ChromaDB for '{db_name}': {e}")
        return False

# --- 3. 메인 실행 로직 ---
if __name__ == "__main__":
    logging.info("-" * 50)
    logging.info("뉴스 데이터 파이프라인 시작")
    logging.info("이 프로그램은 크롤링과 임베딩, DB 저장만 수행합니다.")
    logging.info("-" * 50)
    
    # 환경 변수에서 키워드 가져오기
    keywords_str = os.getenv("GITHUB_KEYWORDS", "")
    keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
    
    if not keywords:
        logging.error("Error: No keywords provided via environment variable (GITHUB_KEYWORDS) or default.")
        sys.exit(1) # 키워드가 없으면 종료
            
    logging.info(f"Keywords to process: {keywords}")

    # 'data' 폴더가 없으면 생성 (CSV 저장용)
    if not os.path.exists("./data"):
        os.makedirs("./data")
        logging.info("Created './data' directory for CSV storage.")

    for i, keyword in enumerate(keywords):
        try:
            db_name = f"DB{i+1}"
            db_path = f"./chroma_db_{db_name}"
            
            logging.info(f"\n--- 키워드: '{keyword}' 크롤링 및 DB 업데이트 시작 ---")
            
            if os.path.exists(db_path):
                logging.info(f"Database for '{db_name}' found. Starting update process...")
            else:
                logging.info(f"Database for '{db_name}' not found. Creating new database...")
            
            news_df = get_all_news(keyword) # get_all_news 함수 호출
            
            if not news_df.empty:
                csv_file_path = f"./data/{db_name}_naver_news_with_content.csv"
                news_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
                logging.info(f"Saved crawled data to {csv_file_path}")
                
                # DB 저장 시도
                if not create_and_store_embeddings(news_df, db_name):
                    logging.error(f"Error: Failed to create/store embeddings for '{keyword}'. Skipping to next keyword.")
            else:
                logging.info(f"No news data collected for '{keyword}'. Skipping embedding and DB storage.")
        
        except Exception as e:
            logging.critical(f"CRITICAL ERROR: An unexpected error occurred during processing keyword '{keyword}': {e}")
            import traceback
            traceback.print_exc() # 상세 스택 트레이스 출력
            # 특정 키워드 처리 중 오류 발생해도 다른 키워드 계속 진행
            # sys.exit(1) # 모든 작업 중단하고 GitHub Actions 실패 처리하려면 이 줄 주석 해제

    logging.info("\n--- 모든 키워드 처리 완료 ---")
