import pandas as pd
import urllib.request
import json
import os
import re
import time
import shutil # shutil 모듈 임포트 추가
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter # 이 모듈은 워크플로우에서 설치됩니다.
from langchain.docstore.document import Document
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import sys # sys 모듈 임포트 추가

# --- 1. 설정 (환경 변수 사용) ---
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")

# 크롤링할 키워드 및 각 키워드에 할당할 DB 이름
KEYWORDS_TO_PROCESS = {
    "AI": "DB1",
    "반도체": "DB2",
    "경제": "DB3",
    "기술": "DB4",
    "환경": "DB5",
    "사회": "DB6",
    "정책": "DB7",
    "문화": "DB8",
}

MAX_ARTICLES_PER_KEYWORD = 1000 # 각 키워드당 최대 크롤링할 기사 수 (Naver API 한도 내)
CHUNK_SIZE = 1000 # 텍스트 청크 크기
CHUNK_OVERLAP = 200 # 텍스트 청크 오버랩
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask" # 한국어 임베딩 모델

# --- 2. 헬퍼 함수 ---

def clean_text(text):
    """HTML 태그 및 특수문자 제거"""
    cleaned_text = re.sub('<.*?>', '', text)
    cleaned_text = cleaned_text.replace('&quot;', "'")
    cleaned_text = cleaned_text.replace('<b>', '').replace('</b>', '')
    return cleaned_text

def get_article_content(url):
    """주어진 URL에서 뉴스 기사 본문 크롤링"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            article_content = soup.find('div', {'id': 'dic_area'})
            if article_content:
                return article_content.get_text(strip=True)
            
            article_content_alt = soup.find('div', {'id': 'articleBodyContents'})
            if article_content_alt:
                return article_content_alt.get_text(strip=True)

            return ""
        return ""
    except Exception as e:
        print(f"경고: URL '{url}'에서 기사 내용 가져오기 실패: {e}")
        return ""

def search_naver_news(keyword, start, display):
    """Naver News API를 통해 뉴스 검색"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("오류: Naver Client ID 또는 Client Secret이 설정되지 않았습니다.")
        return None

    encText = urllib.parse.quote(keyword)
    url = f"https://openapi.naver.com/v1/search/news?query={encText}&start={start}&display={display}"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
    
    try:
        response = urllib.request.urlopen(request)
        rescode = response.getcode()
        if rescode == 200:
            response_body = response.read()
            return json.loads(response_body.decode('utf-8'))
        else:
            print(f"오류: Naver API HTTP Error Code {rescode}. 사용량 제한을 확인하세요.")
            return None
    except Exception as e:
        print(f"오류: Naver API 요청 실패: {e}. Client ID/Secret을 확인하세요.")
        return None

def get_all_news_for_keyword(keyword, max_articles):
    """지정된 키워드로 뉴스 기사 크롤링 및 본문 추출"""
    result_all = []
    start = 1
    display = 100
    
    print(f"\n--- 키워드 '{keyword}' 뉴스 크롤링 시작 (최대 {max_articles}개) ---")
    while len(result_all) < max_articles and start <= 1000:
        print(f"  페이지 {start // display + 1} ({start}번째부터) 크롤링 중...")
        result_json = search_naver_news(keyword, start, display)
        
        if result_json and 'items' in result_json:
            for item in result_json['items']:
                if len(result_all) >= max_articles:
                    break
                
                if "news.naver.com" in item['link']:
                    article_text = get_article_content(item['link'])
                    if article_text:
                        result_all.append({
                            'title': clean_text(item['title']),
                            'description': clean_text(item['description']),
                            'content': article_text,
                            'link': item['link']
                            # 'keyword_topic': keyword # 이전 요청에 따라 제거
                        })
                    time.sleep(0.1)
            
            if len(result_json['items']) < display:
                break
            start += display
        else:
            break

    print(f"--- 키워드 '{keyword}'에 대해 총 {len(result_all)}개의 뉴스 기사 크롤링 완료 ---")
    return result_all

def create_and_store_chromadb(documents: List[Document], db_name: str): # keyword 인자 제거
    """LangChain Document 리스트를 받아 ChromaDB를 생성하고 저장"""
    
    db_path = f"./chroma_db_{db_name}"
    
    # 기존 DB가 있다면 삭제 (새로운 데이터를 위해)
    if os.path.exists(db_path):
        print(f"기존 '{db_path}' 폴더를 삭제하고 다시 생성합니다.")
        shutil.rmtree(db_path)

    print(f"--- ChromaDB '{db_name}' 생성 및 임베딩 시작 ---")
    print(f"임베딩 모델: '{EMBEDDING_MODEL_NAME}'")

    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"오류: 임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로드 실패: {e}")
        return False
        
    start_time = time.time()
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_path,
            collection_name=db_name
        )
        vectorstore.persist()
        print(f"ChromaDB '{db_name}'에 {vectorstore._collection.count()}개 문서 청크 저장 완료.")
    except Exception as e:
        print(f"오류: ChromaDB 생성 및 저장 실패: {e}")
        return False
        
    end_time = time.time()
    print(f"ChromaDB '{db_name}' 생성 총 소요 시간: {end_time - start_time:.2f}초")
    return True

# --- 메인 파이프라인 실행 로직 ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("      뉴스 데이터 파이프라인 시작")
    print("="*50)

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("환경 변수 'NAVER_CLIENT_ID'와 'NAVER_CLIENT_SECRET'를 설정해야 합니다.")
        print("GitHub Actions Secrets 또는 로컬 환경 변수에 추가해주세요.")
        sys.exit(1)

    all_processed_dfs = []
    
    for keyword, db_name in KEYWORDS_TO_PROCESS.items():
        news_data = get_all_news_for_keyword(keyword, MAX_ARTICLES_PER_KEYWORD)
        
        if not news_data:
            print(f"경고: 키워드 '{keyword}'에 대해 크롤링된 뉴스가 없습니다. 다음 키워드로 넘어갑니다.")
            continue
            
        df = pd.DataFrame(news_data)
        all_processed_dfs.append(df)
        
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=row['content'],
                metadata={
                    "source": row['link'], 
                    "title": row['title']
                    # "topic": keyword # 이전 요청에 따라 제거
                }
            )
            documents.append(doc)
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        chunked_documents = text_splitter.split_documents(documents)
        print(f"키워드 '{keyword}'의 원본 문서 {len(documents)}개 -> 청크 {len(chunked_documents)}개 생성")

        if not create_and_store_chromadb(chunked_documents, db_name): # keyword 인자 제거
            print(f"오류: 키워드 '{keyword}'에 대한 ChromaDB 생성 실패. 파이프라인 중단.")
            sys.exit(1)

        output_csv_path_keyword = f"data/{db_name}_naver_news_with_content.csv"
        os.makedirs(os.path.dirname(output_csv_path_keyword), exist_ok=True)
        df.to_csv(output_csv_path_keyword, index=False, encoding='utf-8-sig')
        print(f"키워드 '{keyword}'의 크롤링 데이터가 '{output_csv_path_keyword}'로 저장되었습니다.")

    # 모든 키워드의 데이터를 하나의 CSV로 병합하여 저장 (선택 사항이지만 워크플로우에서 참조하므로 유지)
    if all_processed_dfs:
        merged_all_df = pd.concat(all_processed_dfs, ignore_index=True)
        output_merged_csv_path = 'data/merged_all_news.csv'
        os.makedirs(os.path.dirname(output_merged_csv_path), exist_ok=True)
        merged_all_df.to_csv(output_merged_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n모든 키워드의 병합된 데이터가 '{output_merged_csv_path}'로 저장되었습니다.")
    else:
        print("\n모든 키워드에 대해 크롤링된 뉴스가 없어 병합된 CSV 파일이 생성되지 않았습니다.")

    print("\n" + "="*50)
    print("      뉴스 데이터 파이프라인 완료")
    print("="*50)
