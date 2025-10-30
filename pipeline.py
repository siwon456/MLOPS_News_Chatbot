# data_pipeline.py (제가 이전에 드린, 환경 변수를 사용하는 버전)
import urllib.request
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os
import sys
import re
import requests
from bs4 import BeautifulSoup
import time
from typing import List
import uuid
import urllib3
from requests.exceptions import RequestException
from langchain_community.embeddings import SentenceTransformerEmbeddings

# urllib3 경고를 무시하도록 설정
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Naver API 키 설정 (환경 변수 또는 여기에 직접 입력) ---
# 자동화 시에는 환경 변수 사용을 강력히 권장합니다.
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "fJZdaHl9L2pxICMcjk_h") # 실제 키로 교체 필요 (GitHub Secrets에 등록)
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "ApNrRszPZ9") # 실제 키로 교체 필요 (GitHub Secrets에 등록)

# --- HTML 태그 및 특수문자 제거 함수 ---
def clean_text(text):
    """HTML 태그와 불필요한 특수문자를 제거하는 함수"""
    cleaned_text = re.sub('<.*?>', '', text) # HTML 태그 제거
    cleaned_text = cleaned_text.replace('&quot;', "'") # HTML 엔티티 변환
    cleaned_text = cleaned_text.replace('<b>', '').replace('</b>', '') # <b> 태그 제거
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text) # [사진], [앵커] 같은 패턴 제거
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # 여러 공백 하나로 축소 및 양쪽 공백 제거
    return cleaned_text # <--- 핵심 수정: 정제된 텍스트 반환

# ... (나머지 get_article_content, get_all_news 함수는 제가 드린 버전 사용) ...

def create_and_store_embeddings(df, db_name):
    """
    데이터프레임을 임베딩하고 ChromaDB에 저장하는 함수.
    기존 DB가 있으면 새 데이터만 추가하고 중복을 제거합니다.
    """
    try:
        print(f"Initializing SentenceTransformer model for '{db_name}'...")
        model = SentenceTransformer("jhgan/ko-sroberta-multitask") 
        embeddings_function = SentenceTransformerEmbeddings(model=model) # LangChain 임베딩 사용
        
        db_path = f"./chroma_db_{db_name}"
        collection_name = f"{db_name}_news_collection"
        
        if not os.path.exists("./data"):
            os.makedirs("./data")
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name=collection_name)
        
        print(f"Checking for existing documents in '{db_name}' collection...")
        existing_docs_result = collection.get(limit=collection.count(), include=["metadatas"])
        existing_links = {meta.get('link') for meta in existing_docs_result.get('metadatas', []) if 'link' in meta}
        
        print(f"Found {len(existing_links)} existing articles in '{db_name}'.")
        
        df_to_add = df[~df['link'].isin(existing_links)].drop_duplicates(subset=['link'])
        
        if df_to_add.empty:
            print(f"No new articles to add to '{db_name}' database for this run.")
            return True
            
        print(f"Adding {len(df_to_add)} new unique articles to '{db_name}' database.")

        sentences_to_embed = (df_to_add["title"] + " " + df_to_add["description"] + " " + df_to_add["content"]).tolist()
        new_embeddings = embeddings_function.embed_documents(sentences_to_embed) # LangChain의 embed_documents 사용
        
        ids = [str(uuid.uuid4()) for _ in range(len(df_to_add))]
        documents = sentences_to_embed
        metadatas = [{"link": link, "title": title} for link, title in zip(df_to_add['link'].tolist(), df_to_add['title'].tolist())] # 제목도 메타데이터에 추가
        
        collection.add(
            embeddings=new_embeddings,
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Successfully added {len(df_to_add)} new articles to '{db_name}' database. Total articles: {collection.count()}")
        
        return True
    except Exception as e:
        print(f"Error: Embedding or DB storage failed for '{db_name}': {e}")
        return False

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print("-" * 50)
    print("뉴스 데이터 파이프라인 시작")
    print("이 프로그램은 크롤링과 임베딩, DB 저장만 수행합니다.")
    print("-" * 50)
    
    # 환경 변수에서 키워드를 읽어오도록 변경 (자동화 시 활용)
    keywords_str = os.getenv("GITHUB_KEYWORDS", "반도체, 인공지능, LLM") # 기본값 설정
    keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
    
    if not keywords:
        print("Error: No keywords provided via environment variable (GITHUB_KEYWORDS) or default.")
        sys.exit(1)
            
    print(f"Keywords to process: {keywords}")

    for i, keyword in enumerate(keywords):
        db_name = f"DB{i+1}"
        db_path = f"./chroma_db_{db_name}"
        
        print(f"\n--- 키워드: '{keyword}' 크롤링 및 DB 업데이트 시작 ---")
        
        if os.path.exists(db_path):
            print(f"Database for '{db_name}' found. Starting update process...")
        else:
            print(f"Database for '{db_name}' not found. Creating new database...")