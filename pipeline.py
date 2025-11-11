import pandas as pd
import urllib.request
import json
import os
import re
import time
from typing import List, Dict, Any
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import sys
from datetime import datetime

# --- 1. 설정 (환경 변수 사용) ---
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")

# 크롤링할 키워드 리스트
KEYWORDS_TO_CRAWL = [
    "AI", "반도체", "경제", "기술", "환경", "사회", "정책", "문화",
]

MAX_ARTICLES_PER_KEYWORD = 1000 # 각 키워드당 최대 크롤링할 기사 수 (Naver API 한도 내)

# --- 2. 헬퍼 함수 ---

def clean_text(text):
    """HTML 태그 및 특수문자 제거"""
    if not text:
        return ""
    cleaned_text = re.sub('<.*?>', '', text)
    cleaned_text = cleaned_text.replace('&quot;', "'")
    cleaned_text = cleaned_text.replace('<b>', '').replace('</b>', '')
    return cleaned_text

def get_article_content(url):
    """주어진 URL에서 뉴스 기사 본문 크롤링 (실패 시 빈 문자열 반환)"""
    if "news.naver.com" not in url:
        return ""
        
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 네이버 뉴스 본문 영역 ID
            article_content = soup.find('div', {'id': 'dic_area'})
            if article_content:
                return article_content.get_text(strip=True)
            
            # 다른 네이버 뉴스 본문 영역 ID
            article_content_alt = soup.find('div', {'id': 'articleBodyContents'})
            if article_content_alt:
                return article_content_alt.get_text(strip=True)

            return "" # 파싱할 영역을 찾지 못함
        return "" # HTTP 200이 아님
    except Exception as e:
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
    """
    지정된 키워드로 뉴스 기사 크롤링.
    """
    result_all = []
    start = 1
    display = 100
    
    print(f"\n--- 키워드 '{keyword}' 뉴스 크롤링 시작 (최대 {max_articles}개) ---")
    
    # tqdm을 사용하여 진행률 표시 (tqdm 라이브러리가 설치되어 있어야 함: pip install tqdm)
    # Naver API는 최대 1000개(start=1000)까지만 조회가 가능합니다.
    max_start = min(1001, max_articles + display)
    
    with tqdm(total=max_articles, desc=f' ➡️  {keyword}') as pbar:
        while len(result_all) < max_articles and start <= 1000:
            
            result_json = search_naver_news(keyword, start, display)
            
            if result_json and 'items' in result_json:
                items = result_json['items']
                
                for item in items:
                    if len(result_all) >= max_articles:
                        break
                    
                    article_text = get_article_content(item['link'])
                    
                    if not article_text:
                        content_to_use = clean_text(item.get('description', ''))
                    else:
                        content_to_use = article_text

                    if not content_to_use:
                        continue

                    # 날짜 정보 파싱
                    pub_date = item.get('pubDate', '')
                    try: 
                        dt_object = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                        formatted_date = dt_object.strftime('%Y-%m-%d')
                    except ValueError:
                        formatted_date = '날짜 없음'

                    result_all.append({
                        'title': clean_text(item.get('title', '')),
                        'description': clean_text(item.get('description', '')),
                        'content': content_to_use,
                        'link': item.get('link', ''), # 중복 제거를 위한 고유 키
                        'keyword_topic': keyword,
                        'date': formatted_date 
                    })
                    
                    pbar.update(1) # 진행바 1 증가
                    
                    time.sleep(0.01) # Naver API 속도 제한을 위한 약간의 딜레이
                
                if len(items) < display:
                    break # 마지막 페이지
                start += display
            else:
                break # API 오류 또는 결과 없음

    print(f"\n--- 키워드 '{keyword}'에 대해 총 {len(result_all)}개의 뉴스 기사 크롤링 완료 ---")
    return result_all

# --- 메인 파이프라인 실행 로직 (데이터 누적 및 중복 제거) ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("      통합 뉴스 데이터 파이프라인 시작 (CSV 누적 업데이트)")
    print("="*50)

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("환경 변수 'NAVER_CLIENT_ID'와 'NAVER_CLIENT_SECRET'를 설정해야 합니다.")
        sys.exit(1)

    all_crawled_dfs = []
    
    for keyword in KEYWORDS_TO_CRAWL:
        # Naver API는 '최신순'이 아닌 '유사도순'으로 결과를 반환하므로,
        # 매번 실행 시 중복을 제거하기 위해 최대 1000개를 조회하는 것이 합리적입니다.
        news_data = get_all_news_for_keyword(keyword, MAX_ARTICLES_PER_KEYWORD)
        
        if not news_data:
            print(f"경고: 키워드 '{keyword}'에 대해 크롤링된 뉴스가 없습니다.")
            continue
            
        df_keyword = pd.DataFrame(news_data)
        all_crawled_dfs.append(df_keyword)

    if not all_crawled_dfs:
        print("\n모든 키워드에 대해 크롤링된 뉴스가 없어 파이프라인을 종료합니다.")
        sys.exit(0)

    # 1. 새로 수집된 데이터를 하나의 DataFrame으로 병합
    new_df = pd.concat(all_crawled_dfs, ignore_index=True)
    print(f"\n모든 키워드로부터 총 {len(new_df)}개의 뉴스 기사 수집 완료.")
    
    output_merged_csv_path = 'data/merged_all_news.csv'
    final_df = new_df # 최종본을 일단 new_df로 설정

    # 2. 기존 데이터 로드 및 병합 (GitHub 크롤러와 동일한 로직)
    if os.path.exists(output_merged_csv_path):
        print(f"\n기존 데이터 파일 '{output_merged_csv_path}'을 로드합니다.")
        try:
            existing_df = pd.read_csv(output_merged_csv_path)
            
            # 기존 데이터와 새 데이터를 하나로 합칩니다.
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # 3. 중복 제거: 'link' (기사 URL)를 기준으로 최신 데이터(last)만 남깁니다.
            initial_count = len(combined_df)
            final_df = combined_df.drop_duplicates(subset=['link'], keep='last')
            removed_count = initial_count - len(final_df)
            
            print(f"ℹ️ 중복된 뉴스 데이터 {removed_count}개를 제거했습니다.")

        except pd.errors.EmptyDataError:
            print("경고: 기존 파일이 비어 있어 새로운 데이터만 사용합니다.")
        except Exception as e:
            print(f"❌ 기존 파일 로드 중 오류 발생 ({e}). 새로운 데이터만 사용하여 저장합니다.")
    else:
        print("\n기존 데이터 파일이 없습니다. 새로운 데이터만 사용하여 저장합니다.")

    # 4. 최종 데이터 저장
    os.makedirs(os.path.dirname(output_merged_csv_path), exist_ok=True)
    final_df.to_csv(output_merged_csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print(f"✅ 최종 저장 완료: 총 {len(final_df)}개의 고유 뉴스 데이터")
    print(f"파일 경로: '{output_merged_csv_path}'")
    print("="*50)
