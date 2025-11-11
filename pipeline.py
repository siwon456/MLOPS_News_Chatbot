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
    # 네이버 뉴스 링크가 아닌 경우, 파싱이 거의 항상 실패할 것입니다.
    if "news.naver.com" not in url:
        return ""
        
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5) # 타임아웃 5초
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 네이버 뉴스 본문 영역 ID
            article_content = soup.find('div', {'id': 'dic_area'})
            if article_content:
                return article_content.get_text(strip=True)
            
            # 다른 네이버 뉴스 본문 영역 ID (mnews 등)
            article_content_alt = soup.find('div', {'id': 'articleBodyContents'})
            if article_content_alt:
                return article_content_alt.get_text(strip=True)

            return "" # 파싱할 영역을 찾지 못함
        return "" # HTTP 200이 아님
    except Exception as e:
        # print(f"경고: URL '{url}'에서 기사 내용 가져오기 실패: {e}") # 로그가 너무 많아질 수 있으므로 주석 처리
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
    본문 파싱에 실패하면 요약글을 content로 사용하고, 날짜 정보도 포함합니다.
    """
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
                
                article_text = get_article_content(item['link'])
                
                if not article_text:
                    content_to_use = clean_text(item.get('description', ''))
                else:
                    content_to_use = article_text

                if not content_to_use:
                    continue

                # --- 날짜 정보 추가 시작 ---
                pub_date = item.get('pubDate', '') # Naver API에서 pubDate를 가져옴
                # pubDate 형식: 'Sat, 20 Jan 2024 10:00:00 +0900'
                # 필요에 따라 원하는 형식으로 파싱하여 저장할 수 있습니다.
                # 예: YYYY-MM-DD 형식으로 변환 (더 복잡한 날짜 파싱은 datetime 모듈 필요)
                formatted_date = pub_date.split(' ', 4)[3] + ' ' + pub_date.split(' ', 4)[2] + ' ' + pub_date.split(' ', 4)[1] if pub_date else '날짜 없음'
                try: # 'Fri, 26 Apr 2024 10:00:00 +0900' -> '2024-04-26'
                    dt_object = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                    formatted_date = dt_object.strftime('%Y-%m-%d')
                except ValueError:
                    formatted_date = '날짜 없음'
                # --- 날짜 정보 추가 끝 ---

                result_all.append({
                    'title': clean_text(item.get('title', '')),
                    'description': clean_text(item.get('description', '')),
                    'content': content_to_use,
                    'link': item.get('link', ''),
                    'keyword_topic': keyword,
                    'date': formatted_date # 'date' 컬럼 추가
                })
                
                time.sleep(0.05) 
            
            if len(result_json['items']) < display:
                break
            start += display
        else:
            break

    print(f"--- 키워드 '{keyword}'에 대해 총 {len(result_all)}개의 뉴스 기사 크롤링 완료 ---")
    return result_all

# --- 메인 파이프라인 실행 로직 ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("      통합 뉴스 데이터 파이프라인 시작 (CSV만 업데이트)")
    print("="*50)

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("환경 변수 'NAVER_CLIENT_ID'와 'NAVER_CLIENT_SECRET'를 설정해야 합니다.")
        print("GitHub Actions Secrets 또는 로컬 환경 변수에 추가해주세요.")
        sys.exit(1)

    all_crawled_dfs = []
    
    for keyword in KEYWORDS_TO_CRAWL:
        news_data = get_all_news_for_keyword(keyword, MAX_ARTICLES_PER_KEYWORD)
        
        if not news_data:
            print(f"경고: 키워드 '{keyword}'에 대해 크롤링된 뉴스가 없습니다. 다음 키워드로 넘어갑니다.")
            continue
            
        df_keyword = pd.DataFrame(news_data)
        all_crawled_dfs.append(df_keyword)

    if not all_crawled_dfs:
        print("\n모든 키워드에 대해 크롤링된 뉴스가 없어 통합 CSV 파일을 생성할 수 없습니다.")
        sys.exit(1)

    # 모든 키워드의 데이터를 하나의 DataFrame으로 병합
    merged_df = pd.concat(all_crawled_dfs, ignore_index=True)
    print(f"\n모든 키워드로부터 총 {len(merged_df)}개의 뉴스 기사 병합 완료.")

    # 병합된 데이터를 하나의 CSV 파일로 저장
    output_merged_csv_path = 'data/merged_all_news.csv'
    os.makedirs(os.path.dirname(output_merged_csv_path), exist_ok=True)
    merged_df.to_csv(output_merged_csv_path, index=False, encoding='utf-8-sig')
    print(f"모든 키워드의 병합된 데이터가 '{output_merged_csv_path}'로 저장되었습니다.")

    print("\n" + "="*50)
    print("      뉴스 데이터 파이프라인 완료 (CSV 업데이트만)")
    print("="*50)
