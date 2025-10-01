import urllib.request
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
import uuid
import os
import sys
from langchain_core.prompts import PromptTemplate

# --- Naver API 키 설정 (본인의 키로 교체하세요) ---
NAVER_CLIENT_ID = "fJZdaHl9L2pxICMcjk_h"
NAVER_CLIENT_SECRET = "ApNrRszPZ9"

# --- 1단계: 크롤링 함수 ---
def search_naver_news(keyword, start, display):
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
            print(f"Error: HTTP Error Code {rescode}")
            return None
    except Exception as e:
        print(f"Error: API Request Failed: {e}")
        return None

def get_all_news(keyword):
    result_all = []
    start = 1
    display = 100
    while True:
        print(f"Crawling news from page {start}...")
        result_json = search_naver_news(keyword, start, display)
        if result_json and 'items' in result_json:
            result_all.extend(result_json['items'])
            if len(result_json['items']) < display:
                break
            start += display
        else:
            break
    return result_all

# --- 2단계: 임베딩 및 DB 저장 함수 ---
def create_and_store_embeddings(df, keyword):
    try:
        model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        sentences = df["title"].tolist()
        embeddings = model.encode(sentences)
        
        db_path = f"./chroma_db_{keyword}"
        client = chromadb.PersistentClient(path=db_path)
        collection_name = f"{keyword}_news_collection"
        
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        
        collection = client.get_or_create_collection(name=collection_name)
        
        ids = [str(uuid.uuid4()) for _ in range(len(df))]
        documents = df["title"].tolist()
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids
        )
        return True
    except Exception as e:
        print(f"Error: Embedding or DB storage failed: {e}")
        return False

# --- 3단계: RAG 챗봇 로직 ---
def get_qa_chain(keyword):
    db_path = f"./chroma_db_{keyword}"
    client = chromadb.PersistentClient(path=db_path)
    collection_name = f"{keyword}_news_collection"
    embeddings_model = SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings_model
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    llm = Ollama(model="llama3")
    
    # --- 수정된 부분: 프롬프트 템플릿 추가 ---
    CUSTOM_PROMPT = """
    주어진 맥락을 바탕으로 다음 질문에 대해 간결하고 직접적으로 답변하세요.
    "문맥에 따르면", "따라서"와 같은 추가적인 설명은 생략하고 핵심 내용만 서술하세요.
    만약 주어진 맥락에 답이 없다면 "죄송합니다. 제가 가진 정보로는 답변을 드릴 수 없습니다."라고 답하세요.
    {context}
    질문: {question}
    """
    custom_prompt_template = PromptTemplate(template=CUSTOM_PROMPT, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt_template} # 프롬프트 템플릿 적용
    )
    
    return qa_chain

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print("-" * 50)
    print("뉴스 기반 RAG 챗봇 시작")
    print("-" * 50)
    
    search_keyword = input("챗봇에 사용할 뉴스 키워드(영문)를 입력하세요: ").strip()

    if not search_keyword:
        print("Error: A keyword is required.")
        sys.exit(1)

    print("\nStarting news crawling...")
    news_data = get_all_news(search_keyword)

    if news_data:
        df = pd.DataFrame(news_data)
        print(f"\nSuccessfully crawled {len(df)} news articles!")
        
        print("\nBuilding and embedding the database...")
        if create_and_store_embeddings(df, search_keyword):
            print("Database setup complete!")
            
            try:
                qa_chain = get_qa_chain(search_keyword)
                
                print("\n" + "-" * 50)
                print("Chatbot is ready. Type 'exit' to quit.")
                print("-" * 50)
                
                while True:
                    question = input("You: ").strip()
                    if question.lower() == "exit":
                        print("Exiting chatbot.")
                        break
                    
                    print("AI:", end=" ")
                    response = qa_chain.invoke({"query": question})
                    answer = response.get('result', '').strip()
                    
                    if not answer:
                        answer = "죄송합니다. 제가 가진 정보로는 답변을 찾을 수 없습니다."
                        
                    print(answer)
                    print("-" * 50)
            except Exception as e:
                print(f"Error: Chatbot initialization failed: {e}")
                print("Please make sure Ollama is running (ollama run llama3)")

        else:
            print("\nError: Database setup failed.")
    else:
        print("\nError: No news data was crawled.")