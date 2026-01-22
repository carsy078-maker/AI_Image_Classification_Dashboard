# AI_Image_Classification
https://aiimageclassificationdashboardgit-fkeetpgidfdfgnmirrgmvc.streamlit.app/
<img width="2559" height="1267" alt="image" src="https://github.com/user-attachments/assets/88788407-76d0-47ae-87a8-c7e58c740768" />

## 1. 서비스 개요 (Service Overview)

### 1.1. 서비스 소개

본 서비스는 사용자가 영화 정보를 조회하고 리뷰를 작성하면, **AI(인공지능) 모델이 실시간으로 리뷰의 감정(긍정/부정)을 분석**하여 영화의 평점을 자동으로 산출해주는 웹 애플리케이션입니다. 단순한 별점 부여 방식을 넘어, 텍스트 분석을 통해 보다 객관적이고 직관적인 감성 평점을 제공합니다.

### 1.2. 개발 목표

- **사용자 친화적 인터페이스:** Streamlit을 활용하여 직관적인 대시보드 형태의 UI/UX 구현.
- **MSA(Microservices Architecture) 지향:** 프론트엔드(Streamlit)와 백엔드(FastAPI)를 분리하여 확장성과 유지보수성을 고려한 설계.
- **AI 모델 서빙:** Hugging Face의 Pre-trained 모델을 백엔드에 통합하여 실시간 추론 서비스 구축.

### 1.3. 기술 스택 (Tech Stack)

- **Frontend:** Python, Streamlit
- **Backend:** Python, FastAPI, Uvicorn
- **AI Model:** Hugging Face Transformers (`matthewburke/korean_sentiment` 활용), PyTorch
- **Communication:** RESTful API (HTTP Requests)

---

## 2. 서비스 구조도 (Service Architecture)

### 2.1. 시스템 아키텍처

프론트엔드와 백엔드가 분리되어 있으며, REST API를 통해 데이터를 주고받습니다.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/a45ef5e9-e9a0-4e02-b717-0de47379884c" />

- **Frontend (Streamlit):** 사용자의 입력을 받고, 영화 목록 및 리뷰 데이터를 시각화합니다.
- **Backend (FastAPI):** 클라이언트의 요청을 처리하고, 데이터(영화, 리뷰)를 관리하며, AI 모델을 호출하여 감성 분석을 수행합니다.
- **AI Engine:** 입력된 리뷰 텍스트를 분석하여 긍정(1)/부정(0) 레이블과 확신도(Score)를 반환합니다.

### 2.2. 데이터베이스 구조도 (ERD)

본 프로젝트는 In-memory 방식을 사용하여 데이터를 관리하며, 데이터 구조는 다음과 같습니다.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/d910b60c-3ee9-4aec-8b07-6fcc5629dcc2" />

**[데이터 명세서]**

| **Entity** | **Attribute** | **Type** | **Description** |  |
| --- | --- | --- | --- | --- |
| **Movie** | id | Integer (PK) | 영화 고유 ID |  |
|  | title | String | 영화 제목 |  |
|  | release_date | String | 개봉일 |  |
|  | director | String | 감독 |  |
|  | genre | String | 장르 |  |
|  | poster_url | String | 포스터 이미지 URL |  |
| **Review** | id | Integer (PK) | 리뷰 고유 ID |  |
|  | movie_id | Integer (FK) | 영화 ID (Movie 테이블 참조) |  |
|  | author | String | 작성자 닉네임 |  |
|  | content | Text | 리뷰 내용 |  |
|  | sentiment_label | String | 감성 분석 결과 (긍정/부정) |  |
|  | rating | Float | AI가 산출한 평점 (0~10점) |  |

### 2.3. AI 모델 서빙 전략 (Model Serving Strategy)

본 프로젝트는 **Hugging Face**의 Pre-trained 모델(`matthewburke/korean_sentiment`)을 활용하여 감성 분석을 수행합니다. 제한된 서버 리소스 내에서 효율적인 추론을 위해 다음과 같은 서빙 전략을 적용했습니다.

- **Cold Start 최소화 (Memory Caching):**
사용자의 요청이 들어올 때마다 모델을 로드하면 시간이 오래 걸립니다. 이를 방지하기 위해 **서버가 시작될 때 모델과 토크나이저를 미리 메모리에 로드(Load on Startup)** 하여, 실제 요청 시에는 즉시 추론이 가능하도록 대기 시간을 최소화했습니다.
- **실시간 추론 파이프라인 (Real-time Inference):**`FastAPI` 엔드포인트로 텍스트 데이터가 들어오면 **[토큰화(Tokenization) → 모델 연산(Inference) → 후처리(Post-processing)]** 과정을 거쳐, 긍정/부정 라벨과 확신도(Score)를 0.1초 내외의 빠른 속도로 JSON 반환합니다.
- **클라우드 배포 최적화:**`Render`와 같은 클라우드 환경(PaaS)에서의 배포를 고려하여, 의존성 패키지를 `requirements.txt`로 관리하고 컨테이너 환경에서 안정적으로 동작하도록 `Python 3.x` 및 `PyTorch` 환경을 구성했습니다.

---

## 3. FastAPI Docs 전체 캡쳐

FastAPI가 자동으로 생성한 Swagger UI 명세서입니다. API의 엔드포인트와 요청/응답 스키마를 확인할 수 있습니다.

<img width="1440" height="849" alt="image" src="https://github.com/user-attachments/assets/c184862c-068d-4603-953d-b98a185e72ee" />

<img width="1442" height="1218" alt="image" src="https://github.com/user-attachments/assets/af928269-e031-47bc-88e6-fd7cf19f4562" />

<img width="1433" height="1267" alt="image" src="https://github.com/user-attachments/assets/18523378-d263-4fe8-8800-8f8c8ae18359" />

<img width="1436" height="1126" alt="image" src="https://github.com/user-attachments/assets/408ad688-e685-404c-870c-a7c961231253" />

<img width="1433" height="1017" alt="image" src="https://github.com/user-attachments/assets/40d52163-5c36-43b2-b3d4-f4ab822fb467" />

<img width="1429" height="1016" alt="image" src="https://github.com/user-attachments/assets/993e47cd-19bf-418a-9970-74e1a3bad7b6" />

<img width="1432" height="1012" alt="image" src="https://github.com/user-attachments/assets/44611a94-22c2-45c0-b526-0f36defc837b" />

<img width="1434" height="354" alt="image" src="https://github.com/user-attachments/assets/dbc9a57f-b369-4597-b126-42fc7e19d536" />

**[주요 API 설명]**

- **GET** `/movies`: 등록된 전체 영화 목록을 조회합니다. (Get Movies)
- **POST** `/movies`: 새로운 영화 정보를 데이터베이스에 등록합니다. (Create Movie)
- **POST** `/reviews`: 사용자의 리뷰를 저장하고 AI 감성 분석을 수행합니다. (Create Review)
- **GET** `/reviews/{movie_id}`: 특정 영화 ID에 해당하는 리뷰 목록을 불러옵니다. (Get Reviews)
- **GET** `/movies/{movie_id}/average_rating`: 특정 영화의 AI 분석 평점 평균을 계산하여 반환합니다. (Get Average Rating)
- **DELETE** `/movies/{movie_id}`: 등록된 영화 정보를 삭제합니다. (Delete Movie)
- **DELETE** `/reviews/{review_id}`: 등록된 특정 리뷰를 삭제합니다. (Delete Review)

---

## 4. 서비스 동작 캡쳐 이미지

### 4.1. 메인 화면 및 영화 목록

영화 포스터와 정보가 카드 형태로 시각화되어 표시됩니다.

<img width="2558" height="1269" alt="image" src="https://github.com/user-attachments/assets/a4bd126a-472e-4ba3-b09a-2d0944b00da6" />

### 4.2. 영화 추가 기능

사이드바를 통해 새로운 영화 정보, 리뷰를 입력하고 등록하는 화면입니다.

<img width="252" height="564" alt="image" src="https://github.com/user-attachments/assets/eb9a4723-1513-4aa5-9c53-052fab5cd292" />

<img width="254" height="478" alt="image" src="https://github.com/user-attachments/assets/0d99d451-c594-4483-8f82-f7038f9123a5" />

### 4.3. AI 분석 결과

사용자가 사이드바에서 리뷰를 입력하면, AI 모델이 실시간으로 추론하여 **긍정(초록색)/부정(빨간색)으로 시각화된 결과**를 즉시 보여줍니다.

<img width="1305" height="702" alt="image" src="https://github.com/user-attachments/assets/cabc9bc9-7285-4585-94f9-03ff6cbf6a87" />

<img width="1306" height="1074" alt="image" src="https://github.com/user-attachments/assets/aebf1bde-5399-4829-83c0-5ed57453d41a" />

---

### 5. 결론 및 느낀점

본 프로젝트를 통해 프론트엔드와 백엔드를 분리하여 개발하는 모던 웹 애플리케이션 구조를 이해하게 되었습니다. 특히 FastAPI를 활용한 REST API 서버 구축과 Streamlit을 이용한 빠른 UI 프로토타이핑 과정을 경험했습니다. 향후에는 In-memory 데이터 저장 방식을 실제 데이터베이스(SQLite 등) 연동으로 고도화할 계획입니다.
