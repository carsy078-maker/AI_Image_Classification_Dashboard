# AI_Image_Classification
https://aiimageclassificationdashboardgit-fkeetpgidfdfgnmirrgmvc.streamlit.app/
<img width="2559" height="1267" alt="image" src="https://github.com/user-attachments/assets/88788407-76d0-47ae-87a8-c7e58c740768" />

# AI Image Classification Dashboard

사전 학습된 **ViT(Vision Transformer)** 모델을 활용하여, 사용자가 업로드한 사물 이미지를 실시간으로 다중 클래스로 분류하는 End-to-End 웹 대시보드입니다.

---

## 주요 기능 (Key Features)

- **실시간 이미지 분류:** 사용자가 이미지를 업로드하는 즉시 직관적인 UI를 통해 분류 라벨과 신뢰도(Confidence Score)를 반환합니다.
- **Vision Transformer 적용:** Hugging Face의 `google/vit-base-patch16-224` 모델을 연동하여 높은 정확도의 객체 인식 성능을 제공합니다.
- **클라우드 기반 라이브 서비스:** 로컬 환경에 국한되지 않고 Streamlit Cloud를 통해 누구나 접근 가능한 형태의 웹 서비스로 배포되었습니다.

---

## 기술 스택 (Tech Stack)

- **Language:** Python
- **AI/ML:** PyTorch, Hugging Face Transformers (`google/vit-base-patch16-224`)
- **Frontend & Deployment:** Streamlit, Streamlit Cloud

---

## 시스템 아키텍처 (Architecture)

1. **Input:** 사용자가 웹 인터페이스를 통해 이미지 파일(JPG, PNG) 업로드
2. **Preprocessing:** ViT Feature Extractor를 통해 이미지를 224x224 텐서 크기로 정규화 및 리사이징
3. **Inference:** ViT 모델을 통한 다중 클래스 확률 연산
4. **Output:** Top-K 예측 결과를 막대 그래프(Bar Chart)와 함께 시각화하여 렌더링

---

## 트러블슈팅 및 최적화 (Troubleshooting)

**Issue: 클라우드 환경에서의 메모리 초과(OOM) 및 응답 지연**
* **상황:** Streamlit Cloud 배포 초기, 사용자가 이미지를 업로드할 때마다 무거운 딥러닝 가중치 파일(ViT Model)을 새롭게 메모리에 로드하면서 극심한 로딩 지연과 OOM(Out of Memory) 에러가 발생했습니다.

**Solution: 싱글톤(Singleton) 캐싱 패턴 도입**
* Streamlit의 캐싱 데코레이터(`@st.cache_resource`)를 모델 로더 함수에 적용하였습니다. 
* 이를 통해 서버 인스턴스가 실행될 때 모델을 **메모리에 단 한 번만 적재**하고, 이후의 추론 요청은 캐시된 객체를 재사용하도록 파이프라인을 최적화했습니다.

**Result:**
* 불필요한 모델 반복 로드 시간을 완전히 제거하여, 이미지 업로드 즉시 지연 없는(Zero-delay) 실시간 추론 결과를 반환하는 안정적인 라이브 대시보드를 구축했습니다.
