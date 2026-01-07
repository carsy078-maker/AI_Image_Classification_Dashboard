import streamlit as st
from transformers import pipeline
from PIL import Image, ImageOps

# 1. 모델 로딩
# 모델을 처음 불러올 때도 시간이 걸리므로 여기에도 로딩 표시 적용
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

# 앱 시작 시 모델 로딩 상태 표시
with st.spinner("AI 모델을 불러오는 중입니다... 잠시만 기다려주세요!"):
    classifier = load_model()

# 페이지 설정
st.set_page_config(page_title="AI Vision Dashboard", layout="wide")

# --- [이모지 매핑 함수] ---
def get_emoji_for_label(label):
    label = label.lower()
    if "hamster" in label: return "🐹"
    if "dog" in label or "golden retriever" in label: return "🐶"
    if "cat" in label or "tabby" in label: return "🐱"
    if "bird" in label: return "🐦"
    if "pizza" in label: return "🍕"
    if "burger" in label: return "🍔"
    if "car" in label: return "🚗"
    if "flower" in label: return "🌸"
    return "🤖"

# --- [상태 관리] ---
if 'is_analyzed' not in st.session_state:
    st.session_state.is_analyzed = False

def reset_analysis():
    st.session_state.is_analyzed = False

# 2. 사이드바 (설정 메뉴)
with st.sidebar:
    st.header("⚙️ 입력 제어 패널")
    
    input_mode = st.radio(
        "입력 방식 선택",
        ["📂 파일 업로드", "📷 카메라 촬영"],
        on_change=reset_analysis
    )
    
    uploaded_files = [] 
    
    # [모드 1] 파일 업로드
    if input_mode == "📂 파일 업로드":
        uploaded_files = st.file_uploader(
            "파일 찾기", 
            type=["jpg", "png", "jpeg"], 
            accept_multiple_files=True,
            label_visibility="collapsed",
            on_change=reset_analysis 
        )
        st.caption("💡 **Tip:** 여러장의 이미지도 한번에 분석 가능합니다!")
        
        if uploaded_files:
            st.success(f"총 {len(uploaded_files)}장의 이미지가 선택됐습니다.")
            
    # [모드 2] 카메라 촬영 안내
    elif input_mode == "📷 카메라 촬영":
        st.info("👉 오른쪽 메인 화면에서 카메라로 촬영해주세요.")

    st.divider()
    
    st.subheader("🔧 표시 설정")
    # 이모지 ON/OFF 토글
    use_emoji = st.toggle("이모지 표시", value=True)
    
    # 상세 결과 개수 조절 슬라이더
    top_k_slider = st.slider("상세 결과 표시 개수", min_value=1, max_value=10, value=5)
    
    st.divider()
    
    # 분석 실행 버튼
    if st.button("🚀 분석 실행", type="primary", use_container_width=True):
        st.session_state.is_analyzed = True
        
    st.caption("Google ViT Model")

# 3. 메인 화면
st.title("🖼️ AI 이미지 분류")

target_images = []
col1, col2 = st.columns([5.5, 4.5], gap="large")

# --- 왼쪽: 입력 영역 ---
with col1:
    st.subheader("1️⃣ 이미지 입력")
    
    if input_mode == "📷 카메라 촬영":
        camera_file = st.camera_input(
            "사진 촬영", 
            label_visibility="collapsed",
            on_change=reset_analysis 
        )
        if camera_file:
            img = Image.open(camera_file)
            img = ImageOps.exif_transpose(img) 
            target_images.append(img)
            
    else: # 파일 업로드 모드
        if uploaded_files:
            for f in uploaded_files:
                img = Image.open(f)
                img = ImageOps.exif_transpose(img)
                target_images.append(img)
            
            with st.container(border=True):
                st.image(target_images[0], caption=f"대표 이미지 (총 {len(target_images)}장)", use_container_width=True)
        else:
            with st.container(border=True):
                st.info("👈 왼쪽 사이드바에서 이미지를 업로드해주세요.")

# --- 오른쪽: 결과 영역 ---
with col2:
    st.subheader("2️⃣ 분석 결과")
    
    if target_images and st.session_state.is_analyzed:
        
        # 진행률 표시줄 생성
        prog_bar = st.progress(0)
        result_container = st.container(height=600, border=True)
        
        with result_container:
            for i, img in enumerate(target_images):
                st.markdown(f"**Image #{i+1}**")
                
                sub_c1, sub_c2 = st.columns([1, 2])
                with sub_c1:
                    st.image(img, use_container_width=True)
                with sub_c2:
                    # 로딩 표시 추가

                    with st.spinner(f"🔍 {i+1}번째 이미지를 분석하고 있습니다..."):
                        
                        # 모델 추론 (시간이 걸리는 작업)
                        results = classifier(img, top_k=top_k_slider)
                        
                        top_res = results[0]
                        label = top_res['label']
                        
                        if use_emoji:
                            emoji = get_emoji_for_label(label)
                            display_label = f"{emoji} {label}"
                        else:
                            display_label = label
                        
                        st.success(f"**{display_label}** ({top_res['score']:.1%})")
                        
                        with st.expander(f"상세 결과 보기 (Top {top_k_slider})"):
                             for res in results:
                                l = res['label']
                                if use_emoji:
                                    e = get_emoji_for_label(l)
                                    l = f"{e} {l}"
                                    
                                st.write(f"{l}: {res['score']:.2%}")
                                st.progress(res['score'])
                
                st.divider()
                # 진행률 업데이트
                prog_bar.progress((i + 1) / len(target_images))
        
        # 모든 분석이 끝나면 완료 메시지
        st.toast("🎉 모든 이미지 분석이 완료되었습니다!", icon="✅")
                
    elif target_images and not st.session_state.is_analyzed:
         st.warning("👈 준비 완료! 왼쪽 사이드바 하단의 '분석 실행' 버튼을 눌러주세요.")
         
    elif not target_images:
         st.write("분석할 이미지가 없습니다.")