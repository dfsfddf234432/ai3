# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1cFVZwfNNpbp80YAXs_-SRxKhjSQjdBMf")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },
     labels[0]: {
       "texts": ["네이마르는", "세계 최고의", "드리블러"],
       "images": ["https://i.namu.wiki/i/zu4_C_cWy9w94re4fXXqEVKfA0YmcwuNIUAbuf32WQJ3-BHc3XCnAhRuRqdBfDrIvkI_H2vMXxbmODpP2LX6LQ.webp"],
       "videos": ["https://www.youtube.com/watch?v=rgz1Mo231TU"]
     },
      labels[1]: {
       "texts": ["메시는", "세계 최고의", "축구선수"],
       "images": ["https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQBs6nuf-DmxKNAie-5GWHNmYs44747-0BtbwhaV-8krnz88HxVMbpbky06kQNP7TFG5e0ba1e1httBLohXRYw_Ts-0DiqSCjBRxlI_c1Qq&s=10"],
       "videos": ["https://www.youtube.com/watch?v=3NeDP4C_oRM"]
      },
      labels[2]: {
       "texts": ["호날두는", "세계 최고의", "스트라이커"],
       "images": ["data:image/webp;base64,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"],
       "videos": ["https://www.youtube.com/watch?v=qmWz-RoZNSU"]
      },
}
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1cFVZwfNNpbp80YAXs_-SRxKhjSQjdBMf")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },
     labels[0]: {
       "texts": ["네이마르는 세계 최고의 드리블러"],
       "images": ["https://i.namu.wiki/i/zu4_C_cWy9w94re4fXXqEVKfA0YmcwuNIUAbuf32WQJ3-BHc3XCnAhRuRqdBfDrIvkI_H2vMXxbmODpP2LX6LQ.webp"],
       "videos": ["https://www.youtube.com/watch?v=rgz1Mo231TU"]
     },
     labels[1]: {
       "texts": ["메시는 세계 최고의 축구선수"],
       "images": ["https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQBs6nuf-DmxKNAie-5GWHNmYs44747-0BtbwhaV-8krnz88HxVMbpbky06kQNP7TFG5e0ba1e1httBLohXRYw_Ts-0DiqSCjBRxlI_c1Qq&s=10"],
       "videos": ["https://www.youtube.com/watch?v=3NeDP4C_oRM"]
     },
      labels[2]: {
       "texts": ["호날두는", "세계 최고의", "스트라이커"],
       "images": ["data:image/webp;base64,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"],
       "videos": ["https://www.youtube.com/watch?v=qmWz-RoZNSU"]
      },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
