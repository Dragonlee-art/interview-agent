import streamlit as st
st.set_page_config(page_title="AI 면접 에이전트", layout="wide")

import pandas as pd
import whisper
import av
import tempfile
import pdfplumber
import re
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from openai import OpenAI

client = OpenAI(api_key=st.secrets["openai"]["api_key"])

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

st.title("🎤 AI 면접 에이전트")
st.info("지원자 이력서를 맥락 기반으로 분석하고, PDF 기반 질문을 생성하며, 음성 응답까지 기록합니다.")

st.header("1️⃣ 회사 정보 및 이력서 업로드")
col1, col2 = st.columns(2)

with col1:
    core_pdfs = st.file_uploader("🏢 핵심 가치 (PDF)", type=["pdf"], accept_multiple_files=True)
    persona_pdfs = st.file_uploader("👤 인재상 (PDF)", type=["pdf"], accept_multiple_files=True)

with col2:
    jd_pdfs = st.file_uploader("📝 직무 기술서 (PDF)", type=["pdf"], accept_multiple_files=True)
    resume_pdfs = st.file_uploader("📄 지원자 이력서 (PDF)", type=["pdf"], accept_multiple_files=True)

def extract_pdf_text(files):
    result = ""
    if files:
        for uploaded_file in files:
            with pdfplumber.open(uploaded_file) as pdf:
                result += "\n".join([page.extract_text() or "" for page in pdf.pages])
    return result

def truncate_text(text, max_chars=4000):
    return text[:max_chars]

st.header("2️⃣ 질문 자동 생성 (AI 기반)")
num_questions = st.slider("질문 수", 1, 10, 3)

if st.button("🚀 질문 생성"):
    core_text = truncate_text(extract_pdf_text(core_pdfs))
    persona_text = truncate_text(extract_pdf_text(persona_pdfs))
    jd_text = truncate_text(extract_pdf_text(jd_pdfs))
    resume_text = truncate_text(extract_pdf_text(resume_pdfs))

    if not all([core_text, persona_text, jd_text, resume_text]):
        st.warning("모든 PDF 항목을 업로드해야 질문을 생성할 수 있습니다.")
    else:
        prompt = f"""
당신은 기업의 시니어 인사담당자이며, AI 면접 질문 자동 생성 시스템입니다.

아래 4가지 문서를 제공합니다:
1. 회사의 핵심 가치 (Core Values)
2. 채용 인재상 (Ideal Persona)
3. 직무 기술서 (Job Description)
4. 지원자의 이력서 (Resume)

당신의 역할은 이력서 내용을 기반으로 다음과 같은 방식으로 검증 질문을 생성하는 것입니다:

---

📌 질문 설계 지침:

1. **경험 기반 질문**을 생성하세요. 반드시 지원자의 과거 경험이나 프로젝트, 활동을 기반으로 물어야 합니다.
2. 질문은 **핵심 가치**, **인재상**, **직무 요건** 중 하나 이상에 대해 지원자의 적합성을 확인하기 위한 용도로 구성되어야 합니다.
3. 이력서의 특정 문장, 경력 내용, 키워드 등을 활용하여 **교차 분석**한 결과로 질문을 생성하세요.4. 형식은 다음과 같아야 합니다:

각 질문은 반드시 아래 형식으로 출력해 주세요:
===
Q. [질문 내용]
질문 의도: [검증 포인트]
===

질문 수: {num_questions}개
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt + "\n\n핵심 가치:\n" + core_text + "\n\n인재상:\n" + persona_text + "\n\nJD:\n" + jd_text + "\n\n이력서:\n" + resume_text}],
            temperature=0.7,
            max_tokens=1000
        )
        output = response.choices[0].message.content
        st.session_state["questions"] = output
        st.success("질문 생성 완료")

if "questions" in st.session_state and st.session_state["questions"] != "":
    with st.expander("🧾 GPT 응답 원본 보기"):
        st.code(st.session_state.get("questions", "없음"), language="markdown")

    st.header("3️⃣ 실시간 면접 진행 (음성 답변 인식)")

    blocks = st.session_state["questions"].split("===")
    questions_data = []
    for b in blocks:
        lines = b.strip().split("\n")
        q_line = next((l for l in lines if re.search(r"Q[.:]", l)), "")
        intent_line = next((l for l in lines if "질문 의도" in l), "")
        if q_line:
            questions_data.append({
                "question": re.sub(r"^.*Q[.:]", "", q_line).strip(),
                "intent": intent_line.replace("질문 의도:", "").strip()
            })

    class Recorder(AudioProcessorBase):
        def __init__(self) -> None:
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame)
            return frame

    interview_results = []

    for idx, q in enumerate(questions_data):
        st.markdown(f"### ❓ 질문 {idx+1}")
        st.markdown(f"**{q['question']}**")
        st.markdown(f"📌 질문 의도: _{q['intent']}_")

        ctx = webrtc_streamer(
            key=f"webrtc_{idx}",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=Recorder,
        )

        if f"answer_{idx}" not in st.session_state:
            st.session_state[f"answer_{idx}"] = ""

        if ctx.audio_processor and len(ctx.audio_processor.frames) > 0:
            if st.button(f"🧠 질문 {idx+1} - 음성 인식 실행"):
                with st.spinner("Whisper가 인식 중..."):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        audio = b''.join([frame.to_ndarray().tobytes() for frame in ctx.audio_processor.frames])
                        f.write(audio)
                        audio_path = f.name
                    result = model.transcribe(audio_path, language="ko")
                    st.session_state[f"answer_{idx}"] = result["text"]
                    st.success("🎯 인식 완료!")

        st.text_input("🎤 지원자 답변", value=st.session_state[f"answer_{idx}"], key=f"answer_field_{idx}")
        comment_key = f"comment_{idx}"
        st.text_area("📝 면접관 의견", key=comment_key)

        interview_results.append({
            "질문번호": idx + 1,
            "질문": q["question"],
            "질문 의도": q["intent"],
            "지원자 답변": st.session_state[f"answer_{idx}"],
            "면접관 의견": st.session_state.get(comment_key, "")
        })

        st.markdown("---")

    if st.button("⬇️ 면접 결과 CSV 다운로드"):
        df = pd.DataFrame(interview_results)
        csv = df.to_csv(index=False)
        st.download_button("📥 CSV 저장하기", csv, "interview_results.csv", mime="text/csv")
