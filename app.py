import streamlit as st
import whisper
import av
import tempfile
import pandas as pd
import datetime
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from openai import OpenAI
from fpdf import FPDF

st.set_page_config(page_title="면접비서관", layout="wide")
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- 입력 정보
st.sidebar.header("📌 면접 정보 입력")
interviewer = st.sidebar.text_input("면접관 이름", value="홍길동")
department = st.sidebar.text_input("부서명", value="인사팀")
candidate = st.sidebar.text_input("지원자 이름", value="김지원")

# --- 면접 시작 준비
# 1. 면접 시작 준비
st.header("1️⃣ 면접 시작 준비")

with st.expander("🧊 아이스브레이킹 멘트 (면접 시작용)", expanded=False):
    st.success("면접 분위기를 부드럽게 시작해보세요 🤝")
    st.markdown("""
- 오늘 오시느라 길 막히진 않으셨어요?  
- 자기소개 전에 가볍게 최근 즐긴 취미 있으신가요?  
- 오랜만의 면접이라 긴장되실 수 있는데, 편하게 말씀해주세요.  
- 최근 읽은 책이나 감명 깊었던 콘텐츠 있으셨나요?
""")

with st.expander("📋 면접관이 지켜야 할 에티켓 10가지", expanded=False):
    st.info("면접관의 태도는 지원자의 인상을 결정짓는 중요한 요소입니다.")
    st.markdown("""
1️⃣ **경청 태도 유지**  
  지원자의 말을 끝까지 끊지 않고 주의 깊게 듣습니다.

2️⃣ **공정한 질문 구성**  
  모든 지원자에게 동일하거나 유사한 질문을 제공합니다.

3️⃣ **압박 질문 지양**  
  불필요하게 긴장시키거나 위협적인 질문은 피합니다.

4️⃣ **개인 정보 존중**  
  가족관계, 외모, 건강 등 사적인 질문은 하지 않습니다.

5️⃣ **중립적 태도 유지**  
  표정, 어투, 제스처 등으로 판단을 유도하지 않습니다.

6️⃣ **시간 엄수**  
  면접 시간은 계획한 범위 내에서 효율적으로 운영합니다.

7️⃣ **지원자 존중 표현**  
  인사, 경청, 감사 인사 등으로 지원자를 존중합니다.

8️⃣ **지원자 이해 도우미 역할**  
  면접의 맥락이나 질문 의도를 명확히 전달합니다.

9️⃣ **적극적인 메모 활용**  
  평가 기준에 따른 객관적인 메모를 남깁니다.

🔟 **면접 후 피드백 고려**  
  간단한 피드백이나 후속 절차를 안내할 수 있으면 좋습니다.
""")

with st.expander("🎯 좋은 면접 질문 만드는 법", expanded=False):
    st.info("지원자의 역량을 명확히 파악할 수 있도록 질문을 설계해보세요.")
    st.markdown("""
1️⃣ **구체적인 경험을 묻는 질문**  
  “~한 경험이 있으신가요?” 보다는  
  → “프로젝트에서 리더 역할을 맡았던 경험을 말씀해 주세요.” 처럼 구체적으로 유도합니다.

2️⃣ **행동 기반 질문 사용 (STAR 기법)**  
  - Situation (상황)  
  - Task (과제)  
  - Action (행동)  
  - Result (결과)  
  → “갈등 상황에서 어떻게 대처하셨나요?” 등으로 구성

3️⃣ **직무 연결 질문 구성**  
  지원한 직무와 연결된 기술, 태도, 협업 방식에 대해 묻습니다.  
  예) “팀 프로젝트에서 맡은 역할과 문제 해결 방식은 어땠나요?”

4️⃣ **핵심 가치/인재상과 연계**  
  회사에서 중시하는 가치와 지원자의 경험을 연결합니다.  
  예) “정직함을 중요하게 여긴 경험이 있으신가요?”

5️⃣ **열린 질문을 사용**  
  예/아니오로 끝나지 않고, 서술형으로 유도합니다.  
  예) “~에 대해 설명해 주세요”, “~할 때 어떤 방식으로 해결하셨나요?”

6️⃣ **질문의 목적을 스스로 점검**  
  이 질문이 지원자의 어떤 점을 파악하기 위한 질문인지 스스로 이해하고 있어야 합니다.
""")

with st.expander("🗣️ 면접 시작 시 안내 멘트", expanded=True):
    st.success("👋 안녕하세요, 곧 면접을 시작하겠습니다. 아래 내용을 간단히 안내드릴게요.")
    st.markdown("""
- 🎯 이번 면접은 **지원자의 경험과 직무 적합성**을 중심으로 진행됩니다.  
- ⏱️ 총 **소요 시간은 약 1시간**입니다. 중간 휴식 없이 이어질 예정입니다.  
- ❓ 질문은 주로 **경험 기반 질문**으로 구성되어 있으며, 편하게 말씀해주세요.  
- ⌛ **답변은 충분히 생각하신 후 천천히 말씀**하셔도 괜찮습니다.  
- 🔁 **질문이 잘 들리지 않으면 언제든 다시 요청**하셔도 됩니다.  
- 🎙️ 면접은 **음성 녹음이 진행**될 수 있으며, 채용 외 용도로는 사용되지 않습니다.  
- 🛑 **불편하거나 중단을 원하실 경우, 언제든 말씀해주시면 즉시 멈추겠습니다.**  
- 🔐 **모든 정보는 안전하게 보호되며 채용 목적으로만 활용됩니다.**
""")

# --- 질문 입력
st.header("2️⃣ 질문 작성")
if "questions" not in st.session_state:
    st.session_state["questions"] = [""] * 5

for i in range(len(st.session_state["questions"])):
    st.session_state["questions"][i] = st.text_input(f"질문 {i+1}", st.session_state["questions"][i], key=f"q_{i}")

if st.button("➕ 질문 추가"):
    st.session_state["questions"].append("")
    st.experimental_rerun()

# --- 면접 진행
st.header("3️⃣ 실시간 면접 진행")
interview_results = []

class Recorder(AudioProcessorBase):
    def __init__(self): self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame)
        return frame

for idx, question in enumerate(st.session_state["questions"]):
    if not question.strip(): continue

    st.subheader(f"❓ 질문 {idx+1}: {question}")
    ctx = webrtc_streamer(
        key=f"stream_{idx}", mode=WebRtcMode.SENDONLY, audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=Recorder,
    )

    if f"answer_{idx}" not in st.session_state:
        st.session_state[f"answer_{idx}"] = ""
        st.session_state[f"clean_answer_{idx}"] = ""

    if ctx.audio_processor and ctx.audio_processor.frames:
        if st.button(f"🎙️ 질문 {idx+1} 음성 인식"):
            with st.spinner("Whisper가 인식 중..."):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    audio = b''.join([f_.to_ndarray().tobytes() for f_ in ctx.audio_processor.frames])
                    f.write(audio)
                    result = model.transcribe(f.name, language="ko")
                    st.session_state[f"answer_{idx}"] = result["text"]
                    st.success("음성 인식 완료!")

    st.text_area("📝 음성 인식 결과", value=st.session_state[f"answer_{idx}"], key=f"raw_{idx}")

    if st.button(f"🧹 질문 {idx+1} 문법 정리"):
        prompt = f"다음 문장은 음성으로 인입된 답변입니다. 내용의 변화없이 말로 서술한 내용을 깔끔하게 문맥과 맞춤법에 맞게 정리해주세요.:\n{st.session_state[f'answer_{idx}']}"
        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        st.session_state[f"clean_answer_{idx}"] = result.choices[0].message.content.strip()

    st.text_area("✅ 정리된 답변", value=st.session_state[f"clean_answer_{idx}"], key=f"clean_{idx}")
    comment = st.text_area("🗂️ 면접관 메모", key=f"memo_{idx}")

    interview_results.append({
        "질문번호": idx+1,
        "질문": question,
        "원본 답변": st.session_state[f"answer_{idx}"],
        "정리된 답변": st.session_state[f"clean_answer_{idx}"],
        "면접관 메모": comment
    })
    st.markdown("---")

# --- 결과 저장 구간
st.subheader("📤 결과 저장")

col1, col2 = st.columns(2)

# --- Excel 저장
with col1:
    import io
    df = pd.DataFrame(interview_results)
    excel_output = io.BytesIO()
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="면접 결과", index=False)

    st.download_button(
        label="📥 Excel 다운로드",
        data=excel_output.getvalue(),
        file_name="interview_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# --- 면접 기록 저장 및 조회
st.subheader("🗃️ 면접 기록 저장 및 조회")

if "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("📌 면접 기록 저장"):
    st.session_state["history"].append({
        "일시": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        "면접관": interviewer,
        "부서": department,
        "지원자": candidate,
        "기록": interview_results
    })
    st.success("면접 기록 저장 완료!")

with st.expander("📚 면접 히스토리 보기", expanded=False):
    for i, h in enumerate(st.session_state["history"]):
        st.markdown(f"🕓 {h['일시']} - {h['지원자']} ({h['부서']})")
        if st.button(f"🔍 상세보기 {i+1}", key=f"his_{i}"):
            for row in h["기록"]:
                st.write(f"Q{row['질문번호']}: {row['질문']}")
                st.write(f"🧹 정리: {row['정리된 답변']}")
                st.write(f"📝 메모: {row['면접관 메모']}")
                st.markdown("---")
