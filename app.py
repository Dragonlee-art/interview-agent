import streamlit as st
import whisper
import av
import tempfile
import pandas as pd
import datetime
import os
import numpy as np
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import io
import librosa # <-- Add this import for resampling

# --- Streamlit 페이지 설정 ---
st.set_page_config(page_title="면접 Agent", layout="wide")
st.title("면접 Agent 🤖")
st.markdown("지원자 면접을 위한 질문 준비, 실시간 오디오 녹음/텍스트 변환, 기록 기능을 제공합니다.")

# --- Whisper 모델 로드 (캐싱) ---
@st.cache_resource
def load_model():
    # base 모델 로드 (다른 모델 사용 시 모델 이름 변경)
    # return whisper.load_model("base") # 기본
    # return whisper.load_model("small") # 더 정확하지만 느림
    return whisper.load_model("base") # 여기서는 base 모델 유지
    # 로컬 경로에서 모델 로드 시 예: return whisper.load_model("path/to/your/model.pt")


model = load_model()
# st.sidebar.success("✅ Whisper 모델 로드 완료") # --> 위치 이동됨

# --- 입력 정보 ---
st.sidebar.header("📌 면접 정보 입력")
# 1. 면접날짜 자동 노출 추가
today = datetime.date.today()
st.sidebar.write(f"**🗓️ 면접 날짜:** {today.strftime('%Y년 %m월 %d일')}") # 원하는 형식으로 표시

# 세션 상태에서 값을 가져오도록 수정 (페이지 새로고침 시에도 유지)
interviewer = st.sidebar.text_input("면접관 이름", value=st.session_state.get("interviewer", "홍길동"), key="interviewer")
department = st.sidebar.text_input("지원부서", value=st.session_state.get("department", "인사팀"), key="department")
candidate = st.sidebar.text_input("지원자 이름", value=st.session_state.get("candidate", "김지원"), key="candidate")


# --- 면접 시작 준비 ---
st.header("1️⃣ 면접 시작 준비")
# 1. 면접 10분전 안내 추가
st.markdown("_면접 10분전에 꼭 읽어보세요!_") # 안내 문구를 기울임꼴로 표시
# st.info("면접 10분전에 꼭 읽어보세요!") # 또는 이렇게 info 박스로 표시할 수도 있습니다.

# 1. 면접 시작 준비 Expander 초기 상태 변경
# '아이스브레이킹 멘트'는 기본값인 expanded=False 유지
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
⃟표정, 어투, 제스처 등으로 판단을 유도하지 않습니다.

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

with st.expander("🗣️ 면접 시작 시 안내 멘트", expanded=False): # expanded=True -> expanded=False 로 변경
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

# --- 질문 입력 ---
st.header("2️⃣ 질문 작성")
# 2. 후보자에게 할 질문 안내 추가
st.markdown("_후보자에게 할 질문을 작성하세요!_") # 안내 문구를 기울임꼴로 표시
# st.info("후보자에게 할 질문을 작성하세요!") # 또는 이렇게 info 박스로 표시할 수도 있습니다.
if "questions" not in st.session_state:
    # 2. 질문 작성에 기본으로 필드 3개 정도 만들기
    st.session_state["questions"] = ["", "", ""]

# 각 질문에 대한 입력 필드 생성
for i in range(len(st.session_state["questions"])):
    # 질문 내용이 빈 문자열인 경우 기본값을 설정하지 않음 (새 질문 추가 시 빈칸으로 보이게)
    current_value = st.session_state["questions"][i] if i < len(st.session_state["questions"]) else ""
    st.session_state["questions"][i] = st.text_input(f"질문 {i+1}", value=current_value, key=f"q_{i}")

# 질문 추가/삭제 버튼
col_add, col_remove = st.columns([1, 1])
with col_add:
    # 3. 질문 추가 버튼에 강조 표시 (빨간색 배경은 직접 지원되지 않아 이모지로 대체)
    if st.button("✨ 질문 추가", key="add_question_button"): # 이모지 추가 및 고유 key 설정
        st.session_state["questions"].append("")
        # 새로운 질문에 대한 상태 초기화 (None 또는 빈 문자열)
        if "answer_segments" in st.session_state:
            st.session_state["answer_segments"].append(None)
        # 답변 및 메모 텍스트 영역 상태는 해당 질문 UI가 그려질 때 초기화될 것임
        st.rerun() # 상태 변경 후 새로고침

with col_remove:
    # 질문이 0개일 때는 삭제 버튼 비활성화
    if len(st.session_state["questions"]) > 0: # 질문이 0개일 때는 삭제 버튼 비활성화
         if st.button("🗑️ 마지막 질문 삭제", key="remove_question_button"): # 고유 key 설정
            removed_idx = len(st.session_state["questions"]) - 1
            st.session_state["questions"].pop()
            # 삭제된 질문과 관련된 상태(답변, 메모, 세그먼트)도 함께 정리
            if f"answer_{removed_idx}" in st.session_state:
                 del st.session_state[f"answer_{removed_idx}"]
            if f"memo_{removed_idx}" in st.session_state:
                 del st.session_state[f"memo_{removed_idx}"]
            if "answer_segments" in st.session_state and len(st.session_state["answer_segments"]) > removed_idx:
                # 해당 인덱스의 세그먼트 정보 삭제
                st.session_state["answer_segments"].pop() # 마지막 항목 pop
                # 만약 삭제된 질문이 현재 녹음 중인 질문이었다면 상태 초기화
                if st.session_state.get("currently_recording_idx") == removed_idx:
                    st.session_state["currently_recording_idx"] = None
                    # processor 상태도 초기화 필요 (이 부분은 WebRTC 특성상 어려움이 있을 수 있음 - restart 또는 logic 보완 필요)
            st.rerun() # 상태 변경 후 새로고침


# --- 실시간 면접 진행 ---
st.header("3️⃣ 실시간 면접 진행")
# 4. 실시간 면접 진행 설명 추가
st.info("""
**🎤 면접 진행 방법 설명:**

1.  먼저 아래 WebRTC 스트리머의 **'Start' (또는 '연결 시작') 버튼**을 클릭하여 전체 면접 오디오 스트림을 연결합니다. 사이드바에 **'🌐 오디오 스트림 연결됨'** 메시지가 뜨는지 확인하세요.
2.  각 질문별 답변을 녹음할 때는 해당 질문 아래의 **'▶️ 답변 녹음 시작' 버튼**을 클릭합니다. 다른 답변이 녹음 중일 때는 버튼이 비활성화됩니다.
3.  답변을 마친 후 **'⏹️ 답변 녹음 중지' 버튼**을 클릭하여 해당 답변의 녹음 구간을 확정합니다.
4.  녹음이 완료되면 **'🎤 답변 음성 인식' 버튼**이 활성화됩니다. 이 버튼을 눌러 녹음된 답변을 텍스트로 변환합니다.
5.  텍스트 변환 결과는 '지원자 답변' 텍스트 영역에 표시되며, 필요시 직접 수정할 수 있습니다. '면접관 메모' 영역에는 자유롭게 내용을 입력하세요.
6.  모든 질문에 대해 2~5 단계를 반복합니다.
7.  면접 종료 후 '결과 저장 및 기록 관리' 섹션에서 전체 내용을 Excel로 다운로드하거나 면접 기록으로 저장할 수 있습니다.
""")


st.warning("⚠️ 브라우저 탭/창을 닫거나 새로고침하면 녹음 중인 오디오 데이터는 유실됩니다.")


# 오디오 프레임을 수집할 전역 Recorder 클래스
class GlobalRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = [] # 모든 오디오 프레임 저장
        # self.is_recording_answer = False # 이제 이 상태는 Streamlit 세션 상태에서 관리
        self.current_segment_start_idx = -1 # 현재 녹음 중인 답변의 시작 프레임 인덱스
        # WebRTC 스트림 시작 시점의 프레임 인덱스 (오디오 전처리 시 초반 노이즈/버퍼링 프레임 무시 등에 활용될 수 있음)
        self.stream_start_frame_idx = -1

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # 스트림 시작 시점 기록 (recv가 처음 호출될 때)
        if self.stream_start_frame_idx == -1:
             self.stream_start_frame_idx = len(self.frames)

        self.frames.append(frame)

        # 오디오 프레임을 가공 없이 그대로 반환 (여기서는 가공 필요 없음)
        return frame

# 단일 WebRTC 스트리머 인스턴스 생성 (질문 루프 밖)
# key는 Streamlit 앱 내에서 유일해야 하며, 이 인스턴스를 식별하는 데 사용됨
global_ctx = webrtc_streamer(
    key="global_interview_audio_stream", # 단일 스트리머 고유 키
    mode=WebRtcMode.SENDONLY, # 오디오 데이터만 서버(Streamlit)로 보냄
    audio_receiver_size=4096, # <-- 버퍼 크기를 좀 더 늘려봅니다.
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # STUN 서버 설정 (NAT 통과)
     media_stream_constraints={
        "audio": True, # 오디오 제약 조건을 브라우저 기본값에 맡김
        "video": False # 비디오 스트림은 사용하지 않음
    },
    audio_processor_factory=GlobalRecorder, # 커스텀 오디오 프로세서 연결
    async_processing=True
)


# 2. 위스퍼 모델 로드 완료 메시지 위치 이동
# 사이드바에서 오디오 스트림 상태 메시지 바로 위로 이동
# global_ctx가 초기화된 후 상태 표시 블록 이전에 위치
st.sidebar.success("✅ Whisper 모델 로드 완료") # --> 위치 이동됨


# 전역 스트리머 상태 표시 (사이드바)
# global_ctx와 global_ctx.state가 모두 존재할 때 상태 확인
if global_ctx and global_ctx.state:
    if global_ctx.state.playing:
        st.sidebar.success("🌐 오디오 스트림 연결됨")
    # state 객체는 있으나 playing이 아닌 경우 (연결 중 또는 대기 상태)
    # global_ctx.state 객체가 None이 아닌지 추가 확인 (더 안전한 검사)
    elif global_ctx.state is not None and not global_ctx.state.playing:
        st.sidebar.warning("⏳ 오디오 스트림 연결 중...")
    # global_ctx 객체 자체가 아직 초기화되지 않았거나 state가 None인 경우
else:
    # global_ctx가 None일 때 (아직 초기화 전) 상태 메시지
    st.sidebar.warning("⏳ 오디오 스트림 초기화 중...") # 초기 상태 메시지 변경
    # global_ctx가 초기화되었으나 연결 실패 상태일 때의 메시지
    if global_ctx and not global_ctx.state: # 이 조건은 사실상 위의 else에 포함되지만 명시적으로 구분해볼 수도 있습니다.
         st.sidebar.error("❌ 오디오 스트림 연결 실패. 마이크 권한을 확인하거나 페이지 새로고침이 필요할 수 있습니다.")


# 답변 오디오 세그먼트 정보를 저장할 상태 (시작, 종료 프레임 인덱스 튜플 또는 None)
# 질문 목록 길이 변경에 맞춰 이 리스트의 길이도 관리되어야 함 (질문 추가/삭제 버튼 로직에 반영됨)
if "answer_segments" not in st.session_state:
    st.session_state["answer_segments"] = [None] * len(st.session_state["questions"])
# 현재 녹음 중인 답변의 인덱스를 저장할 상태 (None 또는 질문 인덱스 0부터 시작)
if "currently_recording_idx" not in st.session_state:
    st.session_state["currently_recording_idx"] = None


interview_results = [] # 최종 결과(질문, 답변, 메모)를 수집할 리스트 (Excel 저장, 기록 저장에 사용)

# 스트리머가 활성화되면 오디오 프로세서 객체를 가져옴
processor = global_ctx.audio_processor if global_ctx and global_ctx.audio_processor else None


# 각 질문에 대해 반복하며 면접 진행 UI 생성
# 질문 목록이 비어있으면 아래 루프는 실행되지 않음
for idx, question in enumerate(st.session_state["questions"]):
    st.subheader(f"❓ 질문 {idx+1}: {question if question.strip() else ' (질문 내용을 입력해주세요)'}")

    # 질문별 답변, 메모 텍스트 영역의 상태 초기화 (필요 시)
    # key를 사용하여 세션 상태와 직접 연결
    if f"answer_{idx}" not in st.session_state:
         st.session_state[f"answer_{idx}"] = ""
    if f"memo_{idx}" not in st.session_state:
         st.session_state[f"memo_{idx}"] = ""

    # 답변 녹음 및 텍스트 변환 버튼을 위한 컬럼 레이아웃
    col_rec, col_transcribe = st.columns([1, 3]) # 녹음 버튼 컬럼을 작게

    # 오디오 스트리머가 활성화되고 재생 중인 경우에만 녹음/텍스트 변환 컨트롤 표시
    if processor and global_ctx.state.playing:
        # 현재 이 질문에 대한 답변을 녹음 중인 경우
        with col_rec:
            if st.session_state["currently_recording_idx"] == idx:
                # ▶️ 녹음 중지 버튼 표시
                if st.button(f"⏹️ 답변 {idx+1} 녹음 중지", key=f"stop_rec_{idx}"):
                    end_idx = len(processor.frames) # 현재 시점의 누적 프레임 수를 종료 인덱스로
                    start_idx = processor.current_segment_start_idx

                    # 유효한 세그먼트인지 확인 (시작 인덱스가 기록되었고 종료 인덱스보다 큰지)
                    # 최소 길이 제한 등을 추가하여 너무 짧은 오디오는 무시할 수 있음
                    if start_idx != -1 and end_idx > start_idx + 10: # 예: 최소 10프레임 이상
                         st.session_state["answer_segments"][idx] = (start_idx, end_idx)
                         st.session_state[f"answer_{idx}"] = "✅ 답변 녹음 완료. 아래 '음성 인식' 버튼을 눌러 텍스트로 변환하거나 오디오를 확인하세요." # 메시지 수정
                         processor.current_segment_start_idx = -1 # processor의 시작 인덱스 초기화
                         st.session_state["currently_recording_idx"] = None # 현재 녹음 중인 답변 인덱스 초기화
                         st.success(f"✅ 질문 {idx+1} 답변 녹음이 중지되었습니다. 오디오 프레임: {start_idx} ~ {end_idx}")
                         st.rerun() # 상태 업데이트를 위해 재실행
                    else:
                         # 녹음 시작 버튼은 눌렀으나 유의미한 프레임이 캡처되지 않은 경우
                         st.session_state[f"answer_{idx}"] = "⚠ 녹음된 오디오가 너무 짧거나 없습니다. 다시 녹음해 주세요."
                         st.warning(f"⚠ 질문 {idx+1} 녹음된 오디오 프레임이 너무 짧거나 없습니다.")
                         processor.current_segment_start_idx = -1 # processor의 시작 인덱스 초기화
                         st.session_state["currently_recording_idx"] = None # 현재 녹음 중인 답변 인덱스 초기화
                         st.rerun()


            # 현재 녹음 중이 아닌 경우 (다른 답변이 녹음 중이거나 모든 녹음이 중지된 상태)
            elif st.session_state["currently_recording_idx"] is None:
                # ▶️ 녹음 시작 버튼 표시 (현재 녹음 중인 답변이 없을 때만 활성화)
                 if st.button(f"▶️ 답변 {idx+1} 녹음 시작", key=f"start_rec_{idx}"):
                     # 현재 시점의 누적 프레임 수를 시작 인덱스로 기록
                     processor.current_segment_start_idx = len(processor.frames)
                     st.session_state["currently_recording_idx"] = idx # 현재 녹음 중인 답변 인덱스 기록
                     st.session_state[f"answer_{idx}"] = "🎧 답변 녹음 중..." # 사용자에게 피드백
                     st.info(f"▶️ 질문 {idx+1} 답변 녹음이 시작되었습니다. 답변 완료 후 '녹음 중지'를 눌러주세요.")
                     st.rerun() # 상태 업데이트를 위해 재실행
            else:
                 # 다른 답변이 녹음 중일 때는 이 질문의 녹음 시작 버튼 비활성화
                 st.button(f"▶️ 답변 {idx+1} 녹음 시작", key=f"start_rec_{idx}_disabled", disabled=True, help="다른 답변 녹음 중입니다.")


        # 텍스트 변환 버튼 영역
        with col_transcribe:
            # 해당 질문에 대한 답변 세그먼트가 기록된 경우 (None이 아닌 경우)
            if st.session_state["answer_segments"][idx] is not None:
                # 🎤 음성 인식 버튼 표시
                # 현재 다른 답변이 녹음 중일 때는 변환 버튼 비활성화 (처리 중 부하 방지 등)
                is_transcribe_disabled = st.session_state["currently_recording_idx"] is not None
                if st.button(f"🎤 답변 {idx+1} 음성 인식", key=f"transcribe_{idx}", disabled=is_transcribe_disabled, help="다른 답변 녹음 중에는 음성 인식을 할 수 없습니다." if is_transcribe_disabled else None):
                    start_idx, end_idx = st.session_state["answer_segments"][idx]
                    segment_frames = processor.frames[start_idx:end_idx]

                    if not segment_frames:
                        st.warning("⚠ 녹음된 오디오 프레임이 없습니다. 다시 녹음해 주세요.")
                        st.session_state[f"answer_{idx}"] = "⚠ 오디오 프레임 부족 또는 오류."
                    else:
                        with st.spinner(f"🎙️ 질문 {idx+1} 답변 음성 인식 중..."):
                            temp_audio_path = None
                            try:
                                # Determine the actual sample rate from the first frame
                                # Assuming all frames have the same sample rate as the first.
                                original_sample_rate = segment_frames[0].rate if segment_frames else 0

                                # Convert frames to numpy array and handle stereo to mono
                                audio_np_list = []
                                for f_ in segment_frames:
                                    data = f_.to_ndarray()
                                    if f_.layout.name in ["stereo", "stereo_downmix"]:
                                        audio_np_list.append(np.mean(data, axis=0)) # Downmix stereo to mono
                                    else:
                                        audio_np_list.append(data.flatten()) # Assume mono or handle other layouts as mono

                                audio_np_combined = np.concatenate(audio_np_list)

                                # Resample if necessary (Whisper requires 16kHz)
                                target_sample_rate = 16000
                                if original_sample_rate != target_sample_rate and original_sample_rate > 0:
                                    # Ensure audio_np_combined is float type for librosa
                                    audio_np_combined = audio_np_combined.astype(np.float32)
                                    audio_final_for_whisper = librosa.resample(y=audio_np_combined, orig_sr=original_sample_rate, target_sr=target_sample_rate)
                                    st.info(f"🎤 오디오를 {original_sample_rate}Hz에서 {target_sample_rate}Hz로 리샘플링했습니다.")
                                else:
                                    audio_final_for_whisper = audio_np_combined

                                # Normalize and convert to int16 for WAV file
                                if np.max(np.abs(audio_final_for_whisper)) > 0:
                                    audio_final_for_whisper = audio_final_for_whisper / np.max(np.abs(audio_final_for_whisper))
                                audio_int16 = np.int16(audio_final_for_whisper * 32767)

                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                    sf.write(f.name, audio_int16, target_sample_rate, format='WAV', subtype='PCM_16')
                                    temp_audio_path = f.name

                                result = model.transcribe(temp_audio_path, language="ko")
                                st.session_state[f"answer_{idx}"] = result["text"].strip()
                                st.success(f"✅ 질문 {idx+1} 답변 음성 인식 완료!")

                                # 🎧 디버깅: 오디오 파형과 수치 확인
                                st.write("🔍 평균값:", np.mean(audio_final_for_whisper), "최댓값:", np.max(audio_final_for_whisper))
                                st.line_chart(audio_final_for_whisper[:1000])

                            except Exception as e:
                                st.error(f"❌ 질문 {idx+1} 음성 처리 오류: {e}")
                                st.session_state[f"answer_{idx}"] = f"❌ 음성 처리 오류: {e}"
                            finally:
                                if temp_audio_path and os.path.exists(temp_audio_path):
                                    os.remove(temp_audio_path)

                # --- Audio Download Button ---
                # Downloader will also provide 16kHz resampled audio for consistency
                temp_audio_path_download = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_dl:
                        start_idx_dl, end_idx_dl = st.session_state["answer_segments"][idx]
                        segment_frames_dl = processor.frames[start_idx_dl:end_idx_dl]

                        if segment_frames_dl:
                            original_sample_rate_dl = segment_frames_dl[0].rate if segment_frames_dl else 0

                            audio_np_list_dl = []
                            for f_dl_frame in segment_frames_dl:
                                data_dl = f_dl_frame.to_ndarray()
                                if f_dl_frame.layout.name in ["stereo", "stereo_downmix"]:
                                    audio_np_list_dl.append(np.mean(data_dl, axis=0))
                                else:
                                    audio_np_list_dl.append(data_dl.flatten())

                            audio_np_combined_dl = np.concatenate(audio_np_list_dl)

                            # Resample to 16kHz for download as well, for consistency with transcription
                            target_sample_rate_dl = 16000
                            if original_sample_rate_dl != target_sample_rate_dl and original_sample_rate_dl > 0:
                                audio_np_combined_dl = audio_np_combined_dl.astype(np.float32)
                                audio_final_dl = librosa.resample(y=audio_np_combined_dl, orig_sr=original_sample_rate_dl, target_sr=target_sample_rate_dl)
                            else:
                                audio_final_dl = audio_np_combined_dl

                            if np.max(np.abs(audio_final_dl)) > 0:
                                audio_final_dl = audio_final_dl / np.max(np.abs(audio_final_dl))
                            audio_int16_dl = np.int16(audio_final_dl * 32767)

                            sf.write(f_dl.name, audio_int16_dl, target_sample_rate_dl, format='WAV', subtype='PCM_16')
                            temp_audio_path_download = f_dl.name

                    if temp_audio_path_download and os.path.exists(temp_audio_path_download):
                        with open(temp_audio_path_download, "rb") as file:
                            st.download_button(
                                label=f"⬇️ 답변 {idx+1} 오디오 다운로드 (.wav)",
                                data=file,
                                file_name=f"답변_{idx+1}_오디오_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                                mime="audio/wav",
                                key=f"actual_download_btn_{idx}"
                            )
                except Exception as e:
                     st.error(f"❌ 답변 {idx+1} 오디오 다운로드 준비 중 오류 발생: {e}")
                finally:
                    # Files created with delete=False will persist until process ends or manual cleanup.
                    # This is necessary for st.download_button to work correctly.
                    pass


            else:
                 # 오디오 스트리머가 준비되지 않았거나 재생 중이 아닐 때 안내 메시지 표시
                 if len(st.session_state["questions"]) > 0:
                     st.warning("⚠ 오디오 스트림 연결을 기다리거나 사이드바에서 상태를 확인해 주세요.")


        # 음성 인식 결과 (수정 가능) 및 면접관 메모 입력 필드
        st.text_area("🖍️ 지원자 답변 (음성 인식 결과 및 수정)", value=st.session_state[f"answer_{idx}"], key=f"answer_{idx}", height=150)
        st.text_area("🗂️ 면접관 메모", value=st.session_state[f"memo_{idx}"], key=f"memo_{idx}", height=100)

        interview_results.append({
            "질문번호": idx+1,
            "질문": question,
            "지원자 답변": st.session_state[f"answer_{idx}"],
            "면접관 메모": st.session_state[f"memo_{idx}"]
        })
        st.markdown("---") # 각 질문 섹션 구분선

# --- 결과 저장 및 기록 관리 ---
st.header("4️⃣ 결과 저장 및 기록 관리")

# Excel 다운로드 버튼과 기록 관리 버튼을 위한 컬럼 레이아웃
col_excel, col_history = st.columns([1, 1])

with col_excel:
    # interview_results 리스트(현재 면접의 최신 상태)를 DataFrame으로 변환
    df = pd.DataFrame(interview_results)

    # DataFrame을 Excel 파일 형식으로 메모리(BytesIO)에 저장
    excel_output = io.BytesIO()
    # ExcelWriter를 사용하여 xlsx 형식으로 저장
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        # DataFrame을 Excel 시트에 쓰기 (인덱스 제외)
        df.to_excel(writer, sheet_name=f"{st.session_state.get('candidate', '면접결과')}_면접 결과", index=False)
        # Excel 파일 닫기 (writer가 종료될 때 자동 저장)

    # Excel 파일 다운로드 버튼 생성
    st.download_button(
        label="📥 현재 면접 결과 Excel 다운로드", # 버튼 라벨
        data=excel_output.getvalue(),          # 다운로드할 데이터 (BytesIO의 값)
        file_name=f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_{st.session_state.get('candidate', '면접결과')}_면접결과.xlsx", # 다운로드될 파일 이름
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" # 파일 MIME 타입
    )

# --- 면접 기록 저장 및 조회 ---

# 전체 면접 기록을 저장할 리스트를 세션 상태에 초기화
if "history" not in st.session_state:
    st.session_state["history"] = [] # 전체 면접 기록 리스트

# 현재 기록 중인 면접의 상세 정보 표시 상태를 관리할 변수
# None: 아무 기록도 상세 표시 안함, 숫자: 해당 인덱스의 기록 상세 표시
if "showing_history_details" not in st.session_state:
     st.session_state["showing_history_details"] = None

with col_history:
    # 현재 면접 상태를 기록 리스트에 추가하는 버튼
    if st.button("📌 현재 면접 기록 저장"):
        # 현재 interview_results 리스트 (화면에 보이는 최신 상태)를 기록 리스트에 추가
        # 기록 리스트에 추가될 때의 스냅샷을 저장하기 위해 interview_results[:]와 같이 복사본 저장
        st.session_state["history"].append({
            "일시": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), # 저장 시점 일시
            "면접관": st.session_state.get("interviewer", "N/A"), # 세션 상태에서 면접관 이름 가져옴
            "부서": st.session_state.get("department", "N/A"),     # 세션 상태에서 부서명 가져옴
            "지원자": st.session_state.get("candidate", "N/A"),     # 세션 상태에서 지원자 이름 가져옴
            "기록": interview_results[:] # 현재 면접 결과 리스트의 복사본 저장
        })
        st.success("✅ 현재 면접 기록이 저장되었습니다.")

# 저장된 면접 히스토리를 볼 수 있는 Expander
with st.expander("📚 저장된 면접 기록 보기", expanded=False):
    # 저장된 기록이 없을 경우 안내 메시지
    if not st.session_state["history"]:
        st.info("저장된 면접 기록이 없습니다.")
    else:
        # 저장된 기록 리스트를 역순으로 순회하여 최신 기록이 상단에 오도록 함
        # enumerate(reversed(...)) 사용 시 인덱스가 역순이 되므로, range와 reversed 사용
        for i in reversed(range(len(st.session_state["history"]))):
            h = st.session_state["history"][i] # 해당 인덱스의 기록 데이터

            st.markdown(f"---") # 각 기록 섹션 구분선

            # 기록 요약 정보와 상세 보기/닫기 버튼을 위한 컬럼 레이아웃
            col_hist_sum, col_hist_btn = st.columns([3, 1])
            with col_hist_sum:
                 # 기록 요약 정보 표시 (get() 사용으로 키 오류 방지)
                 st.markdown(f"**🕒 일시:** {h.get('일시', 'N/A')}")
                 st.markdown(f"**🧑‍💼 지원자:** {h.get('지원자', 'N/A')} / **🏢 부서:** {h.get('부서', 'N/A')}")
                 st.markdown(f"**👤 면접관:** {h.get(' 면접관', 'N/A')}")

            with col_hist_btn:
                 # 현재 이 기록의 상세 내용을 보고 있는 경우
                 if st.session_state["showing_history_details"] == i:
                     # '상세 보기 닫기' 버튼 표시
                     if st.button(f"➖ 상세 보기 닫기", key=f"hide_his_{i}"):
                          st.session_state["showing_history_details"] = None # 상세 보기 상태 초기화
                          st.rerun() # 상태 변경 반영을 위해 새로고침

                 # 현재 이 기록의 상세 내용을 보고 있지 않은 경우
                 else:
                     if st.button(f"🔍 상세 보기", key=f"show_his_{i}"):
                          st.session_state["showing_history_details"] = i # 상세 보기 상태를 현재 기록 인덱스로 설정
                          st.rerun() # 상태 변경 반영을 위해 새로고침

            # `showing_history_details` 상태가 현재 기록의 인덱스와 일치할 경우에만 상세 내용을 표시
            if st.session_state["showing_history_details"] == i:
                st.markdown("---") # 상세 내용 시작 구분선
                st.subheader("상세 기록")
                # 저장된 기록(질문-답변-메모 리스트)을 순회하며 상세 내용 표시
                for row in h.get("기록", []): # '기록' 키가 없거나 비어있을 경우를 대비하여 [] 반환
                    # 각 항목의 키가 없을 경우 기본값 'N/A' 표시
                    st.markdown(f"**Q{row.get('질문번호', 'N/A')}:** {row.get('질문', 'N/A')}")
                    st.markdown(f"**🖍️ 지원자 답변:** {row.get('지원자 답변', 'N/A')}")
                    st.markdown(f"**🗂️ 면접관 메모:** {row.get('면접관 메모', 'N/A')}")
                    st.markdown("---") # 질문별 구분선