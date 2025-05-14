import streamlit as st
st.set_page_config(page_title="AI 면접 에이전트", layout="wide")

import pandas as pd
import whisper
import av
import tempfile
import pdfplumber
import re
import numpy as np # NumPy 임포트
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import scipy.io.wavfile # SciPy wavfile 임포트

# Google Gemini API 라이브러리 임포트
import google.generativeai as genai

# API 키 설정 (Streamlit secrets 사용)
# st.secrets에 google.api_key로 저장된 키를 사용합니다.
try:
    genai.configure(api_key=st.secrets["google"]["api_key"])
    # 사용할 Gemini 모델 지정
    # 'gemini-1.5-pro-latest' 또는 다른 안정화 버전 사용
    GEMINI_MODEL_NAME = "gemini-1.5-pro" # 또는 'gemini-1.5-pro-latest'
except KeyError:
    st.error("Streamlit secrets에 'google.api_key'가 설정되어 있지 않습니다. API 키를 설정해주세요.")
    st.stop()
except Exception as e:
    st.error(f"Gemini API 설정 중 오류 발생: {e}")
    st.stop()


@st.cache_resource
def load_whisper_model():
    """Whisper 모델을 로드하고 캐싱합니다."""
    st.info("Whisper 모델 로드 중 (처음 실행 시 시간이 걸릴 수 있습니다)...")
    model = whisper.load_model("base") # 'base', 'small', 'medium' 등 선택 가능
    st.success("Whisper 모델 로드 완료!")
    return model

# Whisper 모델 로드
whisper_model = load_whisper_model()


st.title("🎤 AI 면접 에이전트")
st.info("지원자 이력서를 맥락 기반으로 분석하고, PDF 기반 질문을 생성하며, 음성 응답까지 기록합니다.")

st.header("1️⃣ 회사 정보 및 이력서 업로드")
col1, col2 = st.columns(2)

with col1:
    core_pdfs = st.file_uploader("🏢 핵심 가치 (PDF)", type=["pdf"], accept_multiple_files=True, key="core_uploader")
    persona_pdfs = st.file_uploader("👤 인재상 (PDF)", type=["pdf"], accept_multiple_files=True, key="persona_uploader")

with col2:
    jd_pdfs = st.file_uploader("📝 직무 기술서 (PDF)", type=["pdf"], accept_multiple_files=True, key="jd_uploader")
    resume_pdfs = st.file_uploader("📄 지원자 이력서 (PDF)", type=["pdf"], accept_multiple_files=True, key="resume_uploader")

def extract_pdf_text(files):
    """업로드된 PDF 파일들에서 텍스트를 추출합니다."""
    result = ""
    if files:
        for uploaded_file in files:
            # 파일 포인터를 처음으로 되돌립니다 (Streamlit rerun 시 필요)
            uploaded_file.seek(0)
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    # 각 페이지에서 텍스트를 추출하고 합칩니다.
                    page_texts = []
                    for page in pdf.pages:
                         text = page.extract_text()
                         if text:
                              page_texts.append(text)
                    result += "\n\n".join(page_texts) + "\n\n" # 파일 간 구분을 위해 공백 추가
            except Exception as e:
                st.warning(f"PDF 파일 읽기 오류: {uploaded_file.name} - {e}")
                result += f"\n\n[텍스트 추출 실패: {uploaded_file.name}]\n\n"
    return result.strip() # 마지막 공백 제거

# LLM의 컨텍스트 창 제한을 고려하여 텍스트를 자릅니다.
# Gemini 1.5 Pro는 큰 컨텍스트를 지원하지만, API 비용 및 처리 시간을 고려하여
# 적절한 길이로 조절하는 것이 좋습니다. 여기서는 OpenAI 때보다 여유 있게 잡습니다.
def truncate_text(text, max_tokens=128000): # 128K 토큰 기준 예시
    """텍스트를 토큰 기준으로 자릅니다 (대략적인 길이)."""
    # 실제 토큰 계산은 API를 사용해야 정확하지만, 여기서는 글자 수로 대략 추정
    # 한국어는 1토큰당 글자 수가 적으므로, 보수적으로 접근
    # 예를 들어 1토큰 = 1~2 글자로 가정하고, 128000 토큰 * 1.5 글자/토큰 = 약 192000 글자
    max_chars = int(max_tokens * 1.5) # 대략적인 글자 수 제한
    if len(text) > max_chars:
        st.warning(f"텍스트가 너무 길어 {max_chars}자로 잘랐습니다.")
        return text[:max_chars] + "..." # 잘렸음을 표시
    return text

st.header("2️⃣ 질문 자동 생성 (AI 기반)")
num_questions = st.slider("질문 수", 1, 15, 5) # 질문 수 범위 및 기본값 조정

if st.button("🚀 질문 생성", key="generate_button"):
    with st.spinner("AI가 문서를 분석하고 질문을 생성 중입니다..."):
        core_text = extract_pdf_text(core_pdfs)
        persona_text = extract_pdf_text(persona_pdfs)
        jd_text = extract_pdf_text(jd_pdfs)
        resume_text = extract_pdf_text(resume_pdfs)

        if not all([core_text, persona_text, jd_text, resume_text]):
            st.warning("모든 PDF 항목을 업로드해야 질문을 생성할 수 있습니다.")
        else:
            # Gemini 1.5 Pro의 대용량 컨텍스트를 활용하여 좀 더 많은 텍스트를 전달
            # 하지만 너무 길면 비용 및 지연이 발생하므로 적절한 길이 조절 필요
            # 여기서는 truncate_text 함수로 대략적인 최대 길이를 제한
            truncated_core = truncate_text(core_text)
            truncated_persona = truncate_text(persona_text)
            truncated_jd = truncate_text(jd_text)
            truncated_resume = truncate_text(resume_text)


            prompt_text = f"""
            당신은 기업의 시니어 인사담당자이며, AI 면접 질문 자동 생성 시스템입니다.

            아래 4가지 문서를 제공합니다:
            1. 회사의 핵심 가치 (Core Values)
            2. 채용 인재상 (Ideal Persona)
            3. 직무 기술서 (Job Description)
            4. 지원자의 이력서 (Resume)

            당신의 역할은 위 4가지 문서의 내용을 모두 종합적으로 분석하여, 특히 이력서 내용을 기반으로 지원자가 회사의 **핵심 가치**, **인재상**, **직무 요건**에 얼마나 부합하는지 검증하기 위한 **경험 기반 면접 질문**을 생성하는 것입니다.

            ---

            📌 질문 설계 지침:

            1. **경험 기반 질문**을 생성하세요. 반드시 지원자의 **이력서에 언급된 구체적인 과거 경험, 프로젝트, 활동, 경력 사항** 등을 기반으로 질문해야 합니다.
            2. 질문은 **핵심 가치**, **인재상**, **직무 요건** 중 하나 이상과 연관시켜, 지원자의 역량, 행동 방식, 가치관 등을 심층적으로 파악할 수 있도록 구성되어야 합니다.
            3. 이력서의 특정 문장, 경력 내용, 키워드 등을 활용하여 다른 문서들과 **교차 분석**한 결과로 도출된 통찰을 바탕으로 질문을 생성하세요.
            4. 질문은 명확하고 간결해야 하며, 지원자가 자신의 경험을 구체적으로 설명하도록 유도해야 합니다.
            5. 형식은 반드시 다음과 같아야 합니다. 각 질문과 질문 의도를 구분하여 명확하게 제시해주세요.

            각 질문은 반드시 아래 형식으로 출력해 주세요:
            ===
            Q. [질문 내용]
            질문 의도: [이 질문을 통해 검증하려는 핵심 역량, 경험 또는 가치관]
            ===

            요청된 질문 수: {num_questions}개

            ---
            제공된 문서 내용:

            핵심 가치:
            {truncated_core}

            인재상:
            {truncated_persona}

            JD:
            {truncated_jd}

            이력서:
            {truncated_resume}
            """

            try:
                # Gemini API 호출
                model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                response = model.generate_content(
                    prompt_text,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7, # 창의성 조절 (0에 가까울수록 보수적)
                        max_output_tokens=2000 # 생성될 응답의 최대 토큰 수 (질문 수에 따라 조절)
                    )
                )

                # 응답에서 텍스트 추출 및 오류 처리
                if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    output = response.candidates[0].content.parts[0].text
                    if output:
                        st.session_state["questions_raw_output"] = output # 원본 저장
                        # 응답 파싱 및 질문 데이터 구조화
                        blocks = output.strip().split("===")
                        questions_data = []
                        for b in blocks:
                            lines = b.strip().split("\n")
                            q_line = next((l for l in lines if re.search(r"^[Qq][.:]\s*", l)), "") # 'Q.' 또는 'Q:' 등 대소문자 구분 없이 찾기
                            intent_line = next((l for l in lines if "질문 의도:" in l), "")
                            if q_line:
                                questions_data.append({
                                    "question": re.sub(r"^[Qq][.:]\s*", "", q_line).strip(),
                                    "intent": intent_line.replace("질문 의도:", "").strip() if intent_line else "질문 의도 파싱 실패 또는 없음"
                                })
                        st.session_state["questions_data"] = questions_data # 구조화된 질문 데이터 저장

                        # 결과 저장을 위한 session_state 초기화 또는 불러오기
                        if "interview_results_state" not in st.session_state:
                            st.session_state["interview_results_state"] = {}
                        # 현재 생성된 질문 수에 맞춰 결과 상태 공간 확보
                        for i in range(len(questions_data)):
                             if f"answer_{i}" not in st.session_state["interview_results_state"]:
                                  st.session_state["interview_results_state"][f"answer_{i}"] = ""
                             if f"comment_{i}" not in st.session_state["interview_results_state"]:
                                  st.session_state["interview_results_state"][f"comment_{i}"] = ""


                        st.success(f"질문 생성 완료 ({len(questions_data)}개)")
                    else:
                        st.error("AI 응답에서 텍스트를 추출할 수 없습니다. AI가 유효한 내용을 생성하지 않았을 수 있습니다.")
                        st.session_state["questions_raw_output"] = ""
                        st.session_state["questions_data"] = []
                        st.session_state["interview_results_state"] = {} # 초기화

                else:
                    st.error("AI로부터 유효한 응답을 받지 못했습니다. 응답이 차단되었거나 오류가 발생했을 수 있습니다.")
                    # 응답 차단 정보 확인
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                         st.warning(f"프롬프트 피드백: {response.prompt_feedback}")
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'finish_reason'):
                                 st.warning(f"후보 응답 종료 이유: {candidate.finish_reason}")
                            if hasattr(candidate, 'safety_ratings'):
                                 st.warning(f"안전 등급: {candidate.safety_ratings}")

                    st.session_state["questions_raw_output"] = ""
                    st.session_state["questions_data"] = []
                    st.session_state["interview_results_state"] = {} # 초기화

            except Exception as e:
                st.error(f"AI 질문 생성 중 오류 발생: {e}")
                st.session_state["questions_raw_output"] = ""
                st.session_state["questions_data"] = []
                st.session_state["interview_results_state"] = {} # 초기화


# 질문 데이터가 session_state에 있을 경우 면접 UI 표시
if "questions_data" in st.session_state and st.session_state["questions_data"]:
    with st.expander("🧾 AI 응답 원본 보기"):
        st.code(st.session_state.get("questions_raw_output", "없음"), language="markdown")

    st.header("3️⃣ 실시간 면접 진행 (음성 답변 인식)")
    st.warning("아래 마이크 아이콘은 오디오 스트림을 시작/중지합니다. '음성 인식 실행' 버튼은 스트림이 활성화된 동안 수집된 오디오를 변환합니다.")


    # 면접 결과 저장을 위한 session_state 초기화 (이미 위에서 했지만, 혹시나 다시 확인)
    if "interview_results_state" not in st.session_state:
        st.session_state["interview_results_state"] = {}
    # 현재 질문 데이터의 수와 session_state의 길이가 다르면 초기화 (새 질문 생성 시)
    if len(st.session_state["interview_results_state"]) // 2 != len(st.session_state["questions_data"]):
         st.session_state["interview_results_state"] = {}
         for i in range(len(st.session_state["questions_data"])):
              st.session_state["interview_results_state"][f"answer_{i}"] = ""
              st.session_state["interview_results_state"][f"comment_{i}"] = ""


    # WebRTC Audio Processor 정의
    # 각 질문별 오디오를 분리해서 처리하려면 더 복잡한 로직이 필요
    # 여기서는 WebRTC 스트림 자체는 하나로 유지하고, 버튼 클릭 시 전체 또는 특정 시점 오디오 처리
    # 하지만, Streamlit의 Rerun 특성상 AudioProcessorBase의 frames 리스트가 매번 초기화되므로
    # 오디오를 지속적으로 수집하고 특정 시점에 처리하는 방식 구현이 복잡함.
    # 간단하게 현재 구현된 ctx.audio_processor.frames를 사용하는 방식을 따르지만,
    # 실제 사용 시 오디오 수집 및 변환 타이밍 문제가 발생할 수 있음을 인지해야 함.
    class InterviewAudioProcessor(AudioProcessorBase):
         def __init__(self) -> None:
             self.frames = []
             self._samples = [] # NumPy 배열 형태로 샘플 저장

         def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
             # 오디오 프레임을 수집합니다.
             self.frames.append(frame)
             # NumPy 배열 형태로 변환하여 별도로 저장 (WAV 파일 생성에 용이)
             self._samples.append(frame.to_ndarray(format="s16le")) # s16le: 16비트 Little Endian PCM

             # 메모리 사용량 관리를 위해 일정 프레임/샘플 이상 쌓이면 오래된 것 삭제 고려 필요
             # pass # 현재는 모든 프레임을 수집

             return frame

         def get_audio_samples(self) -> np.ndarray:
             """수집된 오디오 샘플을 하나의 NumPy 배열로 반환합니다."""
             if not self._samples:
                  return np.array([], dtype=np.int16)
             return np.concatenate(self._samples, axis=1).T # 채널을 두 번째 축으로 합치고 전치 (샘플 수, 채널 수) 형태

         def clear_samples(self):
             """수집된 오디오 샘플을 비웁니다."""
             self.frames = []
             self._samples = []


    # WebRTC 스트리머는 한 번만 정의
    webrtc_ctx = webrtc_streamer(
        key="interview_audio_stream",
        mode=WebRtcMode.SENDONLY, # 오디오만 전송
        audio_receiver_size=2048, # 리시버 버퍼 사이즈 증가
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False}, # 오디오만 요청
        audio_processor_factory=InterviewAudioProcessor, # 커스텀 오디오 프로세서 사용
    )
    st.info("👆 면접 시작/종료 시 위 마이크 아이콘을 클릭하세요.")

    # 각 질문별 UI 생성
    for idx, q in enumerate(st.session_state["questions_data"]):
        st.markdown(f"#### ❓ 질문 {idx+1}")
        st.markdown(f"**{q['question']}**")
        st.markdown(f"📌 질문 의도: _{q['intent']}_")

        # 음성 인식 버튼
        # 이 버튼을 누르면 현재까지 수집된 오디오를 변환
        if st.button(f"🧠 질문 {idx+1} - 음성 인식 실행", key=f"transcribe_btn_{idx}"):
             if webrtc_ctx.audio_processor:
                 # 오디오 샘플 가져오기
                 audio_samples = webrtc_ctx.audio_processor.get_audio_samples()

                 if audio_samples.size > 0:
                     with st.spinner(f"질문 {idx+1} 답변 인식 중..."):
                         try:
                             # 임시 WAV 파일로 저장 (WAV 헤더 포함)
                             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                 temp_wav_path = f.name
                                 # SciPy를 사용하여 WAV 파일 쓰기
                                 # 샘플링 레이트 정보 필요 (av.AudioFrame에서 얻을 수 있음)
                                 # AudioProcessorBase에서 샘플링 레이트를 저장하거나, 첫 프레임에서 얻어야 함
                                 # 간편하게 44100Hz로 가정 (실제 스트림의 샘플링 레이트를 확인하고 적용 필요)
                                 sample_rate = 44100 # 실제 샘플링 레이트로 변경해야 함!
                                 scipy.io.wavfile.write(temp_wav_path, sample_rate, audio_samples)

                             # Whisper 모델로 변환
                             result = whisper_model.transcribe(temp_wav_path, language="ko") # 한국어 지정
                             transcribed_text = result["text"]

                             # session_state에 결과 저장 및 UI 업데이트
                             st.session_state["interview_results_state"][f"answer_{idx}"] = transcribed_text
                             st.text_area(f"🎤 지원자 답변 {idx+1}", value=transcribed_text, key=f"answer_field_{idx}") # UI 업데이트 트리거

                             st.success(f"✅ 질문 {idx+1} 인식 완료!")

                             # 인식 후 현재까지 쌓인 오디오를 비울지 선택 가능
                             # webrtc_ctx.audio_processor.clear_samples() # 필요시 활성화

                         except Exception as e:
                             st.error(f"😥 질문 {idx+1} 답변 인식 오류: {e}")
                             import traceback
                             st.error(traceback.format_exc()) # 디버깅을 위해 스택 트레이스 출력

                         finally:
                              # 임시 파일 삭제
                              if 'temp_wav_path' in locals() and tempfile.exists(temp_wav_path):
                                   tempfile.remove(temp_wav_path)

                 else:
                      st.warning(f"❓ 질문 {idx+1} 답변 오디오가 수집되지 않았습니다. 마이크 아이콘을 클릭하여 스트림을 시작해주세요.")

        # 지원자 답변 수동 입력/확인 폼 (session_state 값으로 초기화 및 업데이트)
        # key를 사용하여 session_state['interview_results_state'][f"answer_{idx}"]와 연동
        st.session_state["interview_results_state"][f"answer_{idx}"] = st.text_area(
            f"🎤 지원자 답변 {idx+1}",
            value=st.session_state["interview_results_state"].get(f"answer_{idx}", ""),
            key=f"answer_field_{idx}" # key를 사용하여 session_state['interview_results_state'][f"answer_{idx}"]에 자동 저장
        )

        # 면접관 의견 입력 폼 (session_state 값으로 초기화 및 업데이트)
        # key를 사용하여 session_state['interview_results_state'][f"comment_{idx}"]와 연동
        st.session_state["interview_results_state"][f"comment_{idx}"] = st.text_area(
            f"📝 면접관 의견 {idx+1}",
            value=st.session_state["interview_results_state"].get(f"comment_{idx}", ""),
            key=f"comment_field_{idx}" # key를 사용하여 session_state['interview_results_state'][f"comment_{idx}"]에 자동 저장
        )

        st.markdown("---")

    # 면접 결과 다운로드 버튼
    # session_state에 저장된 데이터를 바탕으로 CSV 생성
    if st.button("⬇️ 면접 결과 CSV 다운로드", key="download_button"):
        # session_state에서 현재 질문들에 해당하는 결과 데이터 추출
        final_interview_results = []
        for idx, q in enumerate(st.session_state["questions_data"]):
             final_interview_results.append({
                 "질문번호": idx + 1,
                 "질문": q["question"],
                 "질문 의도": q["intent"],
                 "지원자 답변": st.session_state["interview_results_state"].get(f"answer_{idx}", ""),
                 "면접관 의견": st.session_state["interview_results_state"].get(f"comment_{idx}", "")
                 # AI 평가 결과 필드는 여기에 추가될 수 있습니다.
             })

        if final_interview_results:
            df = pd.DataFrame(final_interview_results)
            csv = df.to_csv(index=False).encode('utf-8') # 한글 깨짐 방지를 위해 utf-8로 인코딩
            st.download_button(
                "📥 CSV 저장하기",
                csv,
                "interview_results.csv",
                "text/csv",
                key='csv_download_button'
            )
            st.success("CSV 파일을 다운로드할 수 있습니다.")
        else:
            st.warning("다운로드할 면접 결과 데이터가 없습니다.")

else:
    # 질문 데이터가 없을 때 다운로드 버튼 비활성화 또는 숨김
    pass # 질문 생성 전에는 다운로드 버튼을 표시하지 않음