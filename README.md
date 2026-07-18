<div align="center">

# 💕 HeartFeed

### AI 연애 상담가 레온

**근거 기반 연애 조언을 제공하는 RAG 파이프라인**

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[한국어](#소개) | [English](#introduction)

</div>

---

## 소개

HeartFeed는 YouTube 연애 상담 영상 분석을 통해 근거 기반의 맞춤형 연애 조언을 제공하는 AI 서비스입니다.

### 주요 기능

- 🔍 **RAG 파이프라인** — YouTube 영상 트랜스크립트에서 관련 근거를 검색
- 🛡️ **안전 라우팅** — 자기 위해, 폭력, 스토킹 위험 자동 감지 및 에스컬레이션
- 🎯 **개인화** — MBTI, 관찰 경향, 사주 기반 맞춤 조언
- 📊 **근거 인용** — 모든 조언에 YouTube 영상 타임스탬프 포함
- 💬 **구체적 예시** — 실전 대화 예시와 근거 인용 제공

## 아키텍처

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   프론트엔드  │────▶│   API 서버    │────▶│   Qdrant    │
│  (Next.js)  │     │  (FastAPI)   │     │  (벡터 DB)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────┴───────┐
                    │   LLM API    │
                    │ (OpenRouter) │
                    └──────────────┘
```

### 파이프라인 단계

1. **안전 라우팅** — 위험 키워드 감지 (자기 위해, 폭력, 스토킹)
2. **개인정보 처리** — 민감 정보 자동 마스킹
3. **복용 계획** — 질문 분석 및 명확화 필요성 판단
4. **근거 검색** — 하이브리드 검색 (밀집 + 희소 벡터)
5. **근거 게이트** — 품질 임계값 기반 필터링
6. **리랭킹** — 크로스 인코더 기반 재순위 매기기
7. **컨텍스트 구성** — 관련 근거 통합
8. **사주 반영** — 선택적 사주 기반 문화적 해석
9. **응답 생성** — LLM 기반 구조화된 조언 생성
10. **인용 검증** — 근거 인용 유효성 확인

## 빠른 시작

### 전제 조건

- Python 3.14+
- Docker (Qdrant 실행용)
- OpenRouter API 키

### 설치

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/heartfeed.git
cd heartfeed

# 가상 환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -e ".[dev]"

# 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 API 키 설정
```

## LLM 프로바이더 설정

**Rescue 기본(권장):** DeepSeek OpenAI 호환 API

```bash
LLM_PROVIDER=openai
LLM_API_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=deepseek-v4-flash
LLM_ALLOW_FREE_LLM=false
```


`LLM_PROVIDER` 환경 변수로 LLM 백엔드를 선택합니다.

### openai (기본값)
정적 API 키를 사용하는 모든 OpenAI 호환 엔드포인트입니다. `LLM_API_KEY` / `LLM_API_URL` / `LLM_MODEL` 과 선택적 폴백(`LLM_FALLBACK_*`)을 설정하세요.

### nous (Hermes 운영 에이전트 토큰 재사용)
운영 서버에 이미 작동 중인 **Hermes agent**의 NousResearch OAuth 토큰을 그대로 재사용합니다. 별도 API 키가 필요 없고, 토큰은 `~/.hermes/auth.json` 에서 읽어 만료(≈1시간) 시 자동 갱신됩니다. 기본 모델은 **`tencent/hy3:free`** 으로, v2 JSON 스키마를 첫 호출부터 안정적으로 지킵니다 (참고: `stepfun/step-3.7-flash:free` 는 JSON을 자주 누락해 부적합).

```bash
# .env
LLM_PROVIDER=nous
LLM_NOUS_MODEL=stepfun/step-3.7-flash:free   # 생략 시 기본값
LLM_NOUS_AUTH_PATH=~/.hermes/auth.json        # Hermes auth.json 경로
```

> Hermes 인증이 만료되면 `hermes auth add nous` 로 재로그인하세요. HeartFeed는 토큰 갱신 실패 시 폴백(`LLM_FALLBACK_*`)으로 전환됩니다.

### 실행

```bash
# Qdrant 시작
docker compose up -d

# API 서버 시작
python -m uvicorn dating_rag.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 테스트

```bash
# 전체 테스트 실행
pytest -x -q

# 특정 테스트 실행
pytest tests/test_safety_router.py -x -q
```

## API 사용법

### v2 Chat 엔드포인트

```bash
curl -X POST http://localhost:8000/v2/chat \
  -H "Content-Type: application/json" \
  -d '{
    "schema_version": "2",
    "request_id": "test-001",
    "conversation_id": "test-001",
    "question": "마음에 담아두고 있는 여인이 있는데 어떻게 다가가야될지 모르겠다.",
    "consent": {
      "process_my_birth_data": false,
      "store_my_data": false,
      "process_partner_birth_data": false
    }
  }'
```

### 응답 구조

```json
{
  "status": "answered",
  "answer": {
    "empathy": "공감 메시지",
    "situation_framing": "상황 프레이밍",
    "actions": [
      {
        "text": "액션 아이템",
        "example": "구체적 대화 예시",
        "evidence_quote": "근거 인용"
      }
    ],
    "boundaries": "주의사항",
    "summary": "요약"
  },
  "citations": [
    {
      "id": "S1",
      "url": "https://youtube.com/watch?v=...",
      "timestamp_seconds": 550
    }
  ]
}
```

## 프로젝트 구조

```
heartfeed/
├── src/dating_rag/
│   ├── api/              # FastAPI 엔드포인트
│   ├── safety/           # 안전 라우팅
│   ├── privacy/          # 개인정보 처리
│   ├── intake/           # 복용 계획
│   ├── retrieval/        # 근거 검색
│   ├── generation/       # LLM 응답 생성
│   ├── orchestration/    # 파이프라인 오케스트레이션
│   └── domain/           # 도메인 모델
├── tests/                # 테스트 스위트
├── config/               # 설정 파일
├── scripts/              # 유틸리티 스크립트
└── ui_reference/         # 프론트엔드 레퍼런스
```

## 테스트 결과

- **341 tests passed** (0 failures)
- 안전 라우팅: 자기 위해, 폭력, 스토키잉 패턴 100% 감지
- OOD 질문 차단: 관계 무관 질문 자동 거부
- LLM 재시도: 503 에러 시 지수 백오프 재시도

## 기여하기

1. Fork 저장소
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 커밋 (`git commit -m 'Add amazing feature'`)
4. 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 라이선스

MIT License - LICENSE 파일 참조

## 감사의 말

- [Qdrant](https://qdrant.tech/) - 벡터 데이터베이스
- [FastAPI](https://fastapi.tiangolo.com/) - 웹 프레임워크
- [OpenRouter](https://openrouter.ai/) - LLM API 게이트웨이
- YouTube 연애 상담 크리에이터들

---

<div align="center">

**만든 ❤️ by HeartFeed Team**

</div>


## Rescue / BRT-14

HeartFeed Rescue narrows the product to a **14-day breakup recovery track**.

```bash
export DATEWISE_PRODUCT_MODE=rescue_brt14
export DATEWISE_APP_ENV=dev
export LLM_ALLOW_FREE_LLM=true   # dev only; forbidden in production rescue without override rules
python -m uvicorn dating_rag.api.app:app --host 0.0.0.0 --port 8000
```

- Track config: `GET /v2/track/brt14` and `config/tracks/brt14.yaml`
- Chat body may include `track: { id: "brt14", day_index, contact_status, primary_goal }`
- Frontend: `ui_reference/v2_heartfeed` routes `/`, `/start`, `/today`, `/pro`
- Tests: `pytest tests/test_rescue_core.py -q`
- Claims audit: `python scripts/audit_claims.py`
