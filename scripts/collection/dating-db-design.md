# 연애 콘텐츠 데이터베이스 구축 설계

## 1. 목표

- SNS에서 연애 관련 인기 동영상 **500개** 수집
- 조회수 **10만 이상**, 영향력 있는 채널
- 영상 → MD 파일 변환 → 데이터베이스화

## 2. 카테고리 분류

| # | 카테고리 | 검색 키워드 (한국어) | 검색 키워드 (영어) |
|---|---------|-------------------|------------------|
| 1 | 대화법 | 연애 대화법, 설레는 대화, 썸 대화 기술 | dating conversation tips, how to flirt |
| 2 | MBTI 연애 | MBTI 연애 유형, MBTI 궁합, MBTI 썸 | MBTI dating compatibility |
| 3 | 남자 심리 | 남자 심리, 남자 호감 신호, 남자 꼬시는법 | how to attract a guy, male psychology dating |
| 4 | 여자 심리 | 여자 심리, 여자 호감 신호, 여자 꼬시는법 | how to attract a girl, female psychology dating |
| 5 | 연애 금지 | 하면 안되는 연애, 헤어지는 이유, 연애 실수 | dating mistakes to avoid, red flags |
| 6 | 고백/프로포즈 | 고백 방법, 프로포즈 Ideas, 썸에서 연애로 | how to confess feelings, proposal ideas |
| 7 | 이별/재회 | 이별 극복, 재회 방법, 헤어진 후 연락 | getting over breakup, how to get ex back |
| 8 | 장거리 연애 | 장거리 연애 tips, 원거리 연애 유지법 | long distance relationship tips |
| 9 | 소개팅/소셜 | 소개팅 tips, 앱 연애, 만남 방법 | dating app tips, first date advice |
| 10 | 연애 상담 | 연애 상담, 사연, 고민 상담 | relationship advice, dating stories |

## 3. 수집 전략

### 소스 우선순위

| 순위 | 플랫폼 | 방법 | 비고 |
|------|--------|------|------|
| 1 | YouTube | `yt-dlp "ytsearch50:키워드"` | 검색 API 무료, 메타데이터 풍부 |
| 2 | TikTok | `yt-dlp "tiktoksearch:키워드"` | 짧은 영상, 바이럴 콘텐츠 |
| 3 | Instagram | 수동 URL 수집 | 해시태그 기반, 로그인 필요 |

### 필터링 기준

```
최소 조회수: 100,000
최소 채널 구독자: 10,000 (선택)
중복 제거: video_id 기반
언어: 한국어 우선, 영어 포함
```

### 수집 파이프라인

```
1단계: 키워드별 YouTube 검색 (yt-dlp ytsearch)
2단계: 메타데이터 수집 (제목, 조회수, 채널, 업로드일)
3단계: 필터링 (조회수 10만+, 중복 제거)
4단계: 카테고리 자동 분류 (제목/설명 키워드 기반)
5단계: yt-dlp로 자막 추출 + MD 변환
6단계: 데이터베이스 인덱스 생성
```

## 4. MD 파일 구조

```markdown
---
id: "dQw4w9WgXcQ"
title: "연애 상담 모음집"
channel: "찰스엔터"
url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
platform: "youtube"
views: 2367475
likes: null
duration: 1800
uploaded: "2024-03-15"
collected: "2026-07-12"
category: "연애 상담"
tags: ["연애상담", "사연", "고민"]
language: "ko"
---

# 연애 상담 모음집

**채널:** 찰스엔터 | **조회수:** 2,367,475 | **길이:** 30:00

## Transcript

[00:00] 안녕하세요 오늘은 연애 상담을...
[00:15] 첫 번째 사연입니다...
```

## 5. 데이터베이스 구조

```
dating-db/
├── index.json              # 전체 인덱스 (500개)
├── categories.json         # 카테고리별 분류
├── channels.json           # 채널별 그룹핑
├── videos/
│   ├── youtube/
│   │   ├── dQw4w9WgXcQ.md
│   │   └── ...
│   ├── tiktok/
│   │   └── ...
│   └── instagram/
│       └── ...
└── stats.json              # 수집 통계
```

### index.json 구조

```json
{
  "version": "1.0",
  "collected": "2026-07-12",
  "total": 500,
  "categories": {
    "대화법": 50,
    "MBTI 연애": 50,
    "남자 심리": 50,
    ...
  },
  "videos": [
    {
      "id": "dQw4w9WgXcQ",
      "title": "연애 상담 모음집",
      "channel": "찰스엔터",
      "platform": "youtube",
      "category": "연애 상담",
      "views": 2367475,
      "mdFile": "videos/youtube/dQw4w9WgXcQ.md"
    }
  ]
}
```

## 6. 구현 계획

### Phase 1: 수집 스크립트
- `scripts/collect-dating-videos.mjs` — YouTube 검색 + 필터링
- 키워드별 50개씩 검색 → 500개+ 확보
- 카테고리 자동 분류

### Phase 2: 다운로드 + MD 변환
- `scripts/collection/download-and-convert.mjs` — yt-dlp 기반 자막 추출
- 병렬 다운로드 (concurrency: 3)
- MD 파일 생성 + 인덱스 업데이트

### Phase 3: 데이터베이스 구축
- 인덱스 JSON 생성
- 카테고리/채널별 그룹핑
- 통계 리포트

## 7. 예상 규모

| 항목 | 수치 |
|------|------|
| 카테고리 | 10개 |
| 키워드/카테고리 | 5-10개 |
| 검색 결과/키워드 | 50개 |
| 총 검색 결과 | 2,500-5,000개 |
| 필터 후 (10만+) | ~800-1,500개 |
| 최종 선정 | 500개 |
| 예상 용량 | ~5-10GB (영상) + ~50MB (MD) |
