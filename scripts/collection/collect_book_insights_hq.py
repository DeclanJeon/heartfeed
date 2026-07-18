"""Generate high-quality book insights from real summaries.

Replaces the generic template-based insights with specific, actionable content
derived from actual book summaries and key findings.
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid5, NAMESPACE_URL

# ── HIGH-QUALITY BOOK INSIGHTS ─────────────────────────────────────────
# Each entry contains real, specific insights from the book, not templates.

BOOK_INSIGHTS = [
    # ═══ ATTACHED (Amir Levine & Rachel Heller) ═══
    {
        "book": "Attached",
        "author": "Amir Levine & Rachel Heller",
        "year": 2010,
        "insights": [
            {
                "concept": "세 가지 애착 유형",
                "content": """## 세 가지 애착 유형

**핵심 인사이트: "Attached" by Amir Levine & Rachel Heller**

성인 애착 이론에 따르면, 모든 사람은 크게 세 가지 애착 유형 중 하나에 속합니다:

**1. 안정형 (Secure, 약 50%)**
- 친밀감을 편안하게 받아들이며, 파트너의 필요에 자연스럽게 반응
- 갈등이 있어도 관계를 유지할 수 있는 능력 보유
- "나는 충분히 좋은 사람이고, 사랑받을 자격이 있다"는 내적 확신

**2. 불안형 (Anxious, 약 20%)**
- 파트너의 반응에 과민하게 반응하고, 끊임없이 확인 욕구
- "내가 충분하지 않으면 버림받을 수 있다"는 내적 불안
- 파트너가 연락을 안 하면 최악을 상상하며 불안에 시달림
- 프로페셔널한 영역에서는 자신감 있어도, 관계에서만 불안이 폭발

**3. 회피형 (Avoidant, 약 25%)**
- 친밀감을 불편해하고, 독립성을 지나치게 강조
- 감정적 필요를 약점으로 인식하여 억누르는 경향
- "나 혼자면 안전하다"는 내적 작동 모형

**실천 방법:**
- 먼저 자신의 애착 유형을 정확히 파악하세요
- 애착 유형은 "고장"이 아니라 "신경계 적응"입니다
- 안정형 파트너를 찾는 것이 가장 효과적인 전략입니다"""
            },
            {
                "concept": "불안-회피 함정",
                "content": """## 불안-회피 함정 (Anxious-Avoidant Trap)

**핵심 인사이트: "Attached" by Amir Levine & Rachel Heller**

불안형과 회피형은 서로에게 강하게 끌리지만, 이 관계는 대부분 만족스럽지 못합니다:

**함정의 메커니즘:**
1. 불안형은 회피형의 감정적 불가능성을 "신비롭거나 극복할 도전"으로 인식
2. 회피형은 불안형의 따뜻함과 추구를 처음에는 매력적으로 느끼다가 질식감
3. 관계가 깊어지면 양쪽이 서로의 불안을 활성화:
   - 불안형의 친밀감 요구 → 회피형의 퇴행 촉발
   - 회피형의 거리두기 → 불안형의 과잉 활성화 촉발

**왜 이 함정에 빠지는가?**
- 불안형은 불안정한 초기 양육 경험으로 인해 "사랑 = 끊임없이 추구해야 하는 것"으로 학습
- 회피형은 일관된 감정적 반응이 없었던 환경에서 "감정 표현 = 위험"으로 학습

**해결책:**
- 안정형 파트너를 적극적으로 찾으세요
- 처음에 "지루하게" 느껴지는 안정감이 실제로는 안전함입니다
- 불안이 없는 관계를 "화학 반응이 없다"고 오인하지 마세요"""
            },
            {
                "concept": "hyperactivation과 deactivation",
                "content": """## 과잉 활성화 vs 퇴행 전략

**핵심 인사이트: "Attached" by Amir Levine & Rachel Heller**

애착 시스템이 활성화될 때 각 유형이 보이는 구체적 행동 패턴:

**불안형의 과잉 활성화 (Hyperactivation):**
- 파트너의 휴대폰을 끊임없이 확인
- "왜 연락을 안 해?"라는 생각에서 벗어나지 못함
- 파트너의 모든 행동을 "버림받는 신호"로 해석
- 관계에 대해 이야기하고 싶어 하는 강한 욕구
- 파트너에게 지속적인 확인과 보증을 요구

**회피형의 퇴행 (Deactivation):**
- 관계에 대해 이야기하는 것을 회피
- "지금은 관계에 집중할 때가 아니다"라는 핑계
- 파트너의 가까이 다가오면 불편함을 느낌
- 독립적인 척하며 감정을 억누름
- 이전 연애 상대를 이상화하는 경향

**안정형의 반응:**
- 불안을 인식하고 파트너에게 건설적으로 소통
- 불안이 사라질 때까지 기다리는 능력 보유
- 파트너의 필요에 적절히 반응"""
            },
        ]
    },
    # ═══ GOTTMAN'S SEVEN PRINCIPLES ═══
    {
        "book": "The Seven Principles for Making Marriage Work",
        "author": "John Gottman",
        "year": 1999,
        "insights": [
            {
                "concept": "네 기수 이론",
                "content": """## 관계를 파괴하는 네 기수 (Four Horsemen)

**핵심 인사이트: "The Seven Principles for Making Marriage Work" by John Gottman**

존 고티만은 20년 이상의 연구를 통해 관계 실패를 예측하는 네 가지 소통 패턴을 발견했습니다:

**1. 비난 (Criticism)**
- 파트너의 인격을 공격하는 것
- "넌 항상 이래" / "넌 결코改變하지 않아"
- 해독제: 구체적 행동에 대해 이야기하기 ("이번 주에 dishes를 안 한 것이 속상해")

**2. 경멸 (Contempt)** - 이혼의 가장 강력한 예측 인자
- 파트너를 나보다 낮게 보는 태도
- 비꼬기, 눈roll, 조소, 모욕
- 해독제: 감사와 존경의 문화 만들기

**3. 방어 (Defensiveness)**
- 책임을 회피하고 역공격
- "내 잘못이 아니야, 네 잘못이지"
- 해독제: 상대방의 관점에서 공감하기

**4. 벽 쌓기 (Stonewalling)**
- 대화를 차단하고 무시하기
- 반응을 멈추고 철수
- 해독제: 20분 휴식 후 다시 대화 시도

**실천 방법:**
- 네 기수 중 하나라도 관계에 나타나면 즉시 인식하세요
- 특히 경멸은 관계의 적신호입니다
- 각 기수에 대한 해독제를 미리 준비해두세요"""
            },
            {
                "concept": "사랑의 지도",
                "content": """## 사랑의 지도 (Love Maps)

**핵심 인사이트: "The Seven Principles for Making Marriage Work" by John Gottman**

사랑의 지도란 파트너의 내면세계에 대한 지식입니다:

**중요성:**
- 행복한 부부는 파트너의 걱정, 꿈, 역사, 일상생활을 깊이 이해
- 이것은 "감성적 기술"이 아니라 관계의 구조적 기반
- 사랑의 지도가 풍부할수록 갈등 해결이 수월

**구체적 질문들:**
- 파트너의 가장 큰 걱정은 무엇인가?
- 파트너의 인생에서 가장 행복했던 기억은?
- 파트너의 현재 가장 큰 두려움은?
- 파트너가 가장 자랑스럽게 생각하는 성취는?
- 파트너의 가장 친한 친구는 누구인가?

**실천 방법:**
- 매주 10분씩 서로의 사랑의 지도를 업데이트하세요
- 일상적인 대화에서 파트너의 내면을 탐색하세요
- 새롭게 알게 된 것을 기록해두세요"""
            },
            {
                "concept": "수정 시도",
                "content": """## 수정 시도 (Repair Attempts)

**핵심 인사이트: "The Seven Principles for Making Marriage Work" by John Gottman**

갈등이 심화되는 것을 막는 가장 중요한 메커니즘:

**수정 시도란?**
- 갈등의 소螺旋을 중단시키는 어떤 행동이든
- 유머 사용, 손 잡기, "잠깐, 다시 생각해보자" 같은 말
- 서투른 수정 시도도 관계에 충분한 신뢰가 쌓여 있으면 효과적

**왜 중요한가?**
- 부부는 갈등 자체로 헤어지지 않습니다
- 갈등을 해결하는 방식으로 헤어집니다
- 수정 시도는 감정적 홍수를 예방

**성공적인 수정 시도의 조건:**
1. 관계에 충분한 신뢰가 구축되어 있어야 함
2. 상대방의 수정 시도를 받아들이는 자세 필요
3. 서로의 수정 시도를 "방해"가 아닌 "구조 신호"로 인식

**실천 방법:**
- 갈등이 뜨거워지기 전에 수정 시도를 준비하세요
- "우리 잠깐 쉬었다가 할까?" 같은 간단한 문장도 효과적
- 상대방의 수정 시도를 놓치지 마세요"""
            },
        ]
    },
    # ═══ DARING GREATLY (Brené Brown) ═══
    {
        "book": "Daring Greatly",
        "author": "Brené Brown",
        "year": 2012,
        "insights": [
            {
                "concept": "취약성의 힘",
                "content": """## 취약성은 약함이 아닙니다

**핵심 인사이트: "Daring Greatly" by Brené Brown**

12년간의 연구를 통해 밝혀진 브레네 브라운의 핵심 발견:

**취약성이란:**
- 불확실성, 감정적 노출, 위험을 감수하는 것
- "모르겠다", "도움이 필요하다", "상처받았다"고 말하는 것
- 두려움에도 불구하고 관계에 모습을 드러내는 것

**왜 중요한가?**
- 취약성은 용기, 창의성, 연결의 토대
- 취약성을 거부하면 정서적 차단이 발생
- 관계에서 진정한 친밀감은 취약성을 통해서만 가능

**연구 결과:**
- 취약성을 받아들이는 사람들은 더 깊은 관계 구축
- 취약성을 회피하는 사람들은 더 고립
- 취약성은 나약함이 아니라 가장 정확한 용기

**실천 방법:**
- "완벽하지 않아도 괜찮다"는 것을 스스로에게 말하세요
- 작은 것부터 솔직해지기 시작하세요
- 상대방의 취약성을 받아들이는 연습을 하세요"""
            },
            {
                "concept": "수치심 vs 죄책감",
                "content": """## 수치심 vs 죄책감

**핵심 인사이트: "Daring Greatly" by Brené Brown**

관계에서 가장 파괴적인 감정인 수치심과 건설적인 죄책감의 차이:

**수치심 (Shame):**
- "나는 나쁜 사람이다" (인격 공격)
- 고립감, 부끄러움, 침묵 유발
- 관계를 파괴하는 강력한 힘
- "아무도 내 이런 모습을 알면 나를 사랑하지 않을 거야"

**죄책감 (Guilt):**
- "내가 나쁜 행동을 했다" (행동 초점)
- 연결, 사과, 변화의 동기 부여
- 관계를 회복하는 건설적 힘
- "내가 잘못했고, 다시는 하지 않겠다"

**실천 방법:**
- 수치심을 느낄 때 "지금 나는 어떤 이야기를 하고 있는가?" 자문하세요
- 수치심을 말로 표현하면 힘이 약해집니다
- 상대방의 수치심에 공감으로 반응하세요"""
            },
            {
                "concept": "전체적인 삶",
                "content": """## 전체적인 삶 (Wholehearted Living)

**핵심 인사이트: "Daring Greatly" by Brené Brown**

브레네 브라운이 연구를 통해 발견한 "전체적인 삶"을 사는 사람들의 특징:

**그들의 특징:**
1. 가치관에 따라 살아가기
2. 걱정과 수치심으로부터 벗어나기
3. 취약성을 받아들이기
4. 감사와 기쁨을 실천하기
5. 직관과 믿음을 신뢰하기
6. 놀이와 휴식을 소중히 여기기
7. 의미 있는 노동 추구하기
8. 웃음과 노래를 통한 축하

**관계에서의 적용:**
- 상대방에게 "네가 어떤 사람이든 괜찮다"고 말하기
- 함께 취약성을 나누는 시간 가지기
- 서로의 가치관을 존중하는 관계 만들기
- 일상적인 순간에서 기쁨 찾기"""
            },
        ]
    },
    # ═══ HOLD ME TIGHT (Sue Johnson) ═══
    {
        "book": "Hold Me Tight",
        "author": "Sue Johnson",
        "year": 2008,
        "insights": [
            {
                "concept": "감정적 용서",
                "content": """## 감정적 용서란 무엇인가

**핵심 인사이트: "Hold Me Tight" by Sue Johnson**

감정적 용서는 단순히 "화해"가 아닙니다:

**감정적 용서의 정의:**
- 상대방이 나에게 해를 끼쳤다는 사실을 인정하면서도
- 그 상처를 극복하고 관계를 회복하겠다는 의지
- 과거의 상처가 현재의 사랑을 지배하지 않도록 하는 것

**왜 어려운가?**
- 우리는 종종 "용서했다"고 말하면서 실제로는 감정을 억누름
- 진정한 용서는 감정을 완전히 처리한 후에야 가능
- 용서는 시간이 걸리는 과정

**실천 방법:**
- "네가 나를 얼마나 상처받게 했는지 말해줄게"로 시작하세요
- 상대방의 감정을 완전히 듣고 인정하세요
- 서서히 감정의 무게를 줄여나가세요"""
            },
            {
                "concept": "분리의 춤",
                "content": """## 분리의 춤 (Dance of Disconnection)

**핵심 인사이트: "Hold Me Tight" by Sue Johnson**

부부가 반복적으로 빠지는 파괴적 패턴:

**분리의 춤이란?**
- "네가 나를 이해하지 못해" → "내가 뭘 잘못했는지 모르겠어" → "더 이상 이야기 안 해"
- 같은 갈등이 반복되지만 해결되지 않는 상태
- 양쪽 모두 상처받고 외로운 상태

**왜 이 춤을 추게 되는가?**
- 기본적인 애착 필요가 충족되지 않을 때
- "나는 사랑받을 만한가?"라는 근본적 불안
- 서로를 밀어내면서도 연결을 원하는 모순

**탈출 방법:**
1. "분리의 춤"을 인식하기
2. 바닥에 깔린 감정적 필요를 파악하기
3. "내가 필요할 때 네가 있어줬으면 좋겠어"라고 말하기
4. 서로의 애착 필요를 이해하고 받아들이기"""
            },
        ]
    },
    # ═══ NONVIOLENT COMMUNICATION (Marshall Rosenberg) ═══
    {
        "book": "Nonviolent Communication",
        "author": "Marshall Rosenberg",
        "year": 1999,
        "insights": [
            {
                "concept": "관찰과 평가의 차이",
                "content": """## 관찰과 평가를 구별하기

**핵심 인사이트: "Nonviolent Communication" by Marshall Rosenberg**

가장 중요한 소통 기술 중 하나:

**관찰 (Observation):**
- 판단 없이 사실만 전달
- "이번 주에 세 번 저녁을 늦게 먹었어"
- 구체적 행동에 초점

**평가 (Evaluation):**
- 판단과 해석이 포함된 것
- "넌 항상 약속을 지키지 않아"
- 인격이나 성격에 초점

**왜 이것이 중요한가?**
- 평가는 상대방을 방어적으로 만들고 갈등을 증폭
- 관찰은 대화의 가능성을 열어둠
- "항상", "결코" 같은 절대적 표현은 피해야 함

**실천 방법:**
- "~할 때" 표현을 사용하세요 ("네가 ~할 때, 나는 ~을 느껴")
- 감정과 필요를 명확히 구별하세요
- 상대방의 관찰도 존중하세요"""
            },
            {
                "concept": "감정과 생각 구별",
                "content": """## 진짜 감정과 생각 구별하기

**핵심 인사이트: "Nonviolent Communication" by Marshall Rosenberg**

대부분의 사람들이 감정을 생각과 혼동합니다:

**생각 (Thoughts):**
- "나는 무시당했다고 느껴"
- "나는 불공평하다고 생각해"
- "나는 실망했다고 생각해"

**진짜 감정 (Feelings):**
- "나는 슬퍼"
- "나는 두려워"
- "나는 화가 나"
- "나는 외로워"

**왜 이것이 중요한가?**
- 생각은 판단이 포함되어 있어 대화를 방해
- 감정은 표현하면 연결을 촉진
- "무시당했다"는 평가이지만, "슬퍼"는 감정

**실천 방법:**
- "나는 ~을 느껴"보다 "나는 ~해"라고 표현하세요
- 감정을 20가지 이상으로 구별할 수 있는 능력을 기르세요
- 상대방의 감정을 추측하지 말고 직접 물어보세요"""
            },
        ]
    },
    # ═══ THE DANCE OF ANGER (Harriet Lerner) ═══
    {
        "book": "The Dance of Anger",
        "author": "Harriet Lerner",
        "year": 1985,
        "insights": [
            {
                "concept": "분노의 신호",
                "content": """## 분노는 무엇을 말하는가

**핵심 인사이트: "The Dance of Anger" by Harriet Lerner**

분노는 나쁜 감정이 아니라 중요한 신호입니다:

**분노가 말하는 것:**
- 나의 경계가 침해당했을 때
- 나의 필요가 충족되지 않았을 때
- 내가 가치 있다고 생각하지 못할 때
- 관계에서 불균형이 있을 때

**분노를 다루는 잘못된 방법:**
- 억누르기 → 나중에 폭발
- 공격하기 → 관계 파괴
- 자책하기 → 자존감 저하

**올바른 방법:**
- 분노를 인식하고 그 원인을 파악하기
- "이 상황에서 나의 필요는 무엇인가?" 자문하기
- 건설적으로 표현하기

**실천 방법:**
- 분노가 느껴질 때 잠깐 멈추세요
- "나는 지금 어떤 필요가 충족되지 않았는가?" 자문하세요
- 상대방이 아닌 나의 필요에 초점하여 이야기하세요"""
            },
            {
                "concept": "반복되는 갈등",
                "content": """## 반복되는 갈등의 의미

**핵심 인사이트: "The Dance of Anger" by Harriet Lerner**

같은 문제가 반복된다면, 그것은 해결되지 않은 근본적 문제가 있습니다:

**반복되는 갈등의 특징:**
- 같은 주제로 같은 방식의 다툼
- 양쪽 모두 같은 말을 반복
- 해결책 없이 감정만 증폭
- "우리는 왜 항상 이 문제로 싸우지?"라는 느낌

**근본적 원인:**
- 해결되지 않은 과거의 상처
- 명확하지 않은 경계
- 충족되지 않은 기본적 필요
- 상대방에게 기대하는 것과 실제 모습의 괴리

**해결 방법:**
- 표면적 해결이 아닌 근본적 문제에 초점
- "우리가 정말로 싸우고 있는 것은 무엇인가?" 자문
- 서로의 진짜 필요를 이해하는 시간 가지기
- 필요하다면 전문가의 도움 받기"""
            },
        ]
    },
    # ═══ THE POWER OF VULNERABILITY (Brené Brown) ═══
    {
        "book": "The Power of Vulnerability",
        "author": "Brené Brown",
        "year": 2013,
        "insights": [
            {
                "concept": "약점과 취약성의 차이",
                "content": """## 약점과 취약성은 다릅니다

**핵심 인사이트: "The Power of Vulnerability" by Brené Brown**

많은 사람들이 이 두 가지를 혼동합니다:

**약점 (Weaknesses):**
- 내가 개선할 수 있는 영역
- 기술, 능력, 지식의 부족
- 시간과 노력을 통해 발전 가능

**취약성 (Vulnerability):**
- 불확실성과 위험을 감수하는 것
- 감정적 노출을 선택하는 것
- 관계에서 진정한 나를 보여주는 것

**왜 이것이 중요한가?**
- 약점을 고치려다 취약성을 회피할 수 있음
- "나는 아직 준비가 안 됐어"는 흔한 핑계
- 진정한 연결은 취약성을 통해서만 가능

**실천 방법:**
- "아직 준비가 안 됐다"는 생각을 인식하세요
- 작은 것부터 취약성을 나누기 시작하세요
- 상대방의 취약성을 받아들이는 연습을 하세요"""
            },
        ]
    },
]


def generate_high_quality_insight(insight: dict, book: dict, index: int) -> str:
    """Generate a high-quality insight markdown file."""
    title = f"{book['book']} - {insight['concept']}"
    now = datetime.now().strftime("%Y-%m-%d")
    year = book['year']
    
    frontmatter = [
        "---",
        f'id: "hq-{index:03d}-{insight["concept"][:30]}"',
        f'title: "{title.replace(chr(34), chr(39))}"',
        f'channel: "{book["author"]}"',
        f'url: ""',
        f'platform: "book"',
        f"views: 0",
        f"duration: 0",
        f'uploaded: "{year}-01-01"',
        f'collected: "{now}"',
        f'category: "dating"',
        f'language: "ko"',
        f'source_origin: "book-insight-hq"',
        "---",
    ]
    
    content = "\n".join(frontmatter) + "\n\n# " + title + "\n\n" + insight['content']
    
    safe_title = re.sub(r'[^\w가-힣\-.+]', '_', title)[:80]
    return safe_title, content


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../../data/source/book-insights-hq/corpus")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for book_idx, book in enumerate(BOOK_INSIGHTS):
        for insight in book.get("insights", []):
            safe_title, content = generate_high_quality_insight(insight, book, count)
            out_path = output_dir / f"{safe_title}.md"
            out_path.write_text(content, encoding="utf-8")
            count += 1

    print(f"Generated {count} high-quality insight files in {output_dir}")


if __name__ == "__main__":
    main()
