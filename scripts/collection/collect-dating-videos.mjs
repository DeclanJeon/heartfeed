#!/usr/bin/env node

/**
 * 연애 콘텐츠 YouTube 검색 + 메타데이터 수집 스크립트
 *
 * Usage: node scripts/collection/collect-dating-videos.mjs [--limit 500] [--min-views 100000] [--output ./data/source]
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

// ---------------------------------------------------------------------------
// 설정
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
const getArg = (name, def) => {
  const idx = args.indexOf(`--${name}`);
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : def;
};

const LIMIT = parseInt(getArg('limit', '500'), 10);
const MIN_VIEWS = parseInt(getArg('min-views', '100000'), 10);
const OUTPUT_DIR = path.resolve(getArg('output', './data/source'));
const SEARCH_PER_KEYWORD = parseInt(getArg('search-count', '50'), 10);
const EXCLUDE_FILE = getArg('exclude', null);

// Load existing IDs to exclude
const EXCLUDE_IDS = new Set();
if (EXCLUDE_FILE && fs.existsSync(EXCLUDE_FILE)) {
  const excluded = JSON.parse(fs.readFileSync(EXCLUDE_FILE, 'utf8'));
  excluded.forEach((id) => EXCLUDE_IDS.add(id));
}

// ---------------------------------------------------------------------------
// 카테고리 + 검색 키워드
// ---------------------------------------------------------------------------

const CATEGORIES = [
  {
    id: 'conversation',
    name: '대화법',
    keywords: [
      '연애 대화법', '설레는 대화 기술', '썸 대화법', '카톡 대화법',
      '연애 잘하는 법 대화', 'dating conversation tips', 'how to flirt texting',
      '대화 잘하는 법', '말 잘하는 법 연애', '연애 말빨', '썸카톡',
      'flirting techniques', 'conversation starters dating',
    ],
  },
  {
    id: 'mbti',
    name: 'MBTI 연애',
    keywords: [
      'MBTI 연애 유형', 'MBTI 궁합 연애', 'MBTI 썸', 'MBTI 연애 상담',
      'MBTI dating compatibility', 'MBTI love types',
      'MBTI 연애 스타일', 'MBTI별 연애 특징', 'MBTI 이별', 'MBTI 소개팅',
      'INFJ 연애', 'ENFP 연애', 'INTJ 연애', 'INFP 연애',
    ],
  },
  {
    id: 'male-psychology',
    name: '남자 심리',
    keywords: [
      '남자 심리 연애', '남자 호감 신호', '남자 꼬시는법', '남자 마음 읽기',
      '남자가 좋아하는 여자', 'how to attract a guy', 'male psychology dating',
      '남자 행동 심리', '남자 카톡 심리', '남자가 좋아할때', '남자 시그널',
      '남자 여자 심리 차이', '남자가 이별 후 느끼는 감정',
      'guy signals he likes you', 'male body language attraction',
    ],
  },
  {
    id: 'female-psychology',
    name: '여자 심리',
    keywords: [
      '여자 심리 연애', '여자 호감 신호', '여자 꼬시는법', '여자 마음 읽기',
      '여자가 좋아하는 남자', 'how to attract a girl', 'female psychology dating',
      '여자 행동 심리', '여자 카톡 심리', '여자가 좋아할때', '여자 시그널',
      '여자가 남자 좋아할때', '여자 관심 표현',
      'signs she likes you', 'female body language attraction',
    ],
  },
  {
    id: 'donts',
    name: '연애 금지사항',
    keywords: [
      '연애 하면 안되는 것', '헤어지는 이유', '연애 실수', '이별 원인',
      '연애 red flag', 'dating mistakes to avoid', 'relationship red flags',
      '연애 최악의 행동', '이별 부르는 습관', '연애 실패 원인',
      'toxic relationship signs', 'relationship deal breakers',
    ],
  },
  {
    id: 'confession',
    name: '고백/프로포즈',
    keywords: [
      '고백 방법', '프로포즈 아이디어', '썸에서 연애로', '고백 성공',
      'how to confess feelings', 'proposal ideas',
      '고백 멘트', '고백 타이밍', '고백 거절', '고백 후기',
      'romantic proposal', 'confession stories',
    ],
  },
  {
    id: 'breakup',
    name: '이별/재회',
    keywords: [
      '이별 극복', '재회 방법', '헤어진 후 연락', '이별 후 마음',
      'getting over breakup', 'how to get ex back',
      '이별 후 남자 심리', '이별 후 여자 심리', '재회 성공', '이별 통보',
      'no contact rule', 'ex comes back', 'breakup recovery',
    ],
  },
  {
    id: 'long-distance',
    name: '장거리 연애',
    keywords: [
      '장거리 연애', '원거리 연애 유지', '장거리 연애 tips',
      'long distance relationship tips',
      '장거리 연애 이별', '장거리 연애 유지법', '장거리 커플',
      'ldr relationship', 'long distance love',
    ],
  },
  {
    id: 'dating-app',
    name: '소개팅/앱 연애',
    keywords: [
      '소개팅 tips', '소개팅 성공', '앱 연애', '데이트 앱',
      'dating app tips', 'first date advice',
      '소개팅 실패', '소개팅 대화', '틴더 사용법', '데이트 앱 추천',
      'hinge tips', 'bumble tips', 'tinder tips',
    ],
  },
  {
    id: 'counseling',
    name: '연애 상담',
    keywords: [
      '연애 상담', '연애 사연', '연애 고민 상담', '연애 상담 모음',
      'relationship advice', 'dating stories',
      '연애 고민', '커플 상담', '연애 썰', '연애 참견',
      'couple counseling', 'relationship problems',
    ],
  },
];

// ---------------------------------------------------------------------------
// 유틸리티
// ---------------------------------------------------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function ytSearch(keyword, count) {
  const cmd = `yt-dlp "ytsearch${count}:${keyword}" --flat-playlist --dump-json --no-warnings 2>/dev/null`;
  try {
    const stdout = execSync(cmd, { encoding: 'utf8', timeout: 30_000 });
    return stdout
      .split('\n')
      .filter(Boolean)
      .map((line) => {
        try { return JSON.parse(line); } catch { return null; }
      })
      .filter(Boolean);
  } catch {
    return [];
  }
}

function classifyCategory(title, description = '') {
  const text = `${title} ${description}`.toLowerCase();
  const rules = [
    [/mbti|엠비티아이/i, 'mbti'],
    [/남자|남성|그놈|그의|male|guy|boyfriend/i, 'male-psychology'],
    [/여자|여성|그녀|그의 여자|female|girl|girlfriend/i, 'female-psychology'],
    [/대화|카톡|톡|연락|conversation|texting|flirt/i, 'conversation'],
    [/고백|프로포즈|고백법|confess|proposal/i, 'confession'],
    [/이별|헤어|재회|이별 후|breakup|ex back/i, 'breakup'],
    [/장거리|원거리|long.?distance/i, 'long-distance'],
    [/소개팅|앱|데이트|소셜|dating app|first date/i, 'dating-app'],
    [/상담|사연|고민|counseling|advice|story/i, 'counseling'],
    [/하면 안|금지|실수|red flag|mistake|dont/i, 'donts'],
  ];
  for (const [re, cat] of rules) {
    if (re.test(text)) return cat;
  }
  return 'counseling'; // 기본값
}

function safeFilename(title) {
  return title
    .replace(/[\r\n]+/g, ' ')
    .replace(/[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s._-]/g, '')
    .slice(0, 80)
    .trim() || 'untitled';
}

// ---------------------------------------------------------------------------
// 메인 수집 로직
// ---------------------------------------------------------------------------

async function collect() {
  console.log(`\nDatewise 연애 콘텐츠 수집기`);
  console.log(`   목표: ${LIMIT}개 | 최소 조회수: ${MIN_VIEWS.toLocaleString()}`);
  console.log(`   출력: ${OUTPUT_DIR}\n`);

  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.mkdirSync(path.join(OUTPUT_DIR, 'videos', 'youtube'), { recursive: true });

  const allVideos = new Map(); // video_id → metadata
  const categoryStats = {};

  for (const cat of CATEGORIES) {
    console.log(`\n📂 [${cat.name}] 검색 중...`);
    categoryStats[cat.id] = { name: cat.name, searched: 0, found: 0 };

    for (const keyword of cat.keywords) {
      const results = ytSearch(keyword, SEARCH_PER_KEYWORD);
      categoryStats[cat.id].searched += results.length;

      for (const r of results) {
        const views = r.view_count ?? r.view_count ?? 0;
        if (views < MIN_VIEWS) continue;
        if (allVideos.has(r.id)) continue;
        if (EXCLUDE_IDS.has(r.id)) continue;

        const detectedCat = classifyCategory(r.title || '', r.description || '');
        allVideos.set(r.id, {
          id: r.id,
          title: r.title || '',
          channel: r.uploader || r.channel || '',
          channel_id: r.channel_id || r.uploader_id || '',
          platform: 'youtube',
          category: detectedCat,
          views,
          likes: r.like_count ?? null,
          duration: r.duration ?? null,
          uploaded: r.upload_date ?? null,
          url: `https://www.youtube.com/watch?v=${r.id}`,
          description: (r.description || '').slice(0, 500),
          thumbnail: r.thumbnail ?? r.thumbnails?.[0]?.url ?? null,
        });
        categoryStats[cat.id].found++;
      }

      process.stdout.write(`  "${keyword}" → ${results.length} results, ${allVideos.size} unique (≥${MIN_VIEWS / 10000}만)\n`);
      await sleep(500); // rate limit
    }
  }

  // 조회수 내림차순 정렬 후 LIMIT개 선정
  const sorted = [...allVideos.values()].sort((a, b) => b.views - a.views);
  const selected = sorted.slice(0, LIMIT);

  console.log(`\n✅ 수집 완료: ${allVideos.size}개 중 ${selected.length}개 선정\n`);

  // 카테고리별 통계
  const finalCatStats = {};
  for (const v of selected) {
    finalCatStats[v.category] = (finalCatStats[v.category] || 0) + 1;
  }

  console.log('📊 카테고리별 분포:');
  for (const [catId, count] of Object.entries(finalCatStats).sort((a, b) => b[1] - a[1])) {
    const catName = CATEGORIES.find((c) => c.id === catId)?.name || catId;
    console.log(`   ${catName}: ${count}개`);
  }

  // 인덱스 저장
  const index = {
    version: '1.0',
    collected: new Date().toISOString().split('T')[0],
    total: selected.length,
    minViews: MIN_VIEWS,
    categories: finalCatStats,
    searchStats: categoryStats,
    videos: selected.map((v) => ({
      id: v.id,
      title: v.title,
      channel: v.channel,
      platform: v.platform,
      category: v.category,
      views: v.views,
      duration: v.duration,
      uploaded: v.uploaded,
      url: v.url,
      mdFile: `videos/youtube/${safeFilename(v.title)}.md`,
    })),
  };

  fs.writeFileSync(
    path.join(OUTPUT_DIR, 'index.json'),
    JSON.stringify(index, null, 2),
  );

  // 채널별 그룹핑
  const channels = {};
  for (const v of selected) {
    if (!channels[v.channel]) channels[v.channel] = [];
    channels[v.channel].push(v.id);
  }
  fs.writeFileSync(
    path.join(OUTPUT_DIR, 'channels.json'),
    JSON.stringify(channels, null, 2),
  );

  // 카테고리별 그룹핑
  const categories = {};
  for (const v of selected) {
    if (!categories[v.category]) categories[v.category] = [];
    categories[v.category].push(v.id);
  }
  fs.writeFileSync(
    path.join(OUTPUT_DIR, 'categories.json'),
    JSON.stringify(categories, null, 2),
  );

  // 전체 메타데이터 저장
  fs.writeFileSync(
    path.join(OUTPUT_DIR, 'metadata.json'),
    JSON.stringify(selected, null, 2),
  );

  console.log(`\n💾 저장 완료:`);
  console.log(`   ${OUTPUT_DIR}/index.json`);
  console.log(`   ${OUTPUT_DIR}/channels.json`);
  console.log(`   ${OUTPUT_DIR}/categories.json`);
  console.log(`   ${OUTPUT_DIR}/metadata.json`);
  console.log(`\n다음 단계: node scripts/download-and-convert.mjs --db ${OUTPUT_DIR}`);
}

collect().catch(console.error);
