#!/usr/bin/env node

/**
 * YouTube 자막 추출 + Markdown 변환 스크립트
 *
 * Usage: node scripts/collection/download-and-convert.mjs --db ./data/source [--concurrency 3] [--skip-download]
 */

import { execFileSync } from 'child_process';
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
const hasFlag = (name) => args.includes(`--${name}`);

const DB_DIR = path.resolve(getArg('db', './data/source'));
const CONCURRENCY = parseInt(getArg('concurrency', '3'), 10);
const SKIP_EXISTING = hasFlag('skip-existing');
const YT_DLP_BIN = getArg('yt-dlp-bin', 'yt-dlp');

// ---------------------------------------------------------------------------
// 유틸리티
// ---------------------------------------------------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function safeFilename(title) {
  return title
    .replace(/[\r\n]+/g, ' ')
    .replace(/[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s._-]/g, '')
    .slice(0, 80)
    .trim() || 'untitled';
}

function formatViews(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function formatDuration(sec) {
  if (!sec) return '??:??';
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

// ---------------------------------------------------------------------------
// MD 생성
// ---------------------------------------------------------------------------

function generateMd(video, transcript = null) {
  const uploaded = video.uploaded
    ? `${video.uploaded.slice(0, 4)}-${video.uploaded.slice(4, 6)}-${video.uploaded.slice(6, 8)}`
    : 'unknown';

  const frontmatter = [
    '---',
    `id: "${video.id}"`,
    `title: "${(video.title || '').replace(/"/g, '\\"')}"`,
    `channel: "${(video.channel || '').replace(/"/g, '\\"')}"`,
    `url: "${video.url}"`,
    `platform: "${video.platform}"`,
    `views: ${video.views}`,
    `likes: ${video.likes ?? 'null'}`,
    `duration: ${video.duration ?? 0}`,
    `uploaded: "${uploaded}"`,
    `collected: "${new Date().toISOString().split('T')[0]}"`,
    `category: "${video.category}"`,
    `language: "${video.url?.includes('youtube.com') ? 'ko' : 'en'}"`,
    '---',
  ].join('\n');

  const header = [
    '',
    `# ${video.title || 'Untitled'}`,
    '',
    `**채널:** ${video.channel || 'Unknown'} | **조회수:** ${formatViews(video.views)} | **길이:** ${formatDuration(video.duration)}`,
    '',
    `**카테고리:** ${video.category} | **업로드:** ${uploaded}`,
    '',
    `> ${video.description?.slice(0, 200) || ''}`,
    '',
  ].join('\n');

  const transcriptSection = transcript
    ? `## Transcript\n\n${transcript}\n`
    : '## Transcript\n\n> 자막을 사용할 수 없습니다.\n';

  return frontmatter + header + transcriptSection;
}

// ---------------------------------------------------------------------------
// YouTube 자막 추출 + Markdown 생성
// ---------------------------------------------------------------------------

function parseVtt(content) {
  const cues = [];
  const cuePattern = /(?:^|\n)\s*(\d{2}:\d{2}(?::\d{2})?\.\d{3})\s*-->\s*\S+[\s\S]*?\n([\s\S]*?)(?=\n\s*\n|$)/g;
  for (const match of content.matchAll(cuePattern)) {
    const timestamp = match[1].split('.')[0];
    const text = match[2]
      .replace(/<[^>]+>/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    if (text) cues.push(`## [${timestamp}]\n${text}`);
  }
  return cues.join('\n\n');
}

function runYtDlp(video, outputDir) {
  try {
    const prefix = path.join(outputDir, video.id);
    execFileSync(
      YT_DLP_BIN,
      [
        '--skip-download',
        '--write-auto-subs',
        '--write-subs',
        '--sub-langs',
        'ko,en',
        '--sub-format',
        'vtt',
        '--output',
        `${prefix}.%(ext)s`,
        video.url,
      ],
      { encoding: 'utf8', timeout: 120_000, stdio: 'pipe' },
    );
    const subtitle = fs.readdirSync(outputDir)
      .filter((name) => name.startsWith(`${video.id}.`) && name.endsWith('.vtt'))
      .sort()[0];
    if (!subtitle) return { success: false, message: 'No subtitles available' };

    const transcript = parseVtt(fs.readFileSync(path.join(outputDir, subtitle), 'utf8'));
    if (!transcript) return { success: false, message: 'Subtitle file was empty' };

    const mdPath = path.join(outputDir, `${safeFilename(video.title)}.md`);
    fs.writeFileSync(mdPath, generateMd(video, transcript), 'utf8');
    return { success: true, filePath: mdPath };
  } catch (error) {
    return { success: false, message: error.message };
  }
}

// ---------------------------------------------------------------------------
// 병렬 실행기
// ---------------------------------------------------------------------------

async function runParallel(items, fn, concurrency) {
  const results = [];
  let idx = 0;

  async function worker() {
    while (idx < items.length) {
      const i = idx++;
      results[i] = await fn(items[i], i);
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()));
  return results;
}

// ---------------------------------------------------------------------------
// 메인 로직
// ---------------------------------------------------------------------------

async function main() {
  const indexPath = path.join(DB_DIR, 'index.json');
  if (!fs.existsSync(indexPath)) {
    console.error(`❌ 인덱스 파일을 찾을 수 없습니다: ${indexPath}`);
    console.error(`   먼저 node scripts/collect-dating-videos.mjs 를 실행하세요.`);
    process.exit(1);
  }

  const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
  console.log(`\nDatewise 연애 콘텐츠 다운로더`);
  console.log(`   DB: ${DB_DIR}`);
  console.log(`   영상: ${index.total}개`);
  console.log(`   동시성: ${CONCURRENCY}`);
  console.log(`   모드: 자막 추출 + MD 변환\n`);

  const mdDir = path.join(DB_DIR, 'videos', 'youtube');
  fs.mkdirSync(mdDir, { recursive: true });

  const stats = { success: 0, failed: 0, skipped: 0 };

  const results = await runParallel(
    index.videos,
    async (video, i) => {
      const filename = safeFilename(video.title);
      const mdPath = path.join(mdDir, `${filename}.md`);

      // 이미 존재하면 스킵
      if (SKIP_EXISTING && fs.existsSync(mdPath)) {
        stats.skipped++;
        process.stdout.write(`  ⏭ [${i + 1}/${index.total}] ${video.title.slice(0, 40)}... (exists)\n`);
        return { ...video, status: 'skipped' };
      }

      process.stdout.write(`  ⬇ [${i + 1}/${index.total}] ${video.title.slice(0, 40)}...\n`);

      // yt-dlp로 자막 추출 + MD 변환
      const result = runYtDlp(video, mdDir);
      if (result.success) {
        stats.success++;
        return { ...video, status: 'complete', mdFile: result.filePath };
      } else {
        // 자막 없으면 메타데이터만으로 MD 생성
        const md = generateMd(video);
        fs.writeFileSync(mdPath, md);
        stats.success++;
        return { ...video, status: 'md-fallback', mdFile: mdPath, error: result.message };
      }

    },
    CONCURRENCY,
  );

  // 통계 저장
  const finalStats = {
    collected: index.collected,
    processed: new Date().toISOString().split('T')[0],
    total: index.total,
    ...stats,
    byCategory: {},
    byChannel: {},
  };

  for (const r of results) {
    if (!r) continue;
    finalStats.byCategory[r.category] = (finalStats.byCategory[r.category] || 0) + 1;
    finalStats.byChannel[r.channel] = (finalStats.byChannel[r.channel] || 0) + 1;
  }

  fs.writeFileSync(
    path.join(DB_DIR, 'stats.json'),
    JSON.stringify(finalStats, null, 2),
  );

  console.log(`\n✅ 완료:`);
  console.log(`   성공: ${stats.success}개`);
  console.log(`   실패: ${stats.failed}개`);
  console.log(`   스킵: ${stats.skipped}개`);
  console.log(`\n💾 ${DB_DIR}/stats.json`);
}

main().catch(console.error);
