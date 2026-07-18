"""Production LLM and product-mode guards for Rescue."""

from __future__ import annotations

import os
import re

_FREE_MARKERS = (":free", "/free", "free/")

_BREAKUP_RE = re.compile(
    r"(?:"
    r"이별|헤어|헤어진|전\s*여친|전\s*남친|전\s*연인|재회|연락\s*참|노컨택|no[\s-]?contact|"
    r"차단|스토리\s*확인|자책|회복|이별\s*후|깨진\s*사이|차였|차였어|바람\s*피|"
    r"breakup|ex[\s-]?boyfriend|ex[\s-]?girlfriend|got\s*dumped|no\s*contact"
    r")",
    re.IGNORECASE,
)

# Non-breakup general dating that Rescue mode should refuse by default
_GENERAL_DATING_OOD_HINT = re.compile(
    r"(?:"
    r"첫\s*데이트|소개팅|썸\s*타|호감\s*표현|고백\s*타이밍|장거리\s*연애\s*팁|"
    r"MBTI\s*궁합|사주\s*궁합|플러팅|첫\s*메시지"
    r")",
    re.IGNORECASE,
)


def is_free_llm_model(model: str | None) -> bool:
    if not model:
        return False
    m = model.strip().lower()
    return any(marker in m for marker in _FREE_MARKERS)


def assert_production_llm_safe(
    *,
    product_mode: str,
    env: str | None,
    provider: str,
    model: str,
    fallback_model: str,
    allow_free_llm: bool,
) -> None:
    """Raise RuntimeError if production would use free LLM paths."""
    env_name = (env or os.getenv("ENV") or os.getenv("APP_ENV") or "dev").lower()
    is_prod = env_name in {"prod", "production"} or product_mode == "rescue_brt14"
    if allow_free_llm and not is_prod:
        return
    if not is_prod and product_mode != "rescue_brt14":
        return
    if allow_free_llm:
        # Explicit override only for non-prod experiments
        if env_name in {"prod", "production"}:
            raise RuntimeError("ALLOW_FREE_LLM cannot be enabled in production")
        return
    bad = []
    if is_free_llm_model(model):
        bad.append(f"primary model looks free-tier: {model}")
    if is_free_llm_model(fallback_model):
        bad.append(f"fallback model looks free-tier: {fallback_model}")
    if provider == "nous" and is_free_llm_model(model):
        bad.append("nous provider with free model is not allowed in rescue/prod")
    if bad:
        raise RuntimeError(
            "Free LLM path forbidden in rescue/production: " + "; ".join(bad)
        )


def is_breakup_related(text: str) -> bool:
    return bool(_BREAKUP_RE.search(text or ""))


def looks_general_dating_not_breakup(text: str) -> bool:
    t = text or ""
    if is_breakup_related(t):
        return False
    return bool(_GENERAL_DATING_OOD_HINT.search(t))
