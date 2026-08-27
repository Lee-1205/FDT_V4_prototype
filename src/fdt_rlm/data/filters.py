from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple


SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}"),
]


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    rejected: int = 0
    reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_kept: int = 0
    tokens_rejected: int = 0

    def reject(self, reason: str, tokens: int = 0) -> None:
        self.rejected += 1
        self.reasons[reason] += 1
        self.tokens_rejected += int(tokens)

    def accept(self, tokens: int = 0) -> None:
        self.kept += 1
        self.tokens_kept += int(tokens)


def normalize_web_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"[ \t]{3,}", " ", text)
    return text.strip()


def normalize_code_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip("\n")


def normalize_markdown(text: str) -> str:
    return normalize_code_text(text)


def repetition_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    counts = Counter(lines)
    return max(counts.values()) / max(len(lines), 1)


def alpha_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(ch.isalpha() for ch in chars) / len(chars)


def looks_minified(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) <= 2 and len(text) > 2000:
        return True
    long_lines = sum(1 for line in lines if len(line) > 500)
    return bool(lines) and long_lines / len(lines) > 0.25


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def filter_web_text(text: str, token_count: int, cfg: Dict) -> Tuple[bool, str]:
    if not text:
        return False, "empty_after_normalization"
    if len(text) < int(cfg.get("min_chars", 200)):
        return False, "too_short"
    if token_count < int(cfg.get("min_tokens", 32)):
        return False, "too_few_tokens"
    if token_count > int(cfg.get("max_tokens", 200000)):
        return False, "too_many_tokens"
    if repetition_ratio(text) > float(cfg.get("max_repetition_ratio", 0.35)):
        return False, "too_repetitive"
    if alpha_ratio(text) < float(cfg.get("min_alpha_ratio", 0.45)):
        return False, "low_alpha_ratio"
    return True, "ok"


def filter_code_text(text: str, token_count: int, cfg: Dict, path: str = "") -> Tuple[bool, str]:
    low_path = (path or "").lower()
    if not text:
        return False, "empty_after_normalization"
    if len(text) < int(cfg.get("min_chars", 20)):
        return False, "too_short"
    if token_count < int(cfg.get("min_tokens", 8)):
        return False, "too_few_tokens"
    if token_count > int(cfg.get("max_tokens", 200000)):
        return False, "too_many_tokens"
    if cfg.get("reject_vendor", True) and ("/vendor/" in low_path or "\\vendor\\" in low_path or "node_modules" in low_path):
        return False, "vendor"
    if cfg.get("reject_generated", True) and ("generated" in low_path or low_path.endswith((".lock", "package-lock.json"))):
        return False, "generated"
    if cfg.get("reject_minified", True) and looks_minified(text):
        return False, "minified"
    if cfg.get("reject_secrets", True) and contains_secret(text):
        return False, "possible_secret"
    return True, "ok"

