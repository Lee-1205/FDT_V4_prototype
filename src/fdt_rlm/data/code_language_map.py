from __future__ import annotations

from pathlib import Path
from typing import Optional


LANGUAGE_GROUPS = {
    "python": {"python", "py"},
    "javascript_typescript": {"javascript", "typescript", "jsx", "tsx", "js", "ts"},
    "c_cpp": {"c", "c++", "cpp", "c header", "c++ header", "h", "hpp", "cc", "cxx"},
    "java": {"java"},
    "rust_go": {"rust", "go", "rs"},
    "structured_docs": {"shell", "bash", "sh", "json", "yaml", "yml", "markdown", "md"},
}

EXTENSION_TO_GROUP = {
    ".py": "python",
    ".js": "javascript_typescript",
    ".jsx": "javascript_typescript",
    ".ts": "javascript_typescript",
    ".tsx": "javascript_typescript",
    ".c": "c_cpp",
    ".cc": "c_cpp",
    ".cpp": "c_cpp",
    ".cxx": "c_cpp",
    ".h": "c_cpp",
    ".hpp": "c_cpp",
    ".java": "java",
    ".rs": "rust_go",
    ".go": "rust_go",
    ".sh": "structured_docs",
    ".bash": "structured_docs",
    ".json": "structured_docs",
    ".yaml": "structured_docs",
    ".yml": "structured_docs",
    ".md": "structured_docs",
}


def map_language(language: Optional[str], path: str = "") -> Optional[str]:
    label = (language or "").strip().lower()
    for group, names in LANGUAGE_GROUPS.items():
        if label in names:
            return group
    suffix = Path(path or "").suffix.lower()
    return EXTENSION_TO_GROUP.get(suffix)

