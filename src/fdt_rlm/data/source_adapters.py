from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .code_language_map import EXTENSION_TO_GROUP, map_language


@dataclass
class Document:
    text: str
    source: str
    bucket: str
    doc_id: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProbeResult:
    name: str
    ok: bool
    fields: List[str] = field(default_factory=list)
    content_field: str = ""
    configs: List[str] = field(default_factory=list)
    warning: str = ""
    sample: Dict = field(default_factory=dict)


def likely_content_field(fields: Iterable[str]) -> str:
    preferred = ["text", "content", "code", "markdown", "body"]
    fields = list(fields)
    lowered = {field.lower(): field for field in fields}
    for key in preferred:
        if key in lowered:
            return lowered[key]
    return ""


def probe_hf_dataset(
    dataset: str,
    config: str = "",
    split: str = "train",
    streaming: bool = True,
    rows: int = 3,
) -> ProbeResult:
    from datasets import get_dataset_config_names, load_dataset

    configs: List[str] = []
    try:
        configs = list(get_dataset_config_names(dataset))
    except Exception:
        configs = []

    try:
        kwargs = {"split": split, "streaming": streaming}
        ds = load_dataset(dataset, config, **kwargs) if config else load_dataset(dataset, **kwargs)
        it = iter(ds)
        sample = next(it)
        fields = list(sample.keys())
        content = likely_content_field(fields)
        warning = ""
        if dataset == "bigcode/the-stack-v2" and "blob_id" in fields and not content:
            warning = "metadata_only_or_external_content_required"
        elif dataset == "bigcode/the-stack-v2" and content != "content":
            warning = "check_swh_aws_content_access"
        return ProbeResult(
            name=dataset,
            ok=True,
            fields=fields,
            content_field=content,
            configs=configs[:50],
            warning=warning,
            sample={key: str(value)[:200] for key, value in sample.items()},
        )
    except Exception as exc:
        return ProbeResult(name=dataset, ok=False, configs=configs[:50], warning=type(exc).__name__ + ": " + str(exc)[:300])


def stream_hf_documents(
    dataset: str,
    source_name: str,
    bucket: str,
    config: str = "",
    split: str = "train",
    content_field: str = "",
    streaming: bool = True,
    language_group: Optional[str] = None,
) -> Iterator[Document]:
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": streaming}
    ds = load_dataset(dataset, config, **kwargs) if config else load_dataset(dataset, **kwargs)
    for idx, row in enumerate(ds):
        field = content_field or likely_content_field(row.keys())
        text = row.get(field, "") if field else ""
        if not isinstance(text, str) or not text.strip():
            continue
        raw_lang = row.get("language") or row.get("lang")
        group = map_language(str(raw_lang), str(row.get("path", ""))) if language_group is None else language_group
        final_bucket = bucket if not bucket.startswith("code") else f"code_{group}" if group else ""
        if bucket.startswith("code") and not final_bucket:
            continue
        doc_id = str(row.get("id") or row.get("content_id") or row.get("blob_id") or row.get("url") or f"{dataset}:{idx}")
        yield Document(text=text, source=source_name, bucket=final_bucket, doc_id=doc_id, metadata=dict(row))


STACK_SMOL_XL_FILES = {
    "python": ["data/python/data.json"],
    "javascript_typescript": ["data/javascript/data.json", "data/typescript/data.json"],
    "c_cpp": ["data/c/data.json", "data/c++/data.json"],
    "java": ["data/java/data.json"],
    "rust_go": ["data/rust/data.json", "data/go/data.json"],
    "structured_docs": ["data/shell/data.json", "data/markdown/data.json", "data/rmarkdown/data.json", "data/powershell/data.json"],
}


def stream_stack_smol_xl_group(group: str) -> Iterator[Document]:
    from datasets import load_dataset

    for file_path in STACK_SMOL_XL_FILES.get(group, []):
        data_file = f"hf://datasets/bigcode/the-stack-smol-xl/{file_path}"
        ds = load_dataset("json", data_files=data_file, split="train", streaming=True)
        for idx, row in enumerate(ds):
            text = row.get("content", "")
            if not isinstance(text, str) or not text.strip():
                continue
            path = row.get("max_stars_repo_path") or row.get("max_issues_repo_path") or row.get("max_forks_repo_path") or file_path
            repo = row.get("max_stars_repo_name") or row.get("max_issues_repo_name") or row.get("max_forks_repo_name") or ""
            doc_id = str(row.get("hexsha") or f"{file_path}:{idx}")
            yield Document(
                text=text,
                source="bigcode/the-stack-smol-xl",
                bucket=f"code_{group}",
                doc_id=doc_id,
                metadata={
                    "path": path,
                    "repo_name": repo,
                    "language": row.get("lang"),
                    "licenses": row.get("max_stars_repo_licenses") or row.get("max_issues_repo_licenses") or row.get("max_forks_repo_licenses"),
                    "hexsha": row.get("hexsha"),
                    "source_file": file_path,
                },
            )


def stream_local_code(local_dir: str | Path) -> Iterator[Document]:
    root = Path(local_dir)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSION_TO_GROUP:
            continue
        group = EXTENSION_TO_GROUP[path.suffix.lower()]
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="latin-1")
            except Exception:
                continue
        rel = str(path.relative_to(root)).replace(os.sep, "/")
        yield Document(
            text=text,
            source="local_code",
            bucket=f"code_{group}",
            doc_id=rel,
            metadata={"path": rel, "language_group": group},
        )


def stream_local_text_file(path: str | Path, bucket: str, source_name: str) -> Iterator[Document]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            text = line.strip()
            if text:
                yield Document(text=text, source=source_name, bucket=bucket, doc_id=f"{path.name}:{idx}", metadata={"path": str(path)})


def stream_local_tar_texts(path: str | Path, bucket: str, source_name: str) -> Iterator[Document]:
    """Stream UTF-8 documents from a directory of TAR archives.

    The legacy OpenWebText download uses ``.txt`` filenames for TAR files, so
    detection is content-based through ``tarfile`` rather than by extension.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Local TAR text directory not found: {root}")
    for archive_path in sorted(item for item in root.iterdir() if item.is_file()):
        if not tarfile.is_tarfile(archive_path):
            continue
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                raw = extracted.read()
                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                yield Document(
                    text=text,
                    source=source_name,
                    bucket=bucket,
                    doc_id=f"{archive_path.name}:{member.name}",
                    metadata={"archive": str(archive_path), "path": member.name},
                )
