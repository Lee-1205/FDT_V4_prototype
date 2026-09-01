from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    pad_token_id: int = 50256
    eos_token_id: int = 50256

    model_type: str = "fdt"
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: int = 4
    max_seq_len: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = True
    use_rope: bool = True
    rope_transition_alpha: float = 1.0
    rope_transition_mode: str = "lerp"  # "lerp", "phase", or "output_blend"
    legacy_position_transition_max_len: int = 0
    # Negative preserves the legacy coupling: scale = 1 - RoPE alpha.
    legacy_position_scale: float = -1.0

    # FDT anchor routing. These fields are ignored by the plain Transformer.
    use_self_attention: bool = False
    num_anchors: int = 128
    top_k: int = 8
    routing_type: str = "cosine"  # "cosine" or "gaussian"
    cosine_temperature: float = 0.25
    sigma_min: float = 0.10
    sigma_max: float = 0.70
    membership_logit_clip: float = 30.0
    aggregation_impl: str = "dense_gather_first"
    routing_backend: str = "matmul"  # "cdist" or "matmul"
    routing_logit_quantization: float = 0.0
    routing_boundary_smoothing_epsilon: float = 0.0
    routing_boundary_extra_candidates: int = 0
    routing_membership_quantization: float = 0.0
    enable_anchor_metrics: bool = True
    anchor_metrics_interval: int = 1
    compute_anchor_metrics_every: int = 1
    enable_diagnostics: bool = True
    use_cached_normalized_anchors: bool = False
    use_incremental_anchor_state: bool = True
    detach_diagnostics: bool = True
    anchor_layer_pattern: str = "alternate"
    anchor_layer_indices: list[int] = field(default_factory=list)
    use_local_mixer: bool = False
    local_mixer_kernel_size: int = 5
    local_attention_window: int = 64
    inference_prefix_stable_group_size: int = 0
    router_dim: int = 128
    anchor_scan_chunk_size: int = 64
    anchor_recency_bias: float = 4.0
    anchor_recency_reference_len: int = 0
    anchor_decode_state_fp32: bool = False
    diversity_margin: float = 0.20
    diversity_max_weight: float = 2.0

    lm_logit_clip: float = 30.0
    diffusion_noise_scale: float = 0.10

    # Optional lossless token-memory/copy path used by fdt_v3_dual_memory.
    exact_pointer_dim: int = 64
    exact_pointer_window: int = 64
    exact_pointer_loss_weight: float = 0.05
    exact_pointer_gate_weight: float = 0.25
    exact_pointer_chunk_size: int = 32
    exact_pointer_chunk_anchors: int = 4
    exact_pointer_candidate_chunks: int = 4
    exact_pointer_anchor_bias_init: float = 2.0

    # FDT v4: RoPE local working memory plus optional exact episodic memory.
    exact_memory_enabled: bool = False
    exact_memory_mode: str = "off"  # off, store, retrieve, or copy
    exact_memory_full_scan_fallback: bool = True
    exact_memory_candidate_cap: int = 16
    exact_memory_fallback_margin: float = 0.0
    exact_memory_copy_cursor: bool = True
    exact_memory_commit_threshold: float = 0.5
    exact_memory_hard_copy: bool = False
    exact_memory_hard_copy_gate_threshold: float = 0.9
    exact_memory_hard_copy_pointer_threshold: float = 0.9
    exact_memory_hard_copy_margin_threshold: float = 1.0
    generation_repetition_scope: str = "generated"
    generation_ngram_order: int = 3
    generation_ngram_penalty: float = 8.0
    generation_ngram_window: int = 96
    generation_ngram_hard_block_after: int = 2
    # Optional model-native correction head for self-generated loop states.
    # Rank zero preserves every pre-v4.1 checkpoint exactly.
    loop_controller_rank: int = 0
    loop_controller_scale: float = 1.0


@dataclass
class TrainConfig:
    tokenizer_name: str = "gpt2"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = ""
    split: str = "train"
    text_field: str = "text"
    streaming: bool = True
    text_file: str = ""

    output_dir: str = "runs/micro_fdt"
    batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    log_every: int = 20
    save_every: int = 500
    seed: int = 42
    precision: str = "fp32"  # "fp32", "fp16", or "bf16"

    lm_weight: float = 1.0
    noise_weight: float = 0.05
    cluster_weight: float = 0.01
    diversity_weight: float = 0.10
    diversity_every: int = 4
    usage_weight: float = 0.005
    entropy_weight: float = 0.003
    entropy_target_ratio_start: float = 0.65
    entropy_target_ratio_end: float = 0.45
    entropy_over_weight: float = 0.20


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)


def load_yaml_like(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
        return data or {}
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {path} as YAML. Install pyyaml or pass CLI args directly."
        ) from exc
