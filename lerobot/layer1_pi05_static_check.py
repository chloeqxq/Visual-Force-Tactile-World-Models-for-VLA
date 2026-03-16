#!/usr/bin/env python

"""
Layer 1 static pipeline check for Pi0.5 on LIBERO.
python /workspace/lerobot/layer1_pi05_static_check.py

This script is intentionally educational:
1. Load one LIBERO sample and print its structure.
2. Reconstruct the official PI05 "state -> prompt text" step.
3. Run the policy with mocked language tokens to validate the model-side forward path.

Why mocked tokens?
The official PI05 preprocessor uses the PaliGemma tokenizer from
`google/paligemma-3b-pt-224`. If that gated repo is not accessible, the
tokenization stage cannot run. This script still lets you understand the
pipeline and verify that the checkpoint + CUDA path are alive.
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def summarize_sample(sample: dict) -> None:
    print("\n[1] Dataset sample structure")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:<38} shape={list(value.shape)!s:<18} dtype={str(value.dtype):<15}")
        else:
            print(f"  {key:<38} type={type(value).__name__} value={value}")


def normalize_state_like_pi05(raw_state: torch.Tensor, dataset: LeRobotDataset) -> torch.Tensor:
    """
    Mirror the official QUANTILES normalization used before state discretization.

    See:
    - `lerobot/policies/pi05/configuration_pi05.py`
    - `lerobot/processor/normalize_processor.py`
    """
    stats = dataset.meta.stats.get(OBS_STATE, {})
    q01 = stats.get("q01")
    q99 = stats.get("q99")
    if q01 is None or q99 is None:
        raise ValueError("Missing q01/q99 stats for observation.state in dataset metadata.")

    q01_t = torch.as_tensor(q01, dtype=torch.float32)
    q99_t = torch.as_tensor(q99, dtype=torch.float32)
    state_t = raw_state.to(torch.float32)

    denom = q99_t - q01_t
    denom = torch.where(denom == 0, torch.tensor(1e-8, dtype=state_t.dtype), denom)
    return 2.0 * (state_t - q01_t) / denom - 1.0


def build_prompt_like_official_pi05(sample: dict, dataset: LeRobotDataset) -> str:
    """
    Reconstruct the text prompt produced by the official PI05 preprocessing chain.

    This mirrors `Pi05PrepareStateTokenizerProcessorStep` before the tokenizer runs.
    """
    if OBS_STATE not in sample:
        raise ValueError(f"Sample is missing `{OBS_STATE}`.")
    if "task" not in sample:
        raise ValueError("Sample is missing `task`.")

    normalized_state = normalize_state_like_pi05(sample[OBS_STATE], dataset)
    state_np = normalized_state.cpu().numpy()
    discretized = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

    cleaned_task = sample["task"].strip().replace("_", " ").replace("\n", " ")
    state_str = " ".join(map(str, discretized.tolist()))
    return f"Task: {cleaned_task}, State: {state_str};\nAction: "


def build_model_batch(sample: dict, policy: PI05Policy, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Build the batch passed directly into `PI05Policy.select_action`.

    Important:
    - Visual tensors are taken from the dataset sample and batched.
    - Official PI05 would produce language tokens from `task + normalized state`.
    - Here we inject empty tokens so the model-side path can still run.
    """
    batch: dict[str, torch.Tensor] = {}

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)

    max_len = policy.config.tokenizer_max_length
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros((1, max_len), dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.zeros((1, max_len), dtype=torch.bool, device=device)
    return batch


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "lerobot/pi05_libero_base"
    dataset_id = "lerobot/libero"

    print("Layer 1: PI05 static pipeline check")
    print(f"Device: {device}")
    print(f"Model:  {model_id}")
    print(f"Data:   {dataset_id}")

    print("\n[0] Loading policy")
    policy = PI05Policy.from_pretrained(model_id).to(device)
    policy.eval()
    print(f"  policy type: {policy.name}")
    print(f"  chunk_size: {policy.config.chunk_size}")
    print(f"  n_action_steps: {policy.config.n_action_steps}")
    print(f"  tokenizer_max_length: {policy.config.tokenizer_max_length}")
    print(f"  expected image keys: {list(policy.config.image_features.keys())}")

    print("\n[0b] Loading dataset sample")
    dataset = LeRobotDataset(dataset_id)
    start_idx = dataset.meta.episodes["dataset_from_index"][0]
    sample = dataset[start_idx]
    summarize_sample(sample)

    print("\n[2] Official PI05 pre-tokenizer prompt")
    prompt = build_prompt_like_official_pi05(sample, dataset)
    print("  The official pipeline does NOT feed raw state directly into PI05Policy.")
    print("  It first normalizes state, discretizes it into 256 bins, and merges it with task text.")
    print("  Prompt preview:")
    print("  ----------------------------------------")
    print(prompt[:500])
    if len(prompt) > 500:
        print("  ...")
    print("  ----------------------------------------")

    print("\n[3] Model-side inference with mocked language tokens")
    print("  This validates: image preprocessing -> policy forward -> action output.")
    print("  This does NOT validate the gated tokenizer download path.")
    batch = build_model_batch(sample, policy, device)

    with torch.inference_mode():
        action = policy.select_action(batch)

    print("\n[4] Result")
    print(f"  action shape: {list(action.shape)}")
    print(f"  first action vector: {action[0].detach().cpu().numpy()}")

    print("\nDone.")
    print("Read this as two linked halves:")
    print("  dataset sample -> state/task -> prompt text  (official preprocessing half)")
    print("  prompt tokens + image -> policy.select_action (model half)")


if __name__ == "__main__":
    main()
