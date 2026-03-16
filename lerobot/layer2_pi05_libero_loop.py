#!/usr/bin/env python

"""
Layer 2: minimal dynamic loop for PI05 + LIBERO.
python /workspace/lerobot/layer2_pi05_libero_loop.py

Goal:
    env.reset()
    -> raw observation
    -> env preprocessing
    -> build policy batch
    -> policy.select_action()
    -> action unnormalization
    -> env.step(action)
    -> next observation

This script is intentionally educational and prints the dataflow at each stage.

Important note:
The official PI05 preprocessing chain uses a gated PaliGemma tokenizer.
To keep the dynamic loop debuggable even without tokenizer access, this script:
1. reconstructs the official prompt text, but
2. injects mocked language tokens when calling the policy.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def flatten_keys(d: dict, prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.extend(flatten_keys(value, name))
        else:
            keys.append(name)
    return keys


def summarize_tensors(batch: dict, title: str) -> None:
    print(f"\n{title}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:<38} shape={list(value.shape)!s:<18} dtype={str(value.dtype):<15}")
        elif isinstance(value, list):
            if value and isinstance(value[0], str):
                preview = value[0][:100]
                print(f"  {key:<38} list[str] first='{preview}'")
            else:
                print(f"  {key:<38} type=list")
        else:
            print(f"  {key:<38} type={type(value).__name__}")


def normalize_quantiles(x: torch.Tensor, q01, q99) -> torch.Tensor:
    q01_t = torch.as_tensor(q01, dtype=torch.float32, device=x.device)
    q99_t = torch.as_tensor(q99, dtype=torch.float32, device=x.device)
    denom = q99_t - q01_t
    denom = torch.where(denom == 0, torch.tensor(1e-8, dtype=x.dtype, device=x.device), denom)
    return 2.0 * (x - q01_t) / denom - 1.0


def unnormalize_quantiles(x: torch.Tensor, q01, q99) -> torch.Tensor:
    q01_t = torch.as_tensor(q01, dtype=torch.float32, device=x.device)
    q99_t = torch.as_tensor(q99, dtype=torch.float32, device=x.device)
    denom = q99_t - q01_t
    denom = torch.where(denom == 0, torch.tensor(1e-8, dtype=x.dtype, device=x.device), denom)
    return (x + 1.0) * denom / 2.0 + q01_t


def build_prompt_from_processed_obs(processed_obs: dict[str, torch.Tensor | list[str]], dataset_meta) -> str:
    state = processed_obs[OBS_STATE]
    task_list = processed_obs["task"]
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"{OBS_STATE} must be a tensor, got {type(state)}")
    if not isinstance(task_list, list) or not task_list or not isinstance(task_list[0], str):
        raise TypeError("task must be list[str]")

    state0 = state[0].to(torch.float32)
    stats = dataset_meta.stats[OBS_STATE]
    normalized_state = normalize_quantiles(state0, stats["q01"], stats["q99"])
    discretized = np.digitize(
        normalized_state.detach().cpu().numpy(),
        bins=np.linspace(-1, 1, 256 + 1)[:-1],
    ) - 1
    cleaned_task = task_list[0].strip().replace("_", " ").replace("\n", " ")
    state_str = " ".join(map(str, discretized.tolist()))
    return f"Task: {cleaned_task}, State: {state_str};\nAction: "


def add_mock_language_tokens(batch: dict[str, torch.Tensor], policy: PI05Policy, device: torch.device) -> None:
    max_len = policy.config.tokenizer_max_length
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros((1, max_len), dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.zeros((1, max_len), dtype=torch.bool, device=device)


def ensure_libero_config() -> None:
    """
    Create ~/.libero/config.yaml non-interactively if LIBERO has not been initialized yet.
    """
    config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero")))
    config_root.mkdir(parents=True, exist_ok=True)
    config_file = config_root / "config.yaml"
    if config_file.exists():
        return

    spec = importlib.util.find_spec("libero.libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("Could not locate installed `libero.libero` package.")

    benchmark_root = Path(spec.submodule_search_locations[0])
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str((benchmark_root / "../datasets").resolve()),
        "assets": str(benchmark_root / "assets"),
    }
    with config_file.open("w") as f:
        yaml.safe_dump(config, f)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "lerobot/pi05_libero_base"
    dataset_id = "lerobot/libero"

    print("Layer 2: PI05 minimal dynamic loop")
    print(f"Device: {device}")

    print("\n[0] Load policy")
    policy = PI05Policy.from_pretrained(model_id).to(device)
    policy.eval()
    policy.reset()

    print("\n[1] Load dataset metadata for normalization stats")
    dataset_meta = LeRobotDatasetMetadata(dataset_id)

    print("\n[2] Build one LIBERO vector env")
    ensure_libero_config()
    env_cfg = LiberoEnvConfig(
        task="libero_10",
        task_ids=[0],
        control_mode="relative",
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    suite_name = next(iter(envs.keys()))
    task_id = next(iter(envs[suite_name].keys()))
    vec_env = envs[suite_name][task_id]
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    try:
        print(f"  suite={suite_name} task_id={task_id}")

        print("\n[3] env.reset()")
        raw_obs, info = vec_env.reset(seed=0)
        print(f"  reset info: {info}")
        print("  raw env observation keys:")
        for key in flatten_keys(raw_obs):
            print(f"    {key}")

        print("\n[4] preprocess_observation(raw_obs)")
        obs = preprocess_observation(raw_obs)
        summarize_tensors(obs, "  after preprocess_observation")

        print("\n[5] add_envs_task(...)")
        obs = add_envs_task(vec_env, obs)
        summarize_tensors(obs, "  after add_envs_task")

        print("\n[6] env_preprocessor = LiberoProcessorStep")
        obs = env_preprocessor(obs)
        summarize_tensors(obs, "  after LiberoProcessorStep")

        print("\n[7] Reconstruct official PI05 prompt text")
        prompt = build_prompt_from_processed_obs(obs, dataset_meta)
        print("  prompt preview:")
        print("  ----------------------------------------")
        print(prompt[:500])
        if len(prompt) > 500:
            print("  ...")
        print("  ----------------------------------------")

        print("\n[8] Inject mocked language tokens")
        model_batch = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                model_batch[key] = value.to(device)
        add_mock_language_tokens(model_batch, policy, device)
        summarize_tensors(model_batch, "  batch sent into PI05Policy.select_action")

        print("\n[9] policy.select_action(model_batch)")
        with torch.inference_mode():
            action_norm = policy.select_action(model_batch)
        print(f"  normalized action shape: {list(action_norm.shape)}")
        print(f"  normalized action: {action_norm[0].detach().cpu().numpy()}")

        print("\n[10] Unnormalize action like official postprocess")
        action_stats = dataset_meta.stats[ACTION]
        action_env = unnormalize_quantiles(action_norm, action_stats["q01"], action_stats["q99"])
        print(f"  env-scale action: {action_env[0].detach().cpu().numpy()}")

        print("\n[11] env.step(action)")
        next_raw_obs, reward, terminated, truncated, step_info = vec_env.step(action_env.detach().cpu().numpy())
        print(f"  reward: {reward}")
        print(f"  terminated: {terminated}")
        print(f"  truncated: {truncated}")
        print(f"  step_info keys: {list(step_info.keys())}")

        print("\n[12] Inspect next observation")
        next_obs = preprocess_observation(next_raw_obs)
        next_obs = add_envs_task(vec_env, next_obs)
        next_obs = env_preprocessor(next_obs)
        summarize_tensors(next_obs, "  next observation after preprocessing")

        print("\nDone.")
        print("This script validates the dynamic single-step loop:")
        print("  env.reset -> obs preprocess -> policy -> action -> env.step -> next obs")
        print("What is still mocked:")
        print("  prompt text -> tokenizer -> language tokens")

    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()
