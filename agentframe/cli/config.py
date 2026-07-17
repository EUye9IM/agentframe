from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".afcli.toml"

DEFAULT_CONFIG = """model = "deepseek/deepseek-v4-flash"
system_prompt = "You are a helpful assistant."
api_key = ""
compress_threshold = 100000
"""


def load_config(path: Path | None = None) -> dict:
    config_path = path or DEFAULT_CONFIG_PATH

    if not config_path.exists():
        config_path.write_text(DEFAULT_CONFIG)
        print(f"[config] Created default config at {config_path}")
        print(f"[config] Edit it and set your api_key or model.")

    raw = config_path.read_bytes()
    config = tomllib.loads(raw.decode())

    config.setdefault("model", "gpt-4o")
    config.setdefault("system_prompt", "You are a helpful assistant.")
    config.setdefault("api_key", "")
    config.setdefault("compress_threshold", 100000)

    api_key = config["api_key"] or os.environ.get("LLM_AUTH_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    config["api_key"] = api_key

    return config
