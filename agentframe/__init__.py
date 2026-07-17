import os as _os

# litellm: 禁用远程模型价格表拉取，用本地缓存即可
_os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from .core.agent import Agent
from .tools.function_tool import function_tool

__version__ = "0.1.0"
