"""Non-secret Responses API settings. Credentials are resolved at call time."""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
import tomllib
from urllib.parse import urlsplit


@dataclass(frozen=True)
class GPTSettings:
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 90.0
    reasoning_effort: str = "low"
    structured_output: bool = True
    max_output_tokens: int | None = None
    history_steps: int = 2
    max_calls: int = 150  # Zero explicitly disables the model-call cap.
    image_max_edge: int = 768
    transport: str = "responses"
    codex_bin: str = "codex"
    service_tier: str | None = None  # Omit to preserve the provider's default speed.

    def __post_init__(self):
        if self.transport not in {"responses", "codex_cli"}:
            raise ValueError("transport must be responses or codex_cli")
        if not isinstance(self.codex_bin, str) or not self.codex_bin.strip() or any(c in self.codex_bin for c in "\n\r\x00"):
            raise ValueError("codex_bin must be an executable name or path")
        if self.transport == "codex_cli" and self.max_output_tokens is not None:
            raise ValueError("codex_cli does not support max_output_tokens")
        url = urlsplit(self.base_url)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment or self.base_url.endswith("/responses")):
            raise ValueError("base_url must be an HTTPS API root without credentials, query or /responses")
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.model):
            raise ValueError("model must be an explicit provider model ID")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("api_key_env must be an environment variable name, never a token")
        if (isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float))
                or not math.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("timeout_s must be a finite positive number")
        if self.reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported reasoning_effort")
        if self.service_tier is not None and (
            not isinstance(self.service_tier, str) or self.service_tier not in {"fast", "ultrafast"}
        ):
            raise ValueError("service_tier must be fast or ultrafast; omit it for the provider default")
        if type(self.structured_output) is not bool:
            raise ValueError("structured_output must be boolean")
        for name, lower, upper in (("history_steps", 0, 20), ("max_calls", 0, 1000), ("image_max_edge", 64, 2048)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if self.max_output_tokens is not None and (type(self.max_output_tokens) is not int or self.max_output_tokens < 32):
            raise ValueError("max_output_tokens must be an integer >= 32 when set")

    def public_dict(self) -> dict:
        return asdict(self)


def load_gpt_settings(path: Path) -> GPTSettings:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) != {"gpt"} or not isinstance(data["gpt"], dict):
        raise ValueError("Model configuration must contain only [gpt]")
    unknown = set(data["gpt"]) - set(GPTSettings.__dataclass_fields__)
    if unknown:
        raise ValueError("Unknown GPT settings; put credentials only in the named environment variable")
    return GPTSettings(**data["gpt"])


def read_api_key(name: str) -> str:
    """Use process env first; Windows user env also supports a recent `setx`."""
    value = os.environ.get(name)
    if value is None and os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
        except FileNotFoundError:
            pass
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {name}; set it in the environment of the Python process")
    value = value.strip()
    if any(char.isspace() for char in value):
        raise ValueError(f"{name} contains whitespace; check the environment value")
    return value
