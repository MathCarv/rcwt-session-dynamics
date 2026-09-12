"""Loopback-only llama.cpp client. No credentials or remote inference fallback."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


# Qwen3.5 model-card guidance for non-thinking general tasks. A fixed
# seed makes the sampled run traceable; it is not an across-seed evaluation.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0}


class LocalModelError(RuntimeError):
    """Inference or metering failed; callers must not silently substitute an agent."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LocalModelError("Redirect refused: inference must remain on loopback")


def validate_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError as exc:
        raise ValueError("Use a literal loopback IP, not a hostname or remote provider") from exc
    if (parsed.scheme != "http" or not address.is_loopback or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/") or parsed.port is None):
        raise ValueError("Expected http://127.0.0.1:PORT (or IPv6 loopback), without credentials")
    return endpoint.rstrip("/")


@dataclass(frozen=True)
class CallResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    model: str
    purpose: str
    finish_reason: str
    timings: dict[str, Any]
    request: dict[str, Any]
    response_id: str
    api_cost_usd: float = 0.0
    reasoning_chars: int = 0
    reasoning_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalModelClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:18085",
                 model: str = "rcwt-local-qwen35-4b", timeout: float = 120.0,
                 seed: int = 20260911):
        self.endpoint = validate_endpoint(endpoint)
        self.model = model
        self.timeout = timeout
        self.seed = seed
        # Ignore HTTP_PROXY/HTTPS_PROXY: even a loopback request must never leave this host.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        self.calls: list[CallResult] = []

    def _request(self, path: str, payload: dict | None = None) -> dict:
        if path not in {"/health", "/v1/models", "/tokenize", "/detokenize", "/v1/chat/completions"}:
            raise ValueError("Unsupported local endpoint")
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.endpoint + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LocalModelError(f"Local request {path} failed: {type(exc).__name__}") from exc
        if not isinstance(result, dict) or "error" in result:
            raise LocalModelError(f"Invalid local response for {path}")
        return result

    def probe(self) -> dict:
        health = self._request("/health")
        models = self._request("/v1/models")
        if health.get("status") != "ok" or self.model not in [m.get("id") for m in models.get("data", [])]:
            raise LocalModelError("The required local model is not ready")
        return {"health": health, "models": models, "endpoint": self.endpoint,
                "remote_inference": False, "api_cost_usd": 0.0}

    def tokenize(self, text: str) -> list[int]:
        tokens = self._request("/tokenize", {"content": text, "add_special": False,
                                             "parse_special": False}).get("tokens")
        if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
            raise LocalModelError("Missing real model token IDs")
        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        content = self._request("/detokenize", {"tokens": tokens}).get("content")
        if not isinstance(content, str):
            raise LocalModelError("Missing detokenized content")
        return content

    def complete(self, messages: list[dict], max_tokens: int, schema: dict | None = None,
                 purpose: str = "") -> CallResult:
        if not 1 <= max_tokens <= 2048:
            raise ValueError("Generation budget must be bounded to 1..2048 tokens")
        thinking = False
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   **SAMPLING, "seed": self.seed, "stream": False,
                   "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": thinking}}
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "sandbox_action", "strict": True, "schema": schema}}
        start = time.perf_counter()
        response = self._request("/v1/chat/completions", payload)
        duration = time.perf_counter() - start
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
            usage = response["usage"]
            prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
            if (not isinstance(content, str) or type(prompt) is not int or type(completion) is not int
                    or prompt <= 0 or not 0 < completion <= max_tokens or not math.isfinite(duration)
                    or response.get("model", self.model) != self.model
                    or (not thinking and choice["message"].get("reasoning_content"))
                    or response.get("timings", {}).get("cache_n", 0) != 0):
                raise ValueError("Invalid metering")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LocalModelError("Missing output or real token metering") from exc
        result = CallResult(content, prompt, completion, duration, response.get("model", self.model),
                            purpose, choice.get("finish_reason", "unknown"),
                            response.get("timings", {}), payload, response.get("id", ""),
                            reasoning_chars=len(choice["message"].get("reasoning_content") or ""),
                            reasoning_sha256=(hashlib.sha256(choice["message"]["reasoning_content"].encode("utf-8")).hexdigest()
                                              if choice["message"].get("reasoning_content") else None))
        self.calls.append(result)
        return result


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
