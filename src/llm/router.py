from __future__ import annotations

import os
import time
import itertools
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    GROQ      = "groq"
    GEMINI    = "gemini"


@dataclass
class LLMConfig:
    provider:    Provider
    model:       str
    max_tokens:  int   = 4096
    temperature: float = 0.2


PROVIDER_DEFAULTS: dict[Provider, LLMConfig] = {
    Provider.ANTHROPIC: LLMConfig(
        provider=Provider.ANTHROPIC,
        model="claude-3-haiku-20240307",
    ),
    Provider.GROQ: LLMConfig(
        provider=Provider.GROQ,
        model="llama-3.3-70b-versatile",
    ),
    Provider.GEMINI: LLMConfig(
        provider=Provider.GEMINI,
        model="gemini-2.0-flash",
    ),
}


class LLMRouter:
    def __init__(self, providers: list[Provider] | None = None):
        self.providers = providers or [p for p in list(Provider) if os.getenv(f"{p.value.upper()}_API_KEY")]
        self._cycle    = itertools.cycle(self.providers)
        self._clients: dict[Provider, object] = {}
        self._init_clients()

    # ── client setup ──────────────────────────────────────────────────

    def _init_clients(self):
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self._clients[Provider.ANTHROPIC] = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                console.print("[green]✓ Anthropic ready[/green]")
            except ImportError:
                console.print("[yellow]! anthropic package missing — run pip install anthropic[/yellow]")

        if os.getenv("GROQ_API_KEY"):
            try:
                import groq
                self._clients[Provider.GROQ] = groq.Groq(
                    api_key=os.getenv("GROQ_API_KEY")
                )
                console.print("[green]✓ Groq ready[/green]")
            except ImportError:
                console.print("[yellow]! groq package missing — run pip install groq[/yellow]")

        if os.getenv("GEMINI_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self._clients[Provider.GEMINI] = genai.GenerativeModel(
                    PROVIDER_DEFAULTS[Provider.GEMINI].model
                )
                console.print("[green]✓ Gemini ready[/green]")
            except ImportError:
                console.print("[yellow]! google-generativeai package missing[/yellow]")

        if not self._clients:
            raise RuntimeError(
                "No API keys found.\n"
                "Copy .env.example → .env and add at least one key:\n"
                "  ANTHROPIC_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY"
            )

    # ── public API ────────────────────────────────────────────────────

    def complete(
        self,
        prompt:   str,
        system:   str = "You are an expert software engineering agent.",
        provider: Provider | None = None,
        retries:  int = 3,
    ) -> str:
        target = provider or self._next_available()
        errors: list[str] = []

        for attempt in range(retries):
            try:
                result = self._call(target, prompt, system)
                return result
            except Exception as exc:
                errors.append(f"[{target.value}] {exc}")
                console.print(
                    f"[yellow]Provider {target.value} failed "
                    f"(attempt {attempt + 1}/{retries}): {exc}[/yellow]"
                )
                target = self._next_available(exclude=target)
                time.sleep(1)

        raise RuntimeError(
            f"All providers failed after {retries} attempts:\n" +
            "\n".join(errors)
        )

    def available_providers(self) -> list[Provider]:
        return list(self._clients.keys())

    # ── internals ─────────────────────────────────────────────────────

    def _next_available(self, exclude: Provider | None = None) -> Provider:
        for _ in range(len(self.providers)):
            p = next(self._cycle)
            if p in self._clients and p != exclude:
                return p
        # fallback: return any available even if same as exclude
        for p in self._clients:
            return p
        raise RuntimeError("No providers available with API keys configured.")

    def _call(self, provider: Provider, prompt: str, system: str) -> str:
        if provider not in self._clients:
            raise ValueError(f"No client for {provider.value} — missing API key?")

        cfg = PROVIDER_DEFAULTS[provider]

        if provider == Provider.ANTHROPIC:
            client = self._clients[provider]
            resp = client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        if provider == Provider.GROQ:
            client = self._clients[provider]
            resp = client.chat.completions.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content

        if provider == Provider.GEMINI:
            client = self._clients[provider]
            resp = client.generate_content(f"{system}\n\n{prompt}")
            return resp.text

        raise ValueError(f"Unknown provider: {provider}")


# ── module-level singleton ─────────────────────────────────────────────

_router: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
