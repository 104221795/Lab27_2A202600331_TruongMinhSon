"""LLM helpers.

Returns an OpenAI-compatible chat model. The lab README uses OpenRouter, but
some local setups use Gemini's OpenAI-compatible endpoint instead.
"""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel
from langchain_openai import ChatOpenAI


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key and os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ["GEMINI_API_KEY"]
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
        base_url = os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    if not api_key:
        raise RuntimeError(
            "No LLM API key is set. Add OPENROUTER_API_KEY or GEMINI_API_KEY to .env"
        )

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def _gemini_direct() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"))


def _content_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = text[:500].replace("\n", "\\n")
    raise ValueError(f"LLM did not return a JSON object. Preview: {preview}")


def _json_mode_messages(schema: type[SchemaT], messages: list[Any]) -> list[Any]:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    task = "\n\n".join(
        f"{getattr(message, 'type', None) or message.get('role', 'message').upper()}:\n"
        f"{getattr(message, 'content', None) or message.get('content', '')}"
        for message in messages
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a JSON API. Return exactly one valid JSON object. "
                "Do not include markdown fences, prose, comments, analysis text, "
                "or a thought prefix."
            ),
        },
        {
            "role": "user",
            "content": (
                "Analyze the task below and respond only with JSON that validates "
                f"against this schema:\n{schema_json}\n\nTask:\n{task}"
            ),
        },
    ]


def _json_llm(temperature: float):
    llm = get_llm(temperature)
    return llm.bind(response_format={"type": "json_object"})


def _invoke_json_fallback(
    schema: type[SchemaT],
    messages: list[Any],
    *,
    temperature: float,
) -> SchemaT:
    try:
        raw = _json_llm(temperature).invoke(_json_mode_messages(schema, messages))
    except Exception:
        raw = get_llm(temperature).invoke(_json_mode_messages(schema, messages))
    return _parse_structured(schema, raw)


async def _ainvoke_json_fallback(
    schema: type[SchemaT],
    messages: list[Any],
    *,
    temperature: float,
) -> SchemaT:
    try:
        raw = await _json_llm(temperature).ainvoke(_json_mode_messages(schema, messages))
    except Exception:
        raw = await get_llm(temperature).ainvoke(_json_mode_messages(schema, messages))
    return _parse_structured(schema, raw)


def _parse_structured(schema: type[SchemaT], raw: Any) -> SchemaT:
    if isinstance(raw, schema):
        return raw
    if isinstance(raw, dict):
        return schema.model_validate(raw)
    return schema.model_validate(_extract_json_object(_content_to_text(raw)))


def invoke_structured(
    schema: type[SchemaT],
    messages: list[Any],
    *,
    temperature: float = 0.2,
) -> SchemaT:
    """Invoke an LLM and return a validated Pydantic object.

    Gemini's OpenAI-compatible endpoint sometimes prefixes JSON with text such
    as ``thought``. This helper handles that provider quirk without weakening the
    rest of the graph.
    """
    if not _gemini_direct():
        try:
            return get_llm(temperature).with_structured_output(schema).invoke(messages)
        except Exception:
            pass

    return _invoke_json_fallback(schema, messages, temperature=temperature)


async def ainvoke_structured(
    schema: type[SchemaT],
    messages: list[Any],
    *,
    temperature: float = 0.2,
) -> SchemaT:
    """Async version of invoke_structured."""
    if not _gemini_direct():
        try:
            return await get_llm(temperature).with_structured_output(schema).ainvoke(messages)
        except Exception:
            pass

    return await _ainvoke_json_fallback(schema, messages, temperature=temperature)
