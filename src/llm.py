"""LLM abstraction layer: Bedrock primary, local fallback."""

import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from src.config import AWS_REGION, BEDROCK_MODEL_ID, USE_LLM

logger = logging.getLogger(__name__)

_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime", region_name=AWS_REGION
        )
    return _bedrock_client


def invoke_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Optional[str]:
    """Call Bedrock Claude. Returns None if LLM is disabled or unavailable."""
    if not USE_LLM:
        return None

    try:
        client = _get_bedrock_client()
        messages = [{"role": "user", "content": prompt}]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system

        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    except (ClientError, NoCredentialsError, Exception) as e:
        logger.warning(f"Bedrock call failed, falling back to local: {e}")
        return None


def invoke_llm_json(
    prompt: str,
    system: str = "",
    max_tokens: int = 1024,
) -> Optional[dict]:
    """Call LLM and parse JSON response. Returns None on failure."""
    response = invoke_llm(prompt, system=system, max_tokens=max_tokens)
    if response is None:
        return None
    try:
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"Failed to parse LLM JSON: {e}")
        return None
