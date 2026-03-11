"""Tests for LLM layer."""

import pytest
from unittest.mock import patch, MagicMock
import json
from src.llm import invoke_llm, invoke_llm_json


class TestInvokeLLM:
    def test_returns_none_when_disabled(self):
        with patch("src.llm.USE_LLM", False):
            result = invoke_llm("test prompt")
            assert result is None

    def test_returns_none_on_error(self):
        with patch("src.llm.USE_LLM", True):
            with patch("src.llm._get_bedrock_client") as mock_client:
                mock_client.return_value.invoke_model.side_effect = Exception("API error")
                result = invoke_llm("test prompt")
                assert result is None

    def test_successful_call(self):
        mock_response = {
            "content": [{"text": "Hello from Claude"}]
        }
        with patch("src.llm.USE_LLM", True):
            with patch("src.llm._get_bedrock_client") as mock_client:
                mock_body = MagicMock()
                mock_body.read.return_value = json.dumps(mock_response).encode()
                mock_client.return_value.invoke_model.return_value = {"body": mock_body}
                result = invoke_llm("test prompt")
                assert result == "Hello from Claude"


class TestInvokeLLMJson:
    def test_parses_json(self):
        response_data = {"title": "Test", "score": 85}
        mock_response = {"content": [{"text": json.dumps(response_data)}]}

        with patch("src.llm.USE_LLM", True):
            with patch("src.llm._get_bedrock_client") as mock_client:
                mock_body = MagicMock()
                mock_body.read.return_value = json.dumps(mock_response).encode()
                mock_client.return_value.invoke_model.return_value = {"body": mock_body}
                result = invoke_llm_json("test prompt")
                assert result == response_data

    def test_handles_markdown_fences(self):
        response_data = {"key": "value"}
        text = f"```json\n{json.dumps(response_data)}\n```"
        mock_response = {"content": [{"text": text}]}

        with patch("src.llm.USE_LLM", True):
            with patch("src.llm._get_bedrock_client") as mock_client:
                mock_body = MagicMock()
                mock_body.read.return_value = json.dumps(mock_response).encode()
                mock_client.return_value.invoke_model.return_value = {"body": mock_body}
                result = invoke_llm_json("test prompt")
                assert result == response_data

    def test_returns_none_on_invalid_json(self):
        mock_response = {"content": [{"text": "not valid json at all"}]}

        with patch("src.llm.USE_LLM", True):
            with patch("src.llm._get_bedrock_client") as mock_client:
                mock_body = MagicMock()
                mock_body.read.return_value = json.dumps(mock_response).encode()
                mock_client.return_value.invoke_model.return_value = {"body": mock_body}
                result = invoke_llm_json("test prompt")
                assert result is None
