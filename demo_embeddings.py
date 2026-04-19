"""Embedding helpers shared by the Streamlit app and database rebuild script."""

from __future__ import annotations

from openai import OpenAI


class OpenAIEmbeddingFunction:
    """Chroma embedding function backed by the OpenAI Python v1+ client."""

    def __init__(self, api_key: str, model_name: str):
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

    def __call__(self, input):
        texts = [input] if isinstance(input, str) else list(input)
        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
