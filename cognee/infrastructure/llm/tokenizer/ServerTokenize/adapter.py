"""
Tokenizer that calls the embedding server's /tokenize endpoint (e.g. vLLM).
Use when EMBEDDING_USE_SERVER_TOKENIZER=true and EMBEDDING_ENDPOINT points to the server.
"""
from typing import List, Any

from ..tokenizer_interface import TokenizerInterface


class ServerTokenizer(TokenizerInterface):
    """
    Count tokens via POST to server /tokenize (vLLM-style: {"prompt": text} -> tokens).
    """

    def __init__(self, tokenize_url: str):
        self.tokenize_url = tokenize_url.rstrip("/")

    def _tokenize(self, text: str) -> List[int]:
        import urllib.request
        import json

        body = json.dumps({"prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            self.tokenize_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        # vLLM may return "tokens" or "token_ids"
        tokens = data.get("tokens") or data.get("token_ids") or data.get("ids") or []
        return tokens if isinstance(tokens, list) else []

    def extract_tokens(self, text: str) -> List[Any]:
        return self._tokenize(text)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def decode_single_token(self, token: int) -> str:
        raise NotImplementedError("Server tokenizer does not decode single tokens")
