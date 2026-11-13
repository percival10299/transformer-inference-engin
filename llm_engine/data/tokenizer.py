from __future__ import annotations

import codecs
from typing import Any


class ByteTokenizer:
    eos_token_id = 256
    vocab_size = 257

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        byte_values = bytes(token_id for token_id in token_ids if 0 <= token_id <= 255)
        return byte_values.decode("utf-8", errors="ignore")

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "byte_utf8",
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
        }


class ByteStreamDecoder:
    """Turns a stream of byte tokens into text, one token at a time.

    One token is one byte, but a character like an emoji spans several bytes.
    Decoding each token on its own would corrupt those characters, so incomplete
    byte sequences are buffered until the character is complete.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")

    def push(self, token_id: int) -> str:
        if not 0 <= token_id <= 255:
            return ""
        return self._decoder.decode(bytes([token_id]))

    def flush(self) -> str:
        return self._decoder.decode(b"", final=True)
