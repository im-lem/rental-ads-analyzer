import time
from dataclasses import dataclass

import ollama
from google import genai
from pydantic import ValidationError

from schema import RentalListing


@dataclass
class ExtractResult:
    listing: RentalListing | None
    valid: bool
    error: str | None
    raw: str
    latency_s: float
    input_tokens: int
    output_tokens: int


class GeminiExtractor:
    def __init__(self, api_key: str, model: str, temperature: float, top_p: float,
                 timeout_s: int = 60):
        self.client = genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout_s * 1000},
        )
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def extract(self, prompt: str, ad: str) -> ExtractResult:
        full_input = prompt + "\n\nAD:\n" + ad

        start = time.perf_counter()
        interaction = self.client.interactions.create(
            model=self.model,
            input=full_input,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": RentalListing.model_json_schema(),
            },
            generation_config={"temperature": self.temperature, "top_p": self.top_p},
        )
        latency = time.perf_counter() - start

        raw = interaction.output_text
        usage = interaction.usage

        try:
            listing = RentalListing.model_validate_json(raw)
            valid, error = True, None
        except ValidationError as e:
            listing, valid, error = None, False, str(e)

        return ExtractResult(listing, valid, error, raw, latency,
                             usage.total_input_tokens, usage.total_output_tokens)


class OllamaExtractor:
    def __init__(self, model: str, temperature: float, top_p: float,
                 timeout_s: int = 300):
        self.client = ollama.Client(timeout=timeout_s)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def extract(self, prompt: str, ad: str) -> ExtractResult:
        full_input = prompt + "\n\nAD:\n" + ad

        start = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": full_input}],
            format=RentalListing.model_json_schema(),
            options={"temperature": self.temperature, "top_p": self.top_p,
             "num_predict": 400},
        )
        latency = time.perf_counter() - start

        raw = response.message.content
        try:
            listing = RentalListing.model_validate_json(raw)
            valid, error = True, None
        except ValidationError as e:
            listing, valid, error = None, False, str(e)

        return ExtractResult(listing, valid, error, raw, latency,
                             response.prompt_eval_count or 0, response.eval_count or 0)


if __name__ == "__main__":
    from pathlib import Path

    ex = OllamaExtractor("qwen2.5:3b", temperature=0.0, top_p=1.0)
    prompt = Path("prompts/v1.txt").read_text(encoding="utf-8")
    ad = Path("data/ads/011.txt").read_text(encoding="utf-8")
    print(ex.extract(prompt, ad))