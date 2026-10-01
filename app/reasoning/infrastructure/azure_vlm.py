import logging

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from app.reasoning.domain.models import AmbiguityCase, VlmAnswer
from app.reasoning.domain.ports import VisionLanguageModel, VlmBudgetExceeded
from app.reasoning.infrastructure.disk_cache import DiskCache, cache_key
from app.reasoning.infrastructure.images import encode_frame
from app.reasoning.infrastructure.prompt_loader import PROMPT_VERSION, load_prompt
from app.shared.config import Settings
from app.shared.domain.states import ActivityState

logger = logging.getLogger(__name__)


class VlmReply(BaseModel):
    """The JSON the model must return; anything else is invalid output."""

    state: ActivityState
    exit_confirmed: bool | None = None
    confidence: float = Field(ge=0, le=1)
    rationale: str


class AzureVlm(VisionLanguageModel):
    """Azure OpenAI vision calls with disk cache, per-video call cap and token accounting.

    `client` is an `openai.AzureOpenAI` (any object with chat.completions.create works).
    Create one instance per video: the cap and the counters are per instance.
    """

    def __init__(self, client, settings: Settings, cache: DiskCache) -> None:
        self._client, self._settings, self._cache = client, settings, cache
        self.calls_made = 0  # real calls only; cache hits are free
        self.total_tokens = 0

    def ask(self, frames: list[np.ndarray], case: AmbiguityCase, strong: bool = False) -> VlmAnswer:
        s = self._settings
        deployment = s.azure_openai_deployment_strong if strong else s.azure_openai_deployment_fast
        images = [encode_frame(f, s.agent_image_max_side) for f in frames]
        system, question = load_prompt("system"), load_prompt(case.kind.value)
        key = cache_key(PROMPT_VERSION, deployment, system, question, *images)

        cached = self._cache.get(key)
        if cached is not None:
            return self._answer(VlmReply.model_validate(cached), tokens=0, cached=True)

        tokens = 0
        for attempt in (1, 2):  # invalid output: retry once
            if self.calls_made >= s.agent_max_calls_per_video:
                raise VlmBudgetExceeded(f"{s.agent_max_calls_per_video} calls used")
            self.calls_made += 1
            response = self._client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": self._content(question, images)},
                ],
                response_format={"type": "json_object"},
                # No temperature: gpt-5 family models only accept the default. Repeat runs stay identical
                # through the disk cache (key = frames + prompt), not through sampling settings.
            )
            tokens += response.usage.total_tokens
            self.total_tokens += response.usage.total_tokens
            try:
                reply = VlmReply.model_validate_json(response.choices[0].message.content)
            except (ValidationError, TypeError, ValueError):
                logger.warning("Invalid VLM output for %s (attempt %d)", case.kind, attempt)
                continue
            self._cache.put(key, reply.model_dump(mode="json"))
            return self._answer(reply, tokens)
        return VlmAnswer(ActivityState.UNKNOWN, None, 0.0, "invalid_vlm_output", tokens)

    @staticmethod
    def _content(question: str, images: list[str]) -> list[dict]:
        parts: list[dict] = [{"type": "text", "text": question}]
        for image in images:
            url = f"data:image/jpeg;base64,{image}"
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})
        return parts

    @staticmethod
    def _answer(reply: VlmReply, tokens: int, cached: bool = False) -> VlmAnswer:
        return VlmAnswer(
            reply.state, reply.exit_confirmed, reply.confidence, reply.rationale, tokens, cached
        )
