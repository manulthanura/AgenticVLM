import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.reasoning.domain.models import AmbiguityCase, CaseKind
from app.reasoning.domain.ports import VlmBudgetExceeded
from app.shared.config import Settings
from app.shared.domain.states import ActivityState as S
from app.shared.domain.time_range import TimeRange

cv2 = pytest.importorskip("cv2")

from app.reasoning.infrastructure.azure_vlm import AzureVlm
from app.reasoning.infrastructure.disk_cache import DiskCache
from app.reasoning.infrastructure.images import encode_frame
from app.reasoning.infrastructure.prompt_loader import load_prompt

CASE = AmbiguityCase(CaseKind.LYING_OUTSIDE_BED, TimeRange(0, 5))
FRAMES = [np.full((100, 200, 3), 90, np.uint8)] * 3
VALID = json.dumps(
    {"state": "lying_in_bed", "exit_confirmed": None, "confidence": 0.9, "rationale": "on the bed"}
)


class FakeClient:
    """Stands in for openai.AzureOpenAI: replays prepared reply texts and records the requests."""

    def __init__(self, *replies, tokens=50):
        self._replies, self._tokens, self.requests = list(replies), tokens, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self._replies.pop(0))
        usage = SimpleNamespace(total_tokens=self._tokens)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def vlm(client, tmp_path, **settings):
    base = {
        "azure_openai_deployment_fast": "fast",
        "azure_openai_deployment_strong": "strong",
        "agent_max_calls_per_video": 3,
    }
    return AzureVlm(client, Settings(**base | settings), DiskCache(tmp_path / "cache"))


def test_valid_reply_becomes_an_answer_and_request_is_cheap(tmp_path):
    client = FakeClient(VALID)
    result = vlm(client, tmp_path).ask(FRAMES, CASE)
    assert (result.state, result.confidence, result.tokens, result.cached) == (
        S.LYING_IN_BED,
        0.9,
        50,
        False,
    )
    (request,) = client.requests
    assert request["model"] == "fast" and request["response_format"] == {"type": "json_object"}
    images = [p for p in request["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(images) == 3 and {p["image_url"]["detail"] for p in images} == {"low"}


def test_same_question_is_served_from_the_cache(tmp_path):
    client, model = FakeClient(VALID), None
    model = vlm(client, tmp_path)
    model.ask(FRAMES, CASE)
    again = model.ask(FRAMES, CASE)
    assert again.cached and again.tokens == 0
    assert len(client.requests) == 1 and model.calls_made == 1


def test_strong_flag_selects_the_strong_deployment(tmp_path):
    client = FakeClient(VALID)
    vlm(client, tmp_path).ask(FRAMES, CASE, strong=True)
    assert client.requests[0]["model"] == "strong"


def test_invalid_reply_is_retried_once(tmp_path):
    client = FakeClient("not json", VALID)
    model = vlm(client, tmp_path)
    result = model.ask(FRAMES, CASE)
    assert result.state == S.LYING_IN_BED
    assert (model.calls_made, result.tokens) == (2, 100)


def test_two_invalid_replies_fall_back_to_unknown(tmp_path):
    out_of_range = json.dumps({"state": "walking", "confidence": 1.5, "rationale": "x"})
    result = vlm(FakeClient("nope", out_of_range), tmp_path).ask(FRAMES, CASE)
    assert (result.state, result.confidence, result.rationale) == (
        S.UNKNOWN,
        0.0,
        "invalid_vlm_output",
    )


def test_call_cap_is_enforced(tmp_path):
    model = vlm(FakeClient(VALID), tmp_path, agent_max_calls_per_video=1)
    model.ask(FRAMES, CASE)
    other_frames = [np.full((100, 200, 3), 10, np.uint8)] * 3  # different frames: no cache hit
    with pytest.raises(VlmBudgetExceeded):
        model.ask(other_frames, CASE)


def test_frames_are_downscaled_before_sending():
    import base64

    encoded = encode_frame(np.zeros((1000, 2000, 3), np.uint8), max_side=512)
    image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert max(image.shape[:2]) == 512


def test_every_case_kind_has_a_prompt_file():
    assert load_prompt("system")
    assert all(load_prompt(kind.value) for kind in CaseKind)
