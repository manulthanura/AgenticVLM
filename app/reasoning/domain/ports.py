from abc import ABC, abstractmethod

import numpy as np

from app.reasoning.domain.models import AmbiguityCase, VlmAnswer


class VlmBudgetExceeded(Exception):
    """The per-video cap on real VLM calls is used up."""


class VisionLanguageModel(ABC):
    @abstractmethod
    def ask(self, frames: list[np.ndarray], case: AmbiguityCase, strong: bool = False) -> VlmAnswer:
        """Answer the question for this kind of case from the frames (BGR).

        `strong=True` uses the more capable, more expensive deployment.
        Must go through the cache and raise VlmBudgetExceeded over the cap.
        Invalid model output is retried once and then returned as an UNKNOWN answer.
        """
