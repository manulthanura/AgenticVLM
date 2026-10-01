from abc import ABC, abstractmethod

import numpy as np

from app.extract.domain.observation import PersonObservation


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray) -> list[PersonObservation]:
        """People in this frame. Frames must be fed in time order so track IDs stay stable."""
