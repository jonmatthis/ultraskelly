
from abc import ABC, abstractmethod
from enum import Enum, auto
from threading import Event
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, SkipValidation

from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager


class DetectorType(Enum):
    """Types of detectors available."""
    BRIGHTNESS = auto()
    POSE = auto()


class NodeParams(BaseModel):
    """Base class for all node parameters."""
    pass


class Node(BaseModel,ABC):
    """Abstract base class for all nodes."""

    pubsub: PubSubTopicManager
    params: NodeParams
    stop_event = SkipValidation[Event()]

    @abstractmethod
    def run(self) -> None:
        """Main node execution loop."""
        pass

    def stop(self) -> None:
        """Stop the node."""
        self.stop_event.set()


@runtime_checkable
class SensorNode(Protocol):
    """Protocol for nodes that produce sensor data."""
    def publish_data(self, data: object) -> None:
        """Publish sensor data."""
        ...


@runtime_checkable
class ActuatorNode(Protocol):
    """Protocol for nodes that consume data to control actuators."""
    def process_command(self, command: object) -> None:
        """Process actuator command."""
        ...


class DetectorNode(Node):
    """Base class for detector nodes that process images."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> tuple[int | None, int | None, float | None]:
        """Detect target in image."""
        pass
