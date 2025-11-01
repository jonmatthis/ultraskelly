import logging

from pydantic import Field, model_validator
from skellycam.core.types.type_overloads import CameraGroupIdString, CameraIdString
from skellytracker.trackers.base_tracker.base_tracker_abcs import TrackedPointIdString
from skellytracker.trackers.charuco_tracker.charuco_observation import CharucoObservation

from ultraskelly.core.pipeline.pipeline_configs import PipelineConfig
from ultraskelly.core.pubsub.pubsub_abcs import TopicMessageABC, create_topic
from ultraskelly.core.tasks.calibration_task.calibration_helpers.charuco_overlay_data import CharucoOverlayData
from ultraskelly.core.types.type_overloads import TrackerTypeString, FrameNumberInt, Point3d, PipelineIdString

logger = logging.getLogger(__name__)

class ProcessFrameNumberMessage(TopicMessageABC):
    frame_number: int = Field(ge=0, description="Frame number to process")

class ShouldCalibrateMessage(TopicMessageABC):
    """Dummy message to signal that calibration should be performed."""
    pass


class PipelineConfigMessage(TopicMessageABC):
    pipeline_config: PipelineConfig


class CameraNodeOutputMessage(TopicMessageABC):
    camera_id: CameraIdString
    frame_number: FrameNumberInt = Field(ge=0)
    tracker_name: TrackerTypeString
    charuco_observation: CharucoObservation | None = None


class AggregationNodeOutputMessage(TopicMessageABC):
    frame_number: FrameNumberInt = Field(ge=0)
    pipeline_id: PipelineIdString
    camera_group_id: CameraGroupIdString
    pipeline_config: PipelineConfig
    camera_node_outputs: dict[CameraIdString, CameraNodeOutputMessage]
    tracked_points3d: dict[TrackedPointIdString, Point3d] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate(self) -> 'AggregationNodeOutputMessage':
        # ensure that camera_node_outputs keys match camera IDs in pipeline_config
        expected_camera_ids = set(self.pipeline_config.camera_node_configs.keys())
        actual_camera_ids = set(self.camera_node_outputs.keys())
        if sorted(expected_camera_ids) != sorted(actual_camera_ids):
            raise ValueError(
                f"Camera IDs in camera_node_outputs {actual_camera_ids} do not match expected camera IDs from pipeline_config {expected_camera_ids}")
        # make sure camera_node_outputs all have the same frame_number as this message
        for cam_output in self.camera_node_outputs.values():
            if cam_output.frame_number != self.frame_number:
                logger.warning(
                    f"CameraNodeOutputMessage for camera {cam_output.camera_id} has frame number {cam_output.frame_number} which does not match AggregationNodeOutputMessage frame number {self.frame_number}")
        return self

    @property
    def charuco_overlay_data(self) -> dict[CameraIdString, CharucoOverlayData]:
        overlay_data: dict[CameraIdString, CharucoOverlayData] = {}
        for camera_id, cam_output in self.camera_node_outputs.items():
            if cam_output.charuco_observation is not None:
                overlay_data[camera_id] = CharucoOverlayData.from_charuco_observation(
                    camera_id=camera_id,
                    observation=cam_output.charuco_observation,
                )
        return overlay_data

    @property
    def camera_ids(self) -> list[CameraIdString]:
        return list(self.camera_node_outputs.keys())


ProcessFrameNumberTopic = create_topic(ProcessFrameNumberMessage)
ShouldCalibrateTopic = create_topic(ShouldCalibrateMessage)
PipelineConfigTopic = create_topic(PipelineConfigMessage)
CameraNodeOutputTopic = create_topic(CameraNodeOutputMessage)
AggregationNodeOutputTopic = create_topic(AggregationNodeOutputMessage)
