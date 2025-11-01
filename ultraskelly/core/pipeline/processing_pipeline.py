import logging
import multiprocessing
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict
from skellycam.core.camera.config.camera_config import CameraConfigs
from skellycam.core.camera_group.camera_group import CameraGroup
from skellycam.core.ipc.pubsub.pubsub_manager import TopicTypes
from skellycam.core.types.type_overloads import CameraIdString, CameraGroupIdString

from ultraskelly.core.pipeline.aggregation_node import AggregationNode
from ultraskelly.core.pipeline.camera_node import CameraNode
from ultraskelly.core.pipeline.frontend_payload import FrontendPayload
from ultraskelly.core.pipeline.pipeline_configs import PipelineConfig
from ultraskelly.core.pipeline.pipeline_ipc import PipelineIPC
from ultraskelly.core.pubsub.pubsub_topics import AggregationNodeOutputTopic, AggregationNodeOutputMessage
from ultraskelly.core.types.type_overloads import PipelineIdString, TopicSubscriptionQueue

logger = logging.getLogger(__name__)



@dataclass
class ProcessingPipeline:
    id: PipelineIdString
    camera_group: CameraGroup
    config: PipelineConfig
    camera_nodes: dict[CameraIdString, CameraNode]
    aggregation_node: AggregationNode
    aggregation_node_subscription: TopicSubscriptionQueue
    ipc: PipelineIPC

    @property
    def alive(self) -> bool:
        return all([camera_node.worker.is_alive() for camera_node in
                    self.camera_nodes.values()]) and self.aggregation_node.worker.is_alive()

    @property
    def camera_group_id(self) -> CameraGroupIdString:
        return self.camera_group.id

    @property
    def camera_ids(self) -> list[CameraIdString]:
        return list(self.camera_nodes.keys())

    @property
    def camera_configs(self) -> CameraConfigs:
        return self.camera_group.configs

    @classmethod
    def from_config(cls,
                    global_kill_flag: multiprocessing.Value,
                    pipeline_config: PipelineConfig,
                    ):
        camera_group = CameraGroup.create(camera_configs=pipeline_config.camera_configs,
                                          global_kill_flag=global_kill_flag,
                                          )
        ipc = PipelineIPC.create(global_kill_flag=camera_group.ipc.global_kill_flag,
                                 shm_topic=camera_group.ipc.pubsub.topics[TopicTypes.SHM_UPDATES]
                                 )
        camera_nodes = {camera_id: CameraNode.create(camera_id=camera_id,
                                                     config=pipeline_config.camera_node_configs[camera_id],
                                                     ipc=ipc)
                        for camera_id, config in camera_group.configs.items()}
        aggregation_node = AggregationNode.create(camera_group_id=camera_group.id,
                                                  config=pipeline_config,
                                                  ipc=ipc,
                                                  )

        return cls(camera_nodes=camera_nodes,
                   aggregation_node=aggregation_node,
                   ipc=ipc,
                   config=pipeline_config,
                   aggregation_node_subscription=ipc.pubsub.topics[
                       AggregationNodeOutputTopic].get_subscription(),
                   camera_group=camera_group,
                   id=str(uuid.uuid4())[:6],
                   )

    def start(self) -> None:
        logger.debug(
            f"Starting Pipeline (id:{self.id} with camera group (id:{self.camera_group_id} for camera ids: {list(self.camera_nodes.keys())}...")
        try:
            logger.debug("Starting camera group...")
            self.camera_group.start()
        except Exception as e:
            logger.error(f"Failed to start camera group: {type(e).__name__} - {e}")
            logger.exception(e)
            raise

        try:
            logger.debug("Starting aggregation node...")
            self.aggregation_node.start()
            logger.debug(f"Aggregation node worker started: alive={self.aggregation_node.worker.is_alive()}")
        except Exception as e:
            logger.error(f"Failed to start aggregation node: {type(e).__name__} - {e}")
            logger.exception(e)
            raise

        for camera_id, camera_node in self.camera_nodes.items():
            try:
                logger.debug(f"Starting camera node {camera_id}...")
                camera_node.start()
                logger.debug(f"Camera node {camera_id} worker started: alive={camera_node.worker.is_alive()}")
            except Exception as e:
                logger.error(f"Failed to start camera node {camera_id}: {type(e).__name__} - {e}")
                logger.exception(e)
                raise

        logger.info(f"All pipeline workers started successfully")

    def shutdown(self):
        logger.debug(f"Shutting down {self.__class__.__name__}...")

        self.ipc.shutdown_pipeline()
        for camera_id, camera_node in self.camera_nodes.items():
            camera_node.shutdown()
        self.aggregation_node.shutdown()
        self.camera_group.close()

    def update_camera_configs(self, camera_configs: CameraConfigs) -> CameraConfigs:
        return self.camera_group.update_camera_settings(requested_configs=camera_configs)

    def get_latest_frontend_payload(self) -> tuple[bytes, FrontendPayload] | None:
        if not self.alive:
            return None
        aggregation_output: AggregationNodeOutputMessage | None = None
        while not self.aggregation_node_subscription.empty():
            aggregation_output = self.aggregation_node_subscription.get()
        if aggregation_output is None:
            return None
        frames_bytearray = self.camera_group.get_frontend_payload_by_frame_number(
            frame_number=aggregation_output.frame_number,
        )

        return (frames_bytearray,
                FrontendPayload.from_aggregation_output(aggregation_output=aggregation_output))
