"""
Gradio-based UI for robot visualization with network topology display.
Provides real-time visualization of bot camera feed, tracking data, and internal pubsub network.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Tuple


import gradio as gr
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import Field, SkipValidation

from ultraskelly.core.bot.base_abcs import Node, NodeParams
from ultraskelly.core.bot.motor.head_node import ServoStateMessage, TargetLocationMessage
from ultraskelly.core.bot.sensory.camera_node import FrameMessage
from ultraskelly.core.bot.sensory.pose_detection_node import (
    SKELETON_CONNECTIONS,
    PoseDataMessage,
)
from ultraskelly.core.pubsub.bot_topics import FrameTopic, PoseDataTopic, ServoStateTopic, TargetLocationTopic
from ultraskelly.core.pubsub.pubsub_manager import PubSubTopicManager
logger = logging.getLogger(__name__)

import cv2
class UINodeParams(NodeParams):
    """Parameters for GradioUINode."""
    
    deadzone: int = Field(default=30, ge=0)
    update_interval: float = Field(default=0.033, gt=0.0, le=1.0)  # ~30 FPS
    network_update_interval: float = Field(default=2.0, gt=0.0)  # Network graph update rate
    port: int = Field(default=7860, ge=1024)
    server_name: str = Field(default="0.0.0.0")
    share: bool = Field(default=False)


@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    name: str
    type: str
    status: str = "idle"
    messages_sent: int = 0
    messages_received: int = 0
    last_update: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0
    error_count: int = 0


class GradioUINode(Node):
    """Sophisticated Gradio UI for robot visualization."""
    
    params: UINodeParams
    
    # Queue subscriptions
    frame_queue: SkipValidation[queue.Queue] = Field(default=None, exclude=True)
    target_queue: SkipValidation[queue.Queue] = Field(default=None, exclude=True)
    servo_state_queue: SkipValidation[queue.Queue] = Field(default=None, exclude=True)
    pose_data_queue: SkipValidation[queue.Queue] = Field(default=None, exclude=True)
    
    # Latest state
    latest_frame: np.ndarray | None = Field(default=None, exclude=True)
    latest_target: TargetLocationMessage | None = Field(default=None, exclude=True)
    latest_servo_state: ServoStateMessage | None = Field(default=None, exclude=True)
    latest_pose_data: PoseDataMessage | None = Field(default=None, exclude=True)
    
    # Metrics tracking
    node_metrics: dict[str, NodeMetrics] = Field(default_factory=dict, exclude=True)
    
    # FPS tracking
    frame_count: int = Field(default=0, exclude=True)
    last_fps_time: float = Field(default_factory=time.time, exclude=True)
    fps: float = Field(default=0.0, exclude=True)
    
    # Gradio app
    app: gr.Blocks | None = Field(default=None, exclude=True)
    
    @classmethod
    def create(cls, *, pubsub: PubSubTopicManager, params: UINodeParams) -> "GradioUINode":
        """Factory method to create and initialize GradioUINode."""
        node = cls(pubsub=pubsub, params=params)
        
        # Subscribe to topics
        node.frame_queue = pubsub.topics[FrameTopic].get_subscription()
        node.target_queue = pubsub.topics[TargetLocationTopic].get_subscription()
        node.servo_state_queue = pubsub.topics[ServoStateTopic].get_subscription()
        node.pose_data_queue = pubsub.topics[PoseDataTopic].get_subscription()
        
        # Initialize node metrics
        node._initialize_metrics()
        
        return node
    
    def _initialize_metrics(self) -> None:
        """Initialize metrics for all known nodes."""
        # This would be populated based on actual node discovery
        # For now, we'll create placeholders
        node_types = [
            ("VisionNode", "sensor"),
            ("PoseDetectorNode", "detector"),
            ("BrightnessDetectorNode", "detector"),
            ("MotorNode", "actuator"),
            ("UINode", "interface"),
        ]
        
        for name, node_type in node_types:
            self.node_metrics[name] = NodeMetrics(name=name, type=node_type)
    
    def _draw_skeleton_on_frame(
        self, 
        frame: np.ndarray, 
        keypoints: np.ndarray, 
        confidence_threshold: float = 0.3
    ) -> None:
        """Draw skeleton on frame for a single person."""
        # Draw bones
        for kp1, kp2 in SKELETON_CONNECTIONS:
            pt1 = keypoints[kp1]
            pt2 = keypoints[kp2]
            
            if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > confidence_threshold:
                x, y = int(kp[0]), int(kp[1])
                # Color by body part
                if i <= 4:  # Head
                    color = (255, 0, 0)
                elif i <= 10:  # Arms
                    color = (0, 255, 0)
                else:  # Legs
                    color = (0, 0, 255)
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
    
    def _process_frame(self) -> np.ndarray | None:
        """Process and annotate the latest frame."""
        if self.latest_frame is None:
            return None
        
        frame = self.latest_frame.copy()
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw detected skeletons
        if self.latest_pose_data and self.latest_pose_data.keypoints is not None:
            for person_keypoints in self.latest_pose_data.keypoints:
                self._draw_skeleton_on_frame(frame, person_keypoints)
        
        # Draw center crosshair
        cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 2)
        cv2.line(frame, (0, center_y), (w, center_y), (255, 255, 255), 2)
        
        # Draw deadzone
        cv2.rectangle(
            frame,
            (center_x - self.params.deadzone, center_y - self.params.deadzone),
            (center_x + self.params.deadzone, center_y + self.params.deadzone),
            (128, 128, 128),
            1,
        )
        
        # Draw target
        if self.latest_target and self.latest_target.x is not None:
            x, y = self.latest_target.x, self.latest_target.y
            
            is_locked = (
                self.latest_servo_state.is_locked_x
                and self.latest_servo_state.is_locked_y
                and self.latest_servo_state.is_locked_roll
                if self.latest_servo_state else False
            )
            
            color = (0, 255, 0) if is_locked else (255, 0, 0)
            cv2.circle(frame, (x, y), 20, color, 3)
            
            # Draw orientation line
            if self.latest_target.angle is not None:
                angle_rad = np.radians(self.latest_target.angle)
                line_length = 40
                end_x = int(x + line_length * np.sin(angle_rad))
                end_y = int(y - line_length * np.cos(angle_rad))
                
                roll_color = (
                    (0, 255, 0)
                    if (self.latest_servo_state and self.latest_servo_state.is_locked_roll)
                    else (255, 0, 255)
                )
                cv2.line(frame, (x, y), (end_x, end_y), roll_color, 3)
        
        # Add status overlay
        self._add_status_overlay(frame)
        
        # Convert to RGB for Gradio
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def _add_status_overlay(self, frame: np.ndarray) -> None:
        """Add status text overlay to frame."""
        if self.latest_servo_state:
            status = (
                "LOCKED"
                if (
                    self.latest_servo_state.is_locked_x
                    and self.latest_servo_state.is_locked_y
                    and self.latest_servo_state.is_locked_roll
                )
                else "TRACKING"
            )
            status_color = (0, 255, 0) if status == "LOCKED" else (255, 255, 0)
            
            cv2.putText(
                frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
            )
            cv2.putText(
                frame, f"FPS: {self.fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
    
    def _create_network_graph(self) -> go.Figure:
        """Create network topology visualization using Plotly."""
        # Create directed graph
        G = nx.DiGraph()
        
        # Define node topology
        nodes_config = [
            ("Camera", "sensor", "#4CAF50"),
            ("PoseDetector", "detector", "#2196F3"),
            ("BrightnessDetector", "detector", "#2196F3"),
            ("MotorControl", "actuator", "#FF9800"),
            ("GradioUI", "interface", "#9C27B0"),
        ]
        
        # Add nodes
        for node_name, node_type, color in nodes_config:
            G.add_node(node_name, type=node_type, color=color)
        
        # Define pubsub connections
        edges = [
            ("Camera", "PoseDetector", "FrameTopic"),
            ("Camera", "BrightnessDetector", "FrameTopic"),
            ("PoseDetector", "MotorControl", "TargetLocationTopic"),
            ("BrightnessDetector", "MotorControl", "TargetLocationTopic"),
            ("MotorControl", "GradioUI", "ServoStateTopic"),
            ("PoseDetector", "GradioUI", "PoseDataTopic"),
            ("Camera", "GradioUI", "FrameTopic"),
        ]
        
        # Add edges with labels
        for source, target, topic in edges:
            G.add_edge(source, target, topic=topic)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Draw arrow
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none',
            )
            edge_traces.append(edge_trace)
            
            # Add arrow head
            arrow = go.Scatter(
                x=[x1],
                y=[y1],
                mode='markers',
                marker=dict(size=8, color='#888', symbol='triangle-up'),
                hoverinfo='none',
            )
            edge_traces.append(arrow)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[node for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=30,
                color=[G.nodes[node]['color'] for node in G.nodes()],
                line=dict(width=2, color='white'),
            ),
            hovertemplate='<b>%{text}</b><br>Type: %{customdata}<extra></extra>',
            customdata=[G.nodes[node]['type'] for node in G.nodes()],
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='Robot System Network Topology',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white'),
            )
        )
        
        return fig
    
    def _get_metrics_dataframe(self) -> pd.DataFrame:
        """Get current node metrics as a DataFrame."""
        metrics_data = []
        for name, metrics in self.node_metrics.items():
            metrics_data.append({
                'Node': name,
                'Type': metrics.type,
                'Status': metrics.status,
                'Messages Sent': metrics.messages_sent,
                'Messages Received': metrics.messages_received,
                'Processing (ms)': f"{metrics.processing_time_ms:.1f}",
                'Errors': metrics.error_count,
            })
        
        return pd.DataFrame(metrics_data)
    
    def _get_servo_status(self) -> dict[str, Any]:
        """Get current servo status."""
        if not self.latest_servo_state:
            return {
                'Pan Angle': 'N/A',
                'Tilt Angle': 'N/A',
                'Roll Angle': 'N/A',
                'X Lock': '❌',
                'Y Lock': '❌',
                'Roll Lock': '❌',
            }
        
        return {
            'Pan Angle': f"{self.latest_servo_state.pan_angle:.1f}°",
            'Tilt Angle': f"{self.latest_servo_state.tilt_angle:.1f}°",
            'Roll Angle': f"{self.latest_servo_state.roll_angle:.1f}°",
            'X Lock': '✅' if self.latest_servo_state.is_locked_x else '❌',
            'Y Lock': '✅' if self.latest_servo_state.is_locked_y else '❌',
            'Roll Lock': '✅' if self.latest_servo_state.is_locked_roll else '❌',
        }
    
    def _update_state_from_queues(self) -> None:
        """Background thread to consume queue updates."""
        while not self.stop_event.is_set():
            try:
                # Drain queues to get latest values
                while True:
                    try:
                        frame_msg: FrameMessage = self.frame_queue.get_nowait()
                        self.latest_frame = frame_msg.frame
                        self.frame_count += 1
                        
                        # Update FPS
                        current_time = time.time()
                        if current_time - self.last_fps_time >= 1.0:
                            self.fps = self.frame_count / (current_time - self.last_fps_time)
                            self.frame_count = 0
                            self.last_fps_time = current_time
                    except queue.Empty:
                        break
                
                while True:
                    try:
                        self.latest_target = self.target_queue.get_nowait()
                    except queue.Empty:
                        break
                
                while True:
                    try:
                        self.latest_servo_state = self.servo_state_queue.get_nowait()
                    except queue.Empty:
                        break
                
                while True:
                    try:
                        self.latest_pose_data = self.pose_data_queue.get_nowait()
                    except queue.Empty:
                        break
                
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error in UI update thread: {e}")
    
    def _build_gradio_app(self) -> gr.Blocks:
        """Build the Gradio application."""
        with gr.Blocks(
            title="Robot Vision & Control Dashboard",
            theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
            css="""
            .gradio-container {
                background: linear-gradient(to bottom, #1a1a2e, #0f0f1e);
            }
            """,
        ) as app:
            # Header
            gr.Markdown(
                """
                # 🤖 Robot Vision & Control Dashboard
                Real-time visualization of robot perception, control, and internal network architecture
                """,
                elem_id="header",
            )
            
            with gr.Tabs():
                # Main Vision Tab
                with gr.Tab("👁️ Vision & Tracking"):
                    with gr.Row():
                        # Left column - Camera feed
                        with gr.Column(scale=2):
                            camera_feed = gr.Image(
                                label="Camera Feed",
                                type="numpy",
                                height=480,
                                interactive=False,
                            )
                            fps_display = gr.Textbox(
                                label="Performance",
                                value="FPS: 0.0",
                                interactive=False,
                            )
                        
                        # Right column - Status panels
                        with gr.Column(scale=1):
                            # Servo status
                            gr.Markdown("### 🎯 Servo Status")
                            servo_status = gr.JSON(
                                label="",
                                value=self._get_servo_status(),
                            )
                            
                            # Target info
                            gr.Markdown("### 📍 Target Information")
                            target_info = gr.JSON(
                                label="",
                                value={
                                    'X': 'N/A',
                                    'Y': 'N/A',
                                    'Angle': 'N/A',
                                    'Status': 'No Target',
                                },
                            )
                            
                            # Detection settings
                            gr.Markdown("### ⚙️ Detection Settings")
                            confidence_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.3,
                                step=0.05,
                                label="Confidence Threshold",
                                interactive=True,
                            )
                
                # Network Architecture Tab
                with gr.Tab("🌐 System Architecture"):
                    with gr.Row():
                        # Network graph
                        network_graph = gr.Plot(
                            label="PubSub Network Topology",
                            value=self._create_network_graph(),
                        )
                    
                    with gr.Row():
                        # Node metrics table
                        metrics_table = gr.DataFrame(
                            label="Node Metrics",
                            value=self._get_metrics_dataframe(),
                            height=300,
                        )
                
                # Performance Tab
                with gr.Tab("📊 Performance Metrics"):
                    with gr.Row():
                        # Message flow chart
                        message_flow_chart = gr.LinePlot(
                            label="Message Flow Rate",
                            x="Time",
                            y="Messages/sec",
                            height=300,
                        )
                    
                    with gr.Row():
                        # Processing time chart
                        processing_chart = gr.BarPlot(
                            label="Node Processing Time",
                            x="Node",
                            y="Time (ms)",
                            height=300,
                        )
                
                # Configuration Tab
                with gr.Tab("⚙️ Configuration"):
                    gr.Markdown("### System Configuration")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Motor Control")
                            gain_slider = gr.Slider(
                                minimum=0.01,
                                maximum=0.2,
                                value=0.08,
                                step=0.01,
                                label="Control Gain",
                            )
                            deadzone_slider = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=30,
                                step=5,
                                label="Deadzone (pixels)",
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### Detection")
                            detector_dropdown = gr.Dropdown(
                                choices=["Pose", "Brightness"],
                                value="Pose",
                                label="Detector Type",
                            )
                            target_dropdown = gr.Dropdown(
                                choices=["Nose", "Left Wrist", "Right Wrist"],
                                value="Nose",
                                label="Target Keypoint",
                            )
            
            # Timer for updates
            timer = gr.Timer(
                value=self.params.update_interval,
                active=True,
            )
            
            # Update functions
            def update_camera_feed() -> Tuple[np.ndarray | None, str]:
                """Update camera feed and FPS."""
                frame = self._process_frame()
                fps_text = f"FPS: {self.fps:.1f} | Processing: OK"
                return frame, fps_text
            
            def update_servo_status() -> Tuple[dict, dict]:
                """Update servo and target status."""
                servo_dict = self._get_servo_status()
                
                target_dict = {
                    'X': 'N/A',
                    'Y': 'N/A',
                    'Angle': 'N/A',
                    'Status': 'No Target',
                }
                
                if self.latest_target and self.latest_target.x is not None:
                    target_dict = {
                        'X': f"{self.latest_target.x}",
                        'Y': f"{self.latest_target.y}",
                        'Angle': f"{self.latest_target.angle:.1f}°" if self.latest_target.angle else 'N/A',
                        'Status': 'Tracking',
                    }
                
                return servo_dict, target_dict
            
            def update_metrics() -> pd.DataFrame:
                """Update node metrics table."""
                return self._get_metrics_dataframe()
            
            # Connect timer to update functions
            timer.tick(
                fn=update_camera_feed,
                inputs=[],
                outputs=[camera_feed, fps_display],
            )
            
            timer.tick(
                fn=update_servo_status,
                inputs=[],
                outputs=[servo_status, target_info],
            )
            
            # Slower update for metrics
            network_timer = gr.Timer(
                value=self.params.network_update_interval,
                active=True,
            )
            
            network_timer.tick(
                fn=update_metrics,
                inputs=[],
                outputs=[metrics_table],
            )
        
        return app
    
    def run(self) -> None:
        """Main UI loop."""
        logger.info(f"Starting GradioUINode on {self.params.server_name}:{self.params.port}")
        
        # Start background update thread
        update_thread = threading.Thread(target=self._update_state_from_queues)
        update_thread.start()
        
        try:
            # Build and launch Gradio app
            self.app = self._build_gradio_app()
            
            # Launch with queue for proper async handling
            self.app.queue()
            self.app.launch(
                server_name=self.params.server_name,
                server_port=self.params.port,
                share=self.params.share,
                prevent_thread_lock=False,  # Let Gradio handle the main thread
            )
            
        except Exception as e:
            logger.error(f"Error launching Gradio app: {e}")
            self.stop_event.set()
        finally:
            self.stop_event.set()
            update_thread.join(timeout=2.0)
            logger.info("GradioUINode stopped")


# Export for use in bot_launcher.py
UINode = GradioUINode
UINodeParams = UINodeParams