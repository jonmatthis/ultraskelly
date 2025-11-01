import logging
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field
from skellycam.core.types.type_overloads import CameraIdString
from skellyforge.ultraskelly_utils.postprocessing_widgets.postprocessing_functions import interpolate_data
from skellytracker.trackers.charuco_tracker.charuco_observation import CharucoObservation

from ultraskelly.core.tasks.calibration_task.point_triangulator import PointTriangulator
from ultraskelly.core.tasks.calibration_task.v1_capture_volume_calibration.anipose_camera_calibration.charuco_groundplane_utils import \
    skellyforge_data, find_good_frame, CharucoVisibilityError, CharucoVelocityError, \
    compute_basis_vectors_of_new_reference
from ultraskelly.core.tasks.calibration_task.v1_capture_volume_calibration.anipose_camera_calibration.ultraskelly_anipose import (
    AniposeCameraGroup, AniposeCharucoBoard
)
from ultraskelly.system.default_paths import get_default_ultraskelly_base_folder_path

logger = logging.getLogger(__name__)

CharucoObservations = dict[CameraIdString, CharucoObservation | None]


def get_calibrations_folder_path() -> Path:
    """Get the main calibrations folder path."""
    return Path(get_default_ultraskelly_base_folder_path()) / "calibrations"


def create_camera_calibration_file_name(recording_name: str) -> str:
    """Create a standardized camera calibration filename."""
    return f"{recording_name}_camera_calibration.toml"


def get_last_successful_calibration_toml_path() -> Path:
    """Get the path for the last successful calibration TOML file."""
    return get_calibrations_folder_path() / "last_successful_camera_calibration.toml"


class CharucoObservationAggregator(BaseModel):
    anipose_camera_ordering: list[CameraIdString]
    # anipose handles camera rows by the ordering of cameras in CameraGroup - so we need to use that ordering when we pass the camera rows to anipose
    individual_camera_rows: dict[CameraIdString, list] = Field(default_factory=dict)

    @classmethod
    def from_charuco_observation_payload(cls,
                                         charuco_observations_by_camera: CharucoObservations,
                                         anipose_camera_ordering: list[CameraIdString]):
        if set(charuco_observations_by_camera.keys()) != set(anipose_camera_ordering):
            raise ValueError("individual_camera_rows and anipose_camera_ordering must have the same camera ids")
        camera_rows = {}
        for camera_id, charuco_observation in charuco_observations_by_camera.items():
            anipose_camera_row = charuco_observation.to_anipose_camera_row() if charuco_observation is not None else None
            camera_rows[camera_id] = [anipose_camera_row] if anipose_camera_row is not None else []

        return cls(individual_camera_rows=camera_rows, anipose_camera_ordering=anipose_camera_ordering)

    @property
    def all_camera_rows(self):
        return [self.individual_camera_rows[anipose_camera_id] for anipose_camera_id in self.anipose_camera_ordering]

    def add_observations(self, charuco_observations_by_camera: CharucoObservations):
        for camera_id, charuco_observation in charuco_observations_by_camera.items():
            anipose_camera_row = charuco_observation.to_anipose_camera_row() if charuco_observation is not None else None
            if anipose_camera_row is not None:
                self.individual_camera_rows[camera_id].append(anipose_camera_row)


class GroundPlaneSuccess:
    def __init__(self, success: bool, error: str | None = None):
        self.success = success
        self.error = error


def anipose_calibration_from_charuco_observations(
        charuco_observations_by_frame: list[CharucoObservations],
        charuco_board: AniposeCharucoBoard,
        anipose_camera_group: AniposeCameraGroup,
        calibration_toml_save_path: Path | str,
        recording_path: Path | None = None,
        pin_camera_0_to_origin: bool = True,
        use_charuco_as_groundplane: bool = False,
        init_intrinsics: bool = True,
        init_extrinsics: bool = True,
        verbose: bool = True,) -> PointTriangulator:
    logger.info(
        f"Starting Anipose calibration with {len(charuco_observations_by_frame)} frames of charuco observations")
    # Aggregate all observations
    charuco_observation_aggregator = CharucoObservationAggregator.from_charuco_observation_payload(
        charuco_observations_by_camera=charuco_observations_by_frame.pop(0),
        anipose_camera_ordering=[camera.name for camera in anipose_camera_group.cameras])
    for charuco_observations_by_camera in charuco_observations_by_frame:
        charuco_observation_aggregator.add_observations(charuco_observations_by_camera)

    if charuco_observation_aggregator is None:
        raise ValueError("No charuco observations were provided for calibration")

    # Perform calibration on aggregated rows
    all_camera_rows = charuco_observation_aggregator.all_camera_rows
    logger.info(f"Aggregated charuco observations for {len(all_camera_rows)} cameras")
    error, merged, charuco_frame_numbers = anipose_camera_group.calibrate_rows(
        all_camera_rows,
        charuco_board,
        init_intrinsics=init_intrinsics,
        init_extrinsics=init_extrinsics,
        verbose=verbose,
    )

    logger.info(f"Anipose calibration completed successfully!")

    logger.info(f"Calibration error: {error}")

    anipose_camera_group.metadata["groundplane_calibration"] = False
    groundplane_success: GroundPlaneSuccess | None = None

    # Apply camera 0 pinning if requested
    if pin_camera_0_to_origin:
        anipose_camera_group = pin_camera_zero_to_origin(camera_group=anipose_camera_group)
        logger.info("Pinned camera 0 to origin")

    # Apply groundplane correction if requested
    if use_charuco_as_groundplane:
        anipose_camera_group, groundplane_success = set_charuco_board_as_groundplane(
            camera_group=anipose_camera_group,
            charuco_board=charuco_board,
            recording_folder_path=recording_path
        )
        if groundplane_success.success:
            anipose_camera_group.metadata["groundplane_calibration"] = True
            logger.info("Successfully set charuco board as groundplane")
        else:
            logger.warning(f"Failed to set groundplane: {groundplane_success.error}")

    # Calculate real-world camera positions and orientations
    get_real_world_matrices(camera_group=anipose_camera_group)

    # Create calibration filename
    anipose_camera_group.dump(calibration_toml_save_path)
    logger.info(f"Saved calibration to: {calibration_toml_save_path}")

    # Save to recording folder if provided
    if recording_path is not None:
        recording_folder_toml_path = Path(recording_path) / calibration_toml_save_path.stem
        anipose_camera_group.dump(recording_folder_toml_path)
        logger.info(f"Saved calibration copy to recording folder: {recording_folder_toml_path}")

    # Save as last successful calibration
    last_successful_calibration_toml_path = get_last_successful_calibration_toml_path()
    anipose_camera_group.dump(last_successful_calibration_toml_path)
    logger.info(f"Saved as last successful calibration: {last_successful_calibration_toml_path}")
    return PointTriangulator.from_toml(toml_path=last_successful_calibration_toml_path)


def pin_camera_zero_to_origin(camera_group: AniposeCameraGroup) -> AniposeCameraGroup:
    """
    Re-express all camera extrinsics relative to camera 0.

    This function performs two operations:
        1. Rotates all camera coordinate systems so that camera 0's orientation becomes the new world frame.
        2. Shifts the world origin to camera 0's position, making it the new coordinate origin (0, 0, 0).

    The result is that:
        - Camera 0 ends up with an identity rotation and zero translation.
        - All other cameras are expressed relative to camera 0's original position and orientation.
    """
    rvecs_new = align_rotations_to_cam0(camera_group=camera_group)
    camera_group.set_rotations(rvecs_new)
    tvecs_new = shift_origin_to_cam0(camera_group=camera_group)
    camera_group.set_translations(tvecs_new)
    return camera_group


def align_rotations_to_cam0(camera_group: AniposeCameraGroup) -> np.ndarray:
    """Align all camera rotations to camera 0's coordinate frame."""
    rvecs = camera_group.get_rotations()
    R0, _ = cv2.Rodrigues(rvecs[0])

    rvecs_new = np.empty_like(rvecs)
    for i in range(rvecs.shape[0]):
        Ri, _ = cv2.Rodrigues(rvecs[i])
        Ri_new, _ = cv2.Rodrigues(Ri @ R0.T)
        rvecs_new[i] = Ri_new.flatten()
    return rvecs_new


def shift_origin_to_cam0(camera_group: AniposeCameraGroup) -> np.ndarray:
    """Shift world origin to camera 0's position."""
    tvecs = camera_group.get_translations()
    rvecs = camera_group.get_rotations()

    camera_0_translation = tvecs[0, :]
    camera_0_rotation = rvecs[0, :]

    R0, _ = cv2.Rodrigues(camera_0_rotation)
    delta_to_origin_world = -R0.T @ camera_0_translation

    new_tvecs = np.zeros_like(tvecs)
    for cam_i in range(tvecs.shape[0]):
        Ri, _ = cv2.Rodrigues(rvecs[cam_i, :])
        delta_to_origin_camera_i = Ri @ delta_to_origin_world
        new_tvecs[cam_i, :] = tvecs[cam_i, :] + delta_to_origin_camera_i
    return new_tvecs


def set_charuco_board_as_groundplane(
        camera_group: AniposeCameraGroup,
        charuco_board: AniposeCharucoBoard,
        recording_folder_path: Path | None = None
) -> tuple[AniposeCameraGroup, GroundPlaneSuccess]:
    """Set the charuco board plane as the world groundplane."""
    logger.info("Getting 2D Charuco data")

    charuco_2d_xy: np.ndarray = camera_group.charuco_2d_data

    if charuco_2d_xy is None:
        error_message = "Charuco 2d data was not retrieved successfully. Check for an error during calibration."
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info(f"Charuco 2d data retrieved successfully with shape: {charuco_2d_xy.shape}")
    charuco_2d_xy = charuco_2d_xy.astype(np.float64)
    num_cameras, num_frames, num_tracked_points, _ = charuco_2d_xy.shape
    charuco_2d_flat = charuco_2d_xy.reshape(num_cameras, -1, 2)

    logger.info("Getting 3d Charuco data")
    charuco_3d_flat = camera_group.triangulate(charuco_2d_flat)
    charuco_3d_xyz = charuco_3d_flat.reshape(num_frames, num_tracked_points, 3)
    logger.info(f"Charuco 3d data reconstructed with shape: {charuco_3d_xyz.shape}")

    charuco_3d_xyz_interpolated =    interpolate_data.interpolate_skeleton_data(
        skeleton_data=charuco_3d_xyz
    )

    try:
        charuco_still_frame_idx = find_good_frame(
            charuco_data=charuco_3d_xyz_interpolated,
            number_of_squares_width=charuco_board.squaresX,
            number_of_squares_height=charuco_board.squaresY,
        )
    except CharucoVisibilityError as e:
        logger.warning("Ground-plane alignment skipped — reverting to original calibration: %s", e, exc_info=True)
        return camera_group, GroundPlaneSuccess(success=False, error=str(e))
    except CharucoVelocityError as e:
        logger.warning("Ground-plane alignment skipped — reverting to original calibration: %s", e, exc_info=True)
        return camera_group, GroundPlaneSuccess(success=False, error=str(e))

    charuco_frame = charuco_3d_xyz_interpolated[charuco_still_frame_idx]

    x_hat, y_hat, z_hat = compute_basis_vectors_of_new_reference(
        charuco_frame,
        number_of_squares_width=charuco_board.squaresX,
        number_of_squares_height=charuco_board.squaresY
    )

    charuco_origin_in_world = charuco_frame[0]
    rmat_charuco_to_world = np.column_stack([x_hat, y_hat, z_hat])

    rvecs_new, tvecs_new = adjust_world_reference_frame_to_charuco(
        camera_group=camera_group,
        charuco_origin_in_world=charuco_origin_in_world,
        rmat_charuco_to_world=rmat_charuco_to_world
    )

    camera_group.set_rotations(rvecs_new)
    camera_group.set_translations(tvecs_new)

    logger.info("Anipose camera calibration data adjusted to set charuco board as ground plane")

    # Recalculate 3D charuco data in new coordinate system
    charuco_3d_flat = camera_group.triangulate(charuco_2d_flat)
    charuco_3d_xyz = charuco_3d_flat.reshape(num_frames, num_tracked_points, 3)
    charuco_3d_xyz_interpolated = skellyforge_data(raw_charuco_data=charuco_3d_xyz)

    # Save charuco 3D data if recording folder is provided
    if recording_folder_path is not None:
        charuco_save_path = recording_folder_path / "output_data" / "charuco_3d_xyz.npy"
        charuco_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(charuco_save_path, charuco_3d_xyz_interpolated)
        logger.info(f"Charuco 3d data saved to {charuco_save_path}")

    return camera_group, GroundPlaneSuccess(success=True)


def adjust_world_reference_frame_to_charuco(
        camera_group: AniposeCameraGroup,
        charuco_origin_in_world: np.ndarray,
        rmat_charuco_to_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust camera extrinsics to use charuco board as world reference frame."""
    tvecs = camera_group.get_translations()
    rvecs = camera_group.get_rotations()

    tvecs_new = np.zeros_like(tvecs)
    rvecs_new = np.zeros_like(rvecs)

    for i in range(tvecs.shape[0]):
        rmat_world_to_cam_i, _ = cv2.Rodrigues(rvecs[i])
        t_delta = rmat_world_to_cam_i @ charuco_origin_in_world
        tvecs_new[i] = t_delta + tvecs[i]

        new_rmat = rmat_world_to_cam_i @ rmat_charuco_to_world
        new_rvec, _ = cv2.Rodrigues(new_rmat)
        rvecs_new[i] = new_rvec.flatten()

    return rvecs_new, tvecs_new


def get_real_world_matrices(camera_group: AniposeCameraGroup) -> tuple[list, list]:
    """Calculate real-world positions and orientations of cameras."""
    rvecs = camera_group.get_rotations()
    tvecs = camera_group.get_translations()

    positions = []
    orientations = []

    for i in range(tvecs.shape[0]):
        rmat_world_to_cam_i, _ = cv2.Rodrigues(rvecs[i])

        rmat_cam_to_world = rmat_world_to_cam_i.T
        t_world = -rmat_cam_to_world @ tvecs[i]

        positions.append(t_world.astype(float).tolist())
        orientations.append(rmat_cam_to_world.astype(float).tolist())

    camera_group.set_world_positions(positions)
    camera_group.set_world_orientations(orientations)

    return positions, orientations
