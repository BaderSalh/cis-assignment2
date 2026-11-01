"""
Data-collection workflow: which robot poses to visit, how to read them, and
how to register the robot base to the tracker.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from data_types.nominal_types import vct3, Frame

from ..robot import Robot
from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody, clock, sample_marker_bodies,most_recent_observed_frame

from .registration import average_frames


@dataclass
class CalibrationData:
    """One calibration data set.

    thetas[j] is the commanded (encoder) joint vector and frames[j] the
    calibration body's observed pose relative to the tracker, F_ee,j (this
    handout's tracker has no jiggle, so the tracker frame is a fixed
    reference). t_collect is the simulated time the collection took.
    """
    thetas: List[np.ndarray] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    t_collect: float = 0.0

    def __len__(self):
        return len(self.thetas)


def measure_home_offsets(rob: Robot, n_homings: int = 5) -> np.ndarray:
    """
    TODO: implement.

    HomeRobot() drives the robot to theta* = 0 and returns the encoder
    values there, i.e. -delta_theta_sys(home) - delta_theta_rnd. Repeat it
    and average to estimate the systematic joint offsets at home (after the
    first homing the move is tiny, so each repeat costs about t_settle).

    input:
        rob - the robot (rob.home_robot())
        n_homings - how many homings to average
    output: estimated delta_theta_sys at home, np.ndarray of 6
    """
    if n_homings <= 0:
        raise ValueError("n_homings must be positive")

    readings = []

    for _ in range(n_homings):
        encoders = rob.home_robot()
        readings.append(np.asarray(encoders, dtype=float).copy())
    return -np.mean(readings, axis = 0)


# this should contain a desired accuracy which will affect the number of calibration poses
def plan_calibration_poses(
    rob: Robot, 
    trk: Tracker, 
    mbee_local_positions: List[vct3], 
    desired_accuracy: float, 
    **kwargs
) -> List[np.ndarray]:
    """
    TODO: implement -- this is a design task.

    Choose the joint vectors the robot will visit during calibration to reach the desired accuracy. Every
    parameter of your kinematic model must be excited by the data (axis
    tilts need travel along each axis; the wrist axes and F_we need wrist
    rotations; the joint distortion needs several stops along each joint),
    every pose must keep all calibration-body markers inside the tracker's
    viewing volume (Tracker.is_visible, using the nominal rob.F_rob and
    rob.fwd_kins -- they are only tens of mm off), and the order of the
    poses determines the motion time. The wrist moves at 0.1 rad/s: a pi/2
    turn costs ~16 s.

    The handout's method 1: decide how closely spaced the samples along an
    axis must be from the error bounds (delta_sys^trk, omega_sys^trk,
    delta_sys^rob, omega_sys^rob), then sweep each axis in turn from the
    home position. Joints 1-3 are decoupled from 4-6.

    input:
        rob - the Robot (nominal kinematics + limits)
        trk - the Tracker (for visibility checks)
        mbee_local_positions - the calibration body's marker layout a*_k
        desired_accuracy - target RMS orientation error in rad (the limiting
            metric); use it to size the plan -- how many distinct poses and
            how often each is read (a pose listed twice in a row is
            re-commanded and re-read, costing only t_settle + delta_t_samp)
        kwargs - whatever your design needs (grid sizes, margins, ...)
    output: ordered List of joint vectors (np.ndarray of 6)
    """
    if not np.isfinite(desired_accuracy) or desired_accuracy <= 0:
        raise ValueError("desired_accuracy must be positive and finite")

    if len(mbee_local_positions) < 3:
        raise ValueError("At least 3 calibration markers are required")

    # reference: Encoder values maintained by other joints when scanning
    # linear_step: The maximum sampling interval of the translation joint, mm
    # angular_step: The maximum sampling interval of the rotating joint, rad
    # repeats: The number of repeated executions and samplings of each configuration

    reference = np.asarray(
        kwargs.get("reference",np.zeros(6)),
        dtype = float,
    ).reshape(6)

    accuracy_ratio = min(1.0, desired_accuracy / 1e-3)

    linear_step = float(
        kwargs.get("linear_step", 100.0 * accuracy_ratio)
    )
    angular_step = float(
        kwargs.get("angular_step", 0.5 * accuracy_ratio)
    )
    repeats = kwargs.get("repeats", 3)

    if (not np.isfinite(linear_step) or linear_step <= 0 or
            not np.isfinite(angular_step) or angular_step <= 0):
        raise ValueError("Sampling steps must be positive and finite")

    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    poses = []

    def append_pose(theta):
        theta = np.asarray(theta, dtype=float).reshape(6)

        if not np.all(np.isfinite(theta)):
            raise ValueError("Joint values must be finite")

        if not rob.within_limits(theta):
            raise ValueError(f"Joint limits violated: {theta}")

        # Calibration body system -> robot base system -> workspace system
        F_world = rob.F_rob * rob.fwd_kins(theta)
        marker_points_world = [
            F_world * point
            for point in mbee_local_positions
        ]

        if not trk.is_visible(marker_points_world):
            raise ValueError(
                f"Calibration markers are not visible at theta={theta}"
            )

        if poses and np.allclose(
            poses[-1], theta, rtol=0.0, atol=1e-12
        ):
            return

        for _ in range(repeats):
            poses.append(theta.copy())

    append_pose(reference)

    for joint in range(6):
        lower = float(rob.theta_min[joint])
        upper = float(rob.theta_max[joint])

        step = linear_step if joint < 3 else angular_step
        n_intervals = max(1, int(np.ceil((upper - lower) / step)))

        values = np.linspace(lower, upper, n_intervals + 1)

        if abs(reference[joint] - upper) < abs(reference[joint] - lower):
            values = values[::-1]

        for value in values:
            theta = reference.copy()
            theta[joint] = value
            append_pose(theta)

        append_pose(reference)

    return poses
    


def collect_calibration_data(rob: Robot, trk: Tracker, mbee_placed: PlacedBody,
                             thetas: List[np.ndarray], repeats: int = 1) -> CalibrationData:
    """
    TODO: implement.

    Visit each joint vector with rob.move_joints, read the calibration body
    with sample_marker_bodies and store its frame relative to the tracker.
    Decide what to do when the body is not visible (None), and whether/how
    to use repeats (re-sampling costs 0.03 s; re-commanding the same pose
    costs t_settle and re-rolls the joint repeatability noise).

    input:
        rob, trk - robot and tracker
        mbee_placed - the robot calibration body (attached)
        thetas - joint vectors to visit, in order
        repeats - readings per pose
    output: CalibrationData
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    data = CalibrationData()
    t_start = clock.elapsed
    for theta in thetas:
        theta = np.asarray(theta, dtype=float).reshape(6)
        rob.move_joints(theta)
        for _ in range(repeats):
            sample_marker_bodies(trk)
            F_observed = most_recent_observed_frame(mbee_placed.body)

            if F_observed is None:
                raise ValueError(f"Calibration body is not visible at theta={theta}")
            data.thetas.append(theta.copy())
            data.frames.append(F_observed)

    data.t_collect = clock.elapsed - t_start
    return data
            

def estimate_base_frame(data: CalibrationData, kins: Robot) -> Frame:
    """
    TODO: implement.

    With the calibration body's frame being exactly F_ee, every sample
    satisfies  F_ee,j = F_X * kins.fwd_kins(theta_j)  for one constant
    unknown F_X (the robot base expressed in the tracker's frame). Estimate
    F_X from all samples (your average_frames may help).

    input:
        data - CalibrationData
        kins - the kinematics to use (the nominal Robot at first; later your
            ImprovedKinematics)
    output: F_X
    """
    if len(data) == 0:
        raise ValueError("Calibration data must not be empty")

    if len(data.thetas) != len(data.frames):
        raise ValueError("Joint vectors and observed frames must match")

    base_estimates = []

    for theta, F_observed in zip(data.thetas, data.frames):
        F_ee = kins.fwd_kins(theta)

        F_X = F_observed * F_ee.inv()
        base_estimates.append(F_X)

    return average_frames(base_estimates)
