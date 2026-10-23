"""
Rigid registration helpers: point-set registration, frame averaging and
marker-body refinement.
"""

from typing import List

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody


def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    TODO: implement (your own -- do not call the simulator's solver).

    Rigid point-set registration: find the least-squares Frame F such that
    F * a[i] ~= b[i] for every corresponding pair. Think about what happens
    with fewer than 3 points, with collinear points, and with a reflection.

    input:
        a - points in the source frame (e.g. a marker body's local layout)
        b - the same points observed in the target frame, same order
    output: Frame mapping a -> b
    """
    
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("At least 3 paired points are required")

    A = np.column_stack([point.vec.ravel() for point in a])
    B = np.column_stack([point.vec.ravel() for point in b])

    a_centroid = np.mean(A, axis=1, keepdims=True)
    b_centroid = np.mean(B, axis=1, keepdims=True)

    a_prime = A - a_centroid
    b_prime = B - b_centroid

    if (np.linalg.matrix_rank(a_prime) < 2 or
            np.linalg.matrix_rank(b_prime) < 2):
        raise ValueError("Points must not be collinear")

    H = a_prime @ b_prime.T

    U, S, V_transpose = np.linalg.svd(H)
    V = V_transpose.T

    D = np.eye(3)
    D[2, 2] = np.linalg.det(V @ U.T)

    R = V @ D @ U.T
    p = b_centroid - R @ a_centroid

    return Frame(Rot(matrix=R), vct3(p.ravel()))


def average_frames(frames: List[Frame]) -> Frame:
    """
    TODO: implement.

    Average of several noisy estimates of the same Frame. Rotations cannot
    simply be averaged element-wise -- the result must be a proper
    rotation matrix.

    input:
        frames - List[Frame], all estimates of the same transform
    output: the averaged Frame
    """
    if frames is None or len(frames) == 0:
        raise ValueError("frames must not be empty")

    positions = np.array([frame.p.vec.ravel() for frame in frames])
    p_avg = np.mean(positions, axis=0)

    rotations = np.array([frame.R.matrix for frame in frames])
    M = np.mean(rotations, axis=0)

    U, S, V_transpose = np.linalg.svd(M)

    D = np.eye(3)
    D[2, 2] = np.linalg.det(U @ V_transpose)
    R_avg = U @ D @ V_transpose

    return Frame(Rot(matrix=R_avg), vct3(p_avg))


def refine_marker_body(trk: Tracker, placed: PlacedBody, n_readings: int = 30) -> List[vct3]:
    """
    TODO: implement.

    "It is possible to improve your estimate of a_m,k through a calibration
    process involving multiple observations of the marker body using the
    tracking camera and then averaging them."

    Take n_readings SampleMarkerBodies readings, use
    most_recent_observed_markers(...) plus your register_points to bring
    each reading's b_m,k back into the body's own frame, average, and push
    the result into the body with update_marker_body(...). Note that each
    reading costs delta_t_samp of simulated time, and that the tracker
    jiggles between readings.

    input:
        trk - Tracker the body is tracked by
        placed - PlacedBody whose body's nominal marker positions should be refined
        n_readings - number of readings to average
    output: the new List[vct3] of marker positions (also stored on the body)
    """
    """ Skipped due to the fact that we have the exact marker readings on the end-effector"""