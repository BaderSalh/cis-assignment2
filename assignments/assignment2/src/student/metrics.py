"""
Pose-error metrics used to report calibration accuracy.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from data_types.nominal_types import Frame


def frame_error(F_a: Frame, F_b: Frame) -> Tuple[float, float]:
    """
    TODO: implement.

    The mismatch between two frames is defined as
    dF = F_a^-1 F_b ~= Frame(I + sk(alpha), epsilon); report its size.

    input:
        F_a - reference Frame (e.g. ground truth)
        F_b - Frame to compare (e.g. your prediction)
    output:
        (angle_rad, dist_mm) - ||alpha|| (the rotation angle of dF.R) and
            ||epsilon|| (the norm of dF.p)
    """
    dF = F_a.inv() * F_b

    alpha = log_so3(dF.R.matrix)
    angle_rad = float(np.linalg.norm(alpha))
    dist_mm = float(dF.p.norm())

    return angle_rad, dist_mm


def summarize_errors(errors: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """
    TODO: implement.
    
    input:
        errors - list of (angle_rad, dist_mm) pairs from frame_error
    output:
        dict with at least the keys
            n, rms_angle, max_angle, rms_pos, max_pos
    """
    if len(errors) == 0:
        raise ValueError("errors must contain at least one pair")

    values = np.asarray(errors, dtype=float)
    angle_rad = values[:, 0]
    dist_mm = values[:, 1]

    return {
        "n": len(values),
        "rms_angle": float(np.sqrt(np.mean(angle_rad ** 2))),
        "max_angle": float(np.max(angle_rad)),
        "rms_pos": float(np.sqrt(np.mean(dist_mm ** 2))),
        "max_pos": float(np.max(dist_mm)),
    }
    
