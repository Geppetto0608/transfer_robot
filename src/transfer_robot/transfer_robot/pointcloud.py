from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def read_xyz_points(message: PointCloud2) -> np.ndarray:
    """Return contiguous float32 XYZ values from a mixed-field PointCloud2.

    Livox merge messages contain FLOAT32 XYZ/intensity, UINT8 tag/line, and a
    FLOAT64 timestamp. Humble's ``read_points_numpy`` rejects such a message
    before applying its field selection, so read the structured array first
    and explicitly combine only XYZ.
    """
    structured = point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=True
    )
    if len(structured) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.column_stack(
        (structured["x"], structured["y"], structured["z"])
    ).astype(np.float32, copy=False)
