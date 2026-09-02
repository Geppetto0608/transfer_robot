import math

import numpy as np
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from transfer_robot.pointcloud import read_xyz_points


def test_read_xyz_from_mixed_livox_fields():
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="tag", offset=16, datatype=PointField.UINT8, count=1),
        PointField(name="line", offset=17, datatype=PointField.UINT8, count=1),
        PointField(name="timestamp", offset=18, datatype=PointField.FLOAT64, count=1),
    ]
    message = point_cloud2.create_cloud(
        Header(frame_id="base_link"),
        fields,
        [
            (1.0, -2.0, 0.7, 42.0, 3, 1, 100.25),
            (math.nan, 0.0, 0.8, 15.0, 2, 4, 101.25),
            (-0.5, 0.3, 0.9, 21.0, 1, 2, 102.25),
        ],
    )

    xyz = read_xyz_points(message)

    assert xyz.dtype == np.float32
    assert xyz.shape == (2, 3)
    assert np.allclose(xyz, [[1.0, -2.0, 0.7], [-0.5, 0.3, 0.9]])
