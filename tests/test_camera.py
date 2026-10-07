from __future__ import annotations

import numpy as np

from nuscenes_fusion.camera import CameraCalibration, nearest_return_per_pixel, project_points


def test_project_points_keeps_only_in_front_and_in_image() -> None:
    # Camera looks along lidar +x; lidar y-left/z-up become camera x-right/y-down.
    to_camera = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    projection = np.array([[100, 0, 50, 0], [0, 100, 40, 0], [0, 0, 1, 0]], dtype=float)
    points = np.array(((10.0, 0.0, 0.0), (10.0, 1.0, 0.5), (-10.0, 0.0, 0.0), (10.0, -9.0, 0.0)))

    pixels, depths, visible = project_points(points, CameraCalibration(projection, to_camera), 100, 80)

    np.testing.assert_array_equal(visible, (True, True, False, False))
    np.testing.assert_array_equal(pixels, ((50, 40), (40, 35)))
    np.testing.assert_allclose(depths, (10.0, 10.0))


def test_nearest_return_per_pixel_keeps_closest_point() -> None:
    pixels = np.array(((5, 5), (5, 5), (6, 5)))
    depths = np.array((9.0, 3.0, 4.0))

    assert sorted(nearest_return_per_pixel(pixels, depths).tolist()) == [1, 2]
