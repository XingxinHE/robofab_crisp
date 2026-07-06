"""Offline smoke-test for N1.7 DROID helper functions.

Run with:
    cd robofab_crisp
    ./.pixi/envs/default/bin/python -m deployment.gr00t.test_1p7_helpers
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)


def compute_eef_9d(cartesian_position: np.ndarray) -> np.ndarray:
    c = np.asarray(cartesian_position, dtype=np.float64).reshape(6)
    xyz = c[:3]
    euler = c[3:6]
    rot_robot = Rotation.from_euler("XYZ", euler).as_matrix()
    rot_mat = rot_robot @ DROID_EEF_ROTATION_CORRECT
    rot6d = rot_mat[:2, :].reshape(6)
    return np.concatenate([xyz, rot6d]).astype(np.float64)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    rot6d_2d = rot6d.reshape(2, 3)
    row1 = rot6d_2d[0]
    row2 = rot6d_2d[1]
    row1 = row1 / np.linalg.norm(row1)
    row2 = row2 - np.dot(row1, row2) * row1
    row2 = row2 / np.linalg.norm(row2)
    row3 = np.cross(row1, row2)
    return np.vstack([row1, row2, row3])


def _eef9d_to_homogeneous(eef_9d: np.ndarray) -> np.ndarray:
    eef_9d = np.asarray(eef_9d, dtype=np.float64).reshape(9)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot6d_to_matrix(eef_9d[3:])
    T[:3, 3] = eef_9d[:3]
    return T


def main() -> int:
    print("Testing compute_eef_9d round-trip...")
    cartesian = np.array([0.4, -0.1, 0.5, 0.1, 0.2, 0.3], dtype=np.float64)
    eef_9d = compute_eef_9d(cartesian)
    assert eef_9d.shape == (9,), f"Expected shape (9,), got {eef_9d.shape}"
    assert np.isfinite(eef_9d).all()
    print("  eef_9d:", eef_9d)

    print("Testing rot6d_to_matrix orthogonality...")
    R = rot6d_to_matrix(eef_9d[3:])
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)
    print("  det(R):", np.linalg.det(R))

    print("Testing relative transform (same pose -> identity)...")
    T_current = _eef9d_to_homogeneous(eef_9d)
    T_target = _eef9d_to_homogeneous(eef_9d)
    T_delta = np.linalg.inv(T_current) @ T_target
    np.testing.assert_allclose(T_delta, np.eye(4), atol=1e-6)
    print("  T_delta is identity as expected")

    print("Testing relative transform magnitude for 1cm world translation...")
    target_cartesian = cartesian.copy()
    target_cartesian[:3] += np.array([0.01, 0.0, 0.0])
    T_delta = np.linalg.inv(_eef9d_to_homogeneous(eef_9d)) @ _eef9d_to_homogeneous(
        compute_eef_9d(target_cartesian)
    )
    # The delta is expressed in the current EE frame, so the vector is rotated,
    # but its magnitude must equal the world translation magnitude.
    np.testing.assert_allclose(np.linalg.norm(T_delta[:3, 3]), 0.01, atol=1e-6)
    print("  |position_delta|:", np.linalg.norm(T_delta[:3, 3]))
    print("  position_delta (EE frame):", T_delta[:3, 3])

    print("Testing pure rotation relative transform...")
    target_cartesian = cartesian.copy()
    target_cartesian[3] += 0.05  # small Euler-X change
    T_delta = np.linalg.inv(_eef9d_to_homogeneous(eef_9d)) @ _eef9d_to_homogeneous(
        compute_eef_9d(target_cartesian)
    )
    np.testing.assert_allclose(T_delta[:3, 3], 0.0, atol=1e-6)
    angle = Rotation.from_matrix(T_delta[:3, :3]).magnitude()
    np.testing.assert_allclose(angle, 0.05, atol=1e-3)
    print("  rotation angle:", angle)

    print("Testing image shape handling...")
    image_chw = np.zeros((3, 256, 256), dtype=np.uint8)
    # simulate _ensure_hwc_uint8 logic
    if image_chw.shape[0] in {1, 3, 4} and image_chw.shape[-1] not in {1, 3, 4}:
        image_hwc = np.moveaxis(image_chw, 0, -1)
    assert image_hwc.shape == (256, 256, 3)
    print("  CHW -> HWC ok")

    print("\nAll offline helper tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
