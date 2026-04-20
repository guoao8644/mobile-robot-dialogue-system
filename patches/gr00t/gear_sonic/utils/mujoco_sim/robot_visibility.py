"""
robot_visibility.py
user_eye カメラの視野内に G1 ロボットが見えるかどうかを判定するユーティリティ。

使い方:
    from gear_sonic.utils.mujoco_sim.robot_visibility import check_robot_in_user_view
    visible = check_robot_in_user_view(mj_model, mj_data)
"""

import numpy as np
import mujoco


# チェックする G1 の体の部位
ROBOT_CHECK_BODIES = ["pelvis", "torso_link"]

# user_eye カメラ名
USER_EYE_CAM = "user_eye"


def check_robot_in_user_view(mj_model, mj_data) -> bool:
    """
    user_eye カメラの視野内に G1 が見えるか判定する。

    アルゴリズム:
    1. G1 の body（pelvis, torso_link）の world 座標を取得
    2. user_eye カメラの FOV 内にあるかチェック
    3. mj_ray で遮蔽物チェック（手前に壁などがある場合は False）

    Returns:
        True: G1 が視野内に見えている
        False: 見えていない（視野外 or 遮蔽）
    """
    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, USER_EYE_CAM)
    if cam_id < 0:
        return False

    # カメラの位置と回転行列
    cam_pos = mj_data.cam_xpos[cam_id].copy()
    cam_mat = mj_data.cam_xmat[cam_id].reshape(3, 3).copy()
    cam_forward = -cam_mat[:, 2]  # MuJoCo カメラは -Z が視線方向
    cam_right   =  cam_mat[:, 0]
    cam_up      =  cam_mat[:, 1]

    # FOV の半角（ラジアン）
    fovy = mj_model.cam_fovy[cam_id]
    half_fov = np.radians(fovy / 2)

    for body_name in ROBOT_CHECK_BODIES:
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue

        target_pos = mj_data.xpos[body_id].copy()
        diff = target_pos - cam_pos
        dist = np.linalg.norm(diff)
        if dist < 0.01:
            continue

        # カメラ座標系への投影
        fwd_proj   = np.dot(diff, cam_forward)
        right_proj = np.dot(diff, cam_right)
        up_proj    = np.dot(diff, cam_up)

        # カメラの後ろにある
        if fwd_proj <= 0:
            continue

        # 水平・垂直の角度チェック
        angle_h = np.arctan2(abs(right_proj), fwd_proj)
        angle_v = np.arctan2(abs(up_proj), fwd_proj)
        if angle_h > half_fov or angle_v > half_fov:
            continue

        # FOV 内 → raycast で遮蔽チェック
        direction = diff / dist
        geom_id = np.array([-1], dtype=np.int32)

        # bodyexclude に user body を指定して除外
        user_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "user")

        hit_dist = mujoco.mj_ray(
            mj_model, mj_data,
            cam_pos, direction,
            None, 1, user_body_id, geom_id
        )

        # hit_dist < 0: 何も当たらない（見えている）
        # hit_dist >= dist*0.9: G1 より手前で当たっていない → 見えている
        if hit_dist < 0 or hit_dist >= dist * 0.9:
            return True

    return False
