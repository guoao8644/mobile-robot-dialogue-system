#!/usr/bin/env python3
"""
g1_walk.py — キーボードで G1 を歩かせる + BONES-SEED 動作再生

起動手順:
  Terminal 1: python gear_sonic/scripts/run_sim_loop.py
  Terminal 2: bash deploy.sh --input-type zmq_manager sim
  Terminal 3: python g1_walk.py

操作:
  W   前進
  S   後退
  A   左旋回
  D   右旋回
  Q   左ストレイフ
  E   右ストレイフ
  B   BONES-SEED 動作再生（wave_right_hand）
  スペース  停止（IDLE）
  X   終了
"""

import glob
import json
import os
import struct
import sys
import termios
import threading
import time
import tty

import numpy as np
import zmq
from scipy.interpolate import interp1d

# ─── 設定 ────────────────────────────────────────
PORT        = 5556
HEADER_SIZE = 1280
SONIC_FPS   = 50
BONES_ROOT = os.environ.get("BONES_ROOT", "/home/unitree-g1/Documents/G1/g1/csv")
MUJOCO_TO_ISAACLAB = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15,
    22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25,
    19, 26, 20, 27, 21, 28
]
# B キーで再生する動作キーワード
MOTION_KEYWORD = "wave_right_hand_360_R_turn_jog_270_R_walk_ff_loop_315"

# ─── ZMQ ─────────────────────────────────────────
ctx  = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind(f"tcp://*:{PORT}")
print(f"[ZMQ] tcp://*:{PORT}")
time.sleep(0.5)

# ─── メッセージ送信 ───────────────────────────────
def send_msg(topic, fields, data):
    header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
    hj = json.dumps(header).encode()
    hb = hj + b"\x00" * (HEADER_SIZE - len(hj))
    sock.send(topic + hb + data)

def send_command(start=True, stop=False, planner=True):
    fields = [
        {"name": "start",   "dtype": "u8", "shape": [1]},
        {"name": "stop",    "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    data = struct.pack("BBB", int(start), int(stop), int(planner))
    send_msg(b"command", fields, data)

def send_planner(mode, movement, facing, speed=-1.0):
    fields = [
        {"name": "mode",     "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing",   "dtype": "f32", "shape": [3]},
        {"name": "speed",    "dtype": "f32", "shape": [1]},
        {"name": "height",   "dtype": "f32", "shape": [1]},
    ]
    data  = struct.pack("<i", mode)
    data += struct.pack("<fff", *movement)
    data += struct.pack("<fff", *facing)
    data += struct.pack("<f", speed)
    data += struct.pack("<f", -1.0)
    send_msg(b"planner", fields, data)

def send_pose_frames(joint_pos, joint_vel, body_quat, frame_index):
    N = len(joint_pos)
    header = {
        "v": 1, "endian": "le", "count": N,
        "fields": [
            {"name": "joint_pos",   "dtype": "f32", "shape": [N, 29]},
            {"name": "joint_vel",   "dtype": "f32", "shape": [N, 29]},
            {"name": "body_quat_w", "dtype": "f32", "shape": [N, 4]},
            {"name": "frame_index", "dtype": "i64", "shape": [N]},
            {"name": "catch_up",    "dtype": "u8",  "shape": [1]},
        ]
    }
    hj = json.dumps(header).encode()
    hb = hj + b"\x00" * (HEADER_SIZE - len(hj))
    fi = np.arange(frame_index, frame_index + N, dtype=np.int64)
    data = (joint_pos.tobytes() + joint_vel.tobytes() +
            body_quat.tobytes() + fi.tobytes() + struct.pack("B", 0))
    sock.send(b"pose" + hb + data)

# ─── BONES-SEED ───────────────────────────────────
def load_motion(keyword, play_fps=40):
    import pandas as pd
    pattern = os.path.join(BONES_ROOT, "**", f"*{keyword}*.csv")
    files = [f for f in glob.glob(pattern, recursive=True)
             if "_M." not in os.path.basename(f)]
    if not files:
        print(f"[BONES] '{keyword}' が見つかりません")
        return None, None, None
    csv_path = sorted(files)[0]
    print(f"[BONES] {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    jp_orig = (df[joint_cols].values * np.pi / 180.0).astype(np.float32)
    jp_orig = jp_orig[:, MUJOCO_TO_ISAACLAB]
    T_orig = len(jp_orig)
    t_orig = np.linspace(0, T_orig / 120.0 * (120.0 / play_fps), T_orig)
    t_new  = np.arange(0, t_orig[-1], 1.0 / SONIC_FPS)
    jp = interp1d(t_orig, jp_orig, axis=0, kind="linear",
                  fill_value="extrapolate")(t_new).astype(np.float32)
    jv = np.zeros_like(jp)
    jv[:-1] = (jp[1:] - jp[:-1]) * SONIC_FPS
    jv[-1]  = jv[-2]
    bq = np.tile([1., 0., 0., 0.], (len(jp), 1)).astype(np.float32)
    return jp, jv, bq

# 動作データを事前ロード
print(f"[BONES] '{MOTION_KEYWORD}' をロード中...")
_jp, _jv, _bq = load_motion(MOTION_KEYWORD)
if _jp is not None:
    print(f"[BONES] {len(_jp)/SONIC_FPS:.1f}秒 ロード完了")

_motion_playing = False
_motion_lock    = threading.Lock()
_frame_counter  = 0

_motion_stop = threading.Event()

def play_motion_thread():
    global _motion_playing, _frame_counter
    if _jp is None:
        print("[BONES] 動作データなし")
        return
    with _motion_lock:
        if _motion_playing:
            print("[BONES] 再生中のためスキップ")
            return
        _motion_playing = True
        _motion_stop.clear()

    print("[BONES] STREAMED MOTION モードに切替...")
    send_command(start=True, stop=False, planner=False)
    time.sleep(0.8)

    print("[BONES] 再生開始")
    CHUNK = 5
    T = len(_jp)
    fi = _frame_counter
    try:
        for i in range(0, T, CHUNK):
            if _motion_stop.is_set():
                print("[BONES] 中断")
                break
            n  = min(CHUNK, T - i)
            t0 = time.perf_counter()
            send_pose_frames(_jp[i:i+n], _jv[i:i+n], _bq[i:i+n], fi)
            fi += n
            wait = n / SONIC_FPS - (time.perf_counter() - t0)
            if wait > 0:
                _motion_stop.wait(timeout=wait)
        _frame_counter = fi
    finally:
        _motion_playing = False

    print("[BONES] 再生完了 → planner モードに戻す")
    time.sleep(0.2)
    send_command(start=True, stop=False, planner=True)
    time.sleep(0.5)

def stop_motion_and_switch_planner():
    """BONES 再生を中断して planner モードに即座に切替"""
    if _motion_playing:
        _motion_stop.set()
        time.sleep(0.1)
    send_command(start=True, stop=False, planner=True)
    time.sleep(0.3)

# ─── キーボード ───────────────────────────────────
def get_key():
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

facing_angle = 0.0
TURN_STEP    = np.radians(10)

def facing_vec(a):
    return [np.cos(a), np.sin(a), 0.0]

# ─── 起動 ─────────────────────────────────────────
print("\n[起動] command: start=True, planner=True を送信...")
send_command(start=True, stop=False, planner=True)
time.sleep(1.5)
send_planner(0, [0, 0, 0], facing_vec(facing_angle))
print("[起動] 完了 — IDLE 状態で待機中")

print("""
操作キー:
  W = 前進      S = 後退
  A = 左旋回    D = 右旋回
  Q = 左横移動  E = 右横移動
  B = BONES-SEED 動作再生
  スペース = 停止
  X = 終了
""")

try:
    while True:
        key = get_key()

        if key in ("x", "\x03"):
            break

        elif key == "w":
            stop_motion_and_switch_planner()
            mv = facing_vec(facing_angle)
            print(f"[W] 前進  facing={np.degrees(facing_angle):.0f}°")
            send_planner(2, mv, mv)
            time.sleep(0.3)
            send_planner(0, [0,0,0], facing_vec(facing_angle))

        elif key == "s":
            stop_motion_and_switch_planner()
            mv = [-np.cos(facing_angle), -np.sin(facing_angle), 0.0]
            print(f"[S] 後退  facing={np.degrees(facing_angle):.0f}°")
            send_planner(2, mv, facing_vec(facing_angle))
            time.sleep(0.3)
            send_planner(0, [0,0,0], facing_vec(facing_angle))

        elif key == "a":
            stop_motion_and_switch_planner()
            facing_angle += TURN_STEP
            fv = facing_vec(facing_angle)
            print(f"[A] 左旋回 → facing={np.degrees(facing_angle):.0f}°")
            for _ in range(5):
                send_planner(2, [0, 0, 0], fv)
                time.sleep(0.1)

        elif key == "d":
            stop_motion_and_switch_planner()
            facing_angle -= TURN_STEP
            fv = facing_vec(facing_angle)
            print(f"[D] 右旋回 → facing={np.degrees(facing_angle):.0f}°")
            for _ in range(5):
                send_planner(2, [0, 0, 0], fv)
                time.sleep(0.1)

        elif key == "q":
            stop_motion_and_switch_planner()
            left = [np.cos(facing_angle + np.pi/2),
                    np.sin(facing_angle + np.pi/2), 0.0]
            print("[Q] 左横移動")
            for _ in range(5):
                send_planner(2, left, facing_vec(facing_angle))
                time.sleep(0.1)

        elif key == "e":
            stop_motion_and_switch_planner()
            right = [np.cos(facing_angle - np.pi/2),
                     np.sin(facing_angle - np.pi/2), 0.0]
            print("[E] 右横移動")
            for _ in range(5):
                send_planner(2, right, facing_vec(facing_angle))
                time.sleep(0.1)

        elif key == "b":
            print("[B] BONES-SEED 動作再生")
            t = threading.Thread(target=play_motion_thread, daemon=True)
            t.start()

        elif key == " ":
            print("[スペース] 停止")
            for _ in range(3):
                send_planner(0, [0, 0, 0], facing_vec(facing_angle))
                time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    print("\n[終了] IDLE に戻す...")
    send_planner(0, [0, 0, 0], facing_vec(facing_angle))
    time.sleep(0.3)
    send_command(start=False, stop=False, planner=True)
    sock.close()
    ctx.term()
    print("終了")
