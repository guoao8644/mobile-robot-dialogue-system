#!/usr/bin/env python3
"""
Kimodo CSV → GEAR-SONIC ZMQ 送信スクリプト

使い方:
  python kimodo_to_sonic.py /tmp/wave.csv

オプション:
  --play-fps   再生速度 (default: 30, 元の50fpsより遅め = ゆっくり再生)
  --repeat     繰り返し回数 (default: 3)
  --port       ZMQ ポート (default: 5556)

BONES-SEED CSV の場合:
  python kimodo_to_sonic.py /path/to/bones.csv --bones-seed

事前準備:
  Terminal 1: source .venv_sim/bin/activate
              python gear_sonic/scripts/run_sim_loop.py
  Terminal 2: cd gear_sonic_deploy
              bash deploy.sh --input-type zmq --zmq-port 5556 sim
              → ] キーで起動
              → ENTER で ZMQ モード有効
"""

import argparse
import json
import struct
import time

import numpy as np
import zmq
from scipy.interpolate import interp1d

# MuJoCo 順序 → IsaacLab 順序（SONIC が要求する順序）
MUJOCO_TO_ISAACLAB = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15,
    22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25,
    19, 26, 20, 27, 21, 28
]

HEADER_SIZE = 1280
SONIC_FPS   = 50


def send_pose(sock, joint_pos, joint_vel, body_quat, frame_index):
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


def load_kimodo_csv(csv_path):
    """Kimodo G1 CSV (qpos 形式) → IsaacLab 順序"""
    qpos = np.loadtxt(csv_path, delimiter=",")
    jp = qpos[:, 7:36].astype(np.float32)
    return jp[:, MUJOCO_TO_ISAACLAB]


def load_bones_seed_csv(csv_path):
    """BONES-SEED G1 CSV (degrees) → IsaacLab 順序"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    jp = (df[joint_cols].values * np.pi / 180.0).astype(np.float32)
    return jp[:, MUJOCO_TO_ISAACLAB]


def main():
    parser = argparse.ArgumentParser(description="Kimodo CSV を GEAR-SONIC に送信")
    parser.add_argument("csv",          help="Kimodo CSV ファイルのパス")
    parser.add_argument("--play-fps",   type=float, default=30,
                        help="再生速度 fps (default: 30)")
    parser.add_argument("--repeat",     type=int,   default=3,
                        help="繰り返し回数 (default: 3)")
    parser.add_argument("--port",       type=int,   default=5556,
                        help="ZMQ ポート (default: 5556)")
    parser.add_argument("--chunk",      type=int,   default=5,
                        help="送信チャンクサイズ (default: 5)")
    parser.add_argument("--bones-seed", action="store_true",
                        help="BONES-SEED CSV モード")
    args = parser.parse_args()

    print(f"[Load] {args.csv}")
    if args.bones_seed:
        jp_orig = load_bones_seed_csv(args.csv)
        print("[Load] BONES-SEED モード")
    else:
        jp_orig = load_kimodo_csv(args.csv)
        print("[Load] Kimodo モード")

    T_orig = len(jp_orig)
    print(f"[Load] {T_orig} フレーム ({T_orig/50:.1f}秒 @ 50fps)")

    # 再生速度に合わせて補間
    t_orig  = np.linspace(0, T_orig / args.play_fps, T_orig)
    t_new   = np.arange(0, t_orig[-1], 1.0 / SONIC_FPS)
    jp      = interp1d(t_orig, jp_orig, axis=0, kind="linear",
                       fill_value="extrapolate")(t_new).astype(np.float32)
    T       = len(jp)

    jv      = np.zeros_like(jp)
    jv[:-1] = (jp[1:] - jp[:-1]) * SONIC_FPS
    jv[-1]  = jv[-2]
    bq      = np.tile([1., 0., 0., 0.], (T, 1)).astype(np.float32)

    print(f"[Play] {T/SONIC_FPS:.1f}秒/回 × {args.repeat}回")

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[ZMQ] tcp://*:{args.port} 接続待機 (1秒)...")
    time.sleep(1.0)

    fi = 0
    try:
        for r in range(args.repeat):
            print(f"  送信 {r+1}/{args.repeat} ...", flush=True)
            for i in range(0, T, args.chunk):
                n   = min(args.chunk, T - i)
                t0  = time.perf_counter()
                send_pose(sock, jp[i:i+n], jv[i:i+n], bq[i:i+n], fi)
                fi += n
                wait = n / SONIC_FPS - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        sock.close()
        ctx.term()
        print("[完成]")


if __name__ == "__main__":
    main()
