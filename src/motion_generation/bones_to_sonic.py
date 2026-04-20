#!/usr/bin/env python3
"""
BONES-SEED G1 CSV → GEAR-SONIC ZMQ 送信スクリプト

使い方:
  python bones_to_sonic.py --motion casual_greeting
  python bones_to_sonic.py --csv /path/to/file.csv
  python bones_to_sonic.py --list-motions

動作カテゴリ例:
  挥手系:   casual_greeting, bye_bye_salute, come_here
  ジェスチャー: dont_come_any_closer_one_hand, beckon
  その他:   wave_right_hand (ファイル名に含まれる)

事前準備:
  Terminal 1: source .venv_sim/bin/activate
              python gear_sonic/scripts/run_sim_loop.py
  Terminal 2: cd gear_sonic_deploy
              bash deploy.sh --input-type zmq --zmq-port 5556 sim
              → ] キーで起動 → ENTER で ZMQ モード有効
"""

import argparse
import glob
import json
import os
import struct
import time

import numpy as np
import pandas as pd
import zmq
from scipy.interpolate import interp1d

# BONES-SEED G1 CSV のルートディレクトリ
BONES_ROOT = os.environ.get("BONES_ROOT", "/home/unitree-g1/Documents/G1/g1/csv")

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


def load_bones_csv(csv_path):
    """BONES-SEED G1 CSV を読み込み IsaacLab 順序に変換"""
    df = pd.read_csv(csv_path)
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    if len(joint_cols) != 29:
        raise ValueError(f"関節数が29ではありません: {len(joint_cols)}")
    jp = (df[joint_cols].values * np.pi / 180.0).astype(np.float32)
    return jp[:, MUJOCO_TO_ISAACLAB]


def find_motion(keyword):
    """キーワードで動作 CSV を検索（ミラーファイル除外）"""
    pattern = os.path.join(BONES_ROOT, "**", f"*{keyword}*.csv")
    files = glob.glob(pattern, recursive=True)
    # _M. で終わるミラーファイルを除外
    files = [f for f in files if "_M." not in os.path.basename(f)]
    return sorted(files)


def list_motions(keyword=""):
    """利用可能な動作を一覧表示"""
    pattern = os.path.join(BONES_ROOT, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    files = [f for f in files if "_M." not in os.path.basename(f)]

    names = set()
    for f in files:
        base = os.path.basename(f)
        # _A123 以降を除去して動作名を取得
        import re
        name = re.sub(r'_[A-Z]\d+\.csv$', '', base)
        name = re.sub(r'_R_\d+_$', '', name)
        if keyword.lower() in name.lower():
            names.add(name)

    for name in sorted(names):
        print(name)


def main():
    parser = argparse.ArgumentParser(
        description="BONES-SEED G1 CSV を GEAR-SONIC に送信"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--motion",       help="動作名キーワード (例: casual_greeting, bye_bye_salute)")
    group.add_argument("--csv",          help="CSV ファイルの直接パス")
    group.add_argument("--list-motions", action="store_true", help="利用可能な動作を一覧表示")

    parser.add_argument("--search",    default="", help="--list-motions のフィルターキーワード")
    parser.add_argument("--play-fps",  type=float, default=40,
                        help="再生速度 fps (default: 40)")
    parser.add_argument("--repeat",    type=int,   default=3,
                        help="繰り返し回数 (default: 3)")
    parser.add_argument("--port",      type=int,   default=5556,
                        help="ZMQ ポート (default: 5556)")
    parser.add_argument("--chunk",     type=int,   default=5,
                        help="送信チャンクサイズ (default: 5)")
    args = parser.parse_args()

    # 一覧表示モード
    if args.list_motions:
        print(f"[BONES-SEED] 動作一覧 (キーワード: '{args.search}'):")
        list_motions(args.search)
        return

    # CSV パスを決定
    if args.csv:
        csv_path = args.csv
    elif args.motion:
        files = find_motion(args.motion)
        if not files:
            print(f"[エラー] '{args.motion}' に一致する動作が見つかりません")
            print(f"  python bones_to_sonic.py --list-motions --search {args.motion}")
            return
        csv_path = files[0]
        print(f"[Find] {len(files)} 件ヒット → {os.path.basename(csv_path)} を使用")
    else:
        parser.print_help()
        return

    # CSV 読み込み
    print(f"[Load] {csv_path}")
    jp_orig = load_bones_csv(csv_path)
    T_orig  = len(jp_orig)
    print(f"[Load] {T_orig} フレーム ({T_orig/120:.1f}秒 @ 120fps)")

    # SONIC 50fps に補間（BONES-SEED は 120fps）
    BONES_FPS = 120
    t_orig  = np.linspace(0, T_orig / BONES_FPS * (BONES_FPS / args.play_fps), T_orig)
    t_new   = np.arange(0, t_orig[-1], 1.0 / SONIC_FPS)
    jp      = interp1d(t_orig, jp_orig, axis=0, kind="linear",
                       fill_value="extrapolate")(t_new).astype(np.float32)
    T       = len(jp)

    jv      = np.zeros_like(jp)
    jv[:-1] = (jp[1:] - jp[:-1]) * SONIC_FPS
    jv[-1]  = jv[-2]
    bq      = np.tile([1., 0., 0., 0.], (T, 1)).astype(np.float32)

    print(f"[Play] {T/SONIC_FPS:.1f}秒/回 × {args.repeat}回")

    # ZMQ 接続
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[ZMQ] tcp://*:{args.port} 接続待機 (1秒)...")
    time.sleep(1.0)

    # 送信
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
