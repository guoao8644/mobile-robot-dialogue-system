#!/usr/bin/env python3
"""
Kimodo テキスト → G1 動作生成スクリプト

使い方:
  /home/unitree-g1/anaconda3/bin/python kimodo_generate.py \
      --text "wave right hand to greet someone" \
      --output /tmp/wave \
      --duration 4.0

生成されるファイル:
  /tmp/wave.csv  ← SONIC に送信するファイル
  /tmp/wave.npz  ← 可視化用
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Kimodo でテキストから G1 動作を生成")
    parser.add_argument("--text",     required=True, help="動作を説明するテキスト")
    parser.add_argument("--output",   default="/tmp/motion", help="出力ファイルのパス（拡張子なし）")
    parser.add_argument("--duration", type=float, default=4.0, help="動作の長さ（秒）default: 4.0")
    parser.add_argument("--steps",    type=int,   default=100, help="拡散ステップ数 default: 100（少ないと速いが品質低下）")
    args = parser.parse_args()

    kimodo_python = "/home/unitree-g1/anaconda3/bin/python"
    kimodo_script = "/home/unitree-g1/Documents/G1/kimodo/kimodo/scripts/generate.py"

    cmd = [
        kimodo_python, kimodo_script,
        args.text,
        "--model", "g1",
        "--duration", str(args.duration),
        "--diffusion_steps", str(args.steps),
        "--output", args.output,
    ]

    print(f"[Kimodo] テキスト: {args.text}")
    print(f"[Kimodo] 長さ: {args.duration}秒")
    print(f"[Kimodo] 出力: {args.output}.csv")
    print(f"[Kimodo] 生成中... (CPU モード、約1〜2分)")
    print()

    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # CPU モード（GPU メモリ節約）

    result = subprocess.run(cmd, env=env)

    if result.returncode == 0:
        print()
        print(f"[完成] {args.output}.csv が生成されました")
        print(f"[次のステップ] python kimodo_to_sonic.py {args.output}.csv")
    else:
        print("[エラー] 生成に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()
