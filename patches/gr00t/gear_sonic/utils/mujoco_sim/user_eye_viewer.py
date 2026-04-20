#!/usr/bin/env python3
"""
user_eye_viewer.py
ZMQ :5555 から画像と robot_in_user_view フラグを受信して tkinter で表示。
P キーで自動起動。] キーで ego_view ↔ user_eye 切替（base_sim 側で切替）。
"""
import base64
import os
import sys
import time
import tkinter as tk

import cv2
import msgpack
import numpy as np
import zmq

ZMQ_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
WIN_SIZE  = 512


def decode_image(data):
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR


def to_tk_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (WIN_SIZE, WIN_SIZE), interpolation=cv2.INTER_NEAREST)
    ok, png = cv2.imencode(".png", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode image for tkinter")
    return tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))


def main():
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY is not set")

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://localhost:{ZMQ_PORT}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 100)
    sock.setsockopt(zmq.RCVHWM, 2)

    root = tk.Tk()
    root.title("camera view")
    root.configure(bg="black")
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))

    label      = tk.Label(root, bg="black")
    label.pack()
    info_var   = tk.StringVar(value="connecting...")
    info_label = tk.Label(root, textvariable=info_var,
                          anchor="w", justify="left",
                          bg="black", fg="#00ff00",
                          font=("Courier", 11))
    info_label.pack(fill="x", padx=4, pady=2)

    t_last             = [time.time()]
    fps_val            = [0.0]
    last_robot_visible = [False]

    def poll():
        try:
            raw  = sock.recv(zmq.NOBLOCK)
            data = msgpack.unpackb(raw, raw=False)
            imgs = data.get("images", {})

            # 送られてきた画像を取得（何でも表示）
            cam_name = None
            img_data = None
            for k, v in imgs.items():
                cam_name, img_data = k, v
                break

            if img_data is not None:
                img = decode_image(img_data)

                now = time.time()
                dt  = now - t_last[0]
                if dt > 0:
                    fps_val[0] = 0.9 * fps_val[0] + 0.1 / dt
                t_last[0] = now

                # robot_in_user_view は ZMQ から取得（座標ベース判定）
                zmq_visible = data.get("robot_in_user_view", None)
                if zmq_visible is not None:
                    last_robot_visible[0] = bool(zmq_visible)

                robot_visible = last_robot_visible[0]

                tk_img = to_tk_image(img)
                label.configure(image=tk_img)
                label.image = tk_img

                color = "#00ff00" if robot_visible else "#ff6600"
                info_label.configure(fg=color)
                info_var.set(
                    f"[{cam_name}]   FPS: {fps_val[0]:.1f}   "
                    f"Robot IN VIEW: {'YES' if robot_visible else 'NO'}"
                )
        except zmq.Again:
            pass
        except Exception as e:
            print(f"[viewer] {e}")

        root.after(16, poll)

    root.after(16, poll)
    root.mainloop()
    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
