#!/usr/bin/env python3
"""
g1_realtime_dialogue.py — G1 ロボット リアルタイム音声対話 + BONES-SEED 動作制御
OpenAI Realtime API (日本語) + BONES-SEED CSV → SONIC ZMQ

動作送信は bones_to_sonic.py と完全同一ロジック（scipy.interpolate 使用）
"""

import argparse
import asyncio
import base64
import glob
import json
import os
import signal
import struct
import sys
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd
import zmq
from scipy.interpolate import interp1d

# ── 設定 ──────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

ZMQ_PORT    = 5556
HEADER_SIZE = 1280
BONES_FPS   = 120
SONIC_FPS   = 50
PLAY_FPS    = 120   # 元の速度（120fps）で再生
CHUNK_SIZE  = 5
WALK_STEP_DURATION = 0.35
TURN_REPEAT_COUNT = 5
TURN_REPEAT_INTERVAL = 0.05

SAMPLE_RATE   = 24000
MIC_RATE      = 48000
MIC_DEVICE_ID = None  # pyaudio: None=デフォルト or デバイス番号 or デバイス名で部分一致
OUT_DEVICE_ID = None  # pyaudio: None=デフォルト or デバイス番号 or デバイス名で部分一致

BONES_ROOT = os.environ.get("BONES_ROOT", "/home/unitree-g1/Documents/G1/g1/csv")

MUJOCO_TO_ISAACLAB = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15,
    22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25,
    19, 26, 20, 27, 21, 28
]

SYSTEM_PROMPT = """あなたはG1ロボットのアシスタントです。
ユーザーの話に自然に反応し、簡潔に日本語で答えてください。
雑談も質問も歓迎です。フレンドリーに話してください。
返答は1〜2文程度の短さを心がけてください。

あなたはロボットとして体を持っています。必ずselect_motionを呼び出してから返答してください。
- 挨拶 → casual_greeting / wave / greetings_hat
- 同意・頷き → nod_in_thought
- 喜び・称賛 → clap_enthusiastic_001_ / triumph_two_handed_001_ / jump_for_joy
- 驚き → shock_both_hands
- 困惑 → don_t_know_with_hands_001_ / don_t_know_no_hands_001_
- 笑い → neutral_laugh_001_
- 感謝・別れ → bow / bye_bye_salute / sending_kisses_two_hands
- 説明・指示 → point_forwards / point_upwards / point_downwards
- 運動・元気 → squat_001_ / jump_for_joy
返答の内容に最も合う動作を必ず1つ選んでください。"""

SYSTEM_PROMPT += """

移動の依頼を受けたら、必ずwalk_commandも呼び出してください。
- 「3歩前へ」「2歩下がって」のように回数がある場合は steps にその回数を入れる
- 回数が明示されていない場合の steps は 1
- 少しだけ、ちょっと、少し下がって なども steps=1 として扱う
- 止まって、ストップ は direction=stop を使う
"""

DIALOGUE_MOTIONS = {
    "casual_greeting":            "カジュアルな挨拶（こんにちは・はじめまして）",
    "bye_bye_salute":             "さようなら・バイバイ・また会いましょう",
    "wave":                       "手を振る（挨拶・呼びかけ）",
    "wave_two_hands":             "両手を振る（元気な挨拶）",
    "greetings_hat":              "丁寧な挨拶",
    "nod_in_thought":             "頷く・同意・なるほど",
    "clap_enthusiastic_001_":     "拍手（すごい・よかった・素晴らしい）",
    "triumph_two_handed_001_":    "両手を挙げて喜ぶ（やった・成功・嬉しい）",
    "raise_your_hand":            "手を挙げる（はい・わかった）",
    "sending_kisses_two_hands":   "両手でキスを送る（ありがとう・大好き）",
    "don_t_know_with_hands_001_": "両手を広げて困惑（わからない・難しい）",
    "don_t_know_no_hands_001_":   "肩をすくめる（どうかな・微妙）",
    "neutral_laugh_001_":         "笑い（面白い・楽しい）",
    "shock_both_hands":           "驚き（えっ・本当に）",
    # "bow":                        "お辞儀（感謝・謝罪）",
    "point_forwards":             "前を指差す・説明する",
    "point_upwards":              "上を指差す・注目して",
    "point_downwards":            "下を指差す",
    "squat_001_":                 "スクワット（運動・元気なポーズ）",
    "jump_for_joy":               "喜びのジャンプ（嬉しい・やった）",
    "neutral_kick_R_001__A543":   "キック（かっこいい・決め技）",
    "idle_loop_001_":             "自然な待機・リラックス",
    "idle_hands_on_back_loop":    "手を後ろに組んで聞く",
    "crossed_arms_idle":          "腕を組んで考える",
    "street_crowd_talking":       "会話中の自然な身振り",
}
DEFAULT_MOTION = "casual_greeting"


def _normalize_device_selector(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return text


def _get_default_device_index(pa, is_input: bool) -> Optional[int]:
    try:
        info = pa.get_default_input_device_info() if is_input else pa.get_default_output_device_info()
        return int(info["index"])
    except Exception:
        return None


def resolve_audio_device(pa, selector, is_input: bool, purpose: str) -> Optional[int]:
    selector = _normalize_device_selector(selector)
    channel_key = "maxInputChannels" if is_input else "maxOutputChannels"
    kind = "input" if is_input else "output"

    if isinstance(selector, int):
        info = pa.get_device_info_by_index(selector)
        if info[channel_key] <= 0:
            raise RuntimeError(f"[{purpose}] device={selector} は {kind} デバイスではありません")
        return selector

    if isinstance(selector, str):
        target = selector.lower()
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info[channel_key] > 0 and target in info["name"].lower():
                return i
        raise RuntimeError(f"[{purpose}] '{selector}' に一致する {kind} デバイスが見つかりません")

    return _get_default_device_index(pa, is_input)


def print_audio_devices():
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        print("[Audio] 利用可能なデバイス一覧")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            ins = int(info.get("maxInputChannels", 0))
            outs = int(info.get("maxOutputChannels", 0))
            print(f"  {i:2d}: {info['name']}  (in={ins}, out={outs})")
    finally:
        pa.terminate()




# ── BONES-SEED 読み込み（bones_to_sonic.py と完全同一）────────

def load_bones_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    joint_cols = [c for c in df.columns if c.endswith("_dof")]
    if len(joint_cols) != 29:
        raise ValueError(f"関節数が29ではありません: {len(joint_cols)}")
    jp = (df[joint_cols].values * np.pi / 180.0).astype(np.float32)
    return jp[:, MUJOCO_TO_ISAACLAB]


def prepare_motion(jp_orig: np.ndarray, play_fps: float = PLAY_FPS) -> tuple:
    """bones_to_sonic.py と同じ補間処理"""
    T_orig = len(jp_orig)
    t_orig = np.linspace(0, T_orig / BONES_FPS * (BONES_FPS / play_fps), T_orig)
    t_new  = np.arange(0, t_orig[-1], 1.0 / SONIC_FPS)
    jp = interp1d(t_orig, jp_orig, axis=0, kind="linear",
                  fill_value="extrapolate")(t_new).astype(np.float32)
    jv = np.zeros_like(jp)
    jv[:-1] = (jp[1:] - jp[:-1]) * SONIC_FPS
    jv[-1]  = jv[-2]
    bq = np.tile([1., 0., 0., 0.], (len(jp), 1)).astype(np.float32)
    return jp, jv, bq


def find_motion(keyword: str) -> list:
    pattern = os.path.join(BONES_ROOT, "**", f"*{keyword}*.csv")
    files = glob.glob(pattern, recursive=True)
    return sorted([f for f in files if "_M." not in os.path.basename(f)])


def scan_motions(motion_dict: dict, bones_root: str = None) -> dict:
    global BONES_ROOT
    if bones_root:
        BONES_ROOT = bones_root
    available = {}
    for name, desc in motion_dict.items():
        files = find_motion(name)
        if files:
            try:
                jp_orig = load_bones_csv(files[0])
                jp, jv, bq = prepare_motion(jp_orig)
                dur = len(jp) / SONIC_FPS
                available[name] = {"jp": jp, "jv": jv, "bq": bq, "desc": desc, "dur": dur}
                print(f"[Motion] ✓ {name}  ({dur:.1f}s)")
            except Exception as e:
                print(f"[Motion] ✗ {name}  ({e})")
        else:
            print(f"[Motion] - {name}  (CSV なし)")
    print(f"\n[Motion] 利用可能: {len(available)}/{len(motion_dict)} 個\n")
    return available


# ── ZMQ 送信（bones_to_sonic.py と完全同一）──────────────────

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


# ── 動作プレイヤー ────────────────────────────────────────────

class MotionPlayer:
    def __init__(self, sock):
        self._sock   = sock
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()
        self._fi     = 0  # フレームカウンター

    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play_once(self, motion: dict, force: bool = False):
        """動作を1回再生。再生中なら無視（force=True の場合は中断して開始）"""
        if not force and self.is_playing():
            return  # 再生中は無視
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(motion,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self, motion: dict):
        """bones_to_sonic.py と完全同一の送信ロジック"""
        jp = motion["jp"]
        jv = motion["jv"]
        bq = motion["bq"]
        T  = len(jp)

        for i in range(0, T, CHUNK_SIZE):
            if self._stop.is_set():
                return
            n  = min(CHUNK_SIZE, T - i)
            t0 = time.perf_counter()
            send_pose(self._sock, jp[i:i+n], jv[i:i+n], bq[i:i+n], self._fi)
            self._fi += n
            wait = n / SONIC_FPS - (time.perf_counter() - t0)
            if wait > 0:
                self._stop.wait(timeout=wait)
        # 動作完了 → WBC が自動的に引き継ぐ（bones_to_sonic.py と同じ）



# ── 歩行コントローラー ────────────────────────────────────────

class WalkerController:
    def __init__(self, sock):
        self._sock         = sock
        self._facing_angle = 0.0
        self._TURN_STEP    = np.radians(10)
        self._planner_mode = False
        self._lock         = threading.Lock()
        self._action_stop  = threading.Event()
        self._action_thread: Optional[threading.Thread] = None

    def _send_msg(self, topic, fields, data):
        header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
        hj = json.dumps(header).encode()
        hb = hj + b"\x00" * (HEADER_SIZE - len(hj))
        self._sock.send(topic + hb + data)

    def send_command(self, start=True, stop=False, planner=True):
        fields = [
            {"name": "start",   "dtype": "u8", "shape": [1]},
            {"name": "stop",    "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
        self._send_msg(b"command", fields, struct.pack("BBB", int(start), int(stop), int(planner)))

    def send_planner(self, mode, movement, facing, speed=-1.0):
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
        data += struct.pack("<ff", speed, -1.0)
        self._send_msg(b"planner", fields, data)

    def _fv(self):
        a = self._facing_angle
        return [np.cos(a), np.sin(a), 0.0]

    def _wait_or_stop(self, seconds: float) -> bool:
        """Return True if an in-flight action was cancelled."""
        return self._action_stop.wait(timeout=seconds)

    def _cancel_action(self, wait: bool = True):
        self._action_stop.set()
        if wait and self._action_thread and self._action_thread.is_alive():
            self._action_thread.join(timeout=1.0)
        self._action_thread = None
        self._action_stop.clear()

    def run_action(self, action):
        self._cancel_action(wait=True)
        self._action_thread = threading.Thread(target=action, daemon=True)
        self._action_thread.start()

    def start_planner(self):
        with self._lock:
            if not self._planner_mode:
                self.send_command(start=True, stop=False, planner=True)
                if self._wait_or_stop(1.0):
                    return
                self.send_planner(0, [0,0,0], self._fv())
                self._planner_mode = True
                print("[Walker] planner モード開始")

    def _ensure_planner(self):
        if not self._planner_mode:
            self.send_command(start=True, stop=False, planner=True)
            if self._wait_or_stop(0.3):
                return False
            self._planner_mode = True
        return True

    def _walk_linear(self, sign: float, steps: int = 1, step_duration: float = WALK_STEP_DURATION):
        steps = max(1, int(steps))
        with self._lock:
            if not self._ensure_planner():
                return False
            for step_index in range(steps):
                a = self._facing_angle
                mv = [sign * np.cos(a), sign * np.sin(a), 0.0]
                self.send_planner(2, mv, self._fv())
                if self._wait_or_stop(step_duration):
                    self.send_planner(0, [0,0,0], self._fv())
                    return False
                self.send_planner(0, [0,0,0], self._fv())
                if step_index < steps - 1 and self._wait_or_stop(0.08):
                    return False
        return True

    def walk_forward(self, steps: int = 1):
        if self._walk_linear(1.0, steps=steps):
            print(f"[Walker] 前進 x{max(1, int(steps))}")

    def walk_backward(self, steps: int = 1):
        if self._walk_linear(-1.0, steps=steps):
            print(f"[Walker] 後退 x{max(1, int(steps))}")

    def turn_left(self, steps: int = 1):
        with self._lock:
            if not self._ensure_planner():
                return False
            for step_index in range(max(1, int(steps))):
                self._facing_angle += self._TURN_STEP
                fv = self._fv()
                for _ in range(TURN_REPEAT_COUNT):
                    self.send_planner(2, [0,0,0], fv)
                    if self._wait_or_stop(TURN_REPEAT_INTERVAL):
                        self.send_planner(0, [0,0,0], self._fv())
                        return False
                self.send_planner(0, [0,0,0], self._fv())
                if step_index < max(1, int(steps)) - 1 and self._wait_or_stop(0.05):
                    return False
        print(f"[Walker] 左旋回 x{max(1, int(steps))} → {np.degrees(self._facing_angle):.0f}°")
        return True

    def turn_right(self, steps: int = 1):
        with self._lock:
            if not self._ensure_planner():
                return False
            for step_index in range(max(1, int(steps))):
                self._facing_angle -= self._TURN_STEP
                fv = self._fv()
                for _ in range(TURN_REPEAT_COUNT):
                    self.send_planner(2, [0,0,0], fv)
                    if self._wait_or_stop(TURN_REPEAT_INTERVAL):
                        self.send_planner(0, [0,0,0], self._fv())
                        return False
                self.send_planner(0, [0,0,0], self._fv())
                if step_index < max(1, int(steps)) - 1 and self._wait_or_stop(0.05):
                    return False
        print(f"[Walker] 右旋回 x{max(1, int(steps))} → {np.degrees(self._facing_angle):.0f}°")
        return True

    def stop(self):
        self._cancel_action(wait=False)
        with self._lock:
            if not self._ensure_planner():
                return
            self.send_planner(0, [0,0,0], self._fv())
        print("[Walker] 停止")

    def switch_to_streaming(self):
        self._cancel_action(wait=True)
        with self._lock:
            self.send_command(start=True, stop=False, planner=False)
            self._planner_mode = False
            self._wait_or_stop(0.8)

    def switch_to_planner(self):
        with self._lock:
            self.send_command(start=True, stop=False, planner=True)
            self._planner_mode = True
            self._wait_or_stop(0.5)


# ── キーボード手動制御 ────────────────────────────────────────

class KeyboardController:
    def __init__(self, walker, player):
        self._walker  = walker
        self._player  = player
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _loop(self):
        import sys, tty, termios, select
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        print("[Keyboard] WASD 手動制御有効 (W=前進 S=後退 A=左旋回 D=右旋回 Space=停止)")
        try:
            tty.setraw(fd)
            while self._running:
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                key = sys.stdin.read(1).lower()
                if key == "\x03":
                    self._running = False
                    os.kill(os.getpid(), signal.SIGINT)
                    return
                if key == "w":
                    self._player.stop()
                    self._walker.run_action(self._walker.walk_forward)
                elif key == "s":
                    self._player.stop()
                    self._walker.run_action(self._walker.walk_backward)
                elif key == "a":
                    self._player.stop()
                    self._walker.run_action(self._walker.turn_left)
                elif key == "d":
                    self._player.stop()
                    self._walker.run_action(self._walker.turn_right)
                elif key == " ":
                    self._player.stop()
                    self._walker.stop()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Realtime API ──────────────────────────────────────────────

class RealtimeDialogue:
    URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

    def __init__(
        self,
        motions: dict,
        player: MotionPlayer,
        walker,
        vad: bool = True,
        mic_device=None,
        out_device=None,
    ):
        self.motions  = motions
        self.player   = player
        self.walker   = walker
        self.vad      = vad
        self.mic_device = _normalize_device_selector(mic_device)
        self.out_device = _normalize_device_selector(out_device)
        self._ws      = None
        self._abuf    = bytearray()

    async def connect(self):
        try:
            import websockets
        except ImportError:
            print("pip install websockets"); sys.exit(1)

        self._ws = await websockets.connect(
            self.URL,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
            max_size=10 * 1024 * 1024,
        )
        print("[Realtime] 接続しました")

        tool_motion = {
            "type": "function",
            "name": "select_motion",
            "description": "会話の中で自然なタイミングで呼び出す。挨拶・感情表現・強調場面など、3〜5回に1回程度の頻度で会話の雰囲気に合った動作を選択する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "motion_name": {
                        "type": "string",
                        "enum": list(self.motions.keys()),
                        "description": "動作名",
                    }
                },
                "required": ["motion_name"],
            },
        }
        tool_walk = {
            "type": "function",
            "name": "walk_command",
            "description": "ロボットを移動させる。歩数や旋回回数が分かる場合は steps に入れる。",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "turn_left", "turn_right", "stop"],
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "前進・後退の歩数、または左右旋回の回数。省略時は1。",
                    },
                },
                "required": ["direction"],
            },
        }

        cfg = {
            "modalities": ["text", "audio"],
            "instructions": SYSTEM_PROMPT,
            "voice": "shimmer",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1", "language": "ja"},
            "tools": [tool_motion, tool_walk],
            "tool_choice": "required",
        }
        if self.vad:
            cfg["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 800,
            }
        await self._ws.send(json.dumps({"type": "session.update", "session": cfg}))

    async def _send(self, msg):
        if self._ws:
            await self._ws.send(json.dumps(msg))

    async def stream_mic(self):
        import pyaudio, subprocess
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        # PipeWire を再起動して音声デバイスを再認識
        subprocess.run(["systemctl", "--user", "restart", "pipewire", "pipewire-pulse", "wireplumber"],
                       check=False, capture_output=True)
        await asyncio.sleep(3)

        pa = pyaudio.PyAudio()
        dev_index = resolve_audio_device(pa, self.mic_device, is_input=True, purpose="マイク")
        if dev_index is None:
            pa.terminate()
            raise RuntimeError("[マイク] 入力デバイスが見つかりません")
        info = pa.get_device_info_by_index(dev_index)
        print(f"[マイク] {info['name']} device={dev_index}, 48000Hz → 24000Hz")

        def cb(in_data, frame_count, time_info, status):
            pcm = np.frombuffer(in_data, dtype=np.int16)
            down = pcm[::2]  # 48000 → 24000
            loop.call_soon_threadsafe(q.put_nowait, down.tobytes())
            return (None, pyaudio.paContinue)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=48000,
            input=True,
            input_device_index=dev_index,
            frames_per_buffer=2048,
            stream_callback=cb,
        )
        stream.start_stream()
        try:
            while True:
                pcm = await q.get()
                await self._send({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                })
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    async def play_audio(self):
        import pyaudio
        pa = pyaudio.PyAudio()

        out_index = resolve_audio_device(pa, self.out_device, is_input=False, purpose="スピーカー")
        if out_index is None:
            pa.terminate()
            raise RuntimeError("[スピーカー] 出力デバイスが見つかりません")
        info = pa.get_device_info_by_index(out_index)
        print(f"[スピーカー] {info['name']} device={out_index}, 48000Hz")

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=48000,
            output=True,
            output_device_index=out_index,
            frames_per_buffer=4096,
        )
        stream.start_stream()
        try:
            while True:
                if self._abuf:
                    chunk = bytes(self._abuf[:4096])
                    self._abuf = self._abuf[4096:]
                    pcm = np.frombuffer(chunk, dtype=np.int16)
                    up = np.repeat(pcm, 2)  # 24000 → 48000
                    stream.write(up.tobytes())
                else:
                    await asyncio.sleep(0.01)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    async def recv_loop(self):
        async for raw in self._ws:
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            t = ev.get("type", "")

            if t == "response.audio.delta":
                b64 = ev.get("delta", "")
                if b64:
                    self._abuf.extend(base64.b64decode(b64))

            elif t == "response.audio_transcript.delta":
                print(ev.get("delta", ""), end="", flush=True)

            elif t == "response.text.delta":
                print(ev.get("delta", ""), end="", flush=True)

            elif t == "response.function_call_arguments.done":
                fname = ev.get("name", "")
                if fname == "select_motion":
                    try:
                        name = json.loads(ev.get("arguments", "{}")).get("motion_name", "")
                        if name in self.motions:
                            m = self.motions[name]
                            print(f"\n[動作] → {name}  ({m['desc']}, {m['dur']:.1f}s)")
                            def _play(motion=m):
                                self.walker.switch_to_streaming()
                                self.player.play_once(motion, force=True)
                                while self.player.is_playing(): time.sleep(0.1)
                                self.walker.switch_to_planner()
                            threading.Thread(target=_play, daemon=True).start()
                        else:
                            print(f"\n[動作] 不明: {name}")
                    except Exception as e:
                        print(f"\n[動作エラー] {e}")
                elif fname == "walk_command":
                    try:
                        args = json.loads(ev.get("arguments", "{}"))
                        direction = args.get("direction", "stop")
                        steps = max(1, min(10, int(args.get("steps", 1) or 1)))
                        print(f"\n[歩行] → {direction} x{steps}")
                        def _walk(d=direction, n=steps):
                            if d == "forward":
                                self.walker.run_action(lambda: self.walker.walk_forward(n))
                            elif d == "backward":
                                self.walker.run_action(lambda: self.walker.walk_backward(n))
                            elif d == "turn_left":
                                self.walker.run_action(lambda: self.walker.turn_left(n))
                            elif d == "turn_right":
                                self.walker.run_action(lambda: self.walker.turn_right(n))
                            else:
                                self.walker.stop()
                        _walk()
                    except Exception as e:
                        print(f"\n[歩行エラー] {e}")

            elif t == "response.output_item.done":
                item = ev.get("item", {})
                if item.get("type") == "function_call":
                    asyncio.create_task(
                        self._submit_function_result(item.get("call_id", "")))

            elif t == "response.audio.done":
                print()

            elif t == "input_audio_buffer.speech_started":
                print("\n🎤 [話し中...]")
                self.player.stop()  # ユーザーが話し始めたら動作を中断

            elif t == "input_audio_buffer.speech_stopped":
                print("✅ [認識中...]")

            elif t == "conversation.item.input_audio_transcription.completed":
                tr = ev.get("transcript", "")
                if tr:
                    print(f"\n👤 ユーザー: {tr}")

            elif t == "response.created":
                print("🤖 G1: ", end="", flush=True)


            elif t == "error":
                print(f"\n[エラー] {ev.get('error', ev)}")

    async def _submit_function_result(self, call_id: str):
        """Function call の結果を送信して音声レスポンスを要求"""
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": "OK",
            }
        })
        await self._send({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "tools": [],          # function call なしで音声のみ返す
                "tool_choice": "none",
            }
        })

    async def ptt_loop(self):
        import pyaudio, keyboard
        pa = pyaudio.PyAudio()

        dev_index = resolve_audio_device(pa, self.mic_device, is_input=True, purpose="マイク")
        if dev_index is None:
            pa.terminate()
            raise RuntimeError("[マイク] 入力デバイスが見つかりません")

        print("スペースキーを長押しで話してください。Ctrl+C で終了。\n")
        while True:
            print(">>> スペースキーを長押し...", end="", flush=True)
            await asyncio.get_event_loop().run_in_executor(None, keyboard.wait, "space")
            print(" [録音中]", end="", flush=True)
            recorded = []

            def cb(in_data, frame_count, time_info, status):
                recorded.append(np.frombuffer(in_data, dtype=np.int16).copy())
                return (None, pyaudio.paContinue)

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=MIC_RATE,
                input=True,
                input_device_index=dev_index,
                frames_per_buffer=2048,
                stream_callback=cb,
            )
            stream.start_stream()
            await asyncio.get_event_loop().run_in_executor(None, keyboard.wait, "space", True)
            stream.stop_stream()
            stream.close()
            print(" [送信中]")
            if recorded:
                pcm = np.concatenate(recorded)[::2]
                i16 = pcm.clip(-32768, 32767).astype(np.int16)
                b64 = base64.b64encode(i16.tobytes()).decode()
                await self._send({"type": "input_audio_buffer.append", "audio": b64})
                await self._send({"type": "input_audio_buffer.commit"})
                await self._send({"type": "response.create"})

    async def run(self):
        await self.connect()
        if self.vad:
            await asyncio.gather(
                self.stream_mic(), self.recv_loop(), self.play_audio()
            )
        else:
            await asyncio.gather(
                self.ptt_loop(), self.recv_loop(), self.play_audio()
            )


# ── エントリポイント ──────────────────────────────────────────

def main():
    global PLAY_FPS, BONES_ROOT, MIC_DEVICE_ID, OUT_DEVICE_ID
    import subprocess
    # PipeWire を再起動して音声デバイスを再認識
    subprocess.run(["systemctl", "--user", "restart", "pipewire", "pipewire-pulse", "wireplumber"],
                   check=False, capture_output=True)
    time.sleep(2)
    p = argparse.ArgumentParser(description="G1 Realtime 音声対話 + BONES-SEED 動作")
    p.add_argument("--ptt",        action="store_true")
    p.add_argument("--mic-device", default=MIC_DEVICE_ID,
                   help="入力デバイス番号、またはデバイス名の一部")
    p.add_argument("--out-device", default=OUT_DEVICE_ID,
                   help="出力デバイス番号、またはデバイス名の一部")
    p.add_argument("--list-audio-devices", action="store_true",
                   help="利用可能な音声デバイス一覧を表示して終了")
    p.add_argument("--bones-root", default=BONES_ROOT)
    p.add_argument("--zmq-port",   type=int,   default=ZMQ_PORT)
    p.add_argument("--play-fps",   type=float, default=PLAY_FPS)
    args = p.parse_args()

    if args.list_audio_devices:
        print_audio_devices()
        return

    PLAY_FPS   = args.play_fps
    BONES_ROOT = args.bones_root
    MIC_DEVICE_ID = _normalize_device_selector(args.mic_device)
    OUT_DEVICE_ID = _normalize_device_selector(args.out_device)

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.zmq_port}")
    print(f"[ZMQ] tcp://*:{args.zmq_port}")
    time.sleep(0.5)

    motions = scan_motions(DIALOGUE_MOTIONS, BONES_ROOT)
    if not motions:
        print("動作が見つかりません。"); sys.exit(1)

    player = MotionPlayer(sock)
    walker = WalkerController(sock)
    walker.start_planner()
    kb = KeyboardController(walker, player)
    kb.start()
    print(f"✅ 起動完了  モード: {'PTT' if args.ptt else 'VAD'}  動作数: {len(motions)}")
    print("⚠️  手動制御: W=前進 S=後退 A=左旋回 D=右旋回 Space=停止\n")

    try:
        asyncio.run(
            RealtimeDialogue(
                motions,
                player,
                walker,
                vad=not args.ptt,
                mic_device=MIC_DEVICE_ID,
                out_device=OUT_DEVICE_ID,
            ).run()
        )
    except KeyboardInterrupt:
        print("\n終了します")
    finally:
        player.stop()
        kb.stop()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
