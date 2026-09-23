#!/usr/bin/env python3
"""
teleop_finger_WSL.py — [WSL VERSION] Browser-previewed hand-pinch teleop
for DAS Finger Controller. Streams annotated video to
http://localhost:8080 and handles hotkeys in the terminal.

This is the WSL version: because WSL usually has no display/X server
attached to the webcam pipeline, video is streamed over HTTP to a
browser on the Windows host and hotkeys are read from the WSL terminal
instead of a native OpenCV window. If you're running on native Linux
(real display + webcam, no WSL passthrough quirks), use teleop_finger.py
instead — same tracking/control logic, but it shows the feed in a real
OpenCV window and reads keys directly, no HTTP server or terminal
raw-mode needed.

Built against the actual gen_finger_con_python_sdk_release source
(scripts/databus.py, start_finger.py) rather than copy-pasted from the
gripper controller's WSL script — the finger controller differs in two
load-bearing ways:
  - Valid distance range is [0.0, 0.2] m (~20cm), not 0.0-0.103 m
  - Serial ports are /dev/ttyFingerLeft / /dev/ttyFingerRight, not
    /dev/ttyDevice*
  - DataBus here takes no gripper_type parameter at all

Modes:
    single finger:
        python3 teleop_finger_WSL.py left
        python3 teleop_finger_WSL.py right

    dual finger: plug in both finger controllers and drive them with
    both hands at once — your left hand drives the left finger, your
    right hand drives the right finger.
        python3 teleop_finger_WSL.py dual

Controls:
    n        -> capture current pinch as CLOSED reference
    f        -> capture current pinch as OPEN reference
    SPACE    -> freeze target(s) (ignore hand tracking until pressed again)
    q / ESC  -> disable motor(s) and quit

Usage: python3 teleop_finger_WSL.py left
       python3 teleop_finger_WSL.py dual --left-port /dev/ttyUSB0 --right-port /dev/ttyUSB1
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import select
import struct
import sys
import termios
import threading
import time
import tty
import urllib.request

import cv2
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from scripts.databus import DataBus  # noqa: E402

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
except ImportError:
    print("Missing dependency: mediapipe. Install with: pip install mediapipe")
    sys.exit(1)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(_here, "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

MIN_DIST = 0.0
MAX_DIST = 0.2  # confirmed range from gen_finger_con_python_sdk_release: databus.py set_target_distance() + README
THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MIDDLE_MCP = 9

SIDE_PORTS = {
    "left": "/dev/ttyFingerLeft",
    "right": "/dev/ttyFingerRight",
}

latest_jpeg = None
jpeg_lock = threading.Lock()

# Swapped in by run_dual()/run_single() before the HTTP server starts, so the
# index page's control hints match whichever mode is actually running.
INDEX_CONTROLS_HTML = (
    "<code>n</code> = close calib, <code>f</code> = open calib, "
    "<code>space</code> = freeze, <code>q</code> = quit"
)


class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = f"""
            <html>
            <head><title>Finger Teleop Stream</title></head>
            <body style="background:#111; color:#eee; font-family:sans-serif; text-align:center;">
                <h2>DAS Finger Controller — Hand Teleop</h2>
                <p>Controls are in your <b>WSL Terminal</b>: {INDEX_CONTROLS_HTML}</p>
                <img src="/stream.mjpg" style="max-width:90%; border:2px solid #555;" />
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            while True:
                with jpeg_lock:
                    frame_bytes = latest_jpeg
                if frame_bytes is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b"\r\n")
                time.sleep(0.033)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        return  # Silence HTTP server logs


class HandTeleopState:
    def __init__(self, smoothing: float):
        self.lock = threading.Lock()
        self.target = 0.05
        self.encoder_value = None
        self.encoder_ts = 0.0
        self.running = True
        self.frozen = False
        self.smoothing = smoothing
        self.smoothed_ratio = None
        self.calib_near = None
        self.calib_far = None

    def set_target(self, value):
        with self.lock:
            if not self.frozen:
                self.target = max(MIN_DIST, min(MAX_DIST, value))

    def toggle_freeze(self):
        with self.lock:
            self.frozen = not self.frozen
            return self.frozen

    def get_target(self):
        with self.lock:
            return self.target

    def update_encoder(self, value):
        with self.lock:
            self.encoder_value = value
            self.encoder_ts = time.time()

    def snapshot_encoder(self):
        with self.lock:
            return self.encoder_value, self.encoder_ts


def control_loop(databus: DataBus, state: HandTeleopState, hz: float):
    interval = 1.0 / hz
    while state.running:
        databus.set_target_distance(state.get_target())
        time.sleep(interval)


def terminal_input_thread(state: HandTeleopState):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while state.running:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                ch = sys.stdin.read(1)
                if ch in ("q", "\x1b"):  # q or Esc
                    state.running = False
                    print("\n[Quit requested from terminal]")
                    break
                elif ch == "n":
                    if state.smoothed_ratio is not None:
                        state.calib_near = state.smoothed_ratio
                        print(f"\n>> Calibrated CLOSED: {state.calib_near:.3f}")
                elif ch == "f":
                    if state.smoothed_ratio is not None:
                        state.calib_far = state.smoothed_ratio
                        print(f"\n>> Calibrated OPEN: {state.calib_far:.3f}")
                elif ch == " ":
                    is_frozen = state.toggle_freeze()
                    print("\n>> FROZEN" if is_frozen else "\n>> LIVE")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def terminal_input_thread_dual(state_left: HandTeleopState, state_right: HandTeleopState):
    """Same as terminal_input_thread, but n/f/space/q apply to both hands
    at once. Calibration is captured independently per hand from whichever
    hand(s) have produced a smoothed ratio so far — see the module-level
    note in teleop_finger.py's _handle_key for the stricter, currently-
    visible-only version of this same idea, which this script does not
    use (ported as-is from teleop_gripper_WSL.py's terminal-based flow)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while state_left.running or state_right.running:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                ch = sys.stdin.read(1)
                if ch in ("q", "\x1b"):  # q or Esc
                    state_left.running = False
                    state_right.running = False
                    print("\n[Quit requested from terminal]")
                    break
                elif ch == "n":
                    msg = []
                    for label, st in (("L", state_left), ("R", state_right)):
                        if st.smoothed_ratio is not None:
                            st.calib_near = st.smoothed_ratio
                            msg.append(f"{label}={st.calib_near:.3f}")
                    if msg:
                        print(f"\n>> Calibrated CLOSED: {', '.join(msg)}")
                elif ch == "f":
                    msg = []
                    for label, st in (("L", state_left), ("R", state_right)):
                        if st.smoothed_ratio is not None:
                            st.calib_far = st.smoothed_ratio
                            msg.append(f"{label}={st.calib_far:.3f}")
                    if msg:
                        print(f"\n>> Calibrated OPEN: {', '.join(msg)}")
                elif ch == " ":
                    is_frozen = state_left.toggle_freeze()
                    state_right.toggle_freeze()
                    print("\n>> FROZEN" if is_frozen else "\n>> LIVE")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _draw_hand(frame, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(
            frame,
            tuple(map(int, pts[a])),
            tuple(map(int, pts[b])),
            (0, 200, 0),
            2,
        )
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 0), -1)


def _update_ratio_from_landmarks(frame, pts, state: HandTeleopState):
    """Shared per-hand math: draws the pinch line/dots and updates
    state.smoothed_ratio."""
    thumb = np.array(pts[THUMB_TIP])
    index = np.array(pts[INDEX_TIP])
    wrist = np.array(pts[WRIST])
    mid_mcp = np.array(pts[MIDDLE_MCP])
    pinch_dist = np.linalg.norm(thumb - index)
    hand_scale = np.linalg.norm(wrist - mid_mcp)

    if hand_scale > 1e-3:
        ratio = pinch_dist / hand_scale
        if state.smoothed_ratio is None:
            state.smoothed_ratio = ratio
        else:
            a = state.smoothing
            state.smoothed_ratio = a * state.smoothed_ratio + (1 - a) * ratio

    cv2.line(
        frame,
        tuple(thumb.astype(int)),
        tuple(index.astype(int)),
        (0, 255, 0),
        2,
    )
    cv2.circle(frame, tuple(thumb.astype(int)), 6, (255, 0, 0), -1)
    cv2.circle(frame, tuple(index.astype(int)), 6, (0, 0, 255), -1)


def _apply_calibration(state: HandTeleopState):
    if (
        state.calib_near is not None
        and state.calib_far is not None
        and state.smoothed_ratio is not None
    ):
        near, far = state.calib_near, state.calib_far
        if abs(far - near) > 1e-6:
            frac = (state.smoothed_ratio - near) / (far - near)
            frac = max(0.0, min(1.0, frac))
            state.set_target(MIN_DIST + frac * (MAX_DIST - MIN_DIST))


def _calib_status(state: HandTeleopState) -> str:
    # Explicit None-checks, not truthiness — a calibrated ratio of exactly
    # 0.0 (unlikely but not impossible) would otherwise read as uncalibrated.
    # (This fixes a truthiness bug present in teleop_gripper_WSL.py's
    # equivalent check.)
    if state.calib_near is not None and state.calib_far is not None:
        return f"CALIBRATED (near={state.calib_near:.2f}, far={state.calib_far:.2f})"
    return "NOT CALIBRATED (press 'n' then 'f')"


def _encoder_str(state: HandTeleopState) -> str:
    encoder, enc_ts = state.snapshot_encoder()
    enc_age = time.time() - enc_ts if enc_ts else None
    if encoder is None:
        return "waiting..."
    if enc_age is not None and enc_age <= 1.0:
        return f"{encoder:.4f} m"
    return f"{encoder:.4f} m (stale)"


def run_single(args):
    """Single-finger hand-pinch teleop."""
    global latest_jpeg

    port = args.port or SIDE_PORTS[args.side]
    state = HandTeleopState(smoothing=args.smoothing)

    def encoder_cb(data):
        try:
            val = struct.unpack(">f", data)[0]
            state.update_encoder(val)
        except Exception:
            pass

    # Note: unlike the gripper controller SDK's DataBus, this SDK's
    # DataBus takes no gripper_type parameter.
    databus = DataBus(
        tty_port=port,
        baudrate=921600,
        encoder_freq=30,
        encoder_callback=encoder_cb,
    )

    threading.Thread(
        target=control_loop, args=(databus, state, args.hz), daemon=True
    ).start()

    # Start HTTP Video Server
    httpd = HTTPServer(("0.0.0.0", args.http_port), VideoStreamHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # Start Terminal Input Listener
    threading.Thread(target=terminal_input_thread, args=(state,), daemon=True).start()

    cap = cv2.VideoCapture(args.webcam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.webcam_index)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    )

    print("=" * 60)
    print("STREAM READY! Open this URL in Chrome/Edge on Windows:")
    print(f"  👉 http://localhost:{args.http_port}")
    print("=" * 60)
    print("CONTROLS (Keep cursor active in THIS terminal):")
    print("  'n'   -> Calibrate Pinch CLOSED")
    print("  'f'   -> Calibrate Pinch OPEN")
    print("  SPACE -> Freeze / Unfreeze")
    print("  'q'   -> Quit")
    print("=" * 60)

    frame_timestamp_ms = 0
    try:
        while state.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                h, w, _ = frame.shape
                pts = [(p.x * w, p.y * h) for p in lm]
                _draw_hand(frame, pts)
                _update_ratio_from_landmarks(frame, pts, state)

            _apply_calibration(state)

            target = state.get_target()
            overlay = [
                f"Target: {target:.4f} m   Encoder: {_encoder_str(state)}",
                f"Calib: {_calib_status(state)}",
                f"{'FROZEN' if state.frozen else 'LIVE'}  |  Terminal: n=near, f=far, space=freeze, q=quit",
            ]
            for i, line in enumerate(overlay):
                cv2.putText(
                    frame,
                    line,
                    (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            _, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )
            with jpeg_lock:
                latest_jpeg = buf.tobytes()

    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        time.sleep(0.1)
        cap.release()
        try:
            landmarker.close()
        except Exception:
            pass
        try:
            databus.disable_motor()
            time.sleep(0.1)
        except Exception:
            pass
        databus.stop()
        httpd.shutdown()
        print("Shutdown complete.")


def run_dual(args):
    """Dual-finger hand-pinch teleop: your left hand drives the left
    finger, your right hand drives the right finger, both from the same
    webcam feed. Uses MediaPipe's handedness classification to route each
    detected hand's pinch ratio to the matching HandTeleopState/DataBus."""
    global latest_jpeg

    left_port = args.left_port or SIDE_PORTS["left"]
    right_port = args.right_port or SIDE_PORTS["right"]

    state_left = HandTeleopState(smoothing=args.smoothing)
    state_right = HandTeleopState(smoothing=args.smoothing)

    def make_encoder_cb(state):
        def _cb(data):
            try:
                val = struct.unpack(">f", data)[0]
                state.update_encoder(val)
            except Exception:
                pass
        return _cb

    databus_left = DataBus(
        tty_port=left_port,
        baudrate=921600,
        encoder_freq=30,
        encoder_callback=make_encoder_cb(state_left),
    )
    databus_right = DataBus(
        tty_port=right_port,
        baudrate=921600,
        encoder_freq=30,
        encoder_callback=make_encoder_cb(state_right),
    )

    threading.Thread(
        target=control_loop, args=(databus_left, state_left, args.hz), daemon=True
    ).start()
    threading.Thread(
        target=control_loop, args=(databus_right, state_right, args.hz), daemon=True
    ).start()

    global INDEX_CONTROLS_HTML
    INDEX_CONTROLS_HTML = (
        "<code>n</code> = close calib (both hands), <code>f</code> = open calib (both hands), "
        "<code>space</code> = freeze both, <code>q</code> = quit"
    )
    httpd = HTTPServer(("0.0.0.0", args.http_port), VideoStreamHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    threading.Thread(
        target=terminal_input_thread_dual, args=(state_left, state_right), daemon=True
    ).start()

    cap = cv2.VideoCapture(args.webcam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.webcam_index)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    )

    print("=" * 60)
    print("DUAL-FINGER STREAM READY! Open this URL in Chrome/Edge on Windows:")
    print(f"  👉 http://localhost:{args.http_port}")
    print(f"  Left finger  <- your LEFT hand  (port {left_port})")
    print(f"  Right finger <- your RIGHT hand (port {right_port})")
    print("=" * 60)
    print("CONTROLS (Keep cursor active in THIS terminal):")
    print("  'n'   -> Calibrate Pinch CLOSED (per visible hand)")
    print("  'f'   -> Calibrate Pinch OPEN (per visible hand)")
    print("  SPACE -> Freeze / Unfreeze both")
    print("  'q'   -> Quit")
    if args.invert_hands:
        print("  (--invert-hands is ON: mediapipe Left/Right swapped)")
    print("=" * 60)

    frame_timestamp_ms = 0
    try:
        while state_left.running and state_right.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            h, w, _ = frame.shape
            seen = {"Left": False, "Right": False}

            if result.hand_landmarks:
                for idx, lm in enumerate(result.hand_landmarks):
                    # MediaPipe classifies handedness assuming a mirrored
                    # (selfie-view) input, which matches the cv2.flip()
                    # above, so "Left"/"Right" here already correspond to
                    # the user's own left/right hand. If yours come out
                    # swapped (camera mounting, etc.), rerun with
                    # --invert-hands.
                    label = "Left"
                    if result.handedness and idx < len(result.handedness):
                        label = result.handedness[idx][0].category_name
                    if args.invert_hands:
                        label = "Right" if label == "Left" else "Left"

                    state = state_left if label == "Left" else state_right
                    seen[label] = True

                    pts = [(p.x * w, p.y * h) for p in lm]
                    _draw_hand(frame, pts)
                    _update_ratio_from_landmarks(frame, pts, state)

            _apply_calibration(state_left)
            _apply_calibration(state_right)

            overlay = [
                f"L target: {state_left.get_target():.4f} m   L encoder: {_encoder_str(state_left)}"
                f"{'  [tracking]' if seen['Left'] else ''}",
                f"L calib: {_calib_status(state_left)}",
                f"R target: {state_right.get_target():.4f} m   R encoder: {_encoder_str(state_right)}"
                f"{'  [tracking]' if seen['Right'] else ''}",
                f"R calib: {_calib_status(state_right)}",
                f"{'FROZEN' if (state_left.frozen or state_right.frozen) else 'LIVE'}  |  "
                "Terminal: n=near, f=far, space=freeze, q=quit",
            ]
            for i, line in enumerate(overlay):
                cv2.putText(
                    frame,
                    line,
                    (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

            _, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )
            with jpeg_lock:
                latest_jpeg = buf.tobytes()

    except KeyboardInterrupt:
        pass
    finally:
        state_left.running = False
        state_right.running = False
        time.sleep(0.1)
        cap.release()
        try:
            landmarker.close()
        except Exception:
            pass
        for databus in (databus_left, databus_right):
            try:
                databus.disable_motor()
                time.sleep(0.1)
            except Exception:
                pass
            databus.stop()
        httpd.shutdown()
        print("Shutdown complete.")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Webcam hand-pinch teleop for DAS Finger Controller (WSL version)")
    parser.add_argument("side", choices=["left", "right", "dual"],
                         help="Which finger to drive. 'dual' teleops left+right "
                              "fingers at once with your left/right hands.")
    parser.add_argument("--port", type=str, default=None,
                         help="Serial port override for single-finger mode (left/right).")
    parser.add_argument("--left-port", type=str, default=None,
                         help="Serial port override for the left finger in 'dual' mode.")
    parser.add_argument("--right-port", type=str, default=None,
                         help="Serial port override for the right finger in 'dual' mode.")
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=0.4)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--invert-hands", action="store_true",
                         help="'dual' mode only: swap which detected hand "
                              "(Left/Right) drives which finger.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.side == "dual":
        run_dual(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()