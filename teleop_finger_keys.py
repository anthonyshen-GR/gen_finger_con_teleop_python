#!/usr/bin/env python3
"""
teleop_finger_keys.py — Keyboard teleoperation for the DAS Finger Controller,
using curses for input (works on WSL2 and native Linux alike — curses
reads keys straight from the terminal itself, no display/X server
required, so there's no WSL-vs-native split needed here).

Built against the actual gen_finger_con_python_sdk_release source
(scripts/databus.py, start_finger.py) — NOT copy-pasted from the gripper
controller scripts, since the finger controller differs in two
load-bearing ways:
  - Valid distance range is [0.0, 0.2] m (~20cm), not 0.0-0.103 m
  - Serial ports are /dev/ttyFingerLeft / /dev/ttyFingerRight, not
    /dev/ttyDevice*
  - DataBus here takes no gripper_type parameter at all

IMPORTANT: run only this ONE script per port. Do not also have
start_finger.py (or any other script) open on the same serial port at
the same time — two processes reading/writing the same port will corrupt
the protocol stream. This applies per-port: in dual mode, just make sure
nothing else is touching either the left or the right serial port.

Modes:
    single finger:
        python3 teleop_finger_keys.py left
        python3 teleop_finger_keys.py right --port /dev/ttyUSB0

    dual finger: plug in both finger controllers and drive the left one
    with WASD and the right one with the arrow keys, at the same time.
        python3 teleop_finger_keys.py dual

Controls, single-finger mode (must have this terminal focused — that's
how curses reads keys):
    Right / Up    -> open (hold to keep moving)
    Left  / Down  -> close (hold to keep moving)
    Space         -> stop / hold current target
    Home          -> jump to full open  (0.200 m)
    End           -> jump to full close (0.000 m)
    R             -> re-center to 0.050 m
    Q             -> disable motor and quit

Controls, dual-finger mode ('dual'):
    Right / Up    -> open  RIGHT finger (hold to keep moving)
    Left  / Down  -> close RIGHT finger (hold to keep moving)
    D / W         -> open  LEFT finger  (hold to keep moving)
    A / S         -> close LEFT finger  (hold to keep moving)
    Space         -> stop / hold both fingers' current targets
    Home          -> jump BOTH to full open  (0.200 m)
    End           -> jump BOTH to full close (0.000 m)
    R             -> re-center BOTH to 0.050 m
    Q             -> disable both motors and quit

Usage:
    python3 teleop_finger_keys.py left
    python3 teleop_finger_keys.py right --port /dev/ttyUSB0
    python3 teleop_finger_keys.py dual --left-port /dev/ttyUSB0 --right-port /dev/ttyUSB1

Requires: pyserial   (pip install pyserial --break-system-packages)
No curses install needed — it's in the Python standard library on Linux.
"""

import argparse
import curses
import os
import struct
import sys
import threading
import time

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from scripts.databus import DataBus  # noqa: E402


MIN_DIST = 0.0
MAX_DIST = 0.2  # confirmed range from gen_finger_con_python_sdk_release: databus.py set_target_distance() + README

SIDE_PORTS = {
    "left": "/dev/ttyFingerLeft",
    "right": "/dev/ttyFingerRight",
}


class TeleopState:
    def __init__(self, start_pos: float, step_rate: float):
        self.lock = threading.Lock()
        self.target = start_pos
        self.direction = 0
        self.step_rate = step_rate
        self.encoder_value = None
        self.encoder_ts = 0.0
        self.running = True
        self.last_key_ts = 0.0  # used to auto-release direction if no key repeats arrive

    def set_direction(self, d: int):
        with self.lock:
            self.direction = d
            self.last_key_ts = time.time()

    def nudge(self, dt: float):
        with self.lock:
            # Auto-stop if no key event has refreshed direction recently.
            # Terminals send repeat key-press events while held, with gaps
            # between them, so a short grace window (~0.35s) keeps motion
            # smooth without it running away after you release the key.
            if self.direction != 0 and (time.time() - self.last_key_ts) > 0.35:
                self.direction = 0
            if self.direction != 0:
                self.target += self.direction * self.step_rate * dt
                self.target = max(MIN_DIST, min(MAX_DIST, self.target))
            return self.target

    def jump_to(self, value: float):
        with self.lock:
            self.target = max(MIN_DIST, min(MAX_DIST, value))
            self.direction = 0
            return self.target

    def update_encoder(self, value: float):
        with self.lock:
            self.encoder_value = value
            self.encoder_ts = time.time()

    def snapshot(self):
        with self.lock:
            return self.target, self.encoder_value, self.encoder_ts


def encoder_callback_factory(state: TeleopState):
    def _cb(record_data: bytes):
        try:
            value = struct.unpack(">f", record_data)[0]
            state.update_encoder(value)
        except Exception:
            pass  # avoid printing from a background thread while curses owns the screen
    return _cb


def control_loop(databus: DataBus, state: TeleopState, hz: float):
    interval = 1.0 / hz
    last = time.time()
    while state.running:
        now = time.time()
        dt = now - last
        last = now
        target = state.nudge(dt)
        databus.set_target_distance(target)
        elapsed = time.time() - now
        time.sleep(max(0.0, interval - elapsed))


def _open_databus(port: str, state: TeleopState):
    # Note: unlike the gripper controller SDK's DataBus, this SDK's
    # DataBus takes no gripper_type parameter — there's no equivalent
    # concept for the finger controller.
    try:
        return DataBus(
            tty_port=port,
            baudrate=921600,
            encoder_freq=30,
            encoder_callback=encoder_callback_factory(state),
        )
    except Exception as e:
        print(f"Failed to open driver on {port}: {e}")
        print("If start_finger.py or another instance of this script is already")
        print("running against this port, close it first — only one process can")
        print("hold the serial port at a time.")
        sys.exit(1)


def run_ui(stdscr, databus: DataBus, state: TeleopState, side: str, port: str):
    """Single-finger UI."""
    curses.curs_set(0)
    stdscr.nodelay(True)   # non-blocking getch
    stdscr.timeout(50)     # ~20Hz redraw / key-poll rate
    stdscr.keypad(True)

    bar_width = 30
    quit_flag = False

    while not quit_flag:
        try:
            key = stdscr.getch()
        except curses.error:
            key = -1

        if key != -1:
            if key in (curses.KEY_RIGHT, curses.KEY_UP):
                state.set_direction(+1)
            elif key in (curses.KEY_LEFT, curses.KEY_DOWN):
                state.set_direction(-1)
            elif key == ord(' '):
                state.set_direction(0)
            elif key == curses.KEY_HOME:
                state.jump_to(MAX_DIST)
            elif key == curses.KEY_END:
                state.jump_to(MIN_DIST)
            elif key in (ord('r'), ord('R')):
                state.jump_to(0.05)
            elif key in (ord('q'), ord('Q')):
                quit_flag = True

        target, encoder, ts = state.snapshot()
        age = time.time() - ts if ts else None
        if encoder is None:
            enc_str = "waiting..."
        elif age is not None and age > 1.0:
            enc_str = f"{encoder:.4f} m (stale {age:.1f}s)"
        else:
            enc_str = f"{encoder:.4f} m"

        filled = int((target - MIN_DIST) / (MAX_DIST - MIN_DIST) * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)

        stdscr.erase()
        stdscr.addstr(0, 0, "DAS Finger Controller — Keyboard Teleop Demo")
        stdscr.addstr(1, 0, f"Side: {side}   Port: {port}")
        stdscr.addstr(2, 0, "Arrows=open/close  Space=stop  Home/End=full open/close  R=center  Q=quit")
        stdscr.addstr(4, 0, f"[{bar}]")
        stdscr.addstr(5, 0, f"target:  {target:.4f} m")
        stdscr.addstr(6, 0, f"encoder: {enc_str}")
        stdscr.refresh()

    return


def run_ui_dual(stdscr, databus_left: DataBus, databus_right: DataBus,
                 state_left: TeleopState, state_right: TeleopState,
                 left_port: str, right_port: str):
    """Dual-finger UI: WASD drives the left finger, arrow keys drive the
    right finger, at the same time. Home/End/R/Space/Q act on both."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)
    stdscr.keypad(True)

    bar_width = 30
    quit_flag = False

    while not quit_flag:
        try:
            key = stdscr.getch()
        except curses.error:
            key = -1

        if key != -1:
            if key in (curses.KEY_RIGHT, curses.KEY_UP):
                state_right.set_direction(+1)
            elif key in (curses.KEY_LEFT, curses.KEY_DOWN):
                state_right.set_direction(-1)
            elif key in (ord('d'), ord('D'), ord('w'), ord('W')):
                state_left.set_direction(+1)
            elif key in (ord('a'), ord('A'), ord('s'), ord('S')):
                state_left.set_direction(-1)
            elif key == ord(' '):
                state_left.set_direction(0)
                state_right.set_direction(0)
            elif key == curses.KEY_HOME:
                state_left.jump_to(MAX_DIST)
                state_right.jump_to(MAX_DIST)
            elif key == curses.KEY_END:
                state_left.jump_to(MIN_DIST)
                state_right.jump_to(MIN_DIST)
            elif key in (ord('r'), ord('R')):
                state_left.jump_to(0.05)
                state_right.jump_to(0.05)
            elif key in (ord('q'), ord('Q')):
                quit_flag = True

        target_l, encoder_l, ts_l = state_left.snapshot()
        target_r, encoder_r, ts_r = state_right.snapshot()

        def _enc_str(encoder, ts):
            age = time.time() - ts if ts else None
            if encoder is None:
                return "waiting..."
            if age is not None and age > 1.0:
                return f"{encoder:.4f} m (stale {age:.1f}s)"
            return f"{encoder:.4f} m"

        filled_l = int((target_l - MIN_DIST) / (MAX_DIST - MIN_DIST) * bar_width)
        bar_l = "#" * filled_l + "-" * (bar_width - filled_l)
        filled_r = int((target_r - MIN_DIST) / (MAX_DIST - MIN_DIST) * bar_width)
        bar_r = "#" * filled_r + "-" * (bar_width - filled_r)

        stdscr.erase()
        stdscr.addstr(0, 0, "DAS Finger Controller — Dual Keyboard Teleop Demo")
        stdscr.addstr(1, 0, f"Left port: {left_port}   Right port: {right_port}")
        stdscr.addstr(2, 0, "WASD=left finger  Arrows=right finger  Space=stop both")
        stdscr.addstr(3, 0, "Home/End=both full open/close  R=center both  Q=quit")
        stdscr.addstr(5, 0, f"L [{bar_l}]")
        stdscr.addstr(6, 0, f"L target:  {target_l:.4f} m")
        stdscr.addstr(7, 0, f"L encoder: {_enc_str(encoder_l, ts_l)}")
        stdscr.addstr(9, 0, f"R [{bar_r}]")
        stdscr.addstr(10, 0, f"R target:  {target_r:.4f} m")
        stdscr.addstr(11, 0, f"R encoder: {_enc_str(encoder_r, ts_r)}")
        stdscr.refresh()

    return


def run_single(args):
    """Single-finger keyboard teleop."""
    port = args.port or SIDE_PORTS[args.side]

    print(f"Connecting to {port} ...")
    state = TeleopState(start_pos=max(MIN_DIST, min(MAX_DIST, args.start)),
                         step_rate=args.step_rate)
    databus = _open_databus(port, state)

    ctrl_thread = threading.Thread(target=control_loop, args=(databus, state, args.hz), daemon=True)
    ctrl_thread.start()

    try:
        curses.wrapper(run_ui, databus, state, args.side, port)
    finally:
        print("\nShutting down: disabling motor and closing serial port...")
        state.running = False
        time.sleep(0.1)
        try:
            databus.disable_motor()
            time.sleep(0.1)
        except Exception:
            pass
        databus.stop()
        print("Done.")


def run_dual(args):
    """Dual-finger keyboard teleop: WASD drives the left finger, arrow
    keys drive the right finger, both at the same time."""
    left_port = args.left_port or SIDE_PORTS["left"]
    right_port = args.right_port or SIDE_PORTS["right"]

    print(f"Connecting to left finger on {left_port} and right finger on {right_port} ...")
    state_left = TeleopState(start_pos=max(MIN_DIST, min(MAX_DIST, args.start)),
                              step_rate=args.step_rate)
    state_right = TeleopState(start_pos=max(MIN_DIST, min(MAX_DIST, args.start)),
                               step_rate=args.step_rate)
    databus_left = _open_databus(left_port, state_left)
    databus_right = _open_databus(right_port, state_right)

    threading.Thread(
        target=control_loop, args=(databus_left, state_left, args.hz), daemon=True
    ).start()
    threading.Thread(
        target=control_loop, args=(databus_right, state_right, args.hz), daemon=True
    ).start()

    try:
        curses.wrapper(
            run_ui_dual, databus_left, databus_right, state_left, state_right,
            left_port, right_port,
        )
    finally:
        print("\nShutting down: disabling motors and closing serial ports...")
        state_left.running = False
        state_right.running = False
        time.sleep(0.1)
        for databus in (databus_left, databus_right):
            try:
                databus.disable_motor()
                time.sleep(0.1)
            except Exception:
                pass
            databus.stop()
        print("Done.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Keyboard teleop demo for DAS Finger Controller (curses input)"
    )
    parser.add_argument("side", choices=["left", "right", "dual"],
                         help="Which finger to drive. 'dual' teleops left+right "
                              "fingers at once: WASD for left, arrow keys for right.")
    parser.add_argument("--port", type=str, default=None,
                         help="Serial port override for single-finger mode (left/right).")
    parser.add_argument("--left-port", type=str, default=None,
                         help="Serial port override for the left finger in 'dual' mode.")
    parser.add_argument("--right-port", type=str, default=None,
                         help="Serial port override for the right finger in 'dual' mode.")
    parser.add_argument("--step-rate", type=float, default=0.08,
                         help="Meters/second of travel while a direction key is held "
                              "(default 0.08 — scaled up from the gripper version's 0.05 "
                              "since the finger's travel range is roughly double)")
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--hz", type=float, default=30.0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.side == "dual":
        run_dual(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()