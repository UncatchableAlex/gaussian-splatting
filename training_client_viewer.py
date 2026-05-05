"""
Script written by Claude and debugged by Alex

Training viewer client for 3DGS network_gui.
Connects to the training loop socket server, requests rendered frames,
and saves them as PNGs or pipes to ffmpeg for a video.

Usage:
    # Save a single snapshot at the current orbit angle
    python training_viewer.py --mode snapshot

    # Save N frames as numbered PNGs (no ffmpeg needed)
    python training_viewer.py --mode pngs --frames 60 --output ./frames/

    # Record a full orbit as an mp4 (requires ffmpeg)
    python training_viewer.py --mode video --frames 120 --output orbit.mp4

    # Keep looping, overwriting snapshot.png — useful to watch live in an
    # image viewer that auto-refreshes (e.g. feh --auto-zoom --reload 1)
    python training_viewer.py --mode live
"""

import socket
import json
import struct
import argparse
import os
import subprocess
import time

import numpy as np

# ── reuse your existing camera helpers ──────────────────────────────────────
from futhark_3dgs.util import look_at, getProjectionMatrix


HOST = "127.0.0.1"
PORT = 6009

# ── camera defaults  ─────────────────────────────────────────
WIDTH  = 980
HEIGHT = 545
FOVX   = 1.4028140929797817
FOVY   = 0.8753571332164317
ZNEAR  = 0.01
ZFAR   = 100.0

ORBIT_RADIUS    = 5.0
ORBIT_ELEVATION = -1.0
ORBIT_TARGET    = np.array([0.0, 0.0, 0.0], dtype=np.float32)
ORBIT_UP        = np.array([0.0, 1.0, 0.0], dtype=np.float32)
FFMPEG          = '/home/mjk711/ffmpeg-master-latest-linux64-gpl-shared/bin/ffmpeg'
RECV_TIMEOUT = 5.0  # seconds to wait before assuming the server is stuck


# ── protocol helpers ─────────────────────────────────────────────────────────

class Reconnected(Exception):
    pass

def connect_with_retry(host: str, port: int, interval: float = 2.0) -> socket.socket:
    print(f"Waiting for training server at {host}:{port} …")
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print("Connected.")
            return sock
        except ConnectionRefusedError:
            sock.close()
            print(f"  not up yet, retrying in {interval}s …", end="\r")
            time.sleep(interval)

def build_message(view_mat: np.ndarray, proj_mat: np.ndarray,
                  do_training: bool = True, keep_alive: bool = True) -> bytes:
    """
    Serialise a camera request to the JSON wire format that network_gui.read()
    expects.  The server negates columns 1 and 2 of the received view matrix,
    so we pre-negate them here so the renderer sees the correct transform.
    """
    # pre-negate cols 1 & 2 so server's flip gives back the original
    wire_view = view_mat.copy()
    wire_view[:, 1] = -wire_view[:, 1]
    wire_view[:, 2] = -wire_view[:, 2]

    wire_proj = proj_mat.copy()
    wire_proj[:, 1] = -wire_proj[:, 1]   # match server expectation

    payload = json.dumps({
        "resolution_x":        WIDTH,
        "resolution_y":        HEIGHT,
        "train":               do_training,
        "fov_y":               float(FOVY),
        "fov_x":               float(FOVX),
        "z_near":              float(ZNEAR),
        "z_far":               float(ZFAR),
        "shs_python":          True,
        "rot_scale_python":    False,
        "keep_alive":          keep_alive,
        "scaling_modifier":    1.0,
        "view_matrix":         wire_view.flatten().tolist(),
        "view_projection_matrix": wire_proj.flatten().tolist(),
    }).encode("utf-8")

    return len(payload).to_bytes(4, "little") + payload


def recv_exact(sock: socket.socket, n: int):
    host, port = sock.getpeername()
    buf = b""
    print(f"reading {n} bytes from socket", end="\n", flush=True)
    sock.settimeout(RECV_TIMEOUT)
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            print(f"Timed out waiting for bytes ({len(buf)}/{n}), retrying...")
            raise Reconnected(connect_with_retry(host, port))
        if not chunk:
            print("Socket closed, reconnecting...")
            sock = connect_with_retry(host, port)
            raise Reconnected(sock)
        buf += chunk
        print(f"{len(buf)}/{n}", end="\n", flush=True)

    sock.settimeout(None)  # restore blocking mode
    return buf


def recv_frame(sock: socket.socket) -> np.ndarray:
    """
    Read one rendered frame back from the server.
    network_gui.send() writes:
        image_bytes  (WIDTH * HEIGHT * 3 bytes, uint8 RGB)
        4-byte LE length of verify string
        verify string (ascii)
    """
    n_pixels = WIDTH * HEIGHT * 3
    raw = recv_exact(sock, n_pixels)

    # read and discard the verify string
    verify = recv_exact(sock, 4)
    vlen = struct.unpack("<I", verify)[0]
    recv_exact(sock, vlen)

    frame = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    return frame


def camera_matrices(angle_rad: float):
    """Return (view_matrix, proj_matrix) for an orbit angle."""
    eye = np.array([
        ORBIT_RADIUS * np.cos(angle_rad),
        ORBIT_ELEVATION,
        ORBIT_RADIUS * np.sin(angle_rad),
    ], dtype=np.float32)

    view = np.array(look_at(eye, ORBIT_TARGET, ORBIT_UP).T, dtype=np.float32)
    proj_only = np.array(getProjectionMatrix(ZNEAR, ZFAR, FOVX, FOVY).T,
                         dtype=np.float32)
    full_proj = view @ proj_only
    return view, full_proj


def save_png(frame: np.ndarray, path: str):
    from PIL import Image
    Image.fromarray(frame, "RGB").save(path)
    print(f"  saved {path}")


# ── modes ────────────────────────────────────────────────────────────────────

def run_snapshot(sock: socket.socket, output: str, angle: float = 0.0):
    view, proj = camera_matrices(angle)
    sock.sendall(build_message(view, proj, keep_alive=True))
    frame = recv_frame(sock)
    save_png(frame, output)


def run_pngs(sock: socket.socket, frames: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(frames):
        angle = 2 * np.pi * (i / frames)
        view, proj = camera_matrices(angle)
        is_last = (i == frames - 1)
        sock.sendall(build_message(view, proj, keep_alive=not is_last))
        frame = recv_frame(sock)
        save_png(frame, os.path.join(output_dir, f"frame_{i:05d}.png"))


def run_video(sock: socket.socket, frames: int, output: str, fps: int = 30, stride : int = 1, orbits : int = 1):
    ffmpeg_cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(fps),
        "-i", "-",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        output,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=None, stderr=None)
    
    i = 0
    while i < frames:
        angle = 2 * np.pi * ((orbits * i) / frames)
        view, proj = camera_matrices(angle)
        is_last = (i == frames - 1)
        try:
            sock.sendall(build_message(view, proj, keep_alive=not is_last))
            print("recv frame", end="\n", flush=True)
            frame = recv_frame(sock)
            print("frame recv'd", end="\n", flush=True)
        except Reconnected as e:
            print("reconnected exception caught. continuing with new socket")
            sock = e.args[0]   # use the new socket, re-send on next iteration
            continue
        if frames % stride == 0:
            print("writing to ffmpeg", end="\n", flush=True)
            proc.stdin.write(frame.tobytes())
            print(f"\r  frame {i}/{frames}", end="\n", flush=True)
        i += 1

    proc.stdin.close()
    proc.wait()
    print(f"\n  saved {output}")


def run_live(sock: socket.socket, output: str, interval: float = 0.5):
    """
    Continuously overwrite a single PNG.  Open it in an auto-refreshing
    viewer (e.g. `feh --auto-zoom --reload 1 snapshot.png`) to watch
    training progress in real time.
    """
    print(f"Live mode — writing to {output}  (Ctrl-C to stop)")
    angle = 0.0
    try:
        while True:
            view, proj = camera_matrices(angle)
            sock.sendall(build_message(view, proj, keep_alive=True))
            frame = recv_frame(sock)
            save_png(frame, output)
            angle = (angle + 0.05) % (2 * np.pi)   # slow drift
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["snapshot", "pngs", "video", "live"],
                        default="snapshot")
    parser.add_argument("--frames", type=int,   default=120)
    parser.add_argument("--fps",    type=int,   default=30)
    parser.add_argument("--output", type=str,   default=None)
    parser.add_argument("--angle",  type=float, default=0.0,
                        help="Orbit angle in radians (snapshot mode only)")
    parser.add_argument("--host",   type=str,   default=HOST)
    parser.add_argument("--port",   type=int,   default=PORT)
    parser.add_argument("--stride", type=int,   default=1)
    parser.add_argument("--orbits", type=int,   default=1,
                        help="How many times around the center of the scene the camera will orbit. Default: 1")
    
    
    args = parser.parse_args()

    # sensible output defaults per mode
    if args.output is None:
        args.output = {
            "snapshot": "snapshot.png",
            "pngs":     "./frames/",
            "video":    "./scene_captures/orbit.mp4",
            "live":     "snapshot.png",
        }[args.mode]

    print(f"Connecting to {args.host}:{args.port} …")
    sock = connect_with_retry(args.host, args.port)
    print("Connected.")

    if args.mode == "snapshot":
        run_snapshot(sock, args.output, args.angle)
    elif args.mode == "pngs":
        run_pngs(sock, args.frames, args.output)
    elif args.mode == "video":
        run_video(sock, args.frames, args.output, args.fps, args.stride, args.orbits)
    elif args.mode == "live":
        run_live(sock, args.output)

    sock.close()


if __name__ == "__main__":
    main()

# srun --partition=gpu --gres=gpu:titanrtx:1 --time=01:00:00 --mem=12G --cpus-per-task=2 --pty bash