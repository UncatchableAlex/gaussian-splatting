import numpy as np
import subprocess
from tqdm import tqdm
from gaussian_renderer import GaussianModel
from futhark_3dgs import Futhark_Rasterization_Server
from futhark_3dgs.util import look_at, getProjectionMatrix
import torch
from utils.sh_utils import eval_sh
import json
import argparse

PLY_PATH = './output/_futhark20000_8k/point_cloud/iteration_20000/point_cloud.ply'
OUTPUT_DIR = "/home/mjk711/gaussian-splatting/scene_captures/futhark20000_low.mp4"

# ── camera defaults  ─────────────────────────────────────────
WIDTH  = 980
HEIGHT = 545
FOVX   = 1.4028140929797817
FOVY   = 0.8753571332164317
ZNEAR  = 0.01
ZFAR   = 100.0

ORBIT_UP        = np.array([0.0, 1.0, 0.0], dtype=np.float32)
FFMPEG          = '/home/mjk711/ffmpeg-master-latest-linux64-gpl-shared/bin/ffmpeg'


tanfovx = np.tan(FOVX * 0.5)
tanfovy = np.tan(FOVY * 0.5)


def to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().astype(np.float32)
    return np.array(v).astype(np.float32)



def camera_matrices(args, target, angle_rad: float):
    """Return (view_matrix, proj_matrix) for an orbit angle."""
    eye = np.array([
        args.radius * np.cos(angle_rad),
        args.elevation,
        args.radius * np.sin(angle_rad),
    ], dtype=np.float32)

    view = np.array(look_at(eye, target, ORBIT_UP).T, dtype=np.float32)
    proj_only = np.array(getProjectionMatrix(ZNEAR, ZFAR, FOVX, FOVY).T,
                         dtype=np.float32)
    full_proj = view @ proj_only
    return view, full_proj



def make_video(args):

    # load the inputs that get fed to the rasterizer.
    pc = GaussianModel(sh_degree=3)
    pc.load_ply(args.input)

    # code stolen from 3dgs to convert spherical harmonics to rbg
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - torch.tensor([0.4679788053035736, 0.34583476185798645, 3.472193956375122], device='cuda').repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    target  = np.array([0.0, args.elevation, 0.0], dtype=np.float32)

    # prep  inputs as numpy arrays with correct dtypes
    inputs = {
        'bg':           np.array([0,0,0],                       dtype=np.float32),
        'means3D':      to_numpy(pc.get_xyz),
        'colors':       to_numpy(colors_precomp),
        'opacities':    to_numpy(pc.get_opacity),
        'scales':       to_numpy(pc.get_scaling),
        'rotations':    to_numpy(pc.get_rotation),
        'viewmatrix':   np.array([],   dtype=np.float32),
        'projmatrix':   np.array([],   dtype=np.float32),
        'tanfovx':      np.float32(tanfovx),
        'tanfovy':      np.float32(tanfovy),
        'image_height': np.int64(HEIGHT),
        'image_width':  np.int64(WIDTH),
    }    

    for key,value in inputs.items():
        try:
            print(key, value.shape)
        except:
            print('not numpy')


    server = Futhark_Rasterization_Server()

    # store each input as a named variable
    for name, value in inputs.items():
        server.put_value(name, value)

    # chatgpt ffmpeg magic
    ffmpeg_cmd = [
        FFMPEG,
        "-y",  # overwrite
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", f"{args.fps}",  # framerate
        "-i", "-",   # stdin
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # pad to satisfy H.264
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        args.output
    ]

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, 
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, # silence annoying ffmpeg stdout/stderr chatter
        stderr=None)

    i = 0
    pbar = tqdm(total=args.frames)

    for i in range(args.frames):
        view_matrix, proj_matrix = camera_matrices(args, target, np.pi * 2 * (i*args.orbits/args.frames))
        
        # free both view and proj matrix
        server.cmd_free('viewmatrix')
        server.cmd_free('projmatrix')

        # put new view and projection matrices in
        server.put_value('viewmatrix', view_matrix)
        server.put_value('projmatrix', proj_matrix)
        
        # call the function: outputs first, then inputs
        server.cmd_call(
            "rasterize",
            'out',                 
            *inputs.keys()
        )
        (_, frame, _) = server.get_value('out')

        # free the result
        server.cmd_free('out')

        # force-feed this batch of rasterized images into ffmpeg
        # ensure uint8 RGB
        if frame.dtype != np.uint8:
            frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)

        ffmpeg_proc.stdin.write(frame.tobytes())
        pbar.update(1)
        

    # close our ffmpeg process
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    pbar.close()


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int,   default=400)
    parser.add_argument("--fps",    type=int,   default=30)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str,   default=None)
    parser.add_argument("--orbits", type=int,   default=1,
                        help="How many times around the center of the scene the camera will orbit. Default: 1")
    parser.add_argument("--elevation", type=float, default=-0.3)
    parser.add_argument("--radius", type=int, default=5)

    
    args = parser.parse_args()
    make_video(args)


if __name__ == "__main__":
    main()

# srun --partition=gpu --gres=gpu:titanrtx:1 --time=01:00:00 --mem=12G --cpus-per-task=2 --pty bash