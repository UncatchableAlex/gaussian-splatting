import numpy as np
import subprocess
from tqdm import tqdm
from gaussian_renderer import GaussianModel
from futhark_3dgs import Futhark_Rasterization_Server
from futhark_3dgs.util import look_at, getProjectionMatrix
import torch
from utils.sh_utils import eval_sh
import json


ply_path = './output/be9fc165-5/point_cloud/iteration_10000/point_cloud.ply'
rasterizer_inps = './submodules/futhark-3dgs/rasterizer_inps'

# the number of frames to render
frames = 400

# the fps of the resulting video
fps = 30

# how far the camera will be from the center of rotation
r = 5

# center of camera path
target = np.array([0,0,0])

# up
up = np.array([0,1,0])

camera_elevation = -1


# how many camera z-axes we are away from our target
lambd = 1

def to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().astype(np.float32)
    return np.array(v).astype(np.float32)


json_name = 'debug_rasterizer_settings.json'

# load the inputs that get fed to the rasterizer.
pc = GaussianModel(sh_degree=3)
pc.load_ply(ply_path)

# code stolen from 3dgs to convert spherical harmonics to rbg
shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
dir_pp = (pc.get_xyz - torch.tensor([0.4679788053035736, 0.34583476185798645, 3.472193956375122], device='cuda').repeat(pc.get_features.shape[0], 1))
dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)


n = 200_000 # how many gaussians we want to render


test_forward = False

view_matrix = np.zeros((4,4), dtype=np.float32)
proj_matrix = np.zeros((4,4), dtype=np.float32)

# load the inputs that get fed to the rasterizer.
inps = {}
with open(f'{rasterizer_inps}/{json_name}', 'r') as f:
    json_data = json.load(f)
    inps.update(json_data)

# prep  inputs as numpy arrays with correct dtypes
inputs = {
    'bg':           np.array([0,0,0],                       dtype=np.float32),
    'means3D':      to_numpy(pc.get_xyz)[:n],
    'colors':       to_numpy(colors_precomp)[:n],
    'opacities':    to_numpy(pc.get_opacity)[:n],
    'scales':       to_numpy(pc.get_scaling)[:n],
    'rotations':    to_numpy(pc.get_rotation)[:n],
    'viewmatrix':   np.array(inps['viewmatrix'],   dtype=np.float32),
    'projmatrix':   np.array(inps['projmatrix'],   dtype=np.float32),
    'tanfovx':      np.float32(inps['tanfovx']),
    'tanfovy':      np.float32(inps['tanfovy']),
    'image_height': np.int64(inps['image_height']),
    'image_width':  np.int64(inps['image_width']),
}    
for key,value in inputs.items():
    try:
        print(key, value.shape)
    except:
        print('not numpy')

# extract the actual projection matrix from the fused projmatrix


fovx = 1.4028140929797817
fovy = 0.8753571332164317
true_proj = getProjectionMatrix(0.01, 100, fovx, fovy).T


server = Futhark_Rasterization_Server()

# store each input as a named variable
for name, value in inputs.items():
    server.put_value(name, value)

height = int(inputs['image_height'])
width = int(inputs['image_width'])

# chatgpt ffmpeg magic
ffmpeg_cmd = [
    "ffmpeg",
    "-y",  # overwrite
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "rgb24",
    "-s", f"{width}x{height}",
    "-r", f"{fps}",  # framerate
    "-i", "-",   # stdin
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # pad to satisfy H.264
    "-an",
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    "output.mp4"
]

ffmpeg_proc = subprocess.Popen(
    ffmpeg_cmd, 
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL, # silence annoying ffmpeg stdout/stderr chatter
    stderr=subprocess.DEVNULL)

i = 0
pbar = tqdm(total=frames)

for i in range(frames):
    eye = np.array([
        r * np.cos(np.pi * 2 * (i/frames)),       # X
        camera_elevation,  # height above ground (constant)
        r * np.sin(np.pi * 2 *  (i/frames))])      # Z
    
    view_matrix = np.array(look_at(eye, target,up).T, np.float32)
    proj_matrix = np.array(view_matrix @ true_proj, np.float32)
    # free both view and proj matrix
    server.cmd_free('viewmatrix')
    server.cmd_free('projmatrix')

    # put new view and projection matrices in
    server.put_value('viewmatrix', view_matrix)
    server.put_value('projmatrix', proj_matrix)
    
    # call the function: outputs first, then inputs
    server.cmd_call(
        "rasterize",
        'radii',
        'pixels',                 
        *inputs.keys()
    )
    frame = server.get_value('pixels')
    # force-feed this batch of rasterized images into ffmpeg

    # ensure uint8 RGB
    if frame.dtype != np.uint8:
        frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)

    ffmpeg_proc.stdin.write(frame.tobytes())
    pbar.update(1)
    
    # free our result
    server.cmd_free('pixels')
    server.cmd_free('radii')

# close our ffmpeg process
ffmpeg_proc.stdin.close()
ffmpeg_proc.wait()
pbar.close()
        