import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer, cat_params_to_optimizer
from flame.flame import FlameHead
import debug


def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md, include_flame=True):
    init_pt_cld = np.load(f"./data/{seq}/init_pt_cld.npz")["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    
    if include_flame:
        # All flame params
        params['flame_rotation'] = torch.nn.Parameter(torch.zeros((3)).cuda().float().contiguous())
        params['flame_translation'] = torch.nn.Parameter(torch.zeros((3)).cuda().float().contiguous())
        params['flame_scale'] = torch.nn.Parameter(torch.zeros((1)).cuda().float().contiguous())
        params['flame_neck_pose'] = torch.nn.Parameter(torch.zeros((3)).cuda().float().contiguous())
        params['flame_jaw_pose'] = torch.nn.Parameter(torch.zeros((3)).cuda().float().contiguous())
        params['flame_eyes_pose'] = torch.nn.Parameter(torch.zeros((6)).cuda().float().contiguous())
        params['flame_shape'] = torch.nn.Parameter(torch.zeros((300)).cuda().float().contiguous())
        params['flame_expr'] = torch.nn.Parameter(torch.zeros((100)).cuda().float().contiguous())
        params['flame_texture'] = torch.nn.Parameter(torch.zeros((100)).cuda().float().contiguous())
        # Light exists but we're not rendering

        # We don't optimize most of them
        params['flame_neck_pose'].requires_grad = False
        params['flame_jaw_pose'].requires_grad = False
        params['flame_eyes_pose'].requires_grad = False
        params['flame_shape'].requires_grad = False
        params['flame_expr'].requires_grad = False
        params['flame_texture'].requires_grad = False

    return params, variables


def initialize_optimizer(params, variables, include_flame=True):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [params[name]], 'name': name, 'lr': lr} for name, lr in lrs.items()]

    if include_flame:
        param_groups.append({'params': [params["flame_rotation"]], 'name': "flame_rotation", 'lr': 0.001})
        param_groups.append({'params': [params["flame_translation"]], 'name': "flame_translation", 'lr': 0.001})
        param_groups.append({'params': [params["flame_scale"]], 'name': "flame_scale", 'lr': 0.001})

    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)

def get_flame_params(flame_data, t):
    flame_params = {}
    flame_params["rotation"] = torch.tensor(flame_data["rotation"][t]).cuda()
    flame_params["translation"] = torch.tensor(flame_data["translation"][t]).cuda()
    flame_params["neck_pose"] = torch.tensor(flame_data["neck_pose"][t]).cuda()
    flame_params["jaw_pose"] = torch.tensor(flame_data["jaw_pose"][t]).cuda()
    flame_params["eyes_pose"] = torch.tensor(flame_data["eyes_pose"][t]).cuda()
    flame_params["expr"] = torch.tensor(flame_data["expr"][t]).cuda()
    flame_params["shape"] = torch.tensor(flame_data["shape"]).cuda()
    flame_params["scale"] = torch.tensor(flame_data["scale"]).cuda()

    return flame_params

def get_flame_mesh_and_landmarks(flame_head, flame_params):
    vertices, landmarks = flame_head(
        flame_params["shape"][None],
        flame_params["expr"][None],
        flame_params["rotation"][None],
        flame_params["neck_pose"][None],
        flame_params["jaw_pose"][None],
        flame_params["eyes_pose"][None],
        flame_params["translation"][None]
    )

    vertices *= flame_params["scale"]
    landmarks *= flame_params["scale"]

    return vertices, landmarks


def insert_flame_mesh_and_params(flame_vertices, params, variables, optimizer):
    sq_dist, _ = o3d_knn(flame_vertices.cpu().numpy(), 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    new_params = {
        'means3D': flame_vertices,
        'rgb_colors': torch.zeros_like(flame_vertices).cuda().float(),
        'seg_colors': torch.zeros_like(flame_vertices).cuda().float(),
        'unnorm_rotations': torch.tensor(np.tile([1, 0, 0, 0], (len(flame_vertices), 1))).cuda().float(),
        'logit_opacities': torch.zeros((len(flame_vertices), 1)).cuda().float(),
        'log_scales': torch.tensor(np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3))).cuda().float(),
    }
    new_params["seg_colors"][:, 2] = 1.0

    cat_params_to_optimizer(new_params, params, optimizer)
    variables["max_2D_radius"] = torch.cat((variables["max_2D_radius"], torch.zeros((len(flame_vertices))).cuda().float()))
    variables["means2D_gradient_accum"] = torch.cat((variables["means2D_gradient_accum"], torch.zeros((len(flame_vertices))).cuda().float()))
    variables["denom"] = torch.cat((variables["denom"], torch.zeros((len(flame_vertices))).cuda().float()))

def train(seq, exp, include_flame=True):
    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))  # metadata
    flame_data = np.load(f"./data/{seq}/tracked_flame_params.npz")
    num_timesteps = len(md['fn'])
    params, variables = initialize_params(seq, md, False)
    optimizer = initialize_optimizer(params, variables, False)
    if include_flame:
        flame_head = FlameHead(len(flame_data["shape"]), len(flame_data["expr"][0])).cuda()
    output_params = []

    if include_flame:
        t = 0
        flame_params_t = get_flame_params(flame_data, t)
        flame_vertices, _ = get_flame_mesh_and_landmarks(flame_head, flame_params_t)
        insert_flame_mesh_and_params(flame_vertices[0], params, variables, optimizer)

    for t in range(num_timesteps):
        dataset = get_dataset(t, md, seq)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = 10000 if is_initial_timestep else 2000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
    save_params(output_params, seq, exp)


if __name__ == "__main__":
    exp_name = "exp1"
    for sequence in ["Subj1Amb"]:
        train(sequence, exp_name)
        torch.cuda.empty_cache()
