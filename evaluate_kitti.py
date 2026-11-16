from __future__ import print_function

import argparse
import os
from collections import OrderedDict, deque

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from configs.config_setting import setting_config as config
from dataset import input_transform
from data_io import read_paths
from checkpoint import load_weights


class AdaptiveProbVPR:
    def __init__(self, db_coords, sigma_ret=3.0, sigma_mot=8.0, r_min=10.0, r_max=300.0):
        self.db_coords = db_coords
        self.sigma_ret = sigma_ret
        self.sigma_mot = sigma_mot
        self.r_min = r_min
        self.r_max = r_max
        self.last_state_pos = None
        self.current_weight = 0.5
        self.window_size = 7

    def get_dynamic_radius(self):
        k = 5.0
        return self.r_min + (self.r_max - self.r_min) * np.exp(-k * self.current_weight)

    def compute_prob_weight(self, retrieval_dist, sliding_window):
        w_ret = np.exp(-(retrieval_dist ** 2) / (2 * self.sigma_ret ** 2))

        if len(sliding_window) < self.window_size:
            w_mot = 0.5
        else:
            pts = np.asarray(sliding_window, dtype=np.float32)
            t = np.arange(len(pts)).reshape(-1, 1)
            reg_x = LinearRegression().fit(t[:-1], pts[:-1, 0])
            reg_y = LinearRegression().fit(t[:-1], pts[:-1, 1])
            pred_pos = np.array([
                reg_x.predict([[self.window_size - 1]])[0],
                reg_y.predict([[self.window_size - 1]])[0],
            ])
            residual = np.linalg.norm(pts[-1] - pred_pos)
            w_mot = np.exp(-(residual ** 2) / (2 * self.sigma_mot ** 2))

        w_total = float(w_ret * w_mot)
        self.current_weight = w_total
        return w_total, float(w_ret), float(w_mot)


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_state_dict[key.replace("module.", "")] = value
    return new_state_dict


def frame_id(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = load_weights(model, checkpoint_path, device)
    if isinstance(checkpoint, dict):
        print("loaded checkpoint:", checkpoint_path, "epoch:", checkpoint.get("epoch", "unknown"))
    else:
        print("loaded checkpoint:", checkpoint_path)


def extract_lidar_descriptor(model, path, device):
    head, tail = os.path.split(path)
    sam_path = os.path.join(head.replace("lidar", "lidarSAM_t"), tail)
    depth_path = os.path.join(head.replace("lidar", "lidar_D50t"), tail)

    sam_pos = np.array(Image.open(sam_path).convert("RGB")).astype(np.float32)
    sam_pos = input_transform()(sam_pos).to(device).unsqueeze(0)
    depth = np.array(Image.open(depth_path)).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    lidar = np.array([depth]).transpose([1, 2, 0]).repeat(3, 2).astype(np.float32)
    lidar = input_transform()(lidar).to(device).unsqueeze(0)

    with torch.no_grad():
        _, desc = model(torch.cat([lidar, sam_pos]))
    return desc[0].cpu().numpy()


def extract_rgb_descriptor(model, path, device):
    head, tail = os.path.split(path)
    sam_path = os.path.join(head.replace("image_2", "imgSAM"), tail)

    sam_q = np.array(Image.open(sam_path).convert("RGB")).astype(np.float32)
    sam_q = input_transform()(sam_q).to(device).unsqueeze(0)
    rgb_img = np.array(Image.open(path)).astype(np.float32)[165:]
    rgb_img = cv2.resize(rgb_img, config.resize_shape)
    rgb = input_transform()(rgb_img).to(device).unsqueeze(0)

    with torch.no_grad():
        _, desc = model(torch.cat([rgb, sam_q]))
    return desc[0].cpu().numpy()


def evaluate_sequence(model, seq, device, topks, threshold, use_probabilistic):
    query_path = f"{config.dataset_toot}/sequences/{seq:02d}/image_2.txt"
    database_path = f"{config.dataset_toot}/sequences/{seq:02d}/lidar.txt"
    pose_path = f"{config.dataset_toot}/poses/{seq:02d}.txt"

    with open(query_path, "r") as f:
        query = read_paths(query_path, config.dataset_toot)
    with open(database_path, "r") as f:
        database = read_paths(database_path, config.dataset_toot)
    with open(pose_path, "r") as f:
        poses = [line.strip().split() for line in f.readlines()]

    deslen = 16384 * 2
    max_topk = min(max(topks), len(database))
    des_list_full = np.zeros((len(database), deslen), dtype="float32")
    db_coords = np.zeros((len(database), 2), dtype=np.float32)

    print(f"Building HCMC database for Seq {seq:02d}...")
    for i in tqdm(range(len(database)), desc=f"build db {seq:02d}"):
        db_idx = frame_id(database[i])
        db_coords[i] = [float(poses[db_idx][config.pose_cols[0]]), float(poses[db_idx][config.pose_cols[1]])]
        des_list_full[i] = extract_lidar_descriptor(model, database[i], device)

    global_faiss = faiss.IndexFlatL2(deslen)
    global_faiss.add(des_list_full)

    vpr_manager = AdaptiveProbVPR(db_coords)
    sliding_window = deque(maxlen=7)
    correct = {k: 0 for k in topks}
    top1_errors = []
    weights = []

    print(f"Testing HCMC sequence {seq:02d}...")
    for i in tqdm(range(len(query)), desc=f"query {seq:02d}"):
        q_des = extract_rgb_descriptor(model, query[i], device).reshape(1, -1).astype("float32")

        m_idx = -1
        match_dist = None
        if use_probabilistic:
            radius = vpr_manager.get_dynamic_radius()
            if vpr_manager.last_state_pos is not None and vpr_manager.current_weight > 0.2:
                dists = np.linalg.norm(db_coords - vpr_manager.last_state_pos, axis=1)
                local_mask = np.where(dists <= radius)[0]
                if len(local_mask) > 0:
                    local_des = des_list_full[local_mask]
                    errs = np.linalg.norm(local_des - q_des, axis=1)
                    order = np.argsort(errs)
                    ranked = local_mask[order[:max_topk]]
                    m_idx = int(ranked[0])
                    match_dist = float(errs[order[0]])

        if m_idx == -1:
            D, I = global_faiss.search(q_des, max_topk)
            ranked = I[0]
            m_idx = int(ranked[0])
            match_dist = float(np.sqrt(max(D[0][0], 0.0)))

        if use_probabilistic:
            curr_raw_pos = db_coords[m_idx]
            sliding_window.append(curr_raw_pos)
            weight, _, _ = vpr_manager.compute_prob_weight(match_dist, list(sliding_window))
            weights.append(weight)

            if vpr_manager.last_state_pos is None:
                vpr_manager.last_state_pos = curr_raw_pos
            else:
                vpr_manager.last_state_pos = (1.0 - weight) * vpr_manager.last_state_pos + weight * curr_raw_pos

            candidate_coords = db_coords[ranked].copy()
            candidate_coords[0] = vpr_manager.last_state_pos
        else:
            candidate_coords = db_coords[ranked]

        q_idx = frame_id(query[i])
        gt_pos = np.array([float(poses[q_idx][config.pose_cols[0]]), float(poses[q_idx][config.pose_cols[1]])], dtype=np.float32)
        errors = np.linalg.norm(candidate_coords - gt_pos, axis=1)
        top1_errors.append(float(errors[0]))

        for k in topks:
            if np.any(errors[:k] < threshold):
                correct[k] += 1

    recalls = {k: correct[k] / len(query) for k in topks}
    recall_text = " | ".join([f"recall@{k} {recalls[k]:.4f}" for k in topks])
    weight_text = f" | mean weight {np.mean(weights):.4f}" if weights else ""
    print(
        f"seq {seq:02d} | queries {len(query)} | {recall_text} | "
        f"top1 mean error {np.mean(top1_errors):.3f}m | "
        f"top1 median error {np.median(top1_errors):.3f}m"
        f"{weight_text}"
    )

    return len(query), recalls


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a train_hcmc.py checkpoint on KITTI.")
    parser.add_argument("--checkpoint", required=True, help="Path to HCMC checkpoint.")
    parser.add_argument("--root", required=True, help="KITTI dataset root.")
    parser.add_argument("--pose-cols", nargs=2, type=int, default=[3, 11])
    parser.add_argument("--seqs", nargs="+", type=int, default=[0, 2, 5, 6, 8], help="KITTI sequences to test.")
    parser.add_argument("--topks", nargs="+", type=int, default=[1, 5, 20], help="Recall@K values.")
    parser.add_argument("--threshold", type=float, default=10.0, help="Distance threshold in meters.")
    parser.add_argument(
        "--no-probabilistic",
        action="store_true",
        help="Disable the adaptive probabilistic local search used by test_debug.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config.dataset_toot = args.root
    config.pose_cols = args.pose_cols
    from models.model import Backbone
    print("PID:", os.getpid())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Backbone(nmf=config.nmf, mlp=config.mlp).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    all_results = []
    for seq in args.seqs:
        all_results.append(
            evaluate_sequence(
                model=model,
                seq=seq,
                device=device,
                topks=args.topks,
                threshold=args.threshold,
                use_probabilistic=not args.no_probabilistic,
            )
        )

    if len(all_results) > 1:
        total = sum(num for num, _ in all_results)
        merged = {}
        for k in args.topks:
            merged[k] = sum(num * recalls[k] for num, recalls in all_results) / total
        recall_text = " | ".join([f"recall@{k} {merged[k]:.4f}" for k in args.topks])
        print(f"all | queries {total} | {recall_text}")


if __name__ == "__main__":
    main()
