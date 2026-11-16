"""Portable HCMC training; mathematical losses/sampling live in hcmc_kitti.py."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoint import load_weights
from hcmc_kitti import KITTIMultiPosGeoNeg, lazy_triplet_loss, hard_aware_cross_modal_contrastive_loss, hard_mid_loss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--train-seq", default="00")
    p.add_argument("--max-frames", type=int, default=3000)
    p.add_argument("--pose-cols", type=int, nargs=2, default=[3, 7],
                   help="Source HCMC sampler used 3,7. Standard KITTI horizontal x,z is 3,11; changing this changes sampling.")
    p.add_argument("--init-checkpoint", default="", help="Initialize model weights; reset epoch and optimizer.")
    p.add_argument("--resume", default="", help="Resume an interrupted training run including optimizer/scheduler.")
    p.add_argument("--backbone-weights", default=None)
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=301)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.4)
    p.add_argument("--mid-margin", type=float, default=0.04)
    p.add_argument("--hcmc-lambda", type=float, default=0.2)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not (args.resume or args.init_checkpoint or args.backbone_weights or args.from_scratch):
        p.error("Pass --init-checkpoint, --resume, --backbone-weights, or --from-scratch")
    if args.resume and args.init_checkpoint:
        p.error("Choose --resume or --init-checkpoint, not both")
    if min(args.epochs, args.batch_size, args.save_every, args.max_frames) < 1:
        p.error("epochs, batch-size, save-every, and max-frames must be positive")
    return args


def main():
    args = parse_args()
    from models.model import Backbone
    if not torch.cuda.is_available():
        raise RuntimeError("HCMC descriptor training requires a CUDA GPU and the Mamba CUDA extensions")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    model = Backbone(pretrained=args.backbone_weights if not (args.resume or args.init_checkpoint) else None).to(device)
    checkpoint = None
    if args.resume or args.init_checkpoint:
        checkpoint = load_weights(model, args.resume or args.init_checkpoint, device)
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    start_epoch = 0
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" not in checkpoint:
            raise ValueError("Use --init-checkpoint for legacy files without scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
    dataset = KITTIMultiPosGeoNeg(root=args.root, seq=args.train_seq,
                                 max_frames=args.max_frames, pose_cols=args.pose_cols)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        sampler=sampler, shuffle=sampler is None, pin_memory=True)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        sums = np.zeros(4)
        bar = tqdm(loader, disable=rank != 0, desc=f"epoch {epoch}")
        for query, pos, neg, sam_q, sam_pos, sam_neg, geo in bar:
            batch, n_pos = pos.shape[:2]
            n_neg = neg.shape[1]
            x = torch.cat([query, pos.flatten(0, 1), neg.flatten(0, 1),
                           sam_q, sam_pos.flatten(0, 1), sam_neg.flatten(0, 1)]).to(device)
            mid, desc = model(x)
            q, p, n = torch.split(desc, [batch, batch*n_pos, batch*n_neg])
            mq, mp, mn = torch.split(mid, [batch, batch*n_pos, batch*n_neg])
            p, n = p.reshape(batch, n_pos, -1), n.reshape(batch, n_neg, -1)
            mp, mn = mp.reshape(batch, n_pos, *mp.shape[1:]), mn.reshape(batch, n_neg, *mn.shape[1:])
            lazy = lazy_triplet_loss(q, p, n, args.margin)
            hcmc = hard_aware_cross_modal_contrastive_loss(q, p, n, geo.to(device))
            mid_loss = hard_mid_loss(mq, mp, mn, args.mid_margin)
            loss = lazy + args.hcmc_lambda*hcmc + mid_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += [loss.item(), lazy.item(), hcmc.item(), mid_loss.item()]
            bar.set_postfix(loss=loss.item())
        scheduler.step()
        if rank == 0:
            print("epoch", epoch, "total/lazy/hcmc/mid", (sums / len(loader)).tolist())
            if epoch % args.save_every == 0 or epoch == args.epochs - 1:
                module = model.module if distributed else model
                torch.save({"epoch": epoch, "state_dict": module.state_dict(),
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "args": vars(args)}, output / f"epoch_{epoch:03d}.pth.tar")
        if distributed:
            torch.distributed.barrier()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
