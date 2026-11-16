"""Shared legacy namespace. Entry points set dataset_toot from --root."""


class setting_config:
    dataset_toot = "data/kitti"
    resize_shape = (800, 128)
    nmf = False
    mlp = False
    pose_cols = (3, 11)
