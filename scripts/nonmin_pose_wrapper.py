# nonmin_pose_wrapper.py

from nonmin_pose import C2PFast
import numpy as np

def c2p_(f0, f1):
    configuration = {
        "th_rank_optimality": 1e-5,
        "th_pure_rot_sdp": 1e-3,
        "th_pure_rot_noisefree_sdp": 1e-4,
    }

    solver = C2PFast(cfg=configuration)
    solution = solver(f0, f1)

    return (
        solution["E01"],
        solution["R01"],
        solution["t01"],
        solution["is_optimal"],
        solution["is_pure_rot"]
    )
