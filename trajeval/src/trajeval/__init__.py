"""trajeval: evaluate mobile-robot and SLAM trajectories against a reference."""

__version__ = "0.2.0"

from .trajectory import Trajectory  # noqa: E402
from .io import read_trajectory, write_trajectory_csv, write_tum, write_kitti  # noqa: E402
from .alignment import Alignment, align, umeyama  # noqa: E402
from .association import synchronize  # noqa: E402
from .metrics import compute_errors, compute_metrics, rpe, summary_stats  # noqa: E402
from .evaluation import EvalConfig, EvalResult, aggregate, evaluate, evaluate_many  # noqa: E402

__all__ = [
    "Trajectory", "read_trajectory", "write_trajectory_csv", "write_tum", "write_kitti",
    "Alignment", "align", "umeyama", "synchronize", "compute_errors", "compute_metrics",
    "rpe", "summary_stats", "EvalConfig", "EvalResult", "aggregate", "evaluate",
    "evaluate_many", "__version__",
]
