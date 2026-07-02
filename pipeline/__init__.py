"""rally3d: 3D table-tennis data from broadcast video.

End-to-end pipeline that turns full-match footage into synchronized metric 3D
training data — ball trajectory, player body pose, and table/camera geometry —
from a single camera. Original orchestration, segmentation, calibration
disambiguation and trajectory refinement, building on open research components
(TT3D calibration/physics, BlurBall detection, MotionBERT lifting, RTMPose).
"""

__version__ = "0.2.0"
