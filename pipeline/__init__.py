"""tt3d-data: scaled table-tennis 3D data generation pipeline.

Wraps the upstream TT3D method (calibration, pose alignment, ball reconstruction),
BlurBall (2D ball detection) and MotionBERT (2D->3D pose lifting) into a batch
pipeline that turns many broadcast matches into a training dataset of ball
position, human body pose and table position.
"""

__version__ = "0.1.0"
