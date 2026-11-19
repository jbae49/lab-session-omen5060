"""Lazy camera-motion imports.

The OMEN/RTX 5060 classroom profile uses static-camera videos and therefore does
not need to import or compile DROID-SLAM/lietorch at startup.  Moving-camera
support remains available after installing those optional extensions.
"""


def run_metric_slam(*args, **kwargs):
    from .masked_droid_slam import run_metric_slam as implementation

    return implementation(*args, **kwargs)


def calibrate_intrinsics(*args, **kwargs):
    from .masked_droid_slam import calibrate_intrinsics as implementation

    return implementation(*args, **kwargs)


def run_slam(*args, **kwargs):
    from .masked_droid_slam import run_slam as implementation

    return implementation(*args, **kwargs)
