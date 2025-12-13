import cv2
import numpy as np

def estimateEssentialMatrix(kps1, kps2, K, ransac_thresh=1.0, prob=0.999):
    """
    Estimate the essential matrix using RANSAC.

    Parameters:
        kps1, kps2 : Matched 2D keypoints (Nx2)
        K          : Camera intrinsic matrix
        ransac_thresh : RANSAC inlier threshold in pixels
        prob          : Desired RANSAC confidence

    Returns:
        E          : Essential matrix (3x3)
        inliers    : Boolean mask of inlier correspondences
    """
    
    # Ensure float32 for OpenCV
    pts1 = kps1.astype(np.float32)
    pts2 = kps2.astype(np.float32)

    # Robust estimation using 5-point algorithm inside RANSAC
    E, inliers = cv2.findEssentialMat(
        pts1, pts2, cameraMatrix=K,
        method=cv2.RANSAC, threshold=ransac_thresh, prob=prob
    )

    if E is None:
        raise ValueError("Essential matrix estimation failed")

    # Convert OpenCV's {0,1} mask to boolean mask
    return E, (inliers.ravel() == 1)


def recoverPose(E, kps1, kps2, K, inliers_mask):
    """
    Recover relative camera pose (R, t) from the essential matrix.

    Parameters:
        E            : Essential matrix
        kps1, kps2   : Original matched keypoints (Nx2)
        K            : Camera intrinsic matrix
        inliers_mask : Boolean array selecting RANSAC inliers

    Returns:
        R : Rotation matrix (3x3)
        t : Translation vector (3x1), unit-norm direction
    """

    # Use only inlier correspondences
    pts1 = kps1[inliers_mask].astype(np.float32)
    pts2 = kps2[inliers_mask].astype(np.float32)

    # recoverPose returns one of the four physically plausible solutions
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t.reshape(3,1)
