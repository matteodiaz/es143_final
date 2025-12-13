import cv2
import numpy as np

def triangulatePoints(kps1, kps2, K, R, t, inliers_mask=None):
    """
    Triangulate 3D points from two calibrated views.

    Parameters:
        kps1, kps2   : (N,2) matched pixel coordinates in image 1 and 2
        K            : (3,3) intrinsic matrix
        R, t         : Relative pose of camera 2 w.r.t camera 1 (world ← cam2)
        inliers_mask : Optional boolean array selecting RANSAC inliers

    Returns:
        pts3D : (M,3) triangulated 3D points in world coordinates
        valid : Boolean mask of points with positive depth and finite values
    """

    # Use inlier correspondences if mask is provided
    if inliers_mask is None:
        pts1 = kps1.astype(np.float32)
        pts2 = kps2.astype(np.float32)
    else:
        pts1 = kps1[inliers_mask].astype(np.float32)
        pts2 = kps2[inliers_mask].astype(np.float32)

    # Projection matrix for camera 1 at origin
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])

    # Projection matrix for camera 2 with pose (R, t)
    P2 = K @ np.hstack([R, t])

    # OpenCV expects (2,N) arrays
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Convert homogeneous coords to Euclidean
    pts3D = (pts4D[:3] / pts4D[3]).T

    # Keep only valid points (positive depth, finite numeric values)
    valid = (pts3D[:,2] > 0) & np.isfinite(pts3D).all(axis=1)
    return pts3D, valid


def triangulateNewPoints(R_new, t_new, R_ref, t_ref, kps_new, kps_ref, matches, K):
    """
    Triangulate new 3D points between an existing reference view and a newly added view.

    Parameters:
        R_new, t_new : Pose of the new camera
        R_ref, t_ref : Pose of the reference camera
        kps_new      : Keypoints in the new image
        kps_ref      : Keypoints in the reference image
        matches      : List of (kp_ref_idx, kp_new_idx) index pairs
        K            : Intrinsic matrix

    Returns:
        pts3D      : (N,3) triangulated points
        valid_mask : Boolean mask indicating positive depth in the new view
    """
    
    # Projection matrices for reference and new camera
    P_ref = K @ np.hstack([R_ref, t_ref])
    P_new = K @ np.hstack([R_new, t_new])

    # Extract matched 2D keypoints (shape 2×N for OpenCV)
    pts_ref = np.array([kps_ref[i] for (i,j) in matches]).T
    pts_new = np.array([kps_new[j] for (i,j) in matches]).T

    # Triangulate in homogeneous coordinates
    pts4D = cv2.triangulatePoints(P_ref, P_new, pts_ref, pts_new)
    pts3D = (pts4D[:3] / pts4D[3]).T

    # Check positive depth in the new camera
    z_new = (R_new @ pts3D.T + t_new).T[:,2]
    valid_mask = z_new > 0

    return pts3D, valid_mask
