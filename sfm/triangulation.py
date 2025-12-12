import cv2
import numpy as np

def triangulatePoints(kps1, kps2, K, R, t, inliers_mask=None):
    if inliers_mask is None:
        pts1 = kps1.astype(np.float32)
        pts2 = kps2.astype(np.float32)
    else:
        pts1 = kps1[inliers_mask].astype(np.float32)
        pts2 = kps2[inliers_mask].astype(np.float32)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([R, t])

    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = (pts4D[:3] / pts4D[3]).T

    valid = (pts3D[:,2] > 0) & np.isfinite(pts3D).all(axis=1)
    return pts3D, valid


def triangulateNewPoints(R_new, t_new, R_ref, t_ref, kps_new, kps_ref, matches, K):
    P_ref = K @ np.hstack([R_ref, t_ref])
    P_new = K @ np.hstack([R_new, t_new])

    pts_ref = np.array([kps_ref[i] for (i,j) in matches]).T
    pts_new = np.array([kps_new[j] for (i,j) in matches]).T

    pts4D = cv2.triangulatePoints(P_ref, P_new, pts_ref, pts_new)
    pts3D = (pts4D[:3] / pts4D[3]).T

    z_new = (R_new @ pts3D.T + t_new).T[:,2]
    valid_mask = z_new > 0

    return pts3D, valid_mask
