import cv2
import numpy as np

def estimateEssentialMatrix(kps1, kps2, K, ransac_thresh=1.0, prob=0.999):
    pts1 = kps1.astype(np.float32)
    pts2 = kps2.astype(np.float32)

    E, inliers = cv2.findEssentialMat(
        pts1, pts2, cameraMatrix=K,
        method=cv2.RANSAC, threshold=ransac_thresh, prob=prob
    )

    if E is None:
        raise ValueError("Essential matrix estimation failed")

    return E, (inliers.ravel() == 1)


def recoverPose(E, kps1, kps2, K, inliers_mask):
    pts1 = kps1[inliers_mask].astype(np.float32)
    pts2 = kps2[inliers_mask].astype(np.float32)

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t.reshape(3,1)
