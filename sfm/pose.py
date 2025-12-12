import cv2
import numpy as np

def estimatePose(kps_new_matched, X_match, K, use_ransac=True):
    if use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            X_match, kps_new_matched, K, None,
            reprojectionError=4.0, iterationsCount=1000,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    else:
        success, rvec, tvec = cv2.solvePnP(X_match, kps_new_matched, K, None)
        inliers = np.arange(len(X_match))

    if not success:
        raise RuntimeError("PnP failed")

    R_new, _ = cv2.Rodrigues(rvec)
    t_new = tvec.reshape(3,1)

    return R_new, t_new, inliers
