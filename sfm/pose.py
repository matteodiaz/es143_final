import cv2
import numpy as np

def estimatePose(kps_new_matched, X_match, K, use_ransac=True, reproj_threshold=8.0):
    """
    Estimate the pose of a new camera using 2D–3D correspondences.

    Parameters:
        kps_new_matched : (N,2) array of 2D keypoint locations in pixels
        X_match         : (N,3) array of corresponding 3D world points
        K               : (3,3) intrinsic matrix
        use_ransac      : enable PnP with RANSAC for robustness
        reproj_threshold: RANSAC reprojection error threshold in pixels

    Returns:
        R        : (3,3) rotation matrix, or None if estimation fails
        t        : (3,1) translation vector, or None
        inliers  : indices of PnP inliers (1D array)
    """
    
    # Require a minimum number of correspondences for stable PnP
    if kps_new_matched.shape[0] < 6 or X_match.shape[0] < 6:
        print(f"[estimatePose] Not enough correspondences: {kps_new_matched.shape[0]}")
        return None, None, np.array([], dtype=int)

    # Ensure both arrays have the same number of rows
    N = min(kps_new_matched.shape[0], X_match.shape[0])
    kps = kps_new_matched[:N]
    X = X_match[:N]
    
    # Robustness check: invalid values break solvePnP
    if not np.isfinite(kps).all() or not np.isfinite(X).all():
        print("[estimatePose] Non-finite values in inputs")
        return None, None, np.array([], dtype=int)
    
    # OpenCV expects float64 and (N,1,*) array shapes
    K = K.astype(np.float64)
    obj_pts = X.astype(np.float64).reshape(-1, 1, 3)
    img_pts = kps.astype(np.float64).reshape(-1, 1, 2)
    distCoeffs = np.zeros((4, 1), dtype=np.float64)
    
    inliers = None
    rvec = None
    tvec = None

    # Primary Method: PnP + RANSAC (robust to outliers)
    if use_ransac and N >= 6:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K,
            distCoeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=reproj_threshold,
            confidence=0.99,
            iterationsCount=2000
        )

        # Accept RANSAC only if sufficient inliers are found
        if success and inliers is not None and len(inliers) >= 6:
            print(f"  PnPRansac succeeded with {len(inliers)} inliers out of {N}")
            
            # Refine using inliers only
            obj_inl = obj_pts[inliers.flatten()]
            img_inl = img_pts[inliers.flatten()]
            
            success_refine, rvec, tvec = cv2.solvePnP(
                obj_inl,
                img_inl,
                K,
                distCoeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success_refine:
                print("[estimatePose] Refinement failed, using original RANSAC result")
        else:
            print(f"[estimatePose] RANSAC failed or too few inliers, trying plain solvePnP")
            inliers = None
    
    # Fallback Method: plain PnP (no RANSAC)
    if rvec is None or tvec is None:
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            distCoeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print("[estimatePose] Plain solvePnP also failed")
            return None, None, np.array([], dtype=int)
        
        print(f"  Plain solvePnP succeeded using all {N} points")
        inliers = np.arange(N, dtype=int).reshape(-1, 1)
    
    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    
    # Flatten OpenCV's inlier indices
    if inliers is not None:
        inliers = inliers.flatten()
    else:
        inliers = np.array([], dtype=int)
    
    return R, t, inliers
