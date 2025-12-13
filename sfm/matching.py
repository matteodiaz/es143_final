import cv2
import numpy as np

def matchDescriptors(descs1, descs2, keypoints1, keypoints2, good_match_percentage=0.15):
    """
    Match ORB descriptors using BFMatcher and return the best matches.
    """
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descs1, descs2)
    matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * good_match_percentage)
    return matches[:keep]


def match_image_to_3D(desc_new, kps_new, point_descriptors, points3D, ratio=0.75):
    """
    Match new image descriptors to existing 3D point descriptors.
    
    Args:
        desc_new: (N, D) descriptors from new image
        kps_new: (N, 2) keypoints from new image
        point_descriptors: (M, D) descriptors of existing 3D points
        points3D: (M, 3) existing 3D points
        ratio: Lowe's ratio test threshold
        
    Returns:
        kps_matched: (K, 2) matched keypoints
        X_matched: (K, 3) corresponding 3D points
    """
    
    if len(point_descriptors) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    
    # BFMatcher with Hamming distance for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # KNN match
    knn_matches = bf.knnMatch(desc_new, point_descriptors, k=2)
    
    kps_list = []
    X_list = []
    
    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue
            
        m, n = match_pair[0], match_pair[1]
        
        # Lowe's ratio test
        if m.distance < ratio * n.distance:
            kps_list.append(kps_new[m.queryIdx])
            X_list.append(points3D[m.trainIdx])
    
    if len(kps_list) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    
    return np.asarray(kps_list, dtype=np.float32), np.asarray(X_list, dtype=np.float32)
