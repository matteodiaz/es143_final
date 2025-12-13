import cv2
import numpy as np

def matchDescriptors(descs1, descs2, keypoints1, keypoints2, good_match_percentage=0.15):
    """
    Match ORB descriptors between two images using brute-force Hamming matching.

    Parameters:
        descs1, descs2 : Descriptor arrays for image 1 and 2
        keypoints1, keypoints2 : Lists of cv2.KeyPoint objects (not used here directly)
        good_match_percentage : Fraction of top matches to retain

    Returns:
        List of cv2.DMatch objects (best N matches sorted by distance)
    """
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Find mutual nearest neighbors
    matches = matcher.match(descs1, descs2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep top percentage of matches for robustness
    keep = int(len(matches) * good_match_percentage)
    return matches[:keep]


def match_image_to_3D(desc_new, kps_new, point_descriptors, points3D, ratio=0.75):
    """
    Match image descriptors to existing 3D point descriptors using KNN + Lowe's ratio test.

    Parameters:
        desc_new          : (N,D) descriptors from the new image
        kps_new           : (N,2) keypoints from the new image
        point_descriptors : (M,D) descriptors already associated with 3D map points
        points3D          : (M,3) 3D points in world coords
        ratio             : Lowe's ratio threshold for good matches

    Returns:
        kps_matched : (K,2) matched 2D locations in new image
        X_matched   : (K,3) corresponding matched 3D points
    """

    # No 3D points in the map yet → cannot match
    if len(point_descriptors) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    knn_matches = bf.knnMatch(desc_new, point_descriptors, k=2)
    
    kps_list = []
    X_list = []

    # Apply Lowe's ratio test for robust matching
    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue
            
        m, n = match_pair[0], match_pair[1]
        
        if m.distance < ratio * n.distance:
            kps_list.append(kps_new[m.queryIdx])
            X_list.append(points3D[m.trainIdx])
    
    if len(kps_list) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    
    return np.asarray(kps_list, dtype=np.float32), np.asarray(X_list, dtype=np.float32)
