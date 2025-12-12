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


def match_image_to_3D(desc_new, kps_new, point_descriptors, points3D):
    """
    Match new image descriptors to existing 3D point descriptors.
    Returns:
        kps_new_matched: Nx2 2D points
        X_match: Nx3 3D points
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_new, point_descriptors)

    matches = sorted(matches, key=lambda x: x.distance)

    kps_out = []
    X_out = []

    for m in matches:
        kps_out.append(kps_new[m.queryIdx])
        X_out.append(points3D[m.trainIdx])

    return np.array(kps_out, dtype=np.float32), np.array(X_out, dtype=np.float32)
