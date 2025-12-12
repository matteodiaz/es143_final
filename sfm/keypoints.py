import cv2
import numpy as np

def detectKeypoints(image, detector=None):
    """
    Detect keypoints and descriptors using ORB.
    Returns:
        kps: Nx2 array of pixel coords
        descriptors: NxD descriptor array
        keypoint_objs: list of cv2.KeyPoint objects
    """
    if detector is None:
        detector = cv2.ORB_create(nfeatures=2000)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    kps = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    
    return kps, descriptors, keypoints
