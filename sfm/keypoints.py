import cv2
import numpy as np

def detectKeypoints(image, detector=None):
    """
    Detect keypoints and compute descriptors for an image.

    Parameters:
        image    : Input RGB or grayscale image
        detector : Optional OpenCV feature detector (default ORB)

    Returns:
        kps            : (N,2) array of keypoint pixel coordinates
        descriptors    : (N,D) array of descriptors
        keypoint_objs  : list of cv2.KeyPoint objects (for drawMatches)
    """

    # Default to ORB with reasonable feature count
    if detector is None:
        detector = cv2.ORB_create(nfeatures=2000)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute keypoints and descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    # Extract (x, y) pixel locations in a compact array
    kps = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    
    return kps, descriptors, keypoints
