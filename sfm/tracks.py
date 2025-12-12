import numpy as np

def updateTracksWithNewPoints(points3D, tracks, X_new, matches, valid_mask, new_view_id):
    for idx, ok in enumerate(valid_mask):
        if not ok:
            continue

        pt3 = X_new[idx]
        track_id = len(points3D)

        points3D = np.vstack([points3D, pt3])

        ref_kp, new_kp = matches[idx]
        tracks[track_id] = [
            ("ref_view", ref_kp),
            (new_view_id, new_kp)
        ]

    return points3D, tracks
