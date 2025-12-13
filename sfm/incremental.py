from .features import match_image_to_3D

def add_view_incremental(
    view_id,
    points3D,
    point_descriptors,
    tracks,
    kp_to_track,
    cameras,
    video_images,
    video_kps,
    video_descs,
    video_matches,
    K,
    ref_view=None,
    match_ratio=0.75,
    min_matches=20,
    reproj_threshold=8.0
):
    """
    Incrementally add a new view to the reconstruction.
    Steps:
        1. Match descriptors to 3D points (2D–3D)
        2. Estimate pose via PnPRANSAC
        3. Triangulate new points with reference view
        4. Update global tracks + point cloud
    """

    if ref_view is None:
        ref_view = view_id - 1   # natural incremental SFM

    print(f"\n Adding view {view_id} (ref = {ref_view})")

    # --- Step 1: 2D-3D Matching ---
    kps_new = video_kps[view_id]
    descs_new = video_descs[view_id]

    kps_new_matched, X_match = match_image_to_3D(
        descs_new,
        kps_new,
        point_descriptors,
        points3D,
        ratio=match_ratio
    )

    print(f"  2D–3D matches: {len(kps_new_matched)}")

    if len(kps_new_matched) < min_matches:
        print("  Not enough 2D–3D correspondences for PnP.")
        return points3D, point_descriptors, tracks, kp_to_track, cameras

    # --- Step 2: Pose Estimation ---
    R_new, t_new, inliers = estimatePose(
        kps_new_matched,
        X_match,
        K,
        use_ransac=True,
        reproj_threshold=reproj_threshold
    )

    if R_new is None:
        print("  Pose estimation failed.")
        return points3D, point_descriptors, tracks, kp_to_track, cameras

    cameras[view_id] = {"R": R_new, "t": t_new}
    print(f"  Pose OK (inliers: {len(inliers)})")

    # --- Step 3: Triangulate new points ---
    R_ref = cameras[ref_view]["R"]
    t_ref = cameras[ref_view]["t"]
    kps_ref = video_kps[ref_view]

    matches_pair = video_matches[ref_view]  # FIXED
    matches_idx = [(m.queryIdx, m.trainIdx) for m in matches_pair]

    X_new, valid_mask = triangulateNewPoints(
        R_new, t_new,
        R_ref, t_ref,
        kps_new,
        kps_ref,
        matches_idx,
        K
    )

    print(f"  Valid triangulated: {valid_mask.sum()}")

    # --- Step 4: Insert new points into map ---
    num_added = 0
    for idx, ok in enumerate(valid_mask):
        if not ok:
            continue

        kp_ref_idx, kp_new_idx = matches_idx[idx]

        # If this 2D point is already linked to an existing 3D point, update track
        if (ref_view, kp_ref_idx) in kp_to_track:
            track_id = kp_to_track[(ref_view, kp_ref_idx)]
            if (view_id, kp_new_idx) not in tracks[track_id]:
                tracks[track_id].append((view_id, kp_new_idx))
                kp_to_track[(view_id, kp_new_idx)] = track_id
            continue

        # Otherwise: new 3D point
        new_id = len(points3D)
        points3D = np.vstack([points3D, X_new[idx]])

        tracks[new_id] = [
            (ref_view, kp_ref_idx),
            (view_id, kp_new_idx)
        ]

        kp_to_track[(ref_view, kp_ref_idx)] = new_id
        kp_to_track[(view_id, kp_new_idx)] = new_id

        desc = descs_new[kp_new_idx].astype(np.uint8)
        point_descriptors = np.vstack([point_descriptors, desc[None]])

        num_added += 1

    print(f"  Added {num_added} new 3D points → total = {len(points3D)}")

    return points3D, point_descriptors, tracks, kp_to_track, cameras
