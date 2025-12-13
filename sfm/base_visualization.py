import numpy as np
import plotly.graph_objects as go

def base_camera_frustum(R, t, scale=0.2, color='red', name='camera'):
    """
    Generate a simple 3D camera frustum for visualization.
    
    Parameters:
        R, t : Camera rotation and translation (world ← camera)
        scale : Frustum size
        color : Plot color
        name  : Legend name
    
    Returns:
        List of Plotly Scatter3d traces representing the frustum edges and center.
    """
    
    # Camera center in world coords
    C = -R.T @ t

    # Frustum corners in camera coords
    frustum_cam = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
        [1, -1, 1],
    ]) * scale

    # Transform frustum to world coords
    frustum_world = (R.T @ frustum_cam.T).T + C.T

    edges = [
        (0,1),(0,2),(0,3),(0,4),
        (1,2),(2,3),(3,4),(4,1)
    ]

    traces = []
    for i,j in edges:
        traces.append(go.Scatter3d(
            x=[frustum_world[i,0], frustum_world[j,0]],
            y=[frustum_world[i,1], frustum_world[j,1]],
            z=[frustum_world[i,2], frustum_world[j,2]],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False
        ))

    # Camera center marker
    center = go.Scatter3d(
        x=[C[0,0]], y=[C[1,0]], z=[C[2,0]],
        mode='markers',
        marker=dict(size=6, color=color),
        name=name
    )

    return traces + [center]

def plot_base_scene(points3D, R2, t2):
    """
    Visualize the two-view SfM initialization:
    - Triangulated 3D points
    - Camera 1 at the origin
    - Camera 2 recovered from the essential matrix
    """
    
    fig = go.Figure()

    # Plot 3D points as point cloud
    fig.add_trace(go.Scatter3d(
        x=points3D[:,0],
        y=points3D[:,1],
        z=points3D[:,2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='3D points'
    ))

    # First camera (reference frame)
    R1 = np.eye(3)
    t1 = np.zeros((3,1))
    for tr in base_camera_frustum(R1, t1, color='green', name='Camera 1'):
        fig.add_trace(tr)

    # Second camera (relative pose from SfM)
    for tr in base_camera_frustum(R2, t2, color='red', name='Camera 2'):
        fig.add_trace(tr)

    fig.update_layout(
        title="SfM Verification: Cameras + Triangulated Points",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=700
    )

    fig.show()
