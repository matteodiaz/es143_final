import plotly.graph_objs as go
import numpy as np

def camera_frustum(R, t, scale=0.5, color='blue', name='Camera'):
    """
    Construct a simple 3D camera frustum for visualization.

    Parameters:
        R, t   : Camera rotation and translation (world ← camera)
        scale  : Controls frustum size
        color  : Line color
        name   : Label for the legend

    Returns:
        A Plotly Scatter3d object representing the frustum edges.
    """
    
    # Camera center in world coordinates
    t = t.reshape(3, 1)
    C = -R.T @ t
    C = C.flatten()
    
    # Define frustum corners in camera coordinates
    frustum_points = np.array([
        [0, 0, 0],           # Camera center
        [-scale, -scale, 2*scale],  # Top-left
        [scale, -scale, 2*scale],   # Top-right
        [scale, scale, 2*scale],    # Bottom-right
        [-scale, scale, 2*scale],   # Bottom-left
    ]).T
    
    # Transform to world coordinates
    world_points = R.T @ (frustum_points - t)
    
    # Define edges of the frustum
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # From center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Rectangle at far end
    ]
    
    # Create line segments
    x_lines, y_lines, z_lines = [], [], []
    for edge in edges:
        for idx in edge:
            x_lines.append(world_points[0, idx])
            y_lines.append(world_points[1, idx])
            z_lines.append(world_points[2, idx])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    
    return go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(color=color, width=3),
        name=name,
        showlegend=True
    )

def plot_reconstruction(points3D, cameras, point_size=2, camera_scale=0.5, title='SfM Reconstruction'):
    """
    Visualize an incremental SfM reconstruction:
    - Sparse 3D point cloud
    - Estimated camera poses as frustums

    Parameters:
        points3D : (N,3) array of 3D point positions
        cameras  : dict mapping view_id → {'R': R, 't': t}
        point_size : Marker size for points
        camera_scale : Frustum size multiplier
        title    : Plot title

    Returns:
        Plotly Figure object for interactive visualization
    """
    
    traces = []
    
    # 3D point cloud
    if len(points3D) > 0:
        traces.append(
            go.Scatter3d(
                x=points3D[:, 0],
                y=points3D[:, 1],
                z=points3D[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=points3D[:, 2],  # Color by depth
                    colorscale='Viridis',
                    opacity=0.6,
                    showscale=False  
                ),
                name=f'3D Points ({len(points3D)})',
                showlegend=False
            )
        )
    
    # Camera color palette
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    # Add cameras
    for i, (view_id, cam) in enumerate(sorted(cameras.items())):
        color = colors[i % len(colors)]
        
        # Camera frustum
        frustum = camera_frustum(
            cam['R'], 
            cam['t'], 
            scale=camera_scale,
            color=color,
            name=f'Camera {view_id}'
        )
        traces.append(frustum)
        
        # Camera center marker
        C = -cam['R'].T @ cam['t'].reshape(3, 1)
        C = C.flatten()
        
        traces.append(
            go.Scatter3d(
                x=[C[0]],
                y=[C[1]],
                z=[C[2]],
                mode='markers',
                marker=dict(size=8, color=color, symbol='diamond'),
                name=f'Cam {view_id} center',
                showlegend=False
            )
        )
    
    # Assemble figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1000,
        height=800,
        showlegend=True
    )
    
    return fig
