import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import streamlit as st
import tempfile
import os
import ssl
import certifi
import io
import matplotlib
import seaborn as sns
from datetime import datetime
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
matplotlib.use('Agg')  # Use non-interactive backend

# Set SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Set page config with a more compact layout
st.set_page_config(
    layout="wide",
    page_title="Advanced Car Tracking & Analysis",
    initial_sidebar_state="collapsed"
)

# Custom CSS to make the layout more compact
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        margin: 0;
    }
    .element-container {
        margin-bottom: 0.5rem;
    }
    .stVideo {
        margin: 0;
    }
    .stImage {
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for storing data between reruns
if 'car_tracks' not in st.session_state:
    st.session_state.car_tracks = defaultdict(list)
if 'next_car_id' not in st.session_state:
    st.session_state.next_car_id = 0
if 'car_ages' not in st.session_state:
    st.session_state.car_ages = {}
if 'frame_number' not in st.session_state:
    st.session_state.frame_number = 0
if 'speeds' not in st.session_state:
    st.session_state.speeds = defaultdict(list)
if 'directions' not in st.session_state:
    st.session_state.directions = defaultdict(list)
if 'time_series' not in st.session_state:
    st.session_state.time_series = {
        'time': [],
        'vehicle_count': [],
        'avg_speed': []
    }
if 'vehicle_types' not in st.session_state:
    st.session_state.vehicle_types = Counter()
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {
        'total_vehicles': 0,
        'avg_speed': 0,
        'max_speed': 0,
        'traffic_density': 0,
        'direction_counts': defaultdict(int)
    }

# Initialize session state for multiple cameras
if 'cameras' not in st.session_state:
    st.session_state.cameras = {}
if 'current_camera' not in st.session_state:
    st.session_state.current_camera = None
if 'animation_frame' not in st.session_state:
    st.session_state.animation_frame = 0

# Load the YOLOv5 model
@st.cache_resource
def load_model():
    try:
        if os.path.exists('yolov5s.pt'):
            st.info("Loading model from local file...")
            return torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
        else:
            st.info("Downloading YOLOv5 model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            torch.save(model.state_dict(), 'yolov5s.pt')
            return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Class ID for car
CAR_CLASS = 2  # COCO dataset ID for car

# Homography matrix (you'll need to calibrate this for your specific camera setup)
H = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

def calculate_homography():
    """Calculate homography matrix using known points in video and their real-world coordinates."""
    # Example points (video coordinates -> real-world coordinates in meters)
    video_points = np.array([
        [100, 100],  # Top-left
        [500, 100],  # Top-right
        [500, 400],  # Bottom-right
        [100, 400]   # Bottom-left
    ], dtype=np.float32)
    
    real_world_points = np.array([
        [0, 0],      # Top-left
        [10, 0],     # Top-right
        [10, 10],    # Bottom-right
        [0, 10]      # Bottom-left
    ], dtype=np.float32)
    
    return cv2.findHomography(video_points, real_world_points)[0]

def transform_point(point, H):
    """Transform a point from video coordinates to real-world coordinates."""
    point = np.array([point[0], point[1], 1])
    transformed = np.dot(H, point)
    return transformed[:2] / transformed[2]

def calculate_speed(pos1, pos2, time_diff, fps):
    """Calculate speed between two positions."""
    distance = np.linalg.norm(np.array(pos2) - np.array(pos1))
    speed = (distance * fps) / time_diff  # meters per second
    return speed * 3.6  # convert to km/h

def calculate_direction(pos1, pos2):
    """Calculate direction of movement."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle

def update_density_map(position):
    """Update traffic density map."""
    x, y = int(position[0] * 10), int(position[1] * 10)
    if 0 <= x < 100 and 0 <= y < 100:
        st.session_state.density_map[y, x] += 1

def plot_flow_map():
    """Plot dynamic traffic flow map with arrows showing movement patterns."""
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a custom colormap for the flow
    colors = [(0, 'blue'), (0.5, 'green'), (1, 'red')]
    cmap = LinearSegmentedColormap.from_list('flow_cmap', colors)
    norm = plt.Normalize(0, 100)  # Normalize speeds from 0 to 100 km/h
    
    # Track if we've plotted any trajectories
    has_trajectories = False
    
    for car_id, trajectory in st.session_state.car_tracks.items():
        if len(trajectory) > 1:
            has_trajectories = True
            trajectory = np.array(trajectory)
            # Calculate average speed for this vehicle
            avg_speed = np.mean(st.session_state.speeds[car_id]) if car_id in st.session_state.speeds else 0
            
            # Plot trajectory with color based on speed
            points = np.array(trajectory)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            for i, segment in enumerate(segments):
                # Calculate direction for arrow
                dx = segment[2] - segment[0]
                dy = segment[3] - segment[1]
                # Plot arrow with color based on speed
                ax.arrow(segment[0], segment[1], dx, dy,
                        head_width=0.5, head_length=0.7, fc=cmap(norm(avg_speed)),
                        ec=cmap(norm(avg_speed)), alpha=0.6)
    
    ax.set_title('Traffic Flow Map')
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Y Position (meters)')
    ax.grid(True)
    
    # Add colorbar only if we have trajectories
    if has_trajectories:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Create colorbar with explicit axes
        cbar = fig.colorbar(sm, ax=ax, label='Speed (km/h)')
    
    # Adjust layout to prevent colorbar overlap
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_time_series():
    """Plot time series of vehicle count and average speed."""
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot vehicle count
    ax1.plot(st.session_state.time_series['time'], 
             st.session_state.time_series['vehicle_count'], 
             'b-', label='Vehicle Count')
    ax1.set_ylabel('Number of Vehicles')
    ax1.grid(True)
    ax1.legend()
    
    # Plot average speed
    ax2.plot(st.session_state.time_series['time'], 
             st.session_state.time_series['avg_speed'], 
             'r-', label='Average Speed')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.grid(True)
    ax2.legend()
    
    plt.suptitle('Traffic Analysis Over Time')
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_vehicle_distribution():
    """Plot vehicle type distribution pie chart."""
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get vehicle type counts
    labels = list(st.session_state.vehicle_types.keys())
    sizes = list(st.session_state.vehicle_types.values())
    
    if sizes:
        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.axis('equal')
        ax.set_title('Vehicle Type Distribution')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_speed_position():
    """Plot speed vs. position scatter plot with density contours."""
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Collect all speed and position data
    positions = []
    speeds = []
    
    for car_id in st.session_state.car_tracks:
        if car_id in st.session_state.speeds and st.session_state.car_tracks[car_id]:
            # Get the latest position and speed for each vehicle
            latest_pos = st.session_state.car_tracks[car_id][-1]
            latest_speed = st.session_state.speeds[car_id][-1] if st.session_state.speeds[car_id] else 0
            positions.append(latest_pos)
            speeds.append(latest_speed)
    
    if positions and speeds:
        positions = np.array(positions)
        speeds = np.array(speeds)
        
        # Create scatter plot
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=speeds, cmap='viridis', 
                           alpha=0.6, s=50)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Speed (km/h)')
        
        ax.set_title('Speed vs. Position Distribution')
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.grid(True)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def get_vehicle_class_name(class_id):
    """Convert YOLO class ID to vehicle type name."""
    vehicle_classes = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        11: 'stop sign',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        27: 'handbag',
        28: 'tie',
        31: 'handbag',
        32: 'umbrella',
        33: 'backpack',
        34: 'tie',
        35: 'suitcase',
        36: 'frisbee',
        37: 'skis',
        38: 'snowboard',
        39: 'sports ball',
        40: 'kite',
        41: 'baseball bat',
        42: 'baseball glove',
        43: 'skateboard',
        44: 'surfboard',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        83: 'book',
        84: 'clock',
        85: 'vase',
        86: 'scissors',
        87: 'teddy bear',
        88: 'hair drier',
        89: 'toothbrush'
    }
    return vehicle_classes.get(class_id, f'unknown_{class_id}')

def get_vehicle_color(class_id):
    """Get color for different vehicle types."""
    vehicle_colors = {
        1: (255, 0, 0),    # Bicycle - Red
        2: (0, 255, 0),    # Car - Green
        3: (0, 0, 255),    # Motorcycle - Blue
        5: (255, 255, 0),  # Bus - Yellow
        7: (255, 0, 255),  # Truck - Magenta
    }
    return vehicle_colors.get(class_id, (0, 255, 255))  # Default to Cyan for unknown types

def update_analysis_data():
    """Update analysis statistics."""
    all_speeds = [speed for speeds in st.session_state.speeds.values() for speed in speeds]
    
    # Calculate traffic density based on number of vehicles and their positions
    if st.session_state.car_tracks:
        # Get all current positions
        all_positions = []
        for trajectory in st.session_state.car_tracks.values():
            if trajectory:
                all_positions.append(trajectory[-1])  # Get the latest position
        
        if all_positions:
            # Calculate average distance between vehicles
            positions = np.array(all_positions)
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)
            
            # Traffic density is inversely proportional to average distance
            avg_distance = np.mean(distances) if distances else float('inf')
            traffic_density = 1.0 / (1.0 + avg_distance)  # Normalized between 0 and 1
        else:
            traffic_density = 0
    else:
        traffic_density = 0
    
    st.session_state.analysis_data.update({
        'total_vehicles': len(st.session_state.car_tracks),
        'avg_speed': np.mean(all_speeds) if all_speeds else 0,
        'max_speed': max(all_speeds) if all_speeds else 0,
        'traffic_density': traffic_density,
        'direction_counts': defaultdict(int),
        'vehicle_types': dict(st.session_state.vehicle_types)
    })
    
    # Update time series data
    current_time = st.session_state.frame_number / 30  # Assuming 30 fps
    st.session_state.time_series['time'].append(current_time)
    st.session_state.time_series['vehicle_count'].append(len(st.session_state.car_tracks))
    st.session_state.time_series['avg_speed'].append(st.session_state.analysis_data['avg_speed'])
    
    # Update direction counts
    for angles in st.session_state.directions.values():
        for angle in angles:
            direction = int(angle / 45) * 45
            st.session_state.analysis_data['direction_counts'][direction] += 1

def update_tracks(detections, frame_number, fps):
    """Update car tracks with new detections."""
    current_cars = []
    for _, row in detections.iterrows():
        if row['class'] in [1, 2, 3, 5, 7]:  # Vehicle classes: bicycle, car, motorcycle, bus, truck
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
            current_cars.append((center, row['confidence'], row['class']))
            # Update vehicle type counter
            st.session_state.vehicle_types[get_vehicle_class_name(row['class'])] += 1
    
    # Update existing tracks
    for car_id in list(st.session_state.car_ages.keys()):
        st.session_state.car_ages[car_id] += 1
        if st.session_state.car_ages[car_id] > 30:
            del st.session_state.car_ages[car_id]
            del st.session_state.car_tracks[car_id]
    
    # Match detections to existing tracks
    if current_cars:
        for center, conf, class_id in current_cars:
            real_world_pos = transform_point(center, H)
            
            min_dist = float('inf')
            best_match = None
            
            for car_id in st.session_state.car_tracks:
                if car_id in st.session_state.car_ages:
                    last_pos = st.session_state.car_tracks[car_id][-1]
                    dist = np.linalg.norm(np.array(real_world_pos) - np.array(last_pos))
                    if dist < min_dist and dist < 5.0:
                        min_dist = dist
                        best_match = car_id
            
            if best_match is not None:
                # Calculate speed and direction
                if len(st.session_state.car_tracks[best_match]) > 0:
                    last_pos = st.session_state.car_tracks[best_match][-1]
                    speed = calculate_speed(last_pos, real_world_pos, 1, fps)
                    direction = calculate_direction(last_pos, real_world_pos)
                    st.session_state.speeds[best_match].append(speed)
                    st.session_state.directions[best_match].append(direction)
                
                st.session_state.car_tracks[best_match].append(real_world_pos)
                st.session_state.car_ages[best_match] = 0
            else:
                st.session_state.car_tracks[st.session_state.next_car_id] = [real_world_pos]
                st.session_state.car_ages[st.session_state.next_car_id] = 0
                st.session_state.next_car_id += 1

def plot_top_down_view():
    """Plot top-down 2D view of vehicle positions and trajectories."""
    plt.close('all')
    
    # Set style
    plt.style.use('dark_background')
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1E1E1E')
    ax.set_facecolor('#2D2D2D')
    
    # Create a custom colormap for the flow
    colors = [(0, '#FF6B6B'), (0.5, '#4ECDC4'), (1, '#45B7D1')]  # Modern color palette
    cmap = LinearSegmentedColormap.from_list('flow_cmap', colors)
    
    # Plot trajectories and current positions
    for car_id, trajectory in st.session_state.car_tracks.items():
        if len(trajectory) > 1:
            trajectory = np.array(trajectory)
            # Get vehicle type and color
            vehicle_type = None
            for class_id in [1, 2, 3, 5, 7]:  # Vehicle classes: bicycle, car, motorcycle, bus, truck
                if get_vehicle_class_name(class_id) in st.session_state.vehicle_types:
                    vehicle_type = class_id
                    break
            
            color = get_vehicle_color(vehicle_type) if vehicle_type else (0, 255, 255)
            # Convert BGR to RGB and adjust for dark theme
            color = (color[2]/255, color[1]/255, color[0]/255)
            
            # Plot trajectory with gradient effect
            points = np.array(trajectory)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Plot trajectory segments with varying alpha
            for i, segment in enumerate(segments):
                alpha = 0.3 + 0.7 * (i / len(segments))  # Fade in effect
                ax.plot(segment[0:2], segment[2:4], '-', color=color, alpha=alpha, linewidth=2)
            
            # Plot current position with vehicle type
            current_pos = trajectory[-1]
            # Add a white border to the scatter point
            ax.scatter(current_pos[0], current_pos[1], color='white', s=120, alpha=0.3)
            ax.scatter(current_pos[0], current_pos[1], color=color, s=100, 
                      label=f'{get_vehicle_class_name(vehicle_type) if vehicle_type else "Unknown"}',
                      edgecolor='white', linewidth=1.5)
    
    # Set up the plot with modern styling
    ax.set_title('Top-Down View', color='white', fontsize=14, pad=20)
    ax.set_xlabel('X Position (meters)', color='white', fontsize=12)
    ax.set_ylabel('Y Position (meters)', color='white', fontsize=12)
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    
    # Customize ticks
    ax.tick_params(colors='white')
    
    # Add legend with unique entries and modern styling
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(), 
                      bbox_to_anchor=(1.05, 1), loc='upper left',
                      frameon=True, facecolor='#2D2D2D', edgecolor='white',
                      labelcolor='white')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add a subtle border
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI for better quality
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_animated_statistics():
    """Create animated statistics visualization with modern styling and interactivity."""
    # Create subplots with specific types for each subplot
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Vehicle Count Over Time', 'Average Speed',
                       'Traffic Density', 'Vehicle Type Distribution',
                       'Speed Distribution', 'Traffic Flow Heatmap'),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "domain"}, {"type": "domain"}],
               [{"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # Get time series data
    times = st.session_state.time_series['time']
    vehicle_counts = st.session_state.time_series['vehicle_count']
    avg_speeds = st.session_state.time_series['avg_speed']
    
    # Add vehicle count line with area fill and animation
    fig.add_trace(
        go.Scatter(
            x=times, 
            y=vehicle_counts,
            mode='lines+markers',
            name='Vehicle Count',
            line=dict(width=3, color='#00ff00'),
            marker=dict(size=8, color='#00ff00'),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)',
            hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Count:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add average speed line with gradient and animation
    fig.add_trace(
        go.Scatter(
            x=times, 
            y=avg_speeds,
            mode='lines+markers',
            name='Average Speed',
            line=dict(width=3, color='#ff00ff'),
            marker=dict(size=8, color='#ff00ff'),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 255, 0.1)',
            hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Speed:</b> %{y:.1f} km/h<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add traffic density as an animated gauge
    density = st.session_state.analysis_data['traffic_density']
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=density * 100,
            title={'text': "Traffic Density", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00ffff"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(0, 255, 0, 0.3)'},
                    {'range': [33, 66], 'color': 'rgba(255, 255, 0, 0.3)'},
                    {'range': [66, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': density * 100
                }
            }
        ),
        row=2, col=1
    )
    
    # Add vehicle type distribution with custom colors and animation
    vehicle_types = list(st.session_state.vehicle_types.keys())
    vehicle_counts = list(st.session_state.vehicle_types.values())
    colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff']
    fig.add_trace(
        go.Pie(
            labels=vehicle_types,
            values=vehicle_counts,
            name="Vehicle Types",
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent',
            textfont_size=14,
            textposition='outside',
            pull=[0.1] * len(vehicle_types),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add speed distribution histogram
    all_speeds = [speed for speeds in st.session_state.speeds.values() for speed in speeds]
    if all_speeds:
        fig.add_trace(
            go.Histogram(
                x=all_speeds,
                name='Speed Distribution',
                marker_color='#00ffff',
                opacity=0.7,
                nbinsx=20,
                hovertemplate='<b>Speed:</b> %{x:.1f} km/h<br><b>Count:</b> %{y}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # Add traffic flow heatmap
    if st.session_state.car_tracks:
        x_coords = []
        y_coords = []
        for trajectory in st.session_state.car_tracks.values():
            if len(trajectory) > 1:
                trajectory = np.array(trajectory)
                x_coords.extend(trajectory[:, 0])
                y_coords.extend(trajectory[:, 1])
        
        if x_coords and y_coords:
            fig.add_trace(
                go.Histogram2d(
                    x=x_coords,
                    y=y_coords,
                    colorscale='Viridis',
                    showscale=True,
                    name='Traffic Flow',
                    hovertemplate='<b>Position:</b> (%{x:.1f}, %{y:.1f})<br><b>Density:</b> %{z}<extra></extra>'
                ),
                row=3, col=2
            )
    
    # Update layout with modern styling
    fig.update_layout(
        height=1200,
        showlegend=True,
        template='plotly_dark',
        title_text="<b>Real-time Traffic Analytics</b>",
        title_font=dict(size=24, color='white'),
        margin=dict(t=120, b=50, l=50, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        ),
        hovermode='closest',
        hoverdistance=100
    )
    
    # Update axes with modern styling
    for i in range(1, 4):
        for j in range(1, 3):
            if i != 2:  # Skip domain subplots
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255,255,255,0.1)',
                    zeroline=False,
                    row=i, col=j
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(255,255,255,0.1)',
                    zeroline=False,
                    row=i, col=j
                )
    
    # Update specific axes labels
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=2)
    fig.update_xaxes(title_text="Speed (km/h)", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="X Position (meters)", row=3, col=2)
    fig.update_yaxes(title_text="Y Position (meters)", row=3, col=2)
    
    # Add annotations for current values
    fig.add_annotation(
        x=0.5, y=1.1,
        text=f"Current Vehicle Count: {vehicle_counts[-1] if vehicle_counts else 0}",
        showarrow=False,
        font=dict(size=14, color='#00ff00'),
        xref="paper", yref="paper"
    )
    
    fig.add_annotation(
        x=0.5, y=1.05,
        text=f"Current Average Speed: {avg_speeds[-1]:.1f} km/h" if avg_speeds else "Current Average Speed: 0 km/h",
        showarrow=False,
        font=dict(size=14, color='#ff00ff'),
        xref="paper", yref="paper"
    )
    
    return fig

def process_video(video_file, camera_id=None):
    """Process the uploaded video file."""
    if video_file is None:
        return
    
    model = load_model()
    if model is None:
        return
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize camera data if new
    if camera_id not in st.session_state.cameras:
        st.session_state.cameras[camera_id] = {
            'car_tracks': defaultdict(list),
            'next_car_id': 0,
            'car_ages': {},
            'frame_number': 0,
            'speeds': defaultdict(list),
            'directions': defaultdict(list),
            'time_series': {
                'time': [],
                'vehicle_count': [],
                'avg_speed': []
            },
            'vehicle_types': Counter(),
            'analysis_data': {
                'total_vehicles': 0,
                'avg_speed': 0,
                'max_speed': 0,
                'traffic_density': 0,
                'direction_counts': defaultdict(int)
            }
        }
    
    # Create layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Real-time Detection")
        video_placeholder = st.empty()
        
        # Add legend for vehicle types with modern styling
        st.markdown("""
        <style>
        .vehicle-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 15px 0;
            padding: 15px;
            border-radius: 10px;
            background: #2D2D2D;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-right: 20px;
            color: white;
        }
        .color-box {
            width: 24px;
            height: 24px;
            margin-right: 8px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 4px;
        }
        </style>
        <div class="vehicle-legend">
            <div class="legend-item">
                <div class="color-box" style="background: rgb(255,0,0);"></div>
                <span>Bicycle</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: rgb(0,255,0);"></div>
                <span>Car</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: rgb(0,0,255);"></div>
                <span>Motorcycle</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: rgb(255,255,0);"></div>
                <span>Bus</span>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: rgb(255,0,255);"></div>
                <span>Truck</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add top-down view with modern styling
        st.markdown("""
        <style>
        .top-down-container {
            margin: 20px 0;
            padding: 20px;
            border-radius: 10px;
            background: #1E1E1E;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        <div class="top-down-container">
            <h3 style="color: white; margin-bottom: 15px;">Top-Down View</h3>
        </div>
        """, unsafe_allow_html=True)
        top_down_placeholder = st.empty()
    
    with col2:
        st.subheader("Analysis")
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Flow Analysis", "Time Series", "Vehicle Stats"])
        
        with analysis_tab1:
            flow_placeholder = st.empty()
            speed_pos_placeholder = st.empty()
        with analysis_tab2:
            time_series_placeholder = st.empty()
            animated_stats_placeholder = st.empty()
        with analysis_tab3:
            vehicle_dist_placeholder = st.empty()
            stats_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        df = results.pandas().xyxy[0]

        update_tracks(df, st.session_state.frame_number, fps)
        update_analysis_data()
        
        # Draw detections with enhanced styling
        for index, row in df.iterrows():
            if row['class'] in [1, 2, 3, 5, 7]:  # Vehicle classes
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
                
                # Get color for this vehicle type
                color = get_vehicle_color(row['class'])
                
                # Draw rectangle with vehicle-specific color and enhanced styling
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Add vehicle type label with matching color and enhanced styling
                vehicle_type = get_vehicle_class_name(row['class'])
                cv2.putText(frame, vehicle_type, (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add coordinates with matching color and enhanced styling
                real_world_pos = transform_point(center, H)
                cv2.putText(frame, f"({real_world_pos[0]:.1f}, {real_world_pos[1]:.1f})", 
                           (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        scale = min(1.0, 800 / max(width, height))
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update top-down view more frequently
        if st.session_state.frame_number % 3 == 0:  # Increased update frequency
            top_down_plot = plot_top_down_view()
            top_down_placeholder.image(top_down_plot, use_container_width=True)
        
        # Update analysis displays
        if st.session_state.frame_number % 30 == 0:
            with analysis_tab1:
                flow_plot = plot_flow_map()
                flow_placeholder.image(flow_plot, use_container_width=True)
                
                speed_pos_plot = plot_speed_position()
                speed_pos_placeholder.image(speed_pos_plot, use_container_width=True)
            
            with analysis_tab2:
                time_series_plot = plot_time_series()
                time_series_placeholder.image(time_series_plot, use_container_width=True)
                
                # Update animated statistics
                animated_stats = create_animated_statistics()
                animated_stats_placeholder.plotly_chart(animated_stats, use_container_width=True)
            
            with analysis_tab3:
                vehicle_dist_plot = plot_vehicle_distribution()
                vehicle_dist_placeholder.image(vehicle_dist_plot, use_container_width=True)
                
                # Display statistics with vehicle types
                stats_data = {
                    'Metric': [
                        'Total Vehicles',
                        'Average Speed',
                        'Max Speed',
                        'Traffic Density'
                    ],
                    'Value': [
                        st.session_state.analysis_data['total_vehicles'],
                        f"{st.session_state.analysis_data['avg_speed']:.1f} km/h",
                        f"{st.session_state.analysis_data['max_speed']:.1f} km/h",
                        f"{st.session_state.analysis_data['traffic_density']:.2f}"
                    ]
                }
                
                # Add vehicle type counts
                for vehicle_type, count in st.session_state.analysis_data['vehicle_types'].items():
                    stats_data['Metric'].append(f'Number of {vehicle_type.capitalize()}s')
                    stats_data['Value'].append(str(count))
                
                stats_df = pd.DataFrame(stats_data)
                stats_placeholder.dataframe(stats_df, use_container_width=True)
        
        st.session_state.frame_number += 1
    
    cap.release()
    os.unlink(tfile.name)

# Streamlit UI
st.title("Advanced Car Tracking & Analysis")

# Camera selection
camera_options = ["Camera 1", "Camera 2", "Camera 3"]
selected_camera = st.selectbox("Select Camera", camera_options)

# File uploader for each camera
uploaded_file = st.file_uploader(f"Upload video for {selected_camera}", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    process_video(uploaded_file, selected_camera)
else:
    st.info("Please upload a video file to begin tracking and analysis.")
