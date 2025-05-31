import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
import os
import ssl
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# Set page config with professional styling
st.set_page_config(
    layout="wide",
    page_title="AI Traffic Analysis",
    initial_sidebar_state="collapsed",
    page_icon="🚗"
)

# Professional CSS styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
    }
    .title-text {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #00ff87, #60efff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle-text {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00ff87, #60efff);
        color: #0f2027;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 135, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'car_tracks' not in st.session_state:
    st.session_state.car_tracks = {}
if 'next_car_id' not in st.session_state:
    st.session_state.next_car_id = 0
if 'frame_number' not in st.session_state:
    st.session_state.frame_number = 0

# Load the YOLOv5 model
@st.cache_resource
def load_model():
    try:
        # Use the official YOLOv5 repository with optimized settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
        model.conf = 0.25  # Set confidence threshold
        model.iou = 0.45   # Set IoU threshold
        
        # Disable gradient computation for inference
        model.eval()
        with torch.no_grad():
            # Warm up the model
            dummy_input = torch.zeros((1, 3, 640, 640))
            model(dummy_input)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Update homography matrix for better perspective
H = np.array([
    [1.2, 0.0, 0.0],
    [0.0, 1.2, 0.0],
    [0.0, 0.0, 1.0]
])

def transform_point(point, H):
    """Transform a point from video coordinates to real-world coordinates."""
    try:
        # Scale the point to match the video dimensions and center it
        x = (point[0] - 960) / 960  # Center and normalize x coordinate
        y = (point[1] - 540) / 540  # Center and normalize y coordinate
        
        # Apply perspective transformation
        scaled_point = np.array([x, y, 1])
        transformed = np.dot(H, scaled_point)
        result = transformed[:2] / transformed[2]
        
        # Scale the result to fit the view
        return result * 5  # Scale to fit the -5 to 5 range
    except Exception as e:
        st.error(f"Error in point transformation: {str(e)}")
        return point[:2]  # Return original point if transformation fails

def plot_top_down_view():
    """Create a professional eagle eye view visualization."""
    plt.close('all')
    plt.style.use('dark_background')
    
    # Create figure with professional styling
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f2027')
    ax.set_facecolor('#203a43')
    
    # Add grid with enhanced styling
    ax.grid(True, linestyle='--', alpha=0.15, color='white')
    
    # Plot vehicle positions with enhanced styling
    for car_id, trajectory in st.session_state.car_tracks.items():
        if len(trajectory) > 0:
            # Convert trajectory to numpy array
            trajectory = np.array(trajectory)
            
            # Plot trajectory line with gradient effect
            if len(trajectory) > 1:
                # Create gradient effect for trajectory
                points = np.array(trajectory)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                for i, segment in enumerate(segments):
                    alpha = 0.3 + 0.7 * (i / len(segments))
                    ax.plot(segment[0:2], segment[2:4], '-', 
                           color='#00ff87', alpha=alpha, linewidth=2.5)
            
            # Plot current position with enhanced glow effect
            current_pos = trajectory[-1]
            
            # Add multiple layers of glow effect
            for size, alpha in [(300, 0.1), (200, 0.2), (100, 0.3)]:
                ax.scatter(current_pos[0], current_pos[1], 
                          color='#60efff', s=size, alpha=alpha, edgecolor='none')
            
            # Add main vehicle marker
            ax.scatter(current_pos[0], current_pos[1], 
                      color='#00ff87', s=150, edgecolor='white', linewidth=1.5)
            
            # Add car ID label with background
            ax.text(current_pos[0], current_pos[1], f'Car {car_id}', 
                   color='white', fontsize=10, ha='center', va='center',
                   bbox=dict(facecolor='#203a43', alpha=0.7, edgecolor='#60efff', 
                           boxstyle='round,pad=0.3'))
    
    # Professional styling for the plot
    ax.set_title('Eagle Eye View - Traffic Analysis', 
                 color='white', fontsize=20, pad=20, fontweight='bold')
    ax.set_xlabel('X Position (meters)', color='white', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Position (meters)', color='white', fontsize=12, labelpad=10)
    
    # Enhanced tick styling
    ax.tick_params(colors='white', labelsize=10)
    
    # Set equal aspect ratio and adjust limits
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    # Add border with gradient
    for spine in ax.spines.values():
        spine.set_color('#60efff')
        spine.set_alpha(0.3)
        spine.set_linewidth(2)
    
    # Add coordinate system indicators
    ax.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='#60efff', ec='#60efff', alpha=0.5)
    ax.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='#60efff', ec='#60efff', alpha=0.5)
    ax.text(1.1, 0, 'X', color='#60efff', fontsize=10)
    ax.text(0, 1.1, 'Y', color='#60efff', fontsize=10)
    
    # Add scale indicator
    ax.plot([-4.5, -4], [-4.5, -4.5], 'w-', linewidth=2)
    ax.text(-4.25, -4.7, '0.5m', color='white', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save with optimized DPI for better performance
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, 
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

def process_video(video_file):
    """Process the uploaded video file with enhanced visualization."""
    if video_file is None:
        return
    
    model = load_model()
    if model is None:
        return
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # More aggressive frame skipping for better performance
    frame_skip = max(2, fps // 15)  # Target 15 FPS for smoother playback
    
    # Create professional layout
    st.markdown("""
        <div class="card">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        video_placeholder = st.empty()
        st.markdown("### Live Video Feed")
        st.markdown("#### Vehicle Tracking Statistics")
        stats_placeholder = st.empty()
        learning_placeholder = st.empty()
    
    with col2:
        st.markdown("### Real-time Eagle Eye View")
        top_down_placeholder = st.empty()
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Initialize tracking variables
    frame_count = 0
    max_track_history = 20  # Reduced history length
    active_cars = set()
    car_positions = {}
    car_speeds = {}
    total_cars = 0
    
    # Learning parameters
    min_distance = 50
    confidence_threshold = 0.25
    speed_history = []
    detection_history = []
    
    # Reset car tracks for new video
    st.session_state.car_tracks = {}
    st.session_state.next_car_id = 0
    
    # Pre-allocate frame buffer
    frame_buffer = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Process frames with optimized settings
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames to maintain performance
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Run inference with optimized settings
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    results = model(frame)
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
            continue
        
        # Clear active cars for this frame
        active_cars.clear()
        
        # Process detections with enhanced visualization
        current_positions = []
        current_detections = 0
        
        # Optimize detection processing
        detections = results.xyxy[0].cpu().numpy()  # Move to CPU once
        for det in detections:
            if int(det[5]) == 2:  # Car class
                current_detections += 1
                xmin, ymin, xmax, ymax = map(int, det[:4])
                confidence = float(det[4])
                
                # Only draw if confidence is high enough
                if confidence > confidence_threshold:
                    # Enhanced bounding box visualization
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 135), 2)
                    
                    # Calculate center point
                    center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
                    
                    # Transform and display coordinates
                    real_world_pos = transform_point(center, H)
                    current_positions.append((real_world_pos, center))
                    
                    # Add debug visualization (only for high confidence detections)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"({real_world_pos[0]:.2f}, {real_world_pos[1]:.2f})", 
                               (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (96, 239, 255), 2)
        
        # Update detection history
        detection_history.append(current_detections)
        if len(detection_history) > 20:  # Reduced history length
            detection_history.pop(0)
        
        # Match current positions with existing tracks
        matched_cars = set()
        for pos, center in current_positions:
            matched = False
            for car_id, last_pos in car_positions.items():
                if car_id not in matched_cars:
                    # Calculate distance between current and last position
                    distance = np.sqrt((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)
                    if distance < min_distance:
                        # Update existing track
                        st.session_state.car_tracks[car_id].append(pos)
                        if len(st.session_state.car_tracks[car_id]) > max_track_history:
                            st.session_state.car_tracks[car_id].pop(0)
                        
                        # Calculate speed
                        if len(st.session_state.car_tracks[car_id]) > 1:
                            prev_pos = st.session_state.car_tracks[car_id][-2]
                            speed = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
                            speed_history.append(speed)
                            if len(speed_history) > 50:  # Reduced history length
                                speed_history.pop(0)
                        
                        car_positions[car_id] = pos
                        active_cars.add(car_id)
                        matched_cars.add(car_id)
                        matched = True
                        break
            
            if not matched:
                # Create new track
                st.session_state.car_tracks[st.session_state.next_car_id] = [pos]
                car_positions[st.session_state.next_car_id] = pos
                active_cars.add(st.session_state.next_car_id)
                st.session_state.next_car_id += 1
                total_cars += 1
        
        # Adaptive learning (reduced frequency)
        if len(speed_history) > 10 and frame_count % 5 == 0:
            avg_speed = np.mean(speed_history)
            std_speed = np.std(speed_history)
            min_distance = max(30, min(100, avg_speed * 2))
            if len(detection_history) > 10:
                detection_std = np.std(detection_history)
                confidence_threshold = max(0.2, min(0.4, 0.25 + detection_std * 0.1))
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update video feed (reduced frequency)
        if frame_count % 2 == 0:
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update eagle eye view less frequently
        if frame_count % 5 == 0:
            try:
                top_down_plot = plot_top_down_view()
                if top_down_plot is not None:
                    top_down_placeholder.image(top_down_plot, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating eagle eye view: {str(e)}")
        
        # Update statistics less frequently
        if frame_count % 3 == 0:
            stats_placeholder.markdown(f"""
                <div style="background-color: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <p style="color: #00ff87; margin: 0;">🚗 Active Cars: {len(active_cars)}</p>
                    <p style="color: #60efff; margin: 0;">📊 Total Unique Cars: {total_cars}</p>
                    <p style="color: white; margin: 0;">🎥 Frame: {frame_count}/{total_frames}</p>
                </div>
            """, unsafe_allow_html=True)
            
            if len(speed_history) > 0:
                learning_placeholder.markdown(f"""
                    <div style="background-color: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                        <p style="color: #00ff87; margin: 0;">📈 Learning Statistics:</p>
                        <p style="color: #60efff; margin: 0;">⚡ Average Speed: {np.mean(speed_history):.2f}</p>
                        <p style="color: white; margin: 0;">🎯 Tracking Distance: {min_distance:.1f}</p>
                        <p style="color: white; margin: 0;">🔍 Confidence: {confidence_threshold:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        frame_count += 1
    
    cap.release()
    os.unlink(tfile.name)
    
    # Display final statistics
    st.markdown("### Analysis Complete")
    st.write(f"Total unique cars detected: {total_cars}")
    if len(speed_history) > 0:
        st.write(f"Average vehicle speed: {np.mean(speed_history):.2f}")

# Professional header
st.markdown("""
    <div class="card">
        <h1 class="title-text">AI Traffic Analysis System</h1>
        <p class="subtitle-text">Real-time vehicle tracking with advanced eagle eye visualization</p>
    </div>
""", unsafe_allow_html=True)

# Professional file uploader
st.markdown("""
    <div class="card">
        <h3 style="color: white; margin: 0 0 1rem 0;">📤 Upload Traffic Video</h3>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    process_video(uploaded_file)
else:
    st.markdown("""
        <div class="card">
            <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">
                Upload a traffic video to begin AI-powered analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
