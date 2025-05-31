# AI Traffic Analysis System 🚗

A sophisticated traffic monitoring solution that combines computer vision and machine learning to provide real-time insights into traffic patterns.

## Features 🌟

- Real-time vehicle detection and tracking
- Dynamic eagle eye view visualization
- Adaptive learning system
- Speed analysis and traffic pattern recognition
- Beautiful, intuitive user interface
- Real-time statistics and analytics

## Technical Stack 🛠️

- YOLOv5 for object detection
- Python for backend processing
- Streamlit for the interactive dashboard
- OpenCV for video processing
- PyTorch for deep learning

## Prerequisites 📋

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- Git

## Installation 📥

1. Clone the repository:
```bash
git clone https://github.com/Madhav1716/AI-powered-Traffic-Analysis-System.git
cd AI-powered-Traffic-Analysis-System
```

2. Create and activate a virtual environment:

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Make sure your virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. Run the application:
```bash
streamlit run final.py
```

3. Using the Application:

   a. Sample Video:
   - The repository includes a sample video (video.mp4)
   - This video demonstrates typical traffic scenarios
   - Use this to test the system's capabilities

   b. Your Own Videos:
   - Click "Browse files" to upload your traffic video
   - Supported formats: MP4, AVI, MOV
   - Recommended video specifications:
     * Resolution: 720p or 1080p
     * Frame rate: 24-30 fps
     * Duration: 1-5 minutes for optimal performance
     * Camera angle: Elevated view of traffic

   c. Video Processing:
   - The system will automatically:
     * Detect and track vehicles
     * Show real-time statistics
     * Display the eagle eye view
     * Calculate vehicle speeds
     * Learn from the video patterns

   d. Best Practices:
   - Use videos with clear visibility
   - Ensure good lighting conditions
   - Avoid shaky camera footage
   - Keep the camera stationary
   - Include multiple lanes if possible

4. Understanding the Output:
   - Live Video Feed: Shows real-time vehicle detection
   - Eagle Eye View: Top-down perspective of traffic flow
   - Statistics: Active cars, total unique cars, and speeds
   - Learning Metrics: System's adaptive parameters

## Project Structure 📁

```
AI-powered-Traffic-Analysis-System/
├── final.py              # Main application file
├── requirements.txt      # Project dependencies
├── README.md            # Project documentation
├── .gitignore           # Git ignore file
├── video.mp4            # Sample traffic video
├── trajectories.png     # Sample visualization output
└── yolov5s.pt          # Pre-trained YOLOv5 model weights
```

## Performance Tips 💡

1. For better performance:
   - Use a CUDA-capable GPU
   - Keep video resolution moderate (720p recommended)
   - Close other resource-intensive applications
   - Use shorter video clips (1-5 minutes)
   - Ensure good lighting in videos

2. If experiencing lag:
   - Reduce video resolution
   - Increase frame skip in the code
   - Use shorter video clips
   - Close other applications
   - Check GPU memory usage

## Troubleshooting 🔧

1. If you get CUDA errors:
   - Make sure you have the correct CUDA version installed
   - Check if your GPU is CUDA-compatible
   - Try running on CPU by modifying the code

2. If the application is slow:
   - Reduce the video resolution
   - Increase the frame skip value
   - Close other applications
   - Check system resources

3. If you get dependency errors:
   - Make sure you're in the virtual environment
   - Try reinstalling requirements: `pip install -r requirements.txt --force-reinstall`

4. If video processing fails:
   - Check video format compatibility
   - Ensure video file is not corrupted
   - Try with the sample video first
   - Check video resolution and length

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- YOLOv5 team for the object detection model
- Streamlit team for the amazing web framework
- OpenCV community for computer vision tools 