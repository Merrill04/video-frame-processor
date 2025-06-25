Video Frame Processing Application
A complete FastAPI + Streamlit application for extracting frames from videos, computing feature vectors, and performing similarity searches using free tier resources.

Features

Video Processing: Upload videos and extract frames at custom intervals
Feature Extraction: Compute color histogram feature vectors for each frame
Vector Database: Store and search feature vectors using cosine similarity
Web API: RESTful API with FastAPI
User Interface: Interactive Streamlit web app
Local Storage: No external dependencies - runs entirely locally

Requirements

Python 3.8+
2GB+ RAM recommended
1GB+ free disk space
Modern web browser

Quick Start

Step 1: Clone/Download the Files
Create a new directory and save all the provided files:
bashmkdir video-frame-processor
cd video-frame-processor
Save these files in your directory:

main.py (FastAPI backend)
streamlit_app.py (Streamlit UI)
requirements.txt (Dependencies)
setup.py (Setup script)

Step 2: Run Setup
bashpython setup.py
This will:

Check Python version
Create necessary directories (uploads/, frames/)
Install all required packages
Create configuration files

Step 3: Start the Applications
Terminal 1 - Start FastAPI Backend:
bashuvicorn main:app --reload
Terminal 2 - Start Streamlit Frontend:
bashstreamlit run streamlit_app.py
Step 4: Access the Applications

Streamlit UI: http://localhost:8501
FastAPI Docs: http://localhost:8000/docs
API Root: http://localhost:8000

How to Use

1. Upload Video

Go to "Upload Video" page in Streamlit
Choose a video file (MP4, AVI, MOV, MKV)
Set frame extraction interval (1-10 seconds)
Click "Process Video"
Wait for processing to complete
View sample extracted frames

2. Search Similar Frames

Go to "Search Similar Frames" page
Upload a query image (JPG, PNG, etc.)
Set number of results to return
Click "Search Similar Frames"
View results with similarity scores

3. Browse All Frames

Go to "Browse Frames" page
View all stored frames organized by video
Expand each video to see its frames

4. Manage Data

Go to "Settings" page
View system status and database info
Clear all data if needed
