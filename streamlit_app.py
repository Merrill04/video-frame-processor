import streamlit as st
import requests
import json
from PIL import Image
import io
import os
from typing import List, Dict, Any

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Video Frame Processor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.feature-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}
.error-box {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #dc3545;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def check_api_status():
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_video(video_file, interval: int):
    try:
        files = {"file": (video_file.name, video_file.getvalue(), video_file.type)}
        params = {"interval": interval}
        response = requests.post(f"{API_BASE_URL}/upload-video/", files=files, params=params)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error uploading video: {str(e)}")
        return None

def search_similar_frames(image_file, top_k: int):
    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        params = {"top_k": top_k}
        response = requests.post(f"{API_BASE_URL}/search-similar/", files=files, params=params)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error searching frames: {str(e)}")
        return None

def get_frame_image(frame_id: str):
    try:
        response = requests.get(f"{API_BASE_URL}/frame/{frame_id}")
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        return None
    except Exception as e:
        st.error(f"Error fetching frame: {str(e)}")
        return None

def list_all_frames():
    try:
        response = requests.get(f"{API_BASE_URL}/frames/list")
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error fetching frames: {str(e)}")
        return None

def clear_all_data():
    try:
        response = requests.delete(f"{API_BASE_URL}/frames/clear")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error clearing data: {str(e)}")
        return False

def main():
    st.markdown('<h1 class="main-header">🎥 Video Frame Processor</h1>', unsafe_allow_html=True)
    
    if not check_api_status():
        st.markdown("""
        <div class="error-box">
        ⚠️ <strong>API Server Not Running</strong><br>
        Please start the FastAPI server first:<br>
        <code>uvicorn main:app --reload</code>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload Video", "Search Similar Frames", "Browse Frames", "Settings"]
    )
    
    if page == "Upload Video":
        upload_video_page()
    elif page == "Search Similar Frames":
        search_frames_page()
    elif page == "Browse Frames":
        browse_frames_page()
    elif page == "Settings":
        settings_page()

def upload_video_page():
    st.header("📤 Upload Video")
    
    st.markdown("""
    <div class="feature-box">
    <strong>Features:</strong>
    <ul>
    <li>Upload MP4, AVI, MOV, or MKV videos</li>
    <li>Extract frames at custom intervals</li>
    <li>Automatic feature vector computation</li>
    <li>Store in vector database for similarity search</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to extract frames"
        )
    
    with col2:
        interval = st.number_input(
            "Frame Interval (seconds)",
            min_value=1,
            max_value=10,
            value=1,
            help="Extract one frame every N seconds"
        )
    
    if uploaded_file is not None:
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")
        
        if st.button("🎬 Process Video", type="primary"):
            with st.spinner("Processing video... This may take a while."):
                result = upload_video(uploaded_file, interval)
                
                if result:
                    st.markdown(f"""
                    <div class="success-box">
                      <strong>Video processed successfully!</strong><br>
                    Video ID: {result.get('video_id', 'N/A')}<br>
                    Total frames extracted: {result.get('total_frames', 0)}<br>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'frames' in result:
                        st.subheader("📸 Sample Extracted Frames")
                        frame_cols = st.columns(min(5, len(result['frames'])))
                        
                        for i, frame_info in enumerate(result['frames'][:5]):
                            with frame_cols[i]:
                                frame_image = get_frame_image(frame_info['frame_id'])
                                if frame_image:
                                    st.image(frame_image, caption=f"Frame {i+1}", use_column_width=True)
                else:
                    st.error("Failed to process video. Please try again.")

def search_frames_page():
    st.header("Search Similar Frames")
    
    st.markdown("""
    <div class="feature-box">
    <strong>How it works:</strong>
    <ul>
    <li>Upload an image to find similar video frames</li>
    <li>Uses color histogram matching</li>
    <li>Returns most similar frames with similarity scores</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to find similar frames"
        )
    
    with col2:
        top_k = st.number_input(
            "Number of results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of similar frames to return"
        )
    
    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Query Image")
            query_image = Image.open(uploaded_image)
            st.image(query_image, use_column_width=True)
        
        with col2:
            if st.button("Search Similar Frames", type="primary"):
                with st.spinner("Searching for similar frames..."):
                    results = search_similar_frames(uploaded_image, top_k)
                    
                    if results and results.get('similar_frames'):
                        st.subheader("Similar Frames Found")
                        
                        for i, frame_result in enumerate(results['similar_frames']):
                            frame_id = frame_result['frame_id']
                            similarity = frame_result['similarity']
                            
                            st.write(f"**Result {i+1}** - Similarity: {similarity:.3f}")
                            
                            frame_image = get_frame_image(frame_id)
                            if frame_image:
                                st.image(frame_image, width=300)
                            
                            with st.expander(f"Frame Details {i+1}"):
                                st.json(frame_result['metadata'])
                            
                            st.divider()
                    
                    else:
                        st.warning("No similar frames found. Try uploading some videos first!")

def browse_frames_page():
    st.header("📂 Browse All Frames")
    
    frames_data = list_all_frames()
    
    if frames_data and frames_data.get('frames'):
        st.write(f"**Total frames stored:** {frames_data['total_frames']}")

        videos = {}
        for frame in frames_data['frames']:
            video_name = frame['metadata'].get('original_video', 'Unknown')
            if video_name not in videos:
                videos[video_name] = []
            videos[video_name].append(frame)
        
        for video_name, video_frames in videos.items():
            with st.expander(f"🎬 {video_name} ({len(video_frames)} frames)"):
                cols = st.columns(4)
                
                for i, frame in enumerate(video_frames[:12]):  # Show max 12 frames per video
                    with cols[i % 4]:
                        frame_image = get_frame_image(frame['frame_id'])
                        if frame_image:
                            st.image(frame_image, caption=frame['frame_id'][:8], use_column_width=True)
                
                if len(video_frames) > 12:
                    st.write(f"... and {len(video_frames) - 12} more frames")
    
    else:
        st.info("No frames stored yet. Upload some videos first!")

def settings_page():
    st.header("Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Database Management")
        
        frames_data = list_all_frames()
        if frames_data:
            st.metric("Total Frames", frames_data.get('total_frames', 0))
        
        st.warning("This action cannot be undone!")
        
        if st.button("Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                with st.spinner("Clearing all data..."):
                    if clear_all_data():
                        st.success("All data cleared successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to clear data.")
    
    with col2:
        st.subheader("System Information")
        
        if check_api_status():
            st.success("API Server: Running")
        else:
            st.error("API Server: Not responding")
        
        st.info(f"**API URL:** {API_BASE_URL}")
        
        st.subheader("Instructions")
        st.markdown("""
        1. **Start FastAPI server:**
           ```bash
           uvicorn main:app --reload
           ```
        
        2. **Start Streamlit app:**
           ```bash
           streamlit run streamlit_app.py
           ```
        
        3. **Upload videos** to extract frames
        
        4. **Search similar frames** using query images
        """)

if __name__ == "__main__":
    main()
