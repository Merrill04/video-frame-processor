from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import json
import uuid
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import shutil
from pathlib import Path
import aiofiles
import asyncio

app = FastAPI(title="Video Frame Processing API", description="Extract and search video frames")

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"
VECTORS_FILE = "feature_vectors.json"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# In-memory vector database (using JSON file for persistence)
class SimpleVectorDB:
    def __init__(self, vectors_file: str):
        self.vectors_file = vectors_file
        self.vectors = self.load_vectors()
    
    def load_vectors(self) -> Dict[str, Any]:
        """Load vectors from JSON file"""
        if os.path.exists(self.vectors_file):
            try:
                with open(self.vectors_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_vectors(self):
        """Save vectors to JSON file"""
        with open(self.vectors_file, 'w') as f:
            json.dump(self.vectors, f, indent=2)
    
    def add_vector(self, frame_id: str, vector: List[float], metadata: Dict[str, Any]):
        """Add a feature vector with metadata"""
        self.vectors[frame_id] = {
            "vector": vector,
            "metadata": metadata
        }
        self.save_vectors()
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors using cosine similarity"""
        if not self.vectors:
            return []
        
        similarities = []
        query_vec = np.array(query_vector).reshape(1, -1)
        
        for frame_id, data in self.vectors.items():
            stored_vec = np.array(data["vector"]).reshape(1, -1)
            similarity = cosine_similarity(query_vec, stored_vec)[0][0]
            
            similarities.append({
                "frame_id": frame_id,
                "similarity": float(similarity),
                "metadata": data["metadata"],
                "vector": data["vector"]
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

# Initialize vector database
vector_db = SimpleVectorDB(VECTORS_FILE)

def extract_color_histogram(image_path: str) -> List[float]:
    """Extract color histogram as feature vector"""
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate histogram for each channel
    hist_r = cv2.calcHist([img_rgb], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([img_rgb], [2], None, [32], [0, 256])
    
    # Normalize and flatten
    hist_r = hist_r.flatten() / hist_r.sum()
    hist_g = hist_g.flatten() / hist_g.sum()  
    hist_b = hist_b.flatten() / hist_b.sum()
    
    # Combine all channels
    feature_vector = np.concatenate([hist_r, hist_g, hist_b])
    return feature_vector.tolist()

def extract_frames_from_video(video_path: str, interval: int = 1) -> List[str]:
    """Extract frames from video at specified interval"""
    video_id = str(uuid.uuid4())[:8]
    frame_paths = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)  # Extract every 'interval' seconds
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_id}_frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(FRAMES_DIR, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return frame_paths

@app.get("/")
async def root():
    return {"message": "Video Frame Processing API", "status": "running"}

@app.post("/upload-video/")
async def upload_video(
    file: UploadFile = File(...),
    interval: int = Query(1, description="Frame extraction interval in seconds")
):
    """Upload video and extract frames with feature vectors"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    try:
        # Extract frames
        frame_paths = extract_frames_from_video(file_path, interval)
        
        if not frame_paths:
            raise HTTPException(status_code=400, detail="No frames could be extracted")
        
        # Process frames and compute feature vectors
        processed_frames = []
        for frame_path in frame_paths:
            try:
                # Compute feature vector
                feature_vector = extract_color_histogram(frame_path)
                
                if feature_vector:
                    frame_id = Path(frame_path).stem
                    metadata = {
                        "original_video": file.filename,
                        "frame_path": frame_path,
                        "video_id": file_id
                    }
                    
                    # Store in vector database
                    vector_db.add_vector(frame_id, feature_vector, metadata)
                    
                    processed_frames.append({
                        "frame_id": frame_id,
                        "frame_path": frame_path,
                        "feature_vector_length": len(feature_vector)
                    })
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                continue
        
        return {
            "message": "Video processed successfully",
            "video_id": file_id,
            "total_frames": len(processed_frames),
            "frames": processed_frames[:10]  # Return first 10 for preview
        }
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up uploaded video
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/search-similar/")
async def search_similar_frames(
    file: UploadFile = File(...),
    top_k: int = Query(5, description="Number of similar frames to return")
):
    """Search for similar frames using an uploaded image"""
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Save uploaded image temporarily
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4()}_{file.filename}")
    
    async with aiofiles.open(temp_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    try:
        # Extract feature vector from uploaded image
        query_vector = extract_color_histogram(temp_path)
        
        if not query_vector:
            raise HTTPException(status_code=400, detail="Could not extract features from image")
        
        # Search for similar frames
        results = vector_db.search(query_vector, top_k)
        
        # Prepare response with frame info
        similar_frames = []
        for result in results:
            frame_info = {
                "frame_id": result["frame_id"],
                "similarity": result["similarity"],
                "metadata": result["metadata"]
            }
            similar_frames.append(frame_info)
        
        return {
            "query_image": file.filename,
            "total_results": len(similar_frames),
            "similar_frames": similar_frames
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching frames: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/frame/{frame_id}")
async def get_frame(frame_id: str):
    """Get frame image by frame ID"""
    
    if frame_id not in vector_db.vectors:
        raise HTTPException(status_code=404, detail="Frame not found")
    
    frame_path = vector_db.vectors[frame_id]["metadata"]["frame_path"]
    
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="Frame file not found")
    
    return FileResponse(frame_path, media_type="image/jpeg")

@app.get("/frames/list")
async def list_frames():
    """List all stored frames"""
    frames = []
    for frame_id, data in vector_db.vectors.items():
        frames.append({
            "frame_id": frame_id,
            "metadata": data["metadata"],
            "vector_length": len(data["vector"])
        })
    
    return {
        "total_frames": len(frames),
        "frames": frames
    }

@app.delete("/frames/clear")
async def clear_all_frames():
    """Clear all stored frames and vectors"""
    try:
        # Remove all frame files
        for frame_id, data in vector_db.vectors.items():
            frame_path = data["metadata"]["frame_path"]
            if os.path.exists(frame_path):
                os.remove(frame_path)
        
        # Clear vector database
        vector_db.vectors = {}
        vector_db.save_vectors()
        
        return {"message": "All frames and vectors cleared successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)