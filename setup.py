#!/usr/bin/env python3

import os
import sys
import subprocess
import platform

def create_directories():
    directories = ['uploads', 'frames', 'logs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_requirements():
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_env_file():
    env_content = """# Environment Configuration
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
DEBUG=True

# Directories
UPLOAD_DIR=uploads
FRAMES_DIR=frames
VECTORS_FILE=feature_vectors.json

# Frame extraction settings
DEFAULT_INTERVAL=1
MAX_FILE_SIZE=100MB
SUPPORTED_FORMATS=mp4,avi,mov,mkv
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✅ Created .env configuration file")

def main():
    print("🚀 Setting up Video Frame Processing Application...")
    print("=" * 50)

    if not check_python_version():
        sys.exit(1)

    create_directories()

    create_env_file()
    
    if not install_requirements():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete! Next steps:")
    print("\n1. Start the FastAPI server:")
    print("   uvicorn main:app --reload")
    print("\n2. In a new terminal, start the Streamlit app:")
    print("   streamlit run streamlit_app.py")
    print("\n3. Open your browser to:")
    print("   - FastAPI docs: http://localhost:8000/docs")
    print("   - Streamlit app: http://localhost:8501")
    print("\n🎥 Happy video processing!")

if __name__ == "__main__":
    main()
