# Intelligent Event Photo Retrieval System

A local web application that allows users to upload a selfie and instantly retrieve all photos from an event where they appear.

## Features
- **Multi-Face Detection**: Handles group photos.
- **Fast Search**: Uses FAISS for efficient Similarity Search.
- **Privacy First**: Runs 100% locally.
- **Simple UI**: Built with Streamlit.

## Prerequisites
- Python 3.8+
- [CMake](https://cmake.org/download/) (Required for building `dlib`/`face_recognition`)
    - **Windows**: Install Visual Studio Build Tools with C++ support if pip install fails.

## Installation

1. **Clone/Download** this project.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `face_recognition` fails to install, ensure you have CMake installed and added to PATH.*

## Usage

### Step 1: Prepare Data
1. Create a folder `data/images`.
2. Place all your event photos (JPG/PNG) into `data/images`.

### Step 2: Index Images
Run the indexing script to detect faces and build the database:
```bash
python index_images.py
```
This enables fast searching later. It effectively "scans" the album.

### Step 3: Start the App
Launch the web interface:
```bash
streamlit run app.py
```
1. Open the URL provided (usually `http://localhost:8501`).
2. Upload a selfie.
3. View your detected photos!

## Project Structure
- `data/`
  - `images/`: Your photo album.
  - `embeddings/`: Stores the computed face indices.
- `search_engine.py`: Core logic for face detection and FAISS.
- `index_images.py`: Script to process photos.
- `app.py`: The web application.

## Troubleshooting
- **dlib error**: Install CMake. On Windows `pip install cmake`. If that fails, install Visual Studio C++ Build Tools.
- **No matches found**: Try adjusting the threshold slider in the sidebar. 0.6 is standard, 0.5 is strict.
