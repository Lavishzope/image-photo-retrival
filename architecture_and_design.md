# Architecture & Design: Intelligent Event Photo Retrieval System

## 1. Problem Understanding
The goal is to solve the inefficiency of manual photo searching in large event albums.
**Core Challenge**: Efficiently searching for a specific person in a dataset of unlabelled structured images containing multiple faces.
**Key Constraints**:
- **Multi-Face**: Photos contain crowds; need to isolate individual faces.
- **Accuracy**: Must be robust to angles and lighting.
- **Speed**: Retrieval must be near-instantaneous for a collection of ~1000s of photos.
- **User Experience**: Simple upload-and-get-results workflow.

**Edge Cases Anticipated**:
- **No Face Detected**: The reference selfie or event photo might have no detectable faces.
- **Multiple Faces in Selfie**: The system should prompt for a clear single-face image or select the largest face.
- **Low Confidence/False Positives**: Look-alikes or very blurry faces. We need a strict enough distance threshold.

## 2. Proposed Approach

### Algorithm Selection
- **Face Detection & Embedding**: We will use `face_recognition` (built on top of `dlib`'s state-of-the-art model).
    - It maps faces to 128-dimensional encodings.
    - It is robust and accurate for general purpose usage.
- **Similarity Search**: `FAISS` (Facebook AI Similarity Search).
    - We will use a Flat Index (L2 distance/Euclidean) for exact matching initially as dataset is small (<10k), but it scales well.
    - Metric: Euclidean Distance. Lower distance = Better match.
    - Threshold: A distance of 0.6 is a common cutoff for distinct faces. We might tune this to 0.5 for higher precision.

### Logic Flow
1.  **Indexing Phase (Offline/Pre-process)**
    - Iterate through all event images.
    - Detect all faces in each image.
    - Compute 128-d encoding for each face.
    - Store: `{index_id: (image_path, bounding_box)}` in a metadata store (Pickle/JSON).
    - Add vectors to FAISS index.

2.  **Retrieval Phase (Real-time)**
    - User uploads selfie.
    - Detect face and compute 128-d encoding.
    - Query FAISS index for vectors with Euclidean distance < Threshold (e.g. 0.6).
    - Retrieve metadata for the matching indices.
    - Display original images with bounding boxes drawn.

## 3. Architecture Diagram

```mermaid
graph TD
    subgraph "Phase 1: Indexing (Batch)"
        A[Event Photos Folder] -->|Load Image| B[Face Detection]
        B -->|Found Faces| C[Embedding Generation 128-d]
        C -->|Add Vector| D[FAISS Index]
        C -->|Save Map| E[Metadata Store JSON/Pickle]
    end

    subgraph "Phase 2: Retrieval (User)"
        U[User] -->|Uploads Selfie| F[Streamlit Web App]
        F -->|Process| G[Face Detection & Embedding]
        G -->|Query Vector| D
        D -->|Return Top-K Indices| F
        F -->|Fetch Path & Box| E
        E -->|Return Data| F
        F -->|Render| H[Results Gallery]
    end
```
