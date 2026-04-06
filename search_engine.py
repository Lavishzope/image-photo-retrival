import face_recognition
import faiss
import numpy as np
import pickle
import os
import cv2  # OpenCV for faster image loading in indexer

class FaceSearchEngine:
    def __init__(self, embedding_dim=128):
        """
        Initializes the FaceSearchEngine.
        
        Args:
            embedding_dim (int): Dimension of the face embeddings (default 128 for dlib).
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []  # List to store metadata (image_path, bounding_box)
        
        # Paths to save/load index and metadata
        self.index_path = "data/embeddings/faiss_index.bin"
        self.metadata_path = "data/embeddings/metadata.pkl"

    def create_index(self, image_folder_path):
        """
        Scans a folder of images, detects faces, expects embeddings and builds a FAISS index.
        
        Args:
            image_folder_path (str): Path to the folder containing event images.
        """
        print(f"Scanning images in {image_folder_path}...")
        
        embeddings_list = []
        self.metadata = []
        image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("No images found in the data/images directory.")
            return

        for idx, filename in enumerate(image_files):
            image_path = os.path.join(image_folder_path, filename)
            
            # Load image using face_recognition (loads as RGB)
            try:
                img = face_recognition.load_image_file(image_path)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            # Detect faces (using HOG-based model by default, or 'cnn' for GPU if available/slower on CPU)
            # We return face locations as (top, right, bottom, left)
            face_locations = face_recognition.face_locations(img, model="hog")
            
            if len(face_locations) > 0:
                # Compute 128-d embeddings for each face
                face_encodings = face_recognition.face_encodings(img, face_locations)
                
                for box, encoding in zip(face_locations, face_encodings):
                    embeddings_list.append(encoding)
                    # Store metadata: actual path and location of the face
                    self.metadata.append({
                        "path": image_path,
                        "box": box  # (top, right, bottom, left)
                    })
                
                print(f"Processed {filename}: Found {len(face_locations)} faces.")
            else:
                print(f"Processed {filename}: No faces found.")

        # If data collected, build the FAISS index
        if embeddings_list:
            embeddings_np = np.array(embeddings_list).astype('float32')
            
            # Create a Flat L2 index (Exact Search)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings_np)
            
            print(f"Index built with {self.index.ntotal} faces.")
            
            # Save index and metadata to disk
            self.save_index()
        else:
            print("No face embeddings generated. Index not created.")

    def save_index(self):
        """Saves the FAISS index and metadata to disk."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print("Index and metadata saved successfully.")

    def load_index(self):
        """Loads the FAISS index and metadata from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            return True
        else:
            return False

    def search(self, query_image_np, threshold=0.6, k=5):
        """
        Searches for faces matching the query image.
        
        Args:
            query_image_np (numpy array): The uploaded selfie image (RGB).
            threshold (float): Distance threshold for a match.
            k (int): Number of nearest neighbors to retrieve (per face found in query).
            
        Returns:
            list: A list of result dictionaries containing 'path', 'box', 'distance'.
        """
        if self.index is None:
            raise Exception("Index not loaded. Run create_index first or load_index.")

        # Detect face in query image
        # Using 'hog' is faster; 'cnn' is more accurate but requires GPU or is slow
        query_face_locations = face_recognition.face_locations(query_image_np)
        query_encodings = face_recognition.face_encodings(query_image_np, query_face_locations)

        if len(query_encodings) == 0:
            return "No faces detected in the reference selfie."
        
        # We assume the user is the PRIMARY face in the selfie. 
        # If multiple, ideally we ask user to separate, but here we'll take the first/largest.
        # face_recognition sorts by default? Not strictly, but usually fine.
        # Let's take the first found face
        query_embedding = query_encodings[0].reshape(1, -1).astype('float32')
        
        # Search in FAISS
        # We ask for plenty of candidates (e.g. 100) and then filter by threshold
        # Since we want ALL photos of the user, k should be large or we iterate.
        # But 'k' in search means closest neighbors. Let's ask for e.g. 50 closest.
        
        distances, indices = self.index.search(query_embedding, k=50)
        
        results = []
        for i, dist in enumerate(distances[0]):
            if dist < threshold:
                idx = indices[0][i]
                if idx != -1:
                    match_data = self.metadata[idx]
                    results.append({
                        "path": match_data["path"],
                        "box": match_data["box"],
                        "distance": dist
                    })
        
        return results
