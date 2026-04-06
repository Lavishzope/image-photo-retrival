import streamlit as st
import cv2
import numpy as np
from PIL import Image
from search_engine import FaceSearchEngine
import os

# Page Config
st.set_page_config(page_title="Intelligent Event Photo Retrieval", layout="wide")

# Title
st.title("📸 Intelligent Event Photo Retrieval")
st.markdown("Upload a clear selfie to find all your photos from the event!")

# Initialize Search Engine
@st.cache_resource
def load_engine():
    engine = FaceSearchEngine()
    loaded = engine.load_index()
    return engine, loaded

engine, is_index_loaded = load_engine()

# Sidebar
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05, help="Lower = Stricter match, Higher = Looser match")

if not is_index_loaded:
    st.error("⚠️ Index not found! Please run `python index_images.py` first to process the event photos.")
else:
    st.sidebar.success(f"Index loaded! {engine.index.ntotal} faces indexed.")

# Main Interface
uploaded_file = st.file_uploader("Upload your Reference Selfie", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display query image
    params_col1, params_col2 = st.columns([1, 2])
    
    with params_col1:
        image = Image.open(uploaded_file)
        # Convert to RGB (standard for face_recognition)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        st.image(image, caption="Your Reference Selfie", use_column_width=True)
        
        # Convert to numpy for backend
        query_image_np = np.array(image)

    if st.button("Find My Photos"):
        with st.spinner("Scanning the gallery..."):
            try:
                results = engine.search(query_image_np, threshold=threshold)
                
                if isinstance(results, str):
                    st.warning(results)
                elif len(results) == 0:
                    st.info("No matching photos found with this threshold. Try increasing it slightly.")
                else:
                    st.success(f"Found {len(results)} matches!")
                    
                    # Display results in a grid
                    cols = st.columns(3)
                    
                    # Group duplicates (if multiple faces in same image match?? unlikely with Top-K unless same person twice)
                    # For UI clarity, let's just go through each result.
                    
                    # Note: results contain {path, box, distance}
                    # We might want to deduplicate images if the user appears twice in same photo (or validly show both)
                    
                    for i, res in enumerate(results):
                        img_path = res['path']
                        box = res['box'] # top, right, bottom, left
                        dist = res['distance']
                        
                        # Load original image
                        if os.path.exists(img_path):
                            # We draw the box
                            match_img = cv2.imread(img_path)
                            match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                            
                            # Draw rectangle
                            top, right, bottom, left = box
                            cv2.rectangle(match_img, (left, top), (right, bottom), (0, 255, 0), 4)
                            
                            with cols[i % 3]:
                                st.image(match_img, caption=f"Match (Dist: {dist:.2f})", use_column_width=True)
                        else:
                            st.write(f"Image not found: {img_path}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("⚠️ **Note**: This system runs locally. Ensure your photos are in `data/images` and you have indexed them.")
