from search_engine import FaceSearchEngine
import os

def main():
    # Define paths
    # Assuming the script is run from project_root
    images_dir = "data/images"
    
    # Check if images exist
    if not os.path.exists(images_dir):
        print(f"Directory {images_dir} does not exist. Creating it...")
        os.makedirs(images_dir)
        print("Please place your event photos in 'data/images' and run this script again.")
        return

    # Check if there are images
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print("No images found in data/images. Please add some photos.")
        return

    engine = FaceSearchEngine()
    engine.create_index(images_dir)
    print("Indexing complete.")

if __name__ == "__main__":
    main()
