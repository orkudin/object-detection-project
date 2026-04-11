import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def download_video():
    """Download a sample video representing traffic for basic pipeline testing."""
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    output_path = os.path.join(os.path.dirname(__file__), "test_runway.mp4")
    
    if os.path.exists(output_path):
        logging.info(f"Video {output_path} already exists. Skipping download.")
        return
        
    logging.info(f"Downloading sample video from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        logging.info(f"Successfully downloaded to {output_path}")
    except Exception as e:
        logging.error(f"Failed to download video: {e}")

if __name__ == "__main__":
    download_video()
