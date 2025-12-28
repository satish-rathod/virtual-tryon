import requests
import json
import time

API_BASE = "http://localhost:8000"
IMAGE_PATH = "backend/test_output/101A6686_generated_20251228_022033.png"

def run():
    print(f"Uploading {IMAGE_PATH}...")
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": ("test_image.png", f, "image/png")}
        resp = requests.post(f"{API_BASE}/api/upload", files=files)
        
    if resp.status_code != 200:
        print(f"Upload failed: {resp.text}")
        return
        
    data = resp.json()
    saree_id = data["saree_id"]
    print(f"Uploaded! Saree ID: {saree_id}")
    
    print("Triggering generation...")
    payload = {"saree_id": saree_id, "mode": "standard"}
    resp = requests.post(f"{API_BASE}/api/generate", json=payload)
    
    if resp.status_code != 200:
        print(f"Generate failed: {resp.text}")
        return
        
    job_data = resp.json()
    job_id = job_data["job_id"]
    print(f"Job triggered! Job ID: {job_id}")
    
    print("Monitoring status...")
    while True:
        resp = requests.get(f"{API_BASE}/api/status/{job_id}")
        status_data = resp.json()
        status = status_data["status"]
        progress = status_data.get("progress", 0)
        stage = status_data.get("current_stage", "unknown")
        
        print(f"Status: {status} | Stage: {stage} | Progress: {progress}%")
        
        if status in ["success", "failed"]:
            break
            
        time.sleep(2)
        
    print(f"Final Status: {status}")
    if status == "success":
        print("Artifacts generated:")
        print(json.dumps(status_data.get("artifacts", []), indent=2))

if __name__ == "__main__":
    run()
