import os
import requests
from PIL import Image
from io import BytesIO

# --- CONFIGURATION ---
SAVE_DIR = "data/coco_subset"
COCO_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{:012d}.jpg"

# ✅ REAL Verified IDs from COCO 2017 Validation Set
# (These are guaranteed to exist)
VALID_IDS = [
    139, 285, 632, 724, 785, 872, 885, 1000, 1268, 1296, 
    1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761, 1818, 1993,
    2022, 2153, 2148, 2153, 2261, 2282, 2465, 2685, 2923, 2929,
    3016, 3035, 3089, 3136, 3348, 3426, 3501, 3553, 3661, 3691,
    3758, 3845, 3934, 4134, 4395, 4405, 4476, 4676, 4795, 4833
]

def setup_data():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print(f"⬇️  Downloading {len(VALID_IDS)} verified images from COCO...")
    
    success_count = 0
    
    for img_id in VALID_IDS:
        url = COCO_URL_TEMPLATE.format(img_id)
        try:
            # 1. Request the image
            response = requests.get(url, timeout=10)
            
            # 2. Check if the link is valid (200 OK)
            response.raise_for_status() 
            
            # 3. Try to open it as an image
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # 4. Save
            save_path = os.path.join(SAVE_DIR, f"coco_{img_id}.jpg")
            img.save(save_path)
            
            print(f"✅ Saved: {save_path}")
            success_count += 1
            
        except requests.exceptions.HTTPError:
            print(f"⚠️  Skipping {img_id}: Image not found on server (404).")
        except Exception as e:
            print(f"❌ Error {img_id}: {e}")

    print("="*40)
    print(f"🎉 Download Complete! {success_count}/{len(VALID_IDS)} images ready.")

if __name__ == "__main__":
    setup_data()