import urllib.request
from pathlib import Path

train_img_dir = Path("/home/zerotwo/Downloads/yolo-training-dpc/dataset/images/train")
train_lbl_dir = Path("/home/zerotwo/Downloads/yolo-training-dpc/dataset/labels/train")

for i in range(1, 11):
    gender = "men" if i % 2 == 0 else "women"
    url = f"https://randomuser.me/api/portraits/{gender}/{i}.jpg"
    
    img_path = train_img_dir / f"stranger_{i}.jpg"
    lbl_path = train_lbl_dir / f"stranger_{i}.txt"
    
    try:
        urllib.request.urlretrieve(url, img_path)
        # Create an empty label file (negative sample)
        lbl_path.touch(exist_ok=True)
        print(f"Downloaded {img_path.name} as unknown person negative sample.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

