import os
import shutil
import uuid

BASE_DIR = os.path.join(os.path.dirname(__file__), "../datasets")
DEST_DIR = os.path.join(BASE_DIR, "unified_dataset")

SOURCES_DIRS = [
    os.path.join(BASE_DIR, r"2-olter/office-wall-clock/v1"),
    os.path.join(BASE_DIR, r"clock-tvzts/clock-buxuu/v11"),
    os.path.join(BASE_DIR, r"muhammadrizkiproject/clock-r8zgr/v7"),
]

splits = ["train", "valid", "test"]
for split in splits:
    os.makedirs(os.path.join(DEST_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, split, "labels"), exist_ok=True)
    
print("Merging datasets...")

for src in SOURCES_DIRS:
    for split in splits:
        img_dir = os.path.join(src, split, "images")
        lbl_dir = os.path.join(src, split, "labels")
        
        if not os.path.exists(img_dir):
            continue # Skip if this split doesn't exist in the source
        
        for img_name in os.listdir(img_dir):
            base_name, ext = os.path.splitext(img_name)
            lbl_name = base_name + '.txt'
            
            # Ensure the label exists before copying the image
            if os.path.exists(os.path.join(lbl_dir, lbl_name)):
                # Generate a short random string to prevent duplicate names
                prefix = uuid.uuid4().hex[:6]
                new_base = f"{prefix}_{base_name}"
                
                # copy image
                shutil.copy(
                    os.path.join(img_dir, img_name),
                    os.path.join(DEST_DIR, split, "images", f"{new_base}{ext}")
                )
                
                # copy label
                shutil.copy(
                    os.path.join(lbl_dir, lbl_name),
                    os.path.join(DEST_DIR, split, "labels", f"{new_base}.txt")
                )

print(f"Done! Unified dataset created at: {DEST_DIR}")