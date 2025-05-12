from pathlib import Path
import shutil

# Root folder
input_root = Path("D:/Skripsi/newprepo/preprocessed_crop/data") # Path data hasil cropped
output_root = Path("D:/Skripsi/newprepo/LIDC-format/data/train") # Output path setelah dibenerin foldernya

# Output folders
imageTr = output_root / "imageTr" # Buat folder imageTr
labelsTr = output_root / "labelsTr" # Buat folder labelsTr
imageTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)

# Global counter untuk memberi ID unik
global_idx = 0

# Iterasi semua file img_*.nii.gz di seluruh subfolder
for img_path in input_root.rglob("img_*.nii.gz"):
    parent = img_path.parent

    # Cari segmen utama
    seg_path = next(parent.glob("seg_*.nii.gz"), None)
    if seg_path is None:
        print(f"[SKIP] No seg found in: {parent}")
        continue

    # Buat nama file seperti: LIDC_0000_0000.nii.gz
    image_name = f"LIDC_{global_idx:04d}_0000.nii.gz"
    label_name = f"LIDC_{global_idx:04d}.nii.gz"

    # Salin file ke target
    shutil.copy(img_path, imageTr / image_name)
    shutil.copy(seg_path, labelsTr / label_name)

    print(f"[OK] {image_name} saved.")
    global_idx += 1
