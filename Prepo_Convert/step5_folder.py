from pathlib import Path
import SimpleITK as sitk
import numpy as np
import shutil

# Root folder
input_root = Path("D:/Skripsi/newprepo/preprocessed_crop/data")
output_root = Path("D:/Skripsi/newprepo/LIDC-new/data/train")

# Output folders
imageTr = output_root / "imageTr"
labelsTr = output_root / "labelsTr"
imageTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)

# Iterate all img_*.nii.gz files
for img_path in input_root.rglob("img_*.nii.gz"):
    parent = img_path.parent

    # Ambil patient id dari folder induknya (LIDC-IDRI-0001/...)
    for part in img_path.parts:
        if "LIDC-IDRI-" in part:
            patient_id_raw = part.replace("LIDC-IDRI-", "")
            break
    else:
        print(f"[SKIP] Patient ID not found in {img_path}")
        continue

    patient_id = f"{int(patient_id_raw):04d}"

    # Nomor image (dari img_0.nii.gz misalnya)
    img_idx = img_path.stem.split('.')[0].split("_")[1]
    image_name = f"LIDC_{patient_id}_{int(img_idx):04d}.nii.gz"
    label_name = f"LIDC_{patient_id}_{int(img_idx):04d}.nii.gz"

    # Copy image ke imageTr
    shutil.copy(img_path, imageTr / image_name)

    # Cari semua seg_0.nii.gz, seg_0_1.nii.gz, dst
    base_seg = parent / f"seg_{img_idx}.nii.gz"
    additional_segs = list(parent.glob(f"seg_{img_idx}_*.nii.gz"))

    if not base_seg.exists():
        print(f"[SKIP] Tidak ada seg untuk {img_path}")
        continue

    # Load segmentasi & rata-rata
    seg_arrays = [sitk.GetArrayFromImage(sitk.ReadImage(base_seg))]
    for seg_path in additional_segs:
        seg_arrays.append(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)))

    avg_seg = np.mean(seg_arrays, axis=0)
    avg_seg = (avg_seg > 0.5).astype(np.uint8)  # Threshold ke binary mask

    # Simpan ke labelsTr
    final_seg_img = sitk.GetImageFromArray(avg_seg)
    final_seg_img.CopyInformation(sitk.ReadImage(base_seg))  # inherit spacing, origin
    sitk.WriteImage(final_seg_img, labelsTr / label_name)

    print(f"[OK] {image_name} dan mask saved.")
