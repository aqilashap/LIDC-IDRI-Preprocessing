import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # atau 'QtAgg' kalau pakai PyQt


test_load = nib.load('D:\\Skripsi\\newprepo\\LIDC-format\\data\\train\\imageTr\\LIDC_0019_0000.nii.gz').get_fdata()
print(test_load.shape)

# Tampilkan slice tengah dari sumbu axial (z)
slice_idx = test_load.shape[2] // 2  # ambil slice ke-16 kalau bentuknya (256, 256, 32)
plt.imshow(test_load[:, :, slice_idx], cmap='gray')
plt.title(f"Slice ke-{slice_idx}")
plt.axis('off')
plt.show()