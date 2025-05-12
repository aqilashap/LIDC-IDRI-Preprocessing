## LIDC-IDRI-Preprocessing
Download LIDC-IDRI dengan classical directory
Pastikan pylidc.conf ada dan berisi path ke dataset LIDC
Pastikan path di setiap step benar dan sesuai dengan keinginan
Step 1: convert dicom ke nifti
Step 2: export label ke annotation.csv
Step 3: split malignancy tapi outputnya csv, memperbarui dari annotation
Step 4: crop image ke 256x256x32 biar cocok untuk dimasukin ke model
Step 5: misahin jadi folder seperti Task06
