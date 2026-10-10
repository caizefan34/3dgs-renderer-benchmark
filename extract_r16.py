import zipfile
import os

zip_path = r'C:\Users\36570\Downloads\HIGS_R16_OPACITY_SEMANTICS_MX.zip'
extract_path = r'C:\Users\36570\3dgs-renderer-benchmark\HIGS_R16'

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
    print("Extracted files:")
    for name in zip_ref.namelist():
        print(f"  {name}")
