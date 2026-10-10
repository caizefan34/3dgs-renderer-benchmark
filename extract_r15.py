import zipfile
import os

zip_path = r'C:\Users\36570\Downloads\HIGS_R15_CAPTURE_ORACLE_MX.zip'
extract_path = r'C:\Users\36570\3dgs-renderer-benchmark\HIGS_R15'

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
    print(f"Extracted to {extract_path}")
    print("Files:")
    for name in zip_ref.namelist():
        print(f"  {name}")