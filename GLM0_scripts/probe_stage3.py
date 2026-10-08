import re
src=open("/root/epic05_d0/scripts/stage3_accounting.py").read()
print("=== first 35 lines ===")
print("\n".join(src.splitlines()[:35]))
print("=== arg/main/scene markers ===")
for i,l in enumerate(src.splitlines(),1):
    if re.search(r"add_argument|def main|__main__|np\.load|read_parquet|D0_BIRTH_TABLE|D0_LINEAGE|D0_POPULATION|sys.argv|scene\s*=|outdir|ARTIFACT", l):
        print(i, l)
