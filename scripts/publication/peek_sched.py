import re
src = open('/mnt/storage_pool/liaoyuanjun/publication_scheduler.py').read()
lines = src.split('\n')
for a, b in [(150, 200), (210, 275)]:
    print(f"===== lines {a}-{b} =====")
    for i in range(a - 1, min(b, len(lines))):
        print(f"{i+1:4d} {lines[i]}")
