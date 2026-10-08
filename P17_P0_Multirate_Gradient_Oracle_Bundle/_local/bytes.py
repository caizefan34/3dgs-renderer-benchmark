import sys
data = open(sys.argv[1], "rb").read()
for i, line in enumerate(data.split(b"\n")[222:230], start=223):
    print(i, repr(line))
# find any weird bytes
weird = [(i, c) for i, c in enumerate(data) if c < 9 or (13 < c < 32)]
print("weird byte count:", len(weird))
print(weird[:10])
