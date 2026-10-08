import sys
data = open(sys.argv[1], "rb").read()
lines = data.split(b"\n")
for i in range(223, 229):
    print(i + 1, lines[i])
