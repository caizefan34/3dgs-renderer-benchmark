data = open(r"C:\Users\36570\3dgs-renderer-benchmark\P17_P0_Multirate_Gradient_Oracle_Bundle\_local\p17_p0_oracle.py", "rb").read()
lines = data.split(b"\n")
print("total lines:", len(lines))
for i in range(223, 229):
    print(i + 1, repr(lines[i]))
