data = open("/root/epic05_cvt0/CVT0_RESULTS.json","rb").read()
print("total bytes:", len(data))
print("around char 1247:")
print(repr(data[1200:1320]))
