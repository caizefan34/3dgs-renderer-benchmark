s=open("build_report.py",encoding="utf-8").read()
s=s.replace('任何"student 处处复现 teacher"的说法','任何「student 处处复现 teacher」的说法')
open("build_report.py","w",encoding="utf-8").write(s)
print("fixed quotes")
