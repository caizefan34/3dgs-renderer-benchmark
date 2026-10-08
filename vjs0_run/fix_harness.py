import io
p="vjs0_harness.py"
s=open(p,encoding="utf-8").read()
old='''    jdump(summary, os.path.join(args.out, "VJS0_SUMMARY.json"))
    P2(f"\\n[summary] linearization_parity_pass={all_ok}")
    log.close()
    P2("HARNESS_DONE")'''
new='''    jdump(summary, os.path.join(args.out, "VJS0_SUMMARY.json"))
    P2(f"\\n[summary] linearization_parity_pass={all_ok}")
    P2("HARNESS_DONE")
    log.close()'''
assert old in s, "anchor not found"
s=s.replace(old,new)
open(p,"w",encoding="utf-8").write(s)
print("fixed trailing log bug")
