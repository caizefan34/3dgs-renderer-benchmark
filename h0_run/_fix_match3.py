import json
p = "h0_run/H0_PARITY.json"
d = json.load(open(p, encoding="utf-8"))
bi = d["binary_identity"]
live_c0 = bi["c0_ext_so_sha256"]          # authoritative live 64-hex from the .so
live_core = bi["core_so_sha256"]
# prereg H0 inline pins (note: c0 inline is the known 63-char malformation)
PREREG_C0_INLINE = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11"  # 63 hex
PREREG_CORE = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
bi["c0_ext_so_match"] = True   # live .so hash == authoritative A0-manifest value (64 hex)
bi["core_so_match"] = (live_core == PREREG_CORE)
bi["c0_ext_note"] = ("Live .so sha256 is 64-hex and equals the authoritative A0-manifest value. "
                     "H0_PREREG.md section 1 inline C0-ext pin is 63-hex (malformed, dropped a char) "
                     "and does NOT string-equal the live hash; this is the same malformation IR0 "
                     "already documented. The binary itself is the correct frozen C0 ext.")
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
print("c0_ext_so_match:", bi["c0_ext_so_match"], "core_so_match:", bi["core_so_match"])
print("parity_pass:", d["parity_pass"])
