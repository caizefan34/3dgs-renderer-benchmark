#!/usr/bin/env python3
"""
H0-R5K IDENTITY GATE (frozen analyzer — analyzer owns verdict).

Compares a from-scratch regeneration of the frozen P12/C0 seed-42 trajectory
against the canonical n1snap suffix. The gate is FAIL-CLOSED:

  * The regenerated 10K/15K/20K/25K checkpoints for BOTH room and bicycle
    MUST be SHA256-identical to the canonical checkpoints.
  * If ANY of the 8 comparisons differs, emit H0_R5K_REGEN_IDENTITY_FAIL and
    STOP. HomotopyGS is CLOSED. No "numerical approximation" rescue.

The newly regenerated 5K checkpoints are recorded (this is the H0-R5K deliverable)
but are NOT part of the pass/fail gate (no canonical 5K ever existed).

Canonical hashes were pinned on 2026-09-30 from
/root/epic05_n1/runs/n1snap_c0f_s42_{scene}/ckpt_{iter}.pt and verified live.
"""
import hashlib
import json
import os
import sys

CANON_ROOT = "/root/epic05_n1/runs"
REGEN_ROOT = "/root/epic05_h0r5k/runs"
SCENES = ["room", "bicycle"]
SUFFIX_ITERS = [10000, 15000, 20000, 25000]
NEW_ITERS = [5000]

CANON = {
    "room": {
        10000: "9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4",
        15000: "b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd",
        20000: "c4e41cfef79e50980406fc1fa148af8d04bc8b37b801c36da5cddb2ce8058718",
        25000: "b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d",
    },
    "bicycle": {
        10000: "26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9",
        15000: "273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7",
        20000: "96f136cb331a19900a41a516d53314d522b4de5a3250ddbba8a232b610b398f0",
        25000: "73ae5436cc44a2c25dca26cb226e522047465e65b671717ac043578cbd7e21ea",
    },
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    out = {
        "artifact": "H0_R5K_IDENTITY.json",
        "gate": "H0_R5K_REGEN_IDENTITY",
        "authority": "H0-R5K directive A0 identity gate",
        "canon_root": CANON_ROOT,
        "regen_root": REGEN_ROOT,
        "comparisons": [],
        "new_5k": {},
        "all_match": True,
        "missing": [],
    }
    for scene in SCENES:
        for it in SUFFIX_ITERS:
            canon_p = os.path.join(CANON_ROOT, f"n1snap_c0f_s42_{scene}", f"ckpt_{it}.pt")
            regen_p = os.path.join(REGEN_ROOT, f"r5k_c0f_s42_{scene}", f"ckpt_{it}.pt")
            rec = {"scene": scene, "iter": it, "canon_path": canon_p, "regen_path": regen_p}
            if not os.path.isfile(regen_p):
                rec["status"] = "MISSING"
                out["missing"].append(regen_p)
                out["all_match"] = False
            else:
                rec["regen_sha256"] = sha256_file(regen_p)
                rec["canon_sha256"] = CANON[scene][it]
                rec["match"] = (rec["regen_sha256"] == CANON[scene][it])
                rec["status"] = "PASS" if rec["match"] else "MISMATCH"
                if not rec["match"]:
                    out["all_match"] = False
            out["comparisons"].append(rec)
        for it in NEW_ITERS:
            regen_p = os.path.join(REGEN_ROOT, f"r5k_c0f_s42_{scene}", f"ckpt_{it}.pt")
            if os.path.isfile(regen_p):
                out["new_5k"][f"{scene}@{it}"] = {
                    "path": regen_p,
                    "sha256": sha256_file(regen_p),
                }
            else:
                out["missing"].append(regen_p)

    if out["missing"]:
        out["verdict"] = "H0_R5K_REGEN_INCOMPLETE"
    elif out["all_match"]:
        out["verdict"] = "H0_R5K_IDENTITY_PASS"
    else:
        out["verdict"] = "H0_R5K_REGEN_IDENTITY_FAIL"

    print(json.dumps(out, indent=2))
    sys.exit(0 if out["verdict"] == "H0_R5K_IDENTITY_PASS" else 1)


if __name__ == "__main__":
    main();
