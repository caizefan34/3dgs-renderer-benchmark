# FINAL STATUS / DELIVERABLES CHECKLIST
# Generated 2026-09-19 10:45 CST+8 by experiment runtime agent
# Purpose: close out R4/R5 work with a verified deliverable index.

## 1. Verdict (final)
Candidate C (real-threshold skip-mask) is REJECTED as a general-purpose optimization:
- mean ΔPSNR −1.44 dB (10/13 losses, 3/13 wins within noise)
- mean ΔSSIM −0.0531 (12/13 negative)
- speed 0.26×–0.63× (all slower than baseline)

## 2. Deliverables index
| Path | Contents | Status |
|------|----------|--------|
| /mnt/r4_13scene_v2/final_results.json | R4 13-scene aggregated metrics | ✅ verified |
| /mnt/r4_13scene_v2/r4_summary_stats.json | R4 summary + speedup stats | ✅ verified |
| /mnt/r4_13scene_v2/grand_decision.md | Final decision log (R4+R5A) | ✅ verified |
| /mnt/r5_a/r5-a-results.json | R5-A multi-seed paired results | ✅ verified |
| /mnt/r5_a/figures/*.png | R5-A plots (NGS vs iter, ΔPSNR) | ✅ present |
| reports/r4/r4-13scan-final.md | Local R4 report (13-scene tables) | ✅ synced |
| reports/r4/r4-aggregate-final.txt | Local aggregate checks | ✅ synced |
| reports/r4/VERDICT-REPORT.md | Final verdict + recommendations | ✅ synced |

## 3. Open questions / next steps (not blocking)
- [ ] Decide whether to allocate a WarmEmit full 26-run matrix (needs fresh disk budget)
- [ ] Disk cleanup: /mnt has ~20G free; old runs' intermediate content is the largest consumer
