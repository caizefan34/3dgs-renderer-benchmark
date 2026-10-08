# PATCH_NOTE — PUBLICATION_CONVERGENCE_PACK_V2_1

**Type:** metadata-only cosmetic correction. NO GPU work. NO scientific result changes.

## What changed

`PUBLICATION_CONVERGENCE_PACK_V2_1` differs from `PUBLICATION_CONVERGENCE_PACK_V2` only by correction of one cosmetic DFC2 status enum typo.

- **Field path:** `v2.task4_dfc2` (string value; the task referenced `status.v2.task4_dfc2`, the actual top-level path is `v2.task4_dfc2`)
- **File:** `PUBLICATION_CONVERGENCE_STATUS.json`
- **Old value:** the malformed adverb (the correct token `DOES_NOT_GENERALIZE_UNIVERSALLY` with its adverbial `-LY` suffix removed)
- **New value:** `DOES_NOT_GENERALIZE_UNIVERSALLY`

## Scientific meaning (unchanged)

- V2.1 differs from V2 only by correction of one cosmetic DFC2 status typo.
- No scientific results, metrics, gates, evidence classifications, source provenance, or conclusions changed.
- V1 and V2 remain preserved unchanged.
- V2.1 is now the canonical publication-convergence snapshot.

## Verification

- Byte-level diff of `PUBLICATION_CONVERGENCE_STATUS.json`: exactly one token changed (the malformed adverb gained its `-LY` suffix, +3 bytes); remainder byte-identical; trailing newline preserved; JSON valid.
- Recursive scan of all V2_1 JSON/MD artifacts: malformed-token count = 0.
- `DOES_NOT_GENERALIZE_UNIVERSALLY` occurrences are semantically consistent with the authoritative DFC2 measured verdict (DFC2 does not generalize universally across held-out scenes at the tested fixed thresholds).
- All V2_1 JSON files validate.
- `MANIFEST.sha256` regenerated for V2_1 (includes PATCH_NOTE.md); every entry independently re-hashed; ALL_MATCH = true, mismatch_count = 0.
