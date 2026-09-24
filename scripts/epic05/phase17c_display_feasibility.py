"""Display C17-2 differential feasibility results."""
import json

with open('results/phase-a100/c17_2_differential_feasibility.json', 'r') as f:
    d = json.load(f)

for name in ["room", "bicycle"]:
    s = d["scenes"][name]
    print("=" * 70)
    print("  %s: %d tiles (%dx%d)" % (name, s["ntiles"], s["tw"], s["th"]))
    print("=" * 70)
    
    ts = s["tile_stats"]
    print("  Tile sizes: p50=%.0f p75=%.0f p90=%.0f p95=%.0f p99=%.0f max=%.0f" % (
        ts["size"]["p50"], ts["size"]["p75"], ts["size"]["p90"],
        ts["size"]["p95"], ts["size"]["p99"], ts["size"]["max"]))
    print("  Empty tiles: %d / %d" % (ts["empty_tiles"], ts["nonempty_tiles"] + ts["empty_tiles"]))
    
    for label in ["h", "v", "ddr", "ddl"]:
        key = "dir_%s" % label
        if key not in s:
            continue
        di = s[key]
        print("\n  Direction %s (%d pairs):" % (label, di["pairs"]))
        print("    len_B: p50=%.0f p75=%.0f p95=%.0f" % (
            di["len_B"]["p50"], di["len_B"]["p75"], di["len_B"]["p95"]))
        print("    len_S (shared): p50=%.0f p75=%.0f p95=%.0f" % (
            di["len_S"]["p50"], di["len_S"]["p75"], di["len_S"]["p95"]))
        print("    len_I (insert): p50=%.0f p75=%.0f p95=%.0f" % (
            di["len_I"]["p50"], di["len_I"]["p75"], di["len_I"]["p95"]))
        print("    rho_I (I/B): p50=%.4f p75=%.4f p90=%.4f p95=%.4f p99=%.4f" % (
            di["rho_I"]["p50"], di["rho_I"]["p75"], di["rho_I"]["p90"],
            di["rho_I"]["p95"], di["rho_I"]["p99"]))
        
        # Model A: reference already known → Cost = |I| + |R| (both same size as insert on avg)
        # Model B: store reference too
        rho = di["rho_I"]
        print("    Model A savings (|I|/|B|, ref free): p50=%.1f%% saved, p95=%.1f%% saved" % (
            (1 - rho["p50"]) * 100, (1 - rho["p95"]) * 100))
        # Model B: store reference + insert - roughly |A| + |I| vs |A| + |B|
        # Cost ratio = (|A|+|I|) / (|A|+|B|) ≈ actually this is storage analysis
        # For just B given A: Cost_delta = |I| + |R| ≈ 2*|I| (since |I|≈|R|)
        print("    Model B (|A|+|I| / |B|): ref needed separately")
        
        if "order_preserving_subseq" in di:
            ops = di["order_preserving_subseq"]
            print("    Order-preserving subseq: %d / %d = %.4f" % (
                ops["count"], ops["total"], ops["ratio"]))
        
        if "insert_blocks" in di:
            ib = di["insert_blocks"]
            print("    Insert blocks: p50=%.0f p75=%.0f p90=%.0f p95=%.0f mean=%.1f" % (
                ib.get("p50", 0), ib.get("p75", 0), ib.get("p90", 0), ib.get("p95", 0), ib.get("mean", 0)))
        
        if "insert_pos_norm" in di:
            ipn = di["insert_pos_norm"]
            print("    Insert pos(norm): p50=%.4f p75=%.4f p90=%.4f p95=%.4f" % (
                ipn.get("p50", 0), ipn.get("p75", 0), ipn.get("p90", 0), ipn.get("p95", 0)))
    
    # Tile path
    tp = s.get("tile_path_consecutive", {})
    if tp:
        ic = tp.get("insert_count", {})
        print("\n  Row-major path (%d pairs):" % tp["pairs"])
        print("    Insert/pair: p50=%.0f p75=%.0f p95=%.0f p99=%.0f" % (
            ic.get("p50", 0), ic.get("p75", 0), ic.get("p95", 0), ic.get("p99", 0)))
    
    # Worst case
    wc = s.get("worst_case", {})
    if wc:
        print("\n  Worst-case (over %d pairs):" % wc["scan_pairs"])
        print("    max |I|=%d max |R|=%d max delta=%d" % (
            wc["max_len_I"], wc["max_len_R"], wc["max_delta_entries"]))
    print()
