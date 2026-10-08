"""Prove StreamingDebtReplayer == debt_replay.replay_debt (bitwise) on
randomized schedules including clone/split/prune topology."""

import random
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lavos0 import debt_replay
from lavos0.streaming_debt import StreamingDebtReplayer


def _random_schedule(rng, n_initial, n_iters):
    consumers = {}
    clone_p, split_p, prune_p = {}, {}, {}
    alive = n_initial
    next_id = n_initial
    for t in range(1, n_iters + 1):
        # consumer set: random subset of alive rows
        n_cons = rng.randint(0, max(1, alive // 2))
        rows = rng.sample(range(alive), min(n_cons, alive))
        consumers[t] = sorted(rows)
        # occasional topology
        if t % 7 == 0 and alive > 4:
            n_ev = rng.randint(1, 3)
            parents = rng.sample(range(alive), min(n_ev, alive))
            clone_p[t] = parents[:]
            alive += len(parents)            # clone: +1 child each
            next_id += len(parents)
        if t % 11 == 0 and alive > 4:
            n_ev = rng.randint(1, 2)
            parents = rng.sample(range(alive), min(n_ev, alive))
            split_p[t] = parents[:]
            alive += len(parents)            # split: +2 children, -1 parent => +1 net
            next_id += 2 * len(parents)
        if t % 13 == 0 and alive > 4:
            n_ev = rng.randint(1, 3)
            victims = rng.sample(range(alive), min(n_ev, alive))
            prune_p[t] = victims[:]
            alive -= len(victims)
    return consumers, clone_p, split_p, prune_p, n_initial


def _stats_dict(s):
    return s.as_dict()


def main():
    rng = random.Random(1234)
    n_fail = 0
    for case in range(40):
        n_initial = rng.randint(2, 12)
        n_iters = rng.randint(5, 40)
        consumers, clone_p, split_p, prune_p, n_init = _random_schedule(
            rng, n_initial, n_iters)
        # frozen offline replay
        ref = debt_replay.replay_debt(
            consumers_by_iter=consumers, n_iters=n_iters, n_initial_rows=n_init,
            clone_parents=clone_p, split_parents=split_p, prune_ids=prune_p)
        # streaming replay
        st = StreamingDebtReplayer(n_init)
        for t in range(1, n_iters + 1):
            st.replay_topology(clone_p.get(t, []), split_p.get(t, []),
                               prune_p.get(t, []))
            st.replay_step(consumers.get(t, []))
        got = st.finalize()
        rd = _stats_dict(ref)
        gd = _stats_dict(got)
        if rd != gd:
            n_fail += 1
            print(f"CASE {case} MISMATCH n_init={n_init} iters={n_iters}")
            for k in rd:
                if rd[k] != gd[k]:
                    print(f"   {k}: ref={rd[k]} got={gd[k]}")
    if n_fail:
        print(f"FAILED: {n_fail} mismatches")
        sys.exit(1)
    print("STREAMING_DEBT_PARITY_OK (40 randomized cases with topology)")


if __name__ == "__main__":
    main()
