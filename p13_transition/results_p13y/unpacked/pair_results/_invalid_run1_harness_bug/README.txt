Run 1 of the room oracle aborted every pair before any Y2/Y3 number was
computed: p13y_oracle.y1_snapshots passed string values into the verbatim
P13-X probe_core.snapshot_from_meta, which formats iteration/anchor_id with
%d. Error: ValueError("Unknown format code 'd' for object of type 'str'")
raised at p13y_probe_core.py:74. Harness bug, not a gate or recipe change;
no threshold or criterion was touched. Files kept verbatim.
