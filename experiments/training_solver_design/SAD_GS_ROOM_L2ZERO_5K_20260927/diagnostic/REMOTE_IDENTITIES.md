# Remote identities verified on EPIC-05

Checkpoint PLYs remain on the remote host; their SHA256 values were recalculated there. The table also records the exact GPU extension and source files loaded by the diagnostic.

| Remote path / identity | SHA256 or value |
|---|---|
| `/root/sadgs_room_author_3k_l2zero_20260927/model/point_cloud/iteration_3000/point_cloud.ply` | `ac55a30eb06011b40534a0df75e8288b2b7ff3dd48c1d54ef72d3f1fe5d5c3d6` |
| `/root/sadgs_room_author_5k_l2zero_20260927/model/point_cloud/iteration_5000/point_cloud.ply` | `3a088855de73bcd4e123d105812791ab75dd621bd19b0dc6cc89789b4c552894` |
| 3K and 5K `model/cameras.json` | `f01df344ca14773bb318fb92a2509bd46f6220fd7e021e97cc32e3e6691cb9a6` (identical) |
| SAD-GS author repository HEAD / status | `029814fb5ed773e4a794cec5115951a64f610ab8` / clean before and after |
| Tracked author source archive | `6f67c7928b20e7eecab2f7102c318fad999ac923e56b00ec6d335a7d9ec9efde` |
| `gaussian_renderer/__init__.py` | `e4605a4e6de29ae2a64017ad610444edb4d9cfeee5e44ea46b3be1a8d286ffcb` |
| `diff-gaussian-rasterization_structgs/cuda_rasterizer/forward.cu` | `046466cb6fea3bf6970674201cf7c56045925ae076f7ab9da09c80ab397b4542` |
| `diff-gaussian-rasterization_structgs/rasterize_points.cu` | `796d9273daca218f67a7d318d0c9f0fd83df87a61451434996feb41fb7e98813` |
| `diff_gaussian_rasterization_structgs/_C.cpython-311-x86_64-linux-gnu.so` | `2e6949550515397332369cc79791b85b3760bd8b7cf2b4b6b6888c991682b79b` |

The script and all four per-view JSONs were copied locally and their local SHA256 values matched the remote values in `SHA256SUMS.txt`. After both `mult` settings completed, GPU3 reported 0 MiB / 0% and no training or diagnostic process remained.
