# Dataset and canonical asset report

All raw sources were downloaded from the URLs pinned in `benchmark/datasets/*.json` and verified before extraction.

## Source archives

| Dataset / scene | Size bytes | Verification |
| --- | ---: | --- |
| Mip-NeRF 360 Garden | 2,986,730,894 | MD5 `4gbh6q8t3OPK6m7XrEVjSw==`; CRC32C `FA9s7A==`; local SHA-256 `9ecd0d11bcc90887cc091d396f70f1faf805e256703fc24af25f88248efc82bd` |
| Mip-NeRF 360 Bicycle | 2,476,191,765 | MD5 `skoTqxFkUaC/ihEEDNNm5w==`; CRC32C `RGY8lw==`; local SHA-256 `19fa7be077881cb614a87162aa49873b0e3a675bd89369d3598df1308a90f5c0` |
| Mip-NeRF 360 Bonsai | 1,395,703,750 | MD5 `FywcDueuwWhOSGjsBgROnQ==`; CRC32C `hUKdOA==`; local SHA-256 `90ca7ed301c9e7dfc2dc6bd043e100c244995e9d83e6ea20e445d1e8ef46eab0` |
| Tanks and Temples Truck / Train | 682,628,995 | SHA-256 `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1` |

Dataset terms remain governed by the source sites linked from the repository manifests; the repository MIT license does not replace those terms.

## Canonical cases

Every case has `status: canonical` and 100 GT files. The assets were prepared at
`b46e8f27fbc3beea89a12f25c35ce8b296f24cd9` and revalidated byte-for-byte by
the Tier A runner at evidence commit
`dc9bb4e9231ae2fdf90fa9c40bcd6e0dbd7d104f`.

| Case | Checkpoint SHA-256 | Camera SHA-256 | GT manifest SHA-256 | Result |
| --- | --- | --- | --- | --- |
| small-garden-1080p | `16701d5e0630dfaca74f8794ed7ce2aa23fa922f87dc09a7e37484e8d3f82d5a` | `8027a05807f073a965c35abbfb8859b8f59903878dba8047e832772c19c08559` | `05264145bca1014ffa36bb762ee91a8d3009052b906e3f72841292483ab86623` | match |
| medium-truck-1080p | `65ecf4058135a030cddd2198326f67172a4101344b0b54a3fa370cf45ea9688c` | `25ac0a8456b0dc9aa9cff46794efcb5d4c3f80375f3a90f54d805f303e82a950` | `4e0b6b7c8789f4881bef0ab51f15e54368c00390a26c59671ded30db4064833c` | match |
| medium-train-1080p | `70e055d46f53636c83547a59c4998a0a27f64e42fceba45c6b78dfc77f52958d` | `ed4cdf5c9c08d5afa3fce647a02e40c15b0f83d8918df151d4944e69643506be` | `8ea146c9ae7649c4f3666df741e3d55e5c2b181537aea4d06e5624672a325c4c` | match |
| large-bicycle-1080p | `64d357cb25bd85f710f8551a18d830f8497277fbd8c5805adfd72ffe9ca78227` | `58290b6f5d84f69659c53c76e9a9c005dfd94af83de93cd18800abbe14b10732` | `062de75381f5b0f0dafcca1ac884c68bd31366f4d9fe3c37797511f93a4deb7d` | match |
| large-bonsai-1080p | `a16af6d8815498ffbf9eb5d5ee93f5bcc9dca34c4e3eb6f7a796ef9e97c0d273` | `3d9fad612031f6a5b26d9c046a6d11f9bf9bce55b24350ea77385dad889104e5` | `abce2134d2beea04150ee2c57b7d7b2ef34ecfcfcfcb2598439d1aa9b9970e5e` | match |

Raw inventories and preparation manifests are under `../datasets/raw/` and `../datasets/processed/`.
