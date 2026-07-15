# Benchmark Suite

The current primary suite is `3dgs-renderer-matrix` version `3.1.0`, defined in [`benchmark/suite.json`](../benchmark/suite.json).

It fixes five mandatory 1080p cases: Garden, Truck, Train, Bicycle, and Bonsai.
Overall rankings require all five.
The authoritative protocol hash is stored beside the suite and must match the bytes of `benchmark/protocol.json`.

`benchmark_suite/` is a compatibility definition for legacy v1/v2 commands and committed historical artifacts.
It must not be used to label new Matrix v3 results.
