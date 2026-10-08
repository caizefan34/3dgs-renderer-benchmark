# First 32-step handoff development probe

Lifetime 32 was selected after seeing the room sibling-coordinate development
curves. This is an adaptive research follow-up, not independent confirmation.
No unseen-view or full-training claim is supported.

- Same frozen checkpoint, fixed 256 split parents, three initial split draws,
  original loss, camera sequence, A/B protocol and 256 continuation steps.
- Each draw compares native, permanent pair_full, pair_transient32, physical
  inheritance .5/.25, physical inheritance .2/.04, and native_repeat. Two
  no-edit controls bring the count to 20 per scene.
- Transient uses full parent shared moments and fresh relative bias clock for
  32 updates; cross raw second moments start at zero and are updated before
  each Adam step. Then decode parameters and map raw moments using the exact
  zero-extended prehistory convention. Resume physical Adam at the old group
  clock. No moment reset or extra search.
- At handoff require every physical raw parameter and the first A image to be
  exactly unchanged, with group clocks unchanged. CPU tests independently check
  the full moment mapping. Cross state is released after handoff.
- Timings include cross accumulation, handoff and its diagnostic render checks;
  so they are conservative prototype timings, not deployment timings.
- The first execution retains horizons 0,1,8,32,128,256 from the matched probes.
  It does not record steps 33/40/64 proposed for detailed post-switch diagnostics.
  Those remain required if this screen shows an opportunity worth extending.
- Same numerical parity gate as the sibling audit; independent native replay
  envelope is available only for garden and hash-checked. Room must pass the
  fixed numerical gate. Initial failure evidence is retained in earlier files.
- Initial room launch on GPU 7 was refused by the clean-GPU check due to a
  foreign process, before CUDA initialization or training. Do not count it as
  a performance run; retry only on a verified clean device.
