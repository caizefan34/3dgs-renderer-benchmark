# Commands executed

```powershell
rg -n -i "selective.?adam|visible.?adam|visibility.?mask|adam.*mask|mask.*adam" baseline src scripts experiments reports configs variants
rg -n "Adam|optimizer|step\\(|radii|visible|isect|meta" baseline/reference_v1/trainer.py baseline/reference_v1/gaussian_model.py
rg -n -i --glob '!reports/**' --glob '!experiments/**' --glob '!tmp_gsplat_src/**' "masked_adam_step|higs_masked_adam|masked-adam" .
```

No benchmark/training command was launched after the early stop.
