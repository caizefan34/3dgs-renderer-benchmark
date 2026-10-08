# Loss parity audit (2026-09-27)

SAD-GS source `train.py:256-259`: `Ll1=l1_loss(image,gt)`, `Ll2=l2_loss(image,gt)`, `ssim_value=fused_ssim(image[None],gt[None])`, `rgb_loss=(1-lambda_dssim)*Ll1 + lambda_dssim*(1-ssim_value) + lambda_l2*Ll2`. `arguments/__init__.py` defaults `lambda_dssim=0.2`, `lambda_l2=2.0`; this run overrides only `lambda_l2=0.0`.

FastGS source `public_baselines/fastgs/train.py:101-103` computes the first two terms without the extra L2. Both `utils/loss_utils.py` define L1 as `torch.abs((network_output-gt)).mean()`.

| Source | SHA256 |
|---|---|
| Workspace FastGS train.py | `00a3d773b97379d55fbd87388faefbf1fdb15d3903e3b288bc4965723377d5e5` |
| Workspace FastGS utils/loss_utils.py | `ed4b3d686a77958103b92a4cedc20811e03198e4cb532bd61b96e64f9e70247e` |
| FastGS baseline protocol matrix | `64c0b38060541292ffddcfff6b8d7b09eb699a32ea63504b0635d28390e1d747` |
| SAD-GS fused SSIM setup.py | `9eea4316eb2af40e179ae6b8514f76383b7255251b83baac5a82b097c930a38e` |
| SAD-GS fused SSIM `ssim.cu` | `0676eaa84ee01ce749c119428ae88512960a315e8dac9db32360c0c67e686d6d` |
| SAD-GS fused SSIM `ext.cpp` | `1b8c96adbf24e93875d142e9dd7dc009f83b11bc4746f4fe60bf2d1eec6b5487` |
| SAD-GS fused SSIM Python wrapper | `d22009b032254779ada28ff5bb57f664ee5ebda82da453ef60b7c18f7ad14dd9` |

Compared the FastGS and SAD-GS fused-SSIM `setup.py`, `ssim.cu`, `ext.cpp`, and Python wrapper after normalizing CRLF/LF; each pair's full text matched exactly. Both call the same `fused_ssim` interface and use its default same-padding mean SSIM map. Scalar objective after `lambda_l2=0` is therefore the same 0.8 L1 + 0.2 D-SSIM expression. No claim is made that the rest of the training/rendering systems match.
