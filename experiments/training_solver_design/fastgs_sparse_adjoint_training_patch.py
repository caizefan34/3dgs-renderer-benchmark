"""Generate a standalone FastGS trainer with a stratified image-gradient mask."""

import argparse
from pathlib import Path


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one source occurrence, found {count}: {old[:80]!r}")
    return source.replace(old, new)


MASK_HELPER = '''def sparse_adjoint_mask(height, width, generator, device):
    tile_class = torch.randint(
        0, 4, ((height + 15) // 16, (width + 15) // 16),
        device=device, generator=generator)
    tile_class = tile_class.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)[:height, :width]
    y = torch.arange(height, device=device)[:, None]
    x = torch.arange(width, device=device)[None, :]
    parity = 2 * (y % 2) + x % 2
    return (parity != tile_class).to(torch.float32) * (4.0 / 3.0)


'''


def patch_source(source):
    source = replace_once(source, "def training(dataset, opt, pipe,", MASK_HELPER + "def training(dataset, opt, pipe,")
    source = replace_once(
        source,
        "    bg = torch.rand((3), device=\"cuda\") if opt.random_background else background\n",
        "    bg = torch.rand((3), device=\"cuda\") if opt.random_background else background\n"
        "    sparse_adj_probability = float(os.environ.get(\"FASTGS_SPARSE_ADJOINT_P\", \"1\"))\n"
        "    if sparse_adj_probability not in (0.75, 1.0):\n"
        "        raise ValueError(\"FASTGS_SPARSE_ADJOINT_P must be 0.75 or 1.0\")\n"
        "    sparse_adj_generator = None\n"
        "    sparse_adj_seed = int(os.environ.get(\"FASTGS_SPARSE_ADJOINT_SEED\", \"20260927\"))\n"
        "    if sparse_adj_probability == 0.75:\n"
        "        sparse_adj_generator = torch.Generator(device=\"cuda\").manual_seed(sparse_adj_seed)\n"
        "    print(\"FASTGS_SPARSE_ADJOINT \"\n"
        "          f\"probability={sparse_adj_probability} seed={sparse_adj_seed} \"\n"
        "          \"mask_vjp_only=true full_loss=true\", flush=True)\n",
    )
    source = replace_once(
        source,
        "        loss.backward()\n",
        "        if sparse_adj_probability == 0.75:\n"
        "            height, width = image.shape[-2:]\n"
        "            sampled_mask = sparse_adjoint_mask(\n"
        "                height, width, sparse_adj_generator, image.device)\n"
        "            image.register_hook(\n"
        "                lambda image_grad, mask=sampled_mask: image_grad * mask.unsqueeze(0).to(image_grad.dtype))\n"
        "        loss.backward()\n",
    )
    source = replace_once(
        source,
        '    print(f"Gaussian number: {gaussians._xyz.shape[0]}")\n',
        '    checkpoint_path = os.path.join(dataset.model_path, f"chkpnt{opt.iterations}.pth")\n'
        "    torch.save((gaussians.capture(opt.optimizer_type), opt.iterations), checkpoint_path)\n"
        '    print(f"Saved final checkpoint: {checkpoint_path}")\n'
        '    print(f"Gaussian number: {gaussians._xyz.shape[0]}")\n',
    )
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("author_source", type=Path)
    parser.add_argument("patched_output", type=Path)
    args = parser.parse_args()
    source = args.author_source.read_text(encoding="utf-8")
    args.patched_output.parent.mkdir(parents=True, exist_ok=True)
    args.patched_output.write_text(patch_source(source), encoding="utf-8")


if __name__ == "__main__":
    main()
