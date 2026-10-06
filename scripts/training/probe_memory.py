"""Peak-GPU-memory probe for choosing a batch size (no optimizer update).

    python scripts/training/probe_memory.py --model-id ... --config ... \\
        --dataset-dir DIR --data-root ROOT --split SPLIT --batch-size 16 [--precision bf16]

Runs one generator and one discriminator forward/backward pass on the
``--batch-size`` longest training utterances (by phoneme count), under the
config's precision unless ``--precision`` overrides it, and reports
torch.cuda.max_memory_allocated. Gradients are discarded; no optimizer step
is taken, so model weights never change. Prints JSON (including any error).
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from hifimobinet.training.config import load_config
from hifimobinet.training.train import build_module


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--model-id", "--config", "--dataset-dir", "--data-root", "--split"):
        p.add_argument(name, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--precision", choices=("32", "bf16"))
    p.add_argument("--run-dir", default="/tmp/hifimobinet-probe")
    args = p.parse_args()
    config = load_config(Path(args.config))
    if config.model_id != args.model_id:
        raise SystemExit("--model-id does not match the config")
    precision = args.precision or str(config.sections["trainer"]["precision"])
    result = {"model_id": config.model_id, "batch_size": args.batch_size, "precision": precision,
              "optimizer_steps": 0}
    try:
        torch.manual_seed(int(config.sections["trainer"]["seed"]))
        module, data = build_module(config, Path(args.dataset_dir), Path(args.data_root), Path(args.split),
                                    Path(args.run_dir))
        module.log = lambda *a, **k: None
        device = torch.device("cuda")
        module.to(device).train()
        order = sorted(range(len(data.train)), key=lambda i: -len(data.full.utterances[data.train.indices[i]].phoneme_ids))
        items = [data.train[i] for i in order[: args.batch_size]]
        batch = data.collate(module.hparams.segment_size)(items)
        batch = type(batch)(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vars(batch).items()})
        result["longest_phoneme_ids"] = int(batch.phoneme_lengths.max())
        result["longest_spec_frames"] = int(batch.spectrogram_lengths.max())
        torch.cuda.reset_peak_memory_stats()
        before = {k: v.detach().clone() for k, v in module.state_dict().items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            loss_g = module.training_step_g(batch)
        loss_g.backward()
        module.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            loss_d = module.training_step_d(batch)
        loss_d.backward()
        module.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        result["loss_g_finite"] = bool(torch.isfinite(loss_g).item())
        result["loss_d_finite"] = bool(torch.isfinite(loss_d).item())
        result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        result["device_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        result["weights_unchanged"] = all(torch.equal(before[k], v) for k, v in module.state_dict().items())
        result["status"] = "ok"
    except Exception as error:  # reported, not hidden
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback_tail"] = traceback.format_exc()[-1500:]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
