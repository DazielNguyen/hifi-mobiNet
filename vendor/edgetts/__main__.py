import argparse
import json
import logging
import pathlib
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)

# PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects the
# pathlib.PosixPath the dataset path hparam gets pickled as inside our own
# checkpoints. Safe to allowlist since we only ever load checkpoints we wrote.
torch.serialization.add_safe_globals([pathlib.PosixPath])


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    parser.add_argument(
        "--resume_from_checkpoint", help="Path to a .ckpt file to resume training from"
    )
    parser.add_argument("--default_root_dir", help="Trainer log/checkpoint directory")
    # `Trainer.add_argparse_args` was removed in PyTorch Lightning 2.0, so the
    # subset of Trainer flags actually used by this project's README/scripts
    # is re-declared explicitly here. Defaults are tuned for a 2x RTX 4070 Ti
    # (12 GB each) + 20-core CPU workstation.
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument(
        "--devices",
        default="2",
        help="Number of GPUs, or comma-separated GPU ids. >1 requires NCCL "
        "(run from WSL2/Linux on Windows hosts -- see TRAINING.md)",
    )
    parser.add_argument(
        "--strategy",
        default="ddp_find_unused_parameters_true",
        help="Set to 'auto' for single-device training",
    )
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        help="Ada Lovelace (40-series) has native bf16 tensor cores; no GradScaler needed",
    )
    parser.add_argument("--max_epochs", type=int, default=10000)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    # TF32 matmul on Ampere+/Ada tensor cores; safe precision/throughput
    # tradeoff for conv/attention-heavy training.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    devices = int(args.devices) if str(args.devices).isdigit() else args.devices
    # Best-checkpoint is always active (independent of --checkpoint-epochs).
    callbacks = [
        ModelCheckpoint(
            filename="best-{epoch}-{val_loss_mel:.4f}",
            monitor="val_loss_mel",
            mode="min",
            save_top_k=1,
            save_last=False,
        )
    ]
    if args.checkpoint_epochs is not None:
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=args.checkpoint_epochs,
                # Disk space is the constraint now, not pick-the-best-by-ear:
                # only keep one rolling checkpoint instead of every epoch.
                # NOTE: save_top_k=0 disables saving entirely, which silently
                # starves save_last too (it only copies an existing save) --
                # use save_top_k=1 with no monitor so a real checkpoint is
                # written and overwritten every epoch.
                save_top_k=1,
                save_last=True,
            )
        )
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=devices,
        strategy=args.strategy if devices != 1 else "auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        # validation_step logs audio via logger.experiment.add_audio, which
        # only TensorBoardLogger's SummaryWriter exposes (CSVLogger doesn't).
        logger=TensorBoardLogger(save_dir=args.default_root_dir),
        callbacks=callbacks,
    )

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing from the single-speaker model).
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )

    trainer.fit(model, ckpt_path=args.resume_from_checkpoint)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
