#!/usr/bin/env python3
"""Trains a single-speaker VITS voice from a banhmi_train.preprocess output
directory (dataset.jsonl + config.json).

Usage:
    python -m banhmi_train.train \
        --dataset-dir training_output/ljspeech/medium \
        --quality medium \
        --batch-size 16 --accelerator gpu --devices 1

    # Or edit one file instead of typing every flag (see configs/*.yaml for
    # examples). Anything in the YAML overrides --quality and any other
    # flag above it -- including VitsModel parameters that have no --flag
    # at all (e.g. resblock_kernel_sizes), since VitsModel's **kwargs
    # catch-all accepts them by name.
    python -m banhmi_train.train \
        --dataset-dir training_output/ljspeech/medium \
        --config configs/banhmi_mb_sequential_111.yaml \
        --batch-size 16 --accelerator gpu --devices 1
"""
import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .vits.training import VitsModel

_LOGGER = logging.getLogger("banhmi_train.train")


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to preprocessed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs", type=int, help="Save checkpoint every N epochs (default: 1)"
    )
    parser.add_argument(
        "--quality", default="medium", choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--config", type=Path,
        help="YAML file of parameter overrides, applied on top of --quality and every other flag",
    )
    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    # cudnn.benchmark only pays off when the *same* shape recurs often enough
    # to amortize its search cost. VitsModel.train_dataloader() buckets
    # batches by cached spectrogram length (LengthBucketBatchSampler,
    # matching banhmi_tts's own sampler) -- an earlier version bucketed by
    # phoneme count instead, which measured WORSE than no bucketing at all
    # (14.2 vs 37.8 samples/s, both benchmark=True): phoneme count only
    # stabilizes TextEncoder/StochasticDurationPredictor shapes, not the
    # PosteriorEncoder/flow, which run on the *full* spectrogram/latent
    # length (SynthesizerTrn.forward() only crops to segment_size afterward)
    # and dominate compute alongside the Generator. Re-measure before
    # changing this again -- see git history for the phoneme-bucketing
    # numbers this replaced.
    torch.backends.cudnn.benchmark = True
    # TF32 matmul on Ampere/Ada tensor cores (RTX 4070 Ti here) -- banhmi_tts
    # sets this too; this codebase was missing it.
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    # LengthBucketBatchSampler shards across DDP ranks itself (num_replicas/
    # rank, set from torch.distributed inside train_dataloader()) -- with
    # PL's default replace_sampler_ddp=True, the Trainer would additionally
    # wrap it in a DistributedSampler, double-sharding the data.
    #
    # NOTE on --strategy: use plain "ddp" (PL 1.7's registry only has an
    # explicit "..._find_unused_parameters_false" alias -- plain "ddp"
    # already defaults find_unused_parameters=True). Don't pass any
    # "_find_unused_parameters_false" variant: this model genuinely needs
    # unused-parameter tolerance. PL wraps model_g/model_d/model_d_mrd/
    # model_d_dur as one DDP unit, but training_step_g's backward only
    # touches model_g's parameters and training_step_d's backward only
    # touches the discriminators' -- every step, DDP sees a backward pass
    # that skips half the wrapped parameters. Confirmed by trying _false:
    # "Expected to have finished reduction in the prior iteration... Parameter
    # indices which did not receive grad". _true costs one extra autograd-
    # graph traversal per step to tolerate that; there's no way to avoid it
    # without splitting model_g/model_d into separate DDP-wrapped modules.
    args.replace_sampler_ddp = False

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See banhmi_train.preprocess.config for the schema
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        sample_rate = int(config["audio"]["sample_rate"])

    trainer = Trainer.from_argparse_args(args)
    if args.checkpoint_epochs is not None:
        # save_top_k=3 + monitor="loss_mel" keeps the 3 best-mel checkpoints
        # (best-*.ckpt) instead of accumulating one file per `every_n_epochs`
        # interval -- still bounded (important for long runs; e.g. 2000
        # epochs writing one file per epoch would be unbounded ~900MB/epoch),
        # but keeps a little retroactive history instead of only 1 (a single
        # best-so-far silently overwrites every prior snapshot, which cost us
        # the ability to bisect exactly when a duration-predictor collapse
        # started in an earlier run -- see SynthesizerTrn.infer()'s own
        # collapse warning, added for the same reason).
        # save_last=True additionally keeps a rolling last.ckpt for resuming
        # a killed/crashed run without losing all progress since the last
        # best checkpoint. Monitors "val_loss_mel" (validation_step-only,
        # non-adversarial reconstruction loss -- matches EdgeTTS's own
        # val_loss_mel) rather than "loss_mel" (also logged from real
        # training steps, so noisier as a "best model" signal).
        trainer.callbacks = [
            ModelCheckpoint(
                # Fixed, resume-stable location -- deliberately NOT the
                # logger's auto-versioned lightning_logs/version_N/checkpoints
                # (every resume bumps N, and PL's ModelCheckpoint refuses to
                # restore best_model_score/best_k_models whenever dirpath
                # differs from the checkpoint being resumed -- observed in
                # practice: a resume could silently save a worse checkpoint
                # as the new "best" because it no longer knew the real best).
                # A stable dirpath means dirpath never changes across
                # resumes, so that restoration actually happens.
                dirpath=str(Path(args.default_root_dir) / "checkpoints"),
                every_n_epochs=args.checkpoint_epochs,
                monitor="val_loss_mel",
                mode="min",
                save_top_k=3,
                save_last=True,
                filename="best-{epoch}-{val_loss_mel:.4f}",
            )
        ]
        _LOGGER.debug(
            "Checkpoints: best (by val_loss_mel) + last, checked every %s epoch(s)",
            args.checkpoint_epochs,
        )

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as config_yaml_file:
            overrides = yaml.safe_load(config_yaml_file) or {}
        _LOGGER.info("Applying overrides from %s: %s", args.config, overrides)
        dict_args.update(overrides)

    model = VitsModel(
        num_symbols=num_symbols,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    # ckpt_path= here (not relying on Trainer(resume_from_checkpoint=...)
    # alone, set from --resume_from_checkpoint above) matters: PL's
    # trainer.fit() only takes the *full* restore path -- including
    # restoring ModelCheckpoint's own best-score tracking -- when ckpt_path
    # is passed directly to fit(). Without this, each resume's
    # ModelCheckpoint starts its "best" comparison from scratch, so a worse
    # checkpoint than one already on disk can get saved as a new "best"
    # (observed in practice across several resumes of the same run).
    trainer.fit(model, ckpt_path=args.resume_from_checkpoint)

    # test_split is held out entirely until now -- a final, unbiased check
    # on data the training process never touched for monitoring/tuning.
    if model._test_dataset is not None and len(model._test_dataset) > 0:
        trainer.test(model)
    else:
        _LOGGER.warning("No test set to evaluate (test_split too small for this dataset size)")


if __name__ == "__main__":
    main()
