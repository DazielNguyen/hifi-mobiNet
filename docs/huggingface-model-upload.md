# Verified private Hugging Face model upload

Date: 2026-10-06. Account: DazielNguyen.

The author chose model storage first and deferred demo hosting.
The model candidate is uploaded at [https://huggingface.co/DazielNguyen/hifi-mobiNet](https://huggingface.co/DazielNguyen/hifi-mobiNet), with private visibility.
Only the owner and authorized collaborators can access it. It is not a public release.

- Initial artifact commit: `4ba371bd3b5c174e76d6f31844052da2e6d6868e`.
- Current model commit: `a4354007630f14b2cd0046a6f6b0dc17dd1f7ac9`.
- Contents: four selected full training checkpoints, four Q05 FP32 ONNX files, configs, redacted logs and notices.
- Final verification covers 47 inventoried files. Hugging Face also created `.gitattributes`.
- Original checkpoint, ONNX, config and log bytes retain their prepared identities.
- The subsequent status commit changes only the card, licensing upload status, inventory and upload receipt.

## Verification

All file sizes matched the pinned remote inventory.
Large files matched server-reported original-file SHA-256 values. They were not all downloaded again.
Small metadata files were downloaded at exact commits and hashed locally.
Sequential-IR ONNX was downloaded at the initial artifact commit. Its size and SHA-256 matched the supplied identity.
The initial and final inventories appear in [the machine-readable record](huggingface-model-upload.json).
These checks establish upload identity and a working download path. They do not establish new synthesis quality or performance.

## Continue

Use `a4354007630f14b2cd0046a6f6b0dc17dd1f7ac9` as `HIFIMOBINET_MODEL_REVISION` for a future pinned runtime.
Private downloads require a token with read access. Keep tokens outside Git and chat.
Demo hosting remains undecided. The account had no PRO subscription at the recorded check.
The current Docker Space creation rule requires a paid account plan. No plan or hardware was purchased.

The dataset and Space candidates remain local. No dataset upload, tag, GitHub push or deployment occurred in this step.
Historical audio remains off by default. Harvard text distribution conditions remain unresolved.
The MIT and scoped audio decisions are unchanged. Remaining component conditions still need review before public release.
Original assets, transfer archives and candidate snapshots remain unchanged.
The manuscript and research evidence were not edited.
