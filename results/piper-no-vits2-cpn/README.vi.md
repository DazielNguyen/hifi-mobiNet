# Piper_no_VITS2_cpn: kết quả train và đánh giá Harvard-720

*English: [README.md](README.md)*

Câu hỏi mà run này phục vụ (tác giả, 2026-10-06): các thành phần VITS2 làm Piper tốt
hơn hay tệ hơn? `Piper_no_VITS2_cpn` là baseline của hifi-mobiNet khi bỏ các thành
phần đó: EdgeTTS Config A, train từ đầu trên dữ liệu và split của baseline. Model
`baseline-resblock2` đã phát hành là Piper có các thành phần VITS2. Công thức train
và bảng đối chiếu thành phần: `docs/training/piper-component-comparison.md`.

Mọi con số dưới đây là **phép đo mới** (2026-10-10/11). Ngoại lệ duy nhất là bốn file
kết quả Harvard lịch sử dùng để so sánh; các file đó giữ nguyên.

## 1. Model được đánh giá

- **Run:** `training_output/piper-no-vits2-cpn` (nằm ngoài Git).
  - Commit `d0c20d8`; 2× RTX 4070 Ti, DDP, bf16.
  - 1.500 epoch, xong lúc 2026-10-10 21:58 +07.
  - `run_record.json` được chép lại trong `training_summary.json`.
- **Quy tắc chọn checkpoint:** chốt trước khi train (`docs/training/next-run.md`): checkpoint có `val_loss_mel` thấp nhất.
  - Được chọn: **epoch 1403, `val_loss_mel` 19.7719**.
  - File: `best-epoch=1403-val_loss_mel=19.7719.ckpt`.
  - SHA-256 `2eeff5354c3320b12a288c0e52bdadbacce2c1ddaa14d5c739d1d4208b0f6003`.
  - Global step 1.100.736.
- **Load:** strict, `weights_only=True`; 789 entry. Hyper-parameter lưu trong checkpoint trùng với `configs/training/piper-no-vits2-cpn.yaml`.
- **Tập test:** Harvard-720 và tập test không được dùng để chọn checkpoint.

## 2. Đường train so với baseline (`val_loss_mel` trên tập validation)

Hai run dùng cùng 100 câu validation và cùng hàm loss. Giá trị lấy từ scalar
TensorBoard của từng run (`training_summary.json`, có kèm chuỗi theo từng epoch).

| Epoch | Piper_no_VITS2_cpn trung bình (SD) | baseline-resblock2 trung bình (SD) | Chênh lệch |
|---|---:|---:|---:|
| 1000–1099 | 20.364 (0.151) | 20.419 (0.216) | −0.056 |
| 1200–1299 | 20.224 (0.152) | 20.233 (0.220) | −0.009 |
| 1300–1399 | 20.160 (0.143) | 20.196 (0.178) | −0.036 |
| 1400–1499 | 20.105 (0.152) | 20.184 (0.216) | −0.078 |
| Best (epoch) | 19.7719 (1403) | 19.7882 (1489) | −0.016 |

Mel loss trên validation của hai run gần như bằng nhau. Độ dao động giữa các epoch
(SD 0.15–0.22) lớn hơn khoảng cách giữa hai run. Số liệu baseline lấy từ log gốc
của run đó (bốn file events, run có resume); checkpoint được chọn của nó là epoch 1489.

## 3. Kết quả Harvard-720

Protocol: protocol Harvard lịch sử, không thay đổi (`docs/evaluation.md`).
- PyTorch FP32 trên CPU, 12 thread; scales 0.667/1.0/0.8; seed 1234 trước mỗi câu.
- UTMOSv2 `fusion_stage3` fold 0 (SHA-256 trọng số `c8149d98…`), một lần dự đoán cho mỗi WAV.
- Whisper `small` (`9ecf7799…`).
- Macro WER bằng JiWER.

Bốn dòng còn lại là kết quả lịch sử đã lưu (`results/historical/harvard/`).

| Model | UTMOSv2 (dự đoán) | Macro WER | Corpus WER | Số câu WER = 0 | RTF trung bình |
|---|---:|---:|---:|---:|---:|
| **Piper_no_VITS2_cpn** (mới) | **3.517** | **0.0612** | 0.0596 | 502 | 0.0421 * |
| baseline-resblock2 (có VITS2) | 3.026 | 0.1476 | 0.1437 | 318 | 0.0418 |
| Parallel-IR (MRF) | 3.568 | 0.1073 | 0.1033 | 381 | 0.0339 |
| Sequential-IR (SEQ) | 3.544 | 0.1222 | 0.1186 | 351 | 0.0308 |
| Piper (checkpoint đã công bố) | 3.451 | 0.1747 | 0.1696 | 228 | 0.0392 |

\* Đo trong một phiên khác với các dòng lịch sử; xem mục 4. Không so sánh trực tiếp.

So sánh ghép cặp theo từng câu (`comparison.json`). Chênh lệch = mới − đối chứng.
Khoảng tin cậy (CI) là bootstrap ghép cặp 95%. Hiệu chỉnh Holm tính trên 12 phép
thử này và chỉ mang tính thăm dò.

| Mới so với | ΔUTMOSv2 [CI 95%] | ΔWER [CI 95%] | Số câu tốt hơn / kém hơn (WER) |
|---|---|---|---|
| baseline-resblock2 | **+0.491** [+0.465, +0.517] | **−0.086** [−0.099, −0.073] | 301 / 76 |
| Parallel-IR | −0.051 [−0.073, −0.029] | −0.046 [−0.057, −0.036] | 236 / 85 |
| Sequential-IR | −0.027 [−0.050, −0.005] (Holm p 0.13) | −0.061 [−0.073, −0.049] | 263 / 74 |
| Piper (đã công bố) | +0.066 [+0.042, +0.090] | −0.113 [−0.127, −0.100] | 399 / 67 |

WER là tỉ lệ từ sai: −0.086 nghĩa là giảm 8,6 điểm phần trăm.

## 4. Các control chạy trong cùng phiên

- **Tính tái lập của synthesis (`session_control.json`).**
  - Chạy lại `synth_banhmi.py` gốc của BanhmiTTS (SHA-256 `23b2548d…`) trên baseline epoch 1489 cho ra **720/720 WAV giống từng byte** với WAV Harvard lịch sử.
  - Thời lượng audio cũng trùng khớp.
  - Như vậy môi trường hiện tại tái tạo đúng synthesis lịch sử.
  - Model mới cũng deterministic: câu 0 có cùng hash WAV ở lần chạy smoke và lần chạy đầy đủ.
- **Độ trôi thời gian.**
  - Hôm đo, cùng model baseline có RTF 0.04524, so với 0.04177 đã lưu (**+8,3%**).
  - Vì vậy RTF đo vào các ngày khác nhau không so sánh được.
  - Đo liền nhau trong phiên này: model mới có RTF 0.04212, baseline 0.04524.
  - Tức model mới **thấp hơn 6,9%**: CI ghép cặp của chênh lệch [−0.0037, −0.0025]; Wilcoxon p 2.7e-25; nhanh hơn ở 503/720 câu.
- **Tính tái lập của scorer (`scorer_control.json`).** Các WAV control của baseline được chấm lại bằng `score.py` của hôm đó.
  - Whisper cho **720/720 transcript giống hệt**, và macro WER 0.1476 trùng giá trị đã lưu.
  - UTMOSv2: trung bình 3.030 so với 3.026 đã lưu. Chênh lệch theo từng câu có SD 0.22, lớn nhất 0.91, và r = 0.74, vì bước cắt ngẫu nhiên (random cropping) không được gắn seed.
  - Do đó, chênh lệch WER và UTMOSv2 trung bình ở mục 3 không phải do scorer bị trôi.

## 5. Đọc kết quả

- **So với baseline có VITS2:** model không có các thành phần VITS2 tốt hơn rõ rệt trên cả hai thước đo chất lượng.
  - UTMOSv2 dự đoán +0,49; WER giảm 8,6 điểm (0,148 → 0,061).
  - Nhanh hơn 6,9% khi đo trong cùng phiên.
  - Mel loss trên validation gần như bằng nhau.
  - Trong thí nghiệm này, thêm gói VITS2 vào Piper **không** làm model tốt hơn, và độ dễ hiểu (WER) kém đi rõ rệt khi có VITS2.
- **So với Parallel-IR và Sequential-IR:** model mới có WER thấp hơn.
  - UTMOSv2 dự đoán của nó thấp hơn một chút so với cả hai. Với SEQ, CI chỉ vừa đủ loại trừ 0, và kết quả không còn ý nghĩa sau hiệu chỉnh Holm.
  - Cả hai decoder IR vẫn nhanh hơn. Để so giữa các phiên, nhân RTF 0.0421 của model mới với hệ số trôi 8,3%: được khoảng 0,039, so với 0,034 của MRF và 0,031 của SEQ.
  - MRF và SEQ được train bằng công thức baseline VITS2: flow export của chúng mang cùng cơ chế attention, xem `docs/evidence/claim-evidence-2026-10-07` (J-C025).
  - Model mới cũng có WER thấp nhất trong cả năm hệ thống trên Harvard-720.

Những gì kết quả này **không** chứng minh:
- **Thành phần VITS2 nào gây ra khác biệt.** Hai model còn khác nhau về codebase, như liệt kê trong bảng đối chiếu thành phần: các guard của SDP, tối ưu tự động hay thủ công, và kernel AdamW fused. Đây không phải ablation từng thành phần. Phép thử sạch là so EdgeTTS Config A với Config C.
- **Kết quả đúng ngoài thiết lập này.** Mỗi model chỉ có một lần train, mỗi câu sinh một lần, WER dùng Whisper-small, và MOS là dự đoán (không phải người chấm).
- **Khác biệt có ý nghĩa qua các seed train khác nhau.** Phép thử ghép cặp và CI chỉ mô tả dao động giữa các câu.

## 6. Chưa làm trong đợt này

- **Đánh giá trên tập test-500 canonical:** chưa làm.
  - Protocol LJ500 lịch sử (`eval_*_seeded_500.py`) chỉ gắn seed một lần cho mỗi lần chạy và chấm điểm ngay trong vòng lặp synthesis, nên độ ngẫu nhiên của scorer làm thay đổi noise của câu kế tiếp.
  - Muốn tái tạo thì phải dùng đúng vòng lặp đó, không dùng được các script Harvard.
  - Thiết kế sạch hơn: sinh audio cho cả hai checkpoint trong cùng một phiên với seed theo từng câu, rồi chấm cả hai trong cùng một lượt scorer.
- **ONNX export, PTQ và đo thời gian trên thiết bị biên:** chưa làm.

## 7. Các file

| File | Nội dung | Loại |
|---|---|---|
| `training_summary.json` | Bản ghi run, hash checkpoint, `val_loss_mel` theo từng epoch của cả hai run, thống kê theo cửa sổ | tóm tắt mới từ log run |
| `harvard720_synth_record.json` | Bản ghi synthesis: hash checkpoint/config/manifest, protocol, môi trường, thời gian và SHA-256 WAV theo từng câu. Đây chính là file `synth.json` được nêu trong `score_record.json` (cùng SHA-256 `72bee8b4…`) | phép đo mới |
| `harvard_piper_no_vits2_cpn_results.json` | Kết quả theo từng câu, theo schema lịch sử, thêm `wav_sha256` | phép đo mới |
| `score_record.json` | Package scorer, hash trọng số, môi trường, hash kết quả | bản ghi phép đo mới |
| `comparison.json` | Tóm tắt cả năm hệ thống và 12 phép thử ghép cặp | phân tích mới |
| `session_control.json` | Tính tái lập synthesis, độ trôi thời gian, so RTF trong cùng phiên | control mới |
| `harvard_baseline_rescored_results.json`, `baseline_rescore_record.json` | Kết quả chấm lại các WAV control của baseline. Bản ghi synthesis mà chúng trích dẫn được giữ cùng run, vì nó chứa đường dẫn của máy | control mới |
| `scorer_control.json` | So điểm chấm lại với điểm baseline đã lưu | control mới |

Các WAV (720 của model mới và 720 control) nằm ngoài Git, trong thư mục run và vùng
WAV lịch sử của máy. Hash của chúng có trong các bản ghi.

## 8. Tái lập

Chạy từ thư mục gốc của clone training. Synthesis dùng venv training. Chấm điểm dùng
môi trường scorer có UTMOSv2, Whisper và JiWER. Phân tích dùng bất kỳ môi trường nào
có SciPy/JiWER.

```sh
python scripts/evaluation/harvard/synthesize.py --config configs/training/piper-no-vits2-cpn.yaml \
  --checkpoint "training_output/piper-no-vits2-cpn/checkpoints/best-epoch=1403-val_loss_mel=19.7719.ckpt" \
  --manifest results/manifests/harvard720.json \
  --wav-dir training_output/piper-no-vits2-cpn/eval/harvard720/wavs --output training_output/piper-no-vits2-cpn/eval/harvard720/synth.json
python scripts/evaluation/harvard/score.py --synth training_output/piper-no-vits2-cpn/eval/harvard720/synth.json \
  --output <dir>/harvard_piper_no_vits2_cpn_results.json --record <dir>/score_record.json
python scripts/evaluation/harvard/compare.py --candidate results/piper-no-vits2-cpn/harvard_piper_no_vits2_cpn_results.json \
  --label piper_no_vits2_cpn --reference-dir results/historical/harvard --output <dir>/comparison.json
python scripts/evaluation/harvard/scorer_control.py --rescored results/piper-no-vits2-cpn/harvard_baseline_rescored_results.json \
  --historical results/historical/harvard/harvard_baseline_results.json --output <dir>/scorer_control.json
python scripts/training/summarize_run.py --run-dir training_output/piper-no-vits2-cpn \
  --reference-logdir <baseline run>/lightning_logs --reference-label baseline-resblock2 \
  --reference-best-checkpoint "best-epoch=1489-val_loss_mel=19.7882.ckpt" --output <dir>/training_summary.json
```

**Kết quả mong đợi:**
- Synthesis là deterministic, nên hash WAV phải trùng.
- Transcript của Whisper đã được tái tạo chính xác trong control.
- Điểm UTMOSv2 theo từng câu thay đổi giữa các lượt chấm.

**Control trong phiên:** `synth_banhmi.py` gốc được chạy với nhãn `baseline_ctrl_20261010`. Sau đó `session_control.py` so WAV và thời gian của nó với baseline lịch sử.

## 9. Cập nhật 2026-10-16: so sánh với Piper-ParIR

`Piper_ParIR_no_VITS2_cpn` dùng backbone, dữ liệu và công thức train của run này, chỉ thay
bằng bộ decoder Parallel-IR đã phát hành (decoder inverted-residual, SnakeBeta, MRD). Báo
cáo đầy đủ: [`../piper-parir-no-vits2-cpn/README.vi.md`](../piper-parir-no-vits2-cpn/README.vi.md).
Các số liệu ở mục 1–8 phía trên giữ nguyên.

| Model | Epoch được chọn (`val_loss_mel`) | UTMOSv2 (dự đoán) | Macro WER | Số câu WER = 0 | RTF trung bình |
|---|---:|---:|---:|---:|---:|
| **Piper_no_VITS2_cpn** (run này) | 1403 (19.7719) | 3.517 | **0.0612** | **502** | 0.0421 |
| Piper-ParIR | 1486 (19.0069) | 3.530 | 0.0726 | 464 | chưa đo |
| baseline-resblock2 (có VITS2) | 1489 (19.7882) | 3.026 | 0.1476 | 318 | 0.0418 |
| Parallel-IR (MRF) | — | **3.568** | 0.1073 | 381 | 0.0339 |
| Sequential-IR (SEQ) | — | 3.544 | 0.1222 | 351 | 0.0308 |
| Piper (đã công bố) | — | 3.451 | 0.1747 | 228 | 0.0392 |

ParIR so với run này, ghép cặp theo từng câu:
- **Validation:** mel loss trên validation thấp hơn rõ rệt: −0,77 trong epoch 1300–1499.
- **WER:** cao hơn một chút, +0,011 [+0,002, +0,021]; Holm p 0,11 trên 10 phép thử của file đó.
- **UTMOSv2:** gần như bằng nhau.
  - Khác phiên: +0,013 [−0,011, +0,037].
  - Cùng phiên, sau khi chấm lại các WAV của run này: +0,025 [+0,002, +0,049], Wilcoxon p 0,097.
- **Độ trôi của scorer:** lần chấm lại đó tái tạo đúng 720/720 transcript Whisper và đúng macro WER của run này. UTMOSv2 trung bình lệch −0,012.

Vì vậy run này vẫn có WER thấp nhất trong sáu hệ thống. RTF của ParIR (và Piper-SeqIR) sẽ
được đo trong cùng một phiên trên máy rảnh, với run này làm control.
