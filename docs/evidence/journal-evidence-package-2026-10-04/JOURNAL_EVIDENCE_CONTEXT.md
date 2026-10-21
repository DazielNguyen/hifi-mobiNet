# HiFi-MobiNet: gói context và bằng chứng cho manuscript journal

Ngày thu thập: **04/10/2026**. Phạm vi: đọc và sao chép bằng chứng có sẵn, tính lại từ dữ liệu đã lưu. **Không** train, export, quantize, benchmark hoặc chạy lại TTS/ASR/UTMOS. Gói này cung cấp context cho audit Q02–Q10; **không** tuyên bố Q02–Q10 đã hoàn tất.

**Tên model dùng trong tài liệu**

| Tên gốc trong code/log | Tên dự kiến trong bài |
|---|---|
| baseline (`resblock="2"`) | HiFi-GAN ResBlock2 |
| MRF (`resblock="mrf"`, `mb_expansion=(1,1,1)`) | Parallel-IR |
| SEQ (`resblock="mb"`, `mb_expansion=(1,1,1)`) | Sequential-IR |
| Piper `lj-med_1000.ckpt` | external published checkpoint |

**Phân loại nguồn**, dùng xuyên suốt tài liệu và ở cột `source_type` của [evidence-manifest.csv](evidence-manifest.csv):

| Nhãn | Nghĩa |
|---|---|
| `original_raw` | Bản sao byte-for-byte của file gốc: JSON/CSV từng mẫu, log, script, manifest, config, metadata, model card. |
| `prior_round_agent_output` | Đầu ra máy đọc được của các lượt audit trước (Q03/Q05…), có script và hash, **chưa chạy lại** trong lượt này. |
| `author_or_agent_note` | Báo cáo Markdown của tác giả hoặc agent: nguồn định hướng, **không** tự xác lập phép đo. |
| `recomputed_this_round` | Kết quả tính lại trong lượt này, có script đi kèm trong [scripts/](scripts/). |
| `original_exploratory` | Script/profile thăm dò lịch sử, không có file kết quả hoặc không rõ model. |

---

## 1. Executive summary

**Có đủ bằng chứng gốc cho:**

- Harvard-720 đủ cả 4 model:
  - 720 dòng/model, cùng ID, text và thứ tự; 0 lỗi.
  - Có JSON từng câu, 2.880 WAV (đã băm SHA-256), script và log.
  - Thống kê đã tính lại khớp với số đã báo.
- Cấu hình huấn luyện đọc trực tiếp từ checkpoint: epoch, callback, optimizer, scheduler, hparams.
- Split thực tế:
  - Baseline/SEQ dùng canonical.
  - MRF dùng một split tái dựng, đã đối chiếu với log TensorBoard ghi lúc train.
- Số tham số và FLOPs decoder: đếm lại chính xác từ checkpoint.
- Cấu trúc graph ONNX: decoder FP32, số node lượng tử theo từng module.
- Calibration 60 câu lịch sử: tái lập được chính xác theo đúng quy tắc của script.

**Thiếu hoặc chưa xác minh:**

- Đánh giá ONNX/INT8 **cuối cùng** trên các graph đã định danh (Q05). Các số ONNX/INT8 hiện có đều là số lịch sử, với những vấn đề sau:
  - graph MRF không khớp checkpoint 1386;
  - scope INT8 của SEQ khác baseline/MRF;
  - calibration có rò rỉ test;
  - bộ chọn scope có lỗi sample rate.
- Benchmark N150/Pi 5: **NOT AVAILABLE**.
- Đánh giá nghe thử: **NOT AVAILABLE**.
- Hash source lúc runtime của từng phiên train, hash checkpoint dùng để resume, split của Piper.
- Phân tích equivalence/non-inferiority: không có margin định trước.
- Biến thiên giữa các lần benchmark và giữa các lần train: mỗi cấu hình chỉ có 1 lần train và 1 lần đo.

**Mức kết luận hiện được hỗ trợ:**

1. **So sánh cấp hệ thống trên PyTorch CPU, Harvard-720:**
   - Sequential-IR có RTF thấp hơn Parallel-IR 9,08% và thấp hơn Piper 21,33%.
   - Sequential-IR có UTMOSv2 *thấp hơn* Parallel-IR 0,024 (p=0,047; sau Holm p=0,094).
   - WER của Sequential-IR *cao hơn* Parallel-IR 0,015, tức 1,5 điểm phần trăm (Wilcoxon p=0,052; CI bootstrap exploratory không chứa 0).
   - So với Piper, Sequential-IR tốt hơn ở cả 3 chỉ số.
2. **Không được hỗ trợ:**
   - "Chất lượng tương đương" giữa SEQ và MRF.
   - "Sequential-IR tốt hơn Parallel-IR về UTMOSv2".
   - "Lợi thế tốc độ giữ nguyên qua ONNX/INT8".
   - Mọi kết luận nhân quả riêng cho decoder: split, recipe ổn định số, activation và MRD vẫn khác nhau.

---

## 2. Inventory và provenance

### 2.1. Repository và đường dẫn đã đọc

| Mục | Giá trị | Nguồn |
|---|---|---|
| Repository | `BanhmiTTS`, HEAD `0a9bec22e314023e7c3a01f52fc8844f9e323e2c` | `git rev-parse HEAD` (04/10) |
| Working tree | `M banhmi_train/export_onnx.py` (sửa ở lượt Q05); untracked: `docs/` (Q03–Q05 + evidence), `.agents/`, `AGENTS.md`, `CLAUDE.md`, `research-skill-setup/`, gói này | `git status --short` |
| Lưu ý commit | Commit hiện tại **không** chứng minh code của các run cũ. WSL `<WSL_PROJECT_HOME>/BanhmiTTS` không có Git. Source lịch sử chỉ có dạng "source candidate" khôi phục từ lịch sử sửa file (Q03 §11). | [author_notes/Q03…md §11](evidence/author_notes/Q03_CAU_HINH_VA_LICH_SU_HUAN_LUYEN.md) |
| Piper (code đối chứng) | `FPT-Graduation-Project/piper`, HEAD `73c04d81d5590ecc46e522de3601ce7fb29fc2be` | [code_excerpts_index.json](evidence/D_architecture/code_excerpts_index.json) |
| Đường dẫn đã đọc thực tế | `<WSL_PROJECT_HOME>/{BanhmiTTS,scripts,baseline_v2_noclip,mrf_111_run_500ep,seq111_final,piper_original,quantized_models,paper_evidence/q05_*}`, `/tmp/{harvard_*,eval_*,seq_final_onnx,rtf_compare,dec_only_*,…}`, `piper/download/…`, `BanhmiTTS/docs/evidence/*`, `paper_revisions/thesis_ptq/*`, `.research-work/` (chỉ để loại trừ), transcript các phiên Claude Code | các file `copied_files`/`misc_copies…` trong `evidence/` |

### 2.2. Môi trường (kiểm tra **hiện nay**, không phải metadata ghi lúc chạy)

Nguồn: [environment_now.json](evidence/A_inventory/environment_now.json).

| Mục | Giá trị quan sát 04/10 |
|---|---|
| Máy | Intel Core i7-14700 (24 luồng logic, có `avx2`, `avx_vnni`), RAM 64 GB, 2× RTX 4070 Ti; WSL Ubuntu 24.04.4 |
| Env chính | Python 3.10.20; torch 2.13.0+cu130; PL 1.7.7; onnx 1.22.0; onnxruntime 1.23.2; numpy 1.26.4; scipy 1.15.3; openai-whisper 20250625; utmosv2 1.3.1.dev0; jiwer 4.0.0; `torch.get_num_threads()` mặc định = 12 |
| `piper_env` | Python 3.10.20; **torch 2.14.1+cu130**; PL 1.7.7; số thread mặc định = 12 |
| Scorer weights | Whisper `small.pt` `9ecf7799…e794`; UTMOSv2 `fold0_s42_best_model.pth` `c8149d98…5e57`. Đây là hash quan sát hiện tại (Q04), không được ghi kèm khi chấm. |

### 2.3. Checkpoint

Nguồn: [checkpoint_metadata_now.json](evidence/A_inventory/checkpoint_metadata_now.json) và [misc_copies_and_checkpoint_stat.json](evidence/A_inventory/misc_copies_and_checkpoint_stat.json). Hash khớp với Q01.

| Model | Đường dẫn | SHA-256 | Epoch / global_step (trong ckpt) | ctime |
|---|---|---|---|---|
| HiFi-GAN ResBlock2 | `<WSL_PROJECT_HOME>/baseline_v2_noclip/checkpoints/best-epoch=1489-val_loss_mel=19.7882.ckpt` | `921b0f6d7bfc82e4…` | 1489 / 1.168.160 | 2026-09-02T20:46Z |
| Parallel-IR | `<WSL_PROJECT_HOME>/mrf_111_run_500ep/checkpoints/best-epoch=1386-val_loss_mel=18.7681.ckpt` | `7ab302dcd80af326…` | 1386 / 1.087.408 | 2026-09-17T13:20Z |
| Sequential-IR | `<WSL_PROJECT_HOME>/seq111_final/checkpoints/best-epoch=1442-val_loss_mel=19.0136.ckpt` | `c4670261a9c775da…` | 1442 / 1.131.312 | 2026-09-29T23:19Z |
| Piper | `<WSL_PROJECT_HOME>/piper_original/lj-med_1000.ckpt`; bản tải về Windows giống byte | `dcf2449bdbdaad09…` (khớp LFS oid công bố) | 999 / 1.114.000 | 2026-09-30T19:38Z |

Hash đầy đủ nằm trong JSON. ctime của cả 4 checkpoint đều trước lần chạy Harvard (03/10 18:52Z), nên nội dung checkpoint lúc chạy chính là nội dung có hash hiện tại. Checkpoint không được sao chép vào gói (~3,4 GB); chỉ kèm hash, metadata và đường dẫn.

### 2.4. Chuỗi liên kết result → script → config → checkpoint → data → runtime

| Kết quả | Script/lệnh | Checkpoint/graph | Dữ liệu | Runtime | Liên kết chưa xác định |
|---|---|---|---|---|---|
| Harvard-720 ×4 (`harvard_*_results.json`) | `run_all.sh` → `synth_banhmi.py`/`synth_piper.py` → `score.py` (bản sao gốc trong `B_harvard/raw/scripts_original/`) | Đường dẫn ckpt có in trong log cho 3 model BanhmiTTS; với Piper, đường dẫn hardcode trong `synth_piper.py`. Hash ở §2.3. | `harvard_testset_720.json` ← `build_harvard_testset.py` ← `harvard.html` (cs.columbia.edu, tải 2026-10-03) | WSL CPU, PyTorch FP32; torch 2.13.0 (BanhmiTTS) / 2.14.1 (Piper) | Số thread lúc chạy không được ghi; có TensorBoard chạy nền (không tải); source BanhmiTTS lúc chạy = cây WSL ngày 03/10, không có hash riêng |
| LJ-500 PyTorch (baseline, SEQ, Piper) | `eval_*_seeded_500.py`, `piper_infer_batch.py` + `score_piper_original.py` | Hardcode trong script | `eval_true_testset_500_baseline.json` | PyTorch CPU | UTMOSv2 nhận float waveform; WER đọc WAV PCM16 |
| LJ-500 PyTorch MRF-1386 | `eval_mrf_final1500_500.py` | ckpt 1386 | **`eval_true_testset_500_mrf.json`** (chỉ 21/500 câu chung) | PyTorch CPU | — |
| ONNX lịch sử (baseline/MRF FP32, PTQ) | `eval_onnx_ptq_500.py` (2 intra-op, 1 inter-op) | Đường dẫn graph có trong log. Theo Q05: graph baseline khớp 422/422 tensor với ckpt 1489; **graph MRF khớp 0/478 tensor với ckpt 1386** | Manifest riêng từng model | ORT CPU | Checkpoint nguồn không được ghi trong kết quả; lệnh export lịch sử của baseline/MRF không được lưu |
| ONNX SEQ FP32/INT8 (30/09) | Transcript 2026-09-30T12:44:50Z: `export_onnx_fixed.py <ckpt 1442>`; 12:45:01Z: `ptq_quantize.py … --op-types Conv,ConvTranspose,MatMul --calibration-sentences 60`; eval 12:45:31Z và 13:41:56Z | `/tmp/seq_final_onnx/*`; Q05 xác nhận 454 tensor khớp ckpt 1442 | Manifest baseline | ORT CPU, 2 thread | Lệnh chỉ có trong transcript, không có log export |
| ONNX FP32 Q05 (04/10) | `run_q05_model.py` | Metadata ONNX ghi hash ckpt + hash source manifest | 6 câu calibration để kiểm parity | ORT 1.23.2, 2/1 thread | Chưa có đánh giá chất lượng/tốc độ trên các graph này |

---

## 3. Mục 2: Harvard-720

### 3.1. File và schema

Bản sao byte gốc: [B_harvard/raw/](evidence/B_harvard/raw/). WAV: [wav_inventory.csv](evidence/B_harvard/wav_inventory.csv), 2.880 dòng, có path, SHA-256, sr, kênh, số frame. WAV **không** được chép vào gói (~308 MB); vẫn nằm ở `<WSL_PROJECT_HOME>/quantized_models/harvard_{model}/` và trong archive Q01.

| File | Trường | Ý nghĩa |
|---|---|---|
| `harvard_testset_720.json` | `idx, list, text, phonemes, phoneme_ids` | 720 câu Harvard (72 list × 10), phonemize bằng `banhmi_phonemize` (espeak en-us); kiểm khớp 50/50 với ID đã lưu của LJ-500 trước khi dùng |
| `harvard_{m}_synth.json` | `idx, text, wav_path, synth_time_s, audio_duration_s, rtf` | Đầu ra bước tổng hợp |
| `harvard_{m}_results.json` | Các trường trên + `asr_text, utmosv2, wer` (không còn `wav_path`) | `wer` là tỷ lệ theo từng câu (0–1, có thể >1) |

Kiểm tra ([harvard_integrity_checks.json](evidence/B_harvard/harvard_integrity_checks.json)) bằng cách so từng phần tử các list `idx`/`text` giữa 4 file results, 4 file synth và manifest:

- `idx` đúng thứ tự 0..719; text khớp thứ tự với manifest ở cả 4 model.
- Giá trị timing trong results giống hệt synth.
- 0 giá trị UTMOS/WER/RTF không hữu hạn; 0 audio dưới 0,2 s; 0 transcript ASR rỗng.
- Thời lượng WAV khớp JSON; mọi WAV là 22.050 Hz, mono, PCM16.
- Log không có Traceback. **Không có câu nào bị lỗi** nên không phải xử lý loại bỏ.

### 3.2. Protocol thực tế (đọc từ script gốc)

| Mục | BanhmiTTS (`synth_banhmi.py`) | Piper (`synth_piper.py`) |
|---|---|---|
| Model | `VitsModel.load_from_checkpoint`, CPU, FP32, `dec.remove_weight_norm()` | `piper_train` gốc, CPU, FP32, `dec.remove_weight_norm()` |
| Inference | `infer(noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8)`, batch 1 | `model(text, lengths, [0.667,1.0,0.8])`, batch 1 |
| Randomness | `torch.manual_seed(1234)` trước **mỗi** câu | Giống bên trái |
| Timer | `perf_counter` bao quanh **chỉ** lời gọi infer, gồm enc_p, dp, flow, dec; không gồm load model, phonemize, ghi WAV | Giống bên trái |
| Mẫu số RTF | `len(wave)/22050` của chính output đó | Giống bên trái |
| Warm-up / lặp | Không warm-up; 1 lần đo mỗi câu | Giống bên trái |
| Tổng hợp | Trung bình RTF từng câu; script tính lại cũng báo ratio-of-sums | — |
| Threads | Không đặt; số thread lúc chạy không ghi (mặc định quan sát hôm nay: 12) | Giống bên trái |

**Scorer** (`score.py`):

- **WER:**
  - Whisper `small`, CPU, `language="en"`, `fp16=False`, truyền **đường dẫn WAV** (Whisper tự resample 16 kHz qua ffmpeg).
  - Decode mặc định, có temperature fallback; RNG không reset theo câu.
  - Chuẩn hóa: `ToLowerCase → RemovePunctuation → RemoveMultipleSpaces → Strip → ReduceToListOfListOfWords`.
  - Báo **macro WER** (trung bình WER từng câu, dạng tỷ lệ).
- **UTMOSv2:**
  - `create_model(pretrained=True)` (`fusion_stage3`, fold 0, s42).
  - Đọc WAV PCM16 bằng soundfile float32 rồi gọi `predict(data, sr=22050)`; hàm tự resample 16 kHz.
  - Crop ngẫu nhiên **không seed**; 1 lượt chấm.
  - Đây là **điểm dự đoán, không phải human MOS**.
- Kiểm lại WER từ transcript đã lưu: sai lệch tối đa giữa giá trị lưu và tính lại = 0 ([harvard_stats_recomputed.json](evidence/B_harvard/recomputed/harvard_stats_recomputed.json), `max_abs_diff_stored_vs_recomputed_wer`).

### 3.3. Kết quả tính lại (`recomputed_this_round`)

Script: [recompute_B_stats.py](scripts/recompute_B_stats.py). Đầu ra: [harvard_stats_recomputed.md](evidence/B_harvard/recomputed/harvard_stats_recomputed.md) và [.json](evidence/B_harvard/recomputed/harvard_stats_recomputed.json).

| Model | Mean RTF | Ratio-of-sums RTF | Mean UTMOSv2 | Macro WER | Corpus WER (expl.) | Câu WER=0 |
|---|---:|---:|---:|---:|---:|---:|
| HiFi-GAN ResBlock2 | 0.04177 | 0.04154 | 3.0260 | 0.1476 | 0.1437 | 318 |
| Parallel-IR | 0.03388 | 0.03373 | 3.5680 | 0.1073 | 0.1033 | 381 |
| Sequential-IR | 0.03081 | 0.03058 | 3.5443 | 0.1222 | 0.1186 | 351 |
| Piper | 0.03916 | 0.03897 | 3.4507 | 0.1747 | 0.1696 | 228 |

| Cặp (x − y) | Metric | Chênh trung bình | Wilcoxon p (đã báo) | Holm p (expl.) | Hiệu bằng 0 bị loại | Bootstrap ghép cặp 95% (expl.) |
|---|---|---:|---:|---:|---:|---|
| SEQ − MRF | RTF | −0.00308 | 7.0e-37 | 7.0e-36 | 0 | [−0.00357, −0.00257] |
| SEQ − MRF | UTMOSv2 | −0.0237 | 0.0471 | 0.0943 | 0 | [−0.0458, −0.0010] |
| SEQ − MRF | WER | +0.0149 | 0.0523 | 0.0943 | **323** | [+0.0019, +0.0277] |
| SEQ − Piper | RTF | −0.00835 | 8.1e-98 | 1.1e-96 | 0 | [−0.00886, −0.00783] |
| SEQ − Piper | UTMOSv2 | +0.0936 | 2.9e-13 | 1.7e-12 | 0 | [+0.070, +0.117] |
| SEQ − Piper | WER | −0.0525 | 4.7e-11 | 2.4e-10 | 248 | [−0.068, −0.037] |
| SEQ − baseline | RTF / UTMOSv2 / WER | −0.0110 / +0.518 / −0.0255 | 8.1e-105 / 3.8e-108 / 1.2e-3 | — | — | xem file |

**Đối chiếu các con số trong ghi chú:**

- **"Nhanh hơn 9%"**: đó là **mức giảm RTF** = 1 − 0.03081/0.03388 = **9,08%**. Mẫu số là RTF trung bình của Parallel-IR. Nếu tính theo throughput (0.03388/0.03081 − 1) thì là **9,98%**; theo ratio-of-sums là 9,34%.
- **"Nhanh hơn 21%"** so với Piper: giảm RTF **21,33%**; throughput tăng **27,1%**.
- **"WER thấp hơn 0,05"**: là **macro WER dạng tỷ lệ**, tức −0,0525 = **5,25 điểm phần trăm tuyệt đối**, không phải giảm 5% tương đối.
- **Kiểm định đã báo** là `scipy.stats.wilcoxon(x, y)` với mặc định scipy 1.15.3: two-sided, `zero_method="wilcox"` (loại các hiệu số bằng 0), không hiệu chỉnh liên tục, `method="auto"`. Đơn vị ghép cặp là câu (n=720). Kèm `mannwhitneyu` two-sided; phép này **không phù hợp** cho dữ liệu ghép cặp và chỉ được giữ để đối chiếu. Không hiệu chỉnh đa phép so sánh: 15 phép được báo cùng lúc; Holm ở bảng trên là exploratory. Không có giá trị thiếu.
- **Nguồn biến thiên được đo:** chỉ biến thiên **giữa câu**. Mỗi câu có 1 lần sinh (seed 1234), 1 lần chấm và 1 lần đo thời gian; mỗi cấu hình có 1 lần train. **Chưa đo** được biến thiên giữa các lần benchmark, giữa các lượt crop của scorer, hay giữa các lần train.
- **Không** kết luận "chất lượng tương đương": không có margin định trước. Hai ước lượng điểm đều nghiêng về Parallel-IR, và CI exploratory của WER không chứa 0.

---

## 4. Mục 3: dataset, split và training

### 4.1. Dataset

- **Nguồn:** LJSpeech, `dataset.jsonl` gồm 13.100 dòng, SHA-256 `f4a72ae0…2295b`.
- **Bất biến từ 28/07:** birth/ctime của file này và ctime của cả 26.200 file cache `.pt`/`.spec.pt` đều trong ngày 28/07, trước run đầu tiên (Q03 §10.3, [FILE_TIMESTAMPS.json](evidence/C_data_split/prior_round_q03/FILE_TIMESTAMPS.json)).
- **Tiền xử lý:**
  - 22.050 Hz; STFT dùng FFT/window 1024, hop 256; linear spectrogram 513 bin.
  - Phonemize bằng espeak en-us qua `banhmi_phonemize`, có chèn blank token.
  - `segment_size` 8192.

### 4.2. Split

| Model | Split | Bằng chứng | Mức |
|---|---|---|---|
| HiFi-GAN ResBlock2 | Canonical 12.500/100/500 (`canonical_split.json`, SHA `e678cf43…`) | `random_split` dùng RNG chung, seed 1234. Log TensorBoard của cả 3 phiên ghi đúng `val[0..4]` của canonical. | Tái dựng + đối chiếu log ghi lúc train |
| Parallel-IR | Split khác. Tái dựng một lần (script lưu kèm), SHA `f002d251…` | Khớp đúng thứ tự với tag TensorBoard của 3 phiên **và** với test manifest tạo 2026-09-15 | Reconstructed and corroborated |
| Sequential-IR | Canonical | Log ghi "Loaded canonical…" ở cả 3 phiên; birth=ctime=mtime của canonical = 2026-09-19T21:55Z, trước phiên SEQ đầu (23/09) | Đọc trực tiếp + timestamp kernel |
| Piper | **Không biết.** hparams: `validation_split=0.05`, `num_test_examples=0`, `max_phoneme_ids=400` | Chỉ có hparams trong checkpoint | — |

**Độ trùng 95,4%** ([data_split_checks_recomputed.json](evidence/C_data_split/data_split_checks_recomputed.json), `split_overlap`):

- Công thức: |train canonical ∩ train MRF tái dựng| / 12.500 = **11.930/12.500 = 0,9544**.
- Canonical test nằm trong train/val/test của MRF: **473/6/21**.
- MRF test nằm trong train/val/test của canonical: 478/1/21.

### 4.3. Resume và chọn checkpoint

- **Lịch resume:** 9 phiên. Baseline resume từ `best-epoch=89` và `best-epoch=870`; MRF và SEQ resume từ `last.ckpt`. Log đều ghi `Restored all states` (Q03 §5, §10–11).
- **Hash checkpoint dùng để resume:** **không khôi phục được**. Đã tìm trên WSL và Windows nhưng không thấy; xem `WSL_/WINDOWS_CHECKPOINT_SEARCH.json`.
- **Chọn checkpoint:** `ModelCheckpoint(monitor="val_loss_mel", mode="min")`, đọc trực tiếp trong checkpoint, cho ra 1489/1386/1442. `val_loss_mel` là mel L1 ×45, **không** phải UTMOS/WER. Piper có `monitor=None`, tức là checkpoint ở mốc epoch, không chọn theo validation.

### 4.4. Harvard có từng được dùng để chọn cấu hình không?

**Không, trong phạm vi hồ sơ đã kiểm** (J-E008, PARTIAL):

- Lần đầu Harvard xuất hiện trong transcript: **2026-10-03T18:50:53Z**. Mọi file Harvard đều có ngày 03–04/10.
- Callback chọn checkpoint đã kết thúc trước đó: 09-02, 09-17 và 09-29.
- Các graph PTQ lịch sử có ngày 09-07 đến 09-30.
- Ngoài phạm vi các phiên đã kiểm: **UNKNOWN**.
- **Lưu ý:** kết quả Harvard **đã ảnh hưởng** đến các đề xuất claim Q02 trong cuộc trò chuyện ngày 04/10.

### 4.5. Kiểm trùng Harvard với dữ liệu huấn luyện

Đối chiếu với 13.100 transcript trong `dataset.jsonl`:

- Trùng nguyên câu: **0**, với cả hai cách chuẩn hóa (alnum-lower và chuẩn hóa WER).
- Câu chung một cụm 6 từ: **0**. Chung cụm 5 từ: 7 câu; chung cụm 4 từ: 37 câu (cụm thông dụng, ví dụ "on the back of the").
- Trùng chuỗi phoneme ID: **0**. Trùng nội bộ trong Harvard: **0**.
- **Chưa kiểm:**
  - transcript huấn luyện của chính Piper;
  - độ tương đồng ở mức audio;
  - dữ liệu train của scorer: Whisper/UTMOSv2 có thể đã gặp các câu Harvard, một bias tác động lên mọi model.

Non-overlap ở mức text **không** chứng minh rằng mọi model chưa từng thấy dữ liệu liên quan.

### 4.6. Bảng cấu hình so sánh

Nguồn: hparams trong checkpoint ([data_split_checks_recomputed.json](evidence/C_data_split/data_split_checks_recomputed.json) `hparams_from_checkpoints`), code ([code_excerpts.md](evidence/D_architecture/code_excerpts.md)) và diff source khôi phục.

| Mục | HiFi-GAN ResBlock2 | Parallel-IR | Sequential-IR | Piper | Trạng thái |
|---|---|---|---|---|---|
| Text encoder | 6 lớp, 2 head, hidden 192, FFN 768 | = | = | = | Đã xác minh (hparams + schema) |
| Duration predictor | SDP flow (NLL) + duration discriminator | = | = | SDP, không có duration discriminator | Đã xác minh |
| Flow | 4× TransformerCoupling + Flip, kernel 5, WN 4 lớp | = | = | 4× ResidualCoupling (không attention) | Đã xác minh |
| MAS | Noise 0,01, decay 2e-6/step (giảm nhanh gấp 2 do `global_step` của PL 1.7, xem báo cáo Piper/VITS2) | = | = | Không noise | Code |
| Decoder | ResBlock2 dense 3/5/7 | 3 nhánh IR depthwise 3/5/7, lấy trung bình | 2 khối IR depthwise k3 nối tiếp | ResBlock2 | Đã xác minh |
| Activation | LeakyReLU 0,1 (kể cả trước `conv_post`) | SnakeBeta | SnakeBeta | LeakyReLU 0,1; **0,01 trước `conv_post`** | Đã xác minh (code) |
| Discriminator | MPD (+scale) + DurDisc | + MRD | + MRD | MPD (+scale) | Đã xác minh |
| Loss | adv + FM + mel×45 + KL + dur + dur-adv | + MRD adv/FM (hệ số 1,0) | = MRF | Không có dur-adv | Code |
| Optimizer | AdamW (0,8; 0,99), eps 1e-9, wd 0,01, lr 2e-4, ExponentialLR 0,999875 | = | = | = (fused=None) | Đã xác minh |
| Precision / batch / thiết bị | bf16 / 16×2 GPU DDP | = | = | fp32 / 22 / 1 GPU | Đã xác minh |
| Seed | 1234 | 1234 | 1234 | 1234 | hparams |
| Gradient clipping | **null (v0) → 1,0 (v1, v2)** | null | null | null | Đã xác minh |
| Clamp `dp` | **`_EXP_CLAMP_MAX` 30 → 10 và thêm clamp `logw` khi infer, trước khi resume từ epoch 870** | 10 (code lúc đó) | 10 | — | Diff khôi phục + timeline (J-E013) |
| Khác biệt code training | — | — | **Bỏ update khi gradient không hữu hạn; kiểm SDP quyết định checkpoint có được làm "best"; sửa duration_predictor; loader canonical** | — | Diff khôi phục (J-E014) |
| Split | Canonical | Riêng (95,4% train chung) | Canonical | Không biết | §4.2 |
| Thời lượng / chọn ckpt | 1500 epoch / best val_loss_mel 1489 | 1500 / 1386 | 1500 / 1442 | 1000 / epoch 999, không monitor | Đã xác minh |
| `num_workers` | 8 | 1 | 1 | 1 | hparams |

**Confounder** cho mọi so sánh giữa các model:

- Baseline khác MRF/SEQ ở activation, MRD, gradient clipping và clamp `dp` từ epoch 870.
- MRF khác SEQ ở split và ở code ổn định số/chọn checkpoint.
- Piper khác ở rất nhiều yếu tố.

→ Chỉ được diễn giải là **so sánh cấp hệ thống**.

---

## 5. Mục 4: ONNX và PTQ

### 5.1. Danh mục artifact

Nguồn: [onnx_inventory_inspected_now.json](evidence/E_onnx_ptq/onnx_inventory_inspected_now.json), gồm 51 graph, đọc bằng `onnx.load` và chỉ đếm node, không chạy inference. Mọi graph đều opset 18 và có 2 node `RandomNormalLike`, trừ graph validation và decoder-only.

| Nhãn | Model | Precision | Đường dẫn | SHA-256 (đầu) | Checkpoint nguồn | Export / commit | Trạng thái |
|---|---|---|---|---|---|---|---|
| final (Q05) | baseline / MRF / SEQ / Piper | FP32 | `<WSL_PROJECT_HOME>/paper_evidence/q05_2026-10-04T000830Z/{m}/{m}_fp32.onnx` | xem JSON (`fdfda3ac…`, `14c9f547…`, `4052629a…`, `2c87a6c3…` theo Q05) | Ghi trong `metadata_props` (hash ckpt, epoch, hash source manifest) | `run_q05_model.py`, opset 18, TorchScript exporter, output `[B,1,samples]` động | Đã định danh; **chưa** có đánh giá chất lượng/tốc độ |
| historical | baseline | FP32 / INT8 | `ptq_sweep/baseline_fp32.onnx`, `baseline_flow_enc_p_dp.onnx` (09-07) | JSON | 422/422 tensor khớp ckpt 1489 (Q05) | Lệnh export không lưu | Bảng lịch sử |
| historical | MRF | FP32 / INT8 | `ptq_sweep/mrf_fp32.onnx`, `mrf_flow_enc_p_dp.onnx` (09-15) | JSON | **0/478 tensor khớp ckpt 1386** | Không lưu | **Không gán cho MRF-1386** |
| historical | SEQ | FP32 / INT8 | `/tmp/seq_final_onnx/seq_final_{fp32,int8}.onnx` (09-30) | JSON | 454 tensor khớp ckpt 1442; lệnh trong transcript | `/tmp/export_onnx_fixed.py` (bản sao trong gói) | Bảng lịch sử |
| exploratory | baseline/SEQ (ckpt trước SEQ final) | FP32 / INT8 | `/tmp/rtf_compare/*` (09-21) | JSON | Không phải ckpt 1442 (trước phiên SEQ cuối) | — | Không dùng |
| exploratory | decoder-only (trọng số chưa train) | FP32 | `/tmp/dec_only_*.onnx` | JSON | Không có ckpt | `export_and_bench_dec_only.py` | Thăm dò Q10 |

### 5.2. Parity PyTorch–ONNX

Chỉ có cho các graph final Q05; đây là `prior_round_agent_output`, **chưa chạy lại** trong lượt này:

- Mỗi model 10 ca: 6 độ dài từ 43 đến 359 phoneme ID, 1 ca đổi seed, 2 ca đổi length scale, 1 ca batch 2.
- Nhiễu được tiêm từ PyTorch vào graph kiểm.
- Ngưỡng định trước: NRMSE ≤ 1e-3 và max-abs ≤ 5e-3. Max NRMSE quan sát: 9,2e-5 (baseline), 1,1e-4 (MRF), 4,4e-5 (SEQ), 7,1e-5 (Piper).
- **Giới hạn:** chỉ áp dụng cho các đầu vào và môi trường ORT 1.23.2 đã thử. ORT không lặp lại được giá trị nhiễu trong cùng một session.

Xem [prior_round_q05/](evidence/E_onnx_ptq/prior_round_q05/).

### 5.3. PTQ lịch sử

- **Script:** [`ptq_quantize.py`](evidence/E_onnx_ptq/scripts_and_logs/ptq_quantize.py). Cấu hình: `quantize_static`, **QOperator**, weight QInt8 **per-channel**, activation QUInt8. Node được chọn theo tiền tố tên `/{submodule}/`; luôn loại `/MatMul` và `/MatMul_1` (phép nhân alignment).
- **Bằng chứng trên graph** (đã xác minh tiền tố `/flow/`, `/enc_p/`, `/dp/`, `/dec/`):

| Graph INT8 | QLinearConv enc_p / dp / flow | QLinearMatMul | Op lượng tử dưới `/dec/` | Conv FP32 còn trong `/dec/` |
|---|---|---:|---|---|
| `baseline_flow_enc_p_dp` | 37 / 32 / 64 | 0 | **0** | 20 Conv + 3 ConvTranspose |
| `mrf_flow_enc_p_dp` | 37 / 32 / 64 | 0 | **0** | 29 Conv + 3 ConvTranspose |
| `seq_final_int8` | 37 / 32 / 64 | **40** | **0** | 20 Conv + 3 ConvTranspose |

  → Decoder giữ FP32 ở cả 3 graph. **Nhưng scope INT8 của SEQ khác** baseline/MRF, vì op-types có thêm MatMul.

- **Calibration 60 câu**, tái lập từ đúng quy tắc của script: `random.Random(42).sample` trên toàn bộ 13.100 dòng ([historical_calibration60_recomputed.json](evidence/E_onnx_ptq/historical_calibration60_recomputed.json)):
  - Canonical: 57 train / 0 val / **3 test**, và 3 câu này cũng thuộc manifest LJ-500 baseline.
  - MRF: 57 train / **2 val** / **1 test**.
  - Harvard: 0.
  - Tiền xử lý: đưa thẳng phoneme ID với scales `[0.667, 1.0, 0.8]`.
- **Chọn scope:** sweep ngày 09-07 (baseline) và 09-15 (MRF) dùng bộ chọn rẻ `eval_wer_cheap.py`. Bộ chọn này **có lỗi sample rate** ở dòng 71: mảng 22,05 kHz được đưa vào Whisper như 16 kHz. Nó còn dùng `base.en`, chuẩn hóa khác, và gán WER=1 khi lỗi (round7 audit).
  - **Phạm vi lỗi:** chỉ bước chọn scope. Bộ đánh giá 500 câu truyền đường dẫn WAV nên không bị.
  - Chưa có lần chạy lại nào; tiêu chí chấp nhận chưa được định trước.
- **Đối chiếu RTF lịch sử MRF 0,0416 và baseline 0,0388:** tính lại khớp (RTF ONNX FP32, 2 thread, mỗi model dùng manifest riêng). Nhưng graph MRF có nguồn không xác định, nên không dùng làm bằng chứng cho Parallel-IR-1386.
- **Thứ hạng tốc độ SEQ qua PyTorch → ONNX FP32 → INT8:** **chưa có bằng chứng final**. Số lịch sử SEQ: 0,0346 (FP32) và 0,0289 (INT8, scope có MatMul), trên manifest baseline.
- PTQ chỉ áp dụng cho backbone, decoder vẫn FP32, nên **không** chứng minh decoder chịu được INT8.
- **Profiling thực thi INT8:** không có. Hai file `onnxruntime_profile__2026-09-19_*.json` không xác định được thuộc model nào.

---

## 6. Mục 5: thiết bị và nghe thử

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Intel N150 | **NOT AVAILABLE**: không có log đo | Chỉ có tài liệu kế hoạch (`CHECKLIST_SUA_LUAN_VAN.md` C06: "4 điều kiện × 2 thiết bị = 8 ô… chưa đo"). Không coi ma trận dự kiến là kết quả. |
| Raspberry Pi 5 | **NOT AVAILABLE** | Như trên |
| Desktop (i7-14700, WSL) | Có đo lịch sử: Harvard/LJ-500 PyTorch, ORT 2 thread, `bench_vnni_*` (3 warm-up + 15 lần, 5 câu) | Không đại diện thiết bị edge; không có peak memory; không có điều kiện nguồn điện/tản nhiệt/tải nền |
| Profiling decoder theo T | Exploratory: decoder **chưa train**, kết quả chỉ in ra console | [exploratory_desktop_profiling/](evidence/F_device_listening/exploratory_desktop_profiling/) |
| Nghe thử / MOS | **NOT AVAILABLE**: không tìm thấy protocol, người tham gia, phản hồi hay phê duyệt | Không thay thế bằng UTMOSv2 |

---

## 7. Kiến trúc, số tham số/FLOPs và normalizing flow

Code kèm số dòng và SHA-256 file: [code_excerpts.md](evidence/D_architecture/code_excerpts.md). Đây là code hiện tại, không phải code lúc train.

- **Decoder baseline so với Piper:**
  - Cùng tham số (1.662.976), cùng shape, cùng MAC.
  - **Khác** activation trước `conv_post`: BanhmiTTS dùng `LeakyReLUActivation._SLOPE = 0.1` (`normalization.py:31`, gọi ở `generator.py:168`); Piper dùng `F.leaky_relu(x)` slope mặc định 0,01 (`models.py:364`).
  - → **Không** giống hệt về phép tính forward (J-E018 CONTRADICTED).
- **Parallel-IR** (`generator.py:101–131`, forward `:159–161`): mỗi tầng có 3 `ResBlockInverted` với kernel 3/5/7, song song, lấy trung bình. Expansion = 1.
- **Sequential-IR** (`generator.py:74–99`, forward `:156–158`): mỗi tầng có `mb_blocks_per_stage=2` khối `ResBlockInverted` nối tiếp, kernel 3, expansion 1 (hparams `mb_expansion=[1,1,1]`). `ResBlockInverted` (`resblocks.py:107–`) gồm: pointwise expand, depthwise (groups = hidden), pointwise project khởi tạo zero, cộng residual.
- **Upsample:** stride 8/8/4, kernel 16/16/8, channel 256 → 128/64/32 (hparams; `generator.py:58–73`).
- **SnakeBeta / MRD:**
  - `activation_cls = SnakeBeta if resblock in ("mb","mrf")` (`generator.py:22, :56`).
  - MRD được dựng khi resblock thuộc nhóm inverted (`training.py:203–206`). **Chỉ dùng lúc train.**
  - Cùng nhóm chỉ dùng lúc train: posterior encoder `enc_q` (7.225.920 tham số), MPD/scale discriminator, duration discriminator.
- **Backbone dùng chung** (BanhmiTTS): `enc_p` 6.341.760, `dp` 1.317.168, `flow` 8.285.568 tham số.

**Số tham số và FLOPs** (`recomputed_this_round`; [recompute_D_counts.py](scripts/recompute_D_counts.py)):

| Model | Decoder params | GFLOPs decoder @T=86 (0,998 s) | GFLOPs / giây chính xác | Tổng inference (enc_p+dp+flow+dec) | Flow params |
|---|---:|---:|---:|---:|---:|
| HiFi-GAN ResBlock2 | 1.662.976 | 3.9016 | 3.9076 | 17.607.472 | 8.285.568 |
| Parallel-IR | 1.154.560 | 1.2306 | 1.2325 | 17.099.056 | 8.285.568 |
| Sequential-IR | 1.107.968 | 0.9846 | 0.9861 | 17.052.464 | 8.285.568 |
| Piper | 1.662.976 | 3.9016 | 3.9076 | 16.412.464 | 7.090.560 |

**Quy ước đếm:**

- Ranh giới: `model_g.*`, gộp weight-norm (bỏ `weight_g`).
- FLOPs = 2 × MAC của Conv1d/ConvTranspose1d; **không** tính activation, bias, tanh hay phép cộng.
- Batch 1, đầu vào latent `[1,192,T]`.
- T=86 là quy ước của script lịch sử (86 × 256 = 22.016 mẫu); cột "giây chính xác" = MAC mỗi frame × 22.050/256.
- Generator được nạp từ checkpoint thật và chạy forward một lần trên input ngẫu nhiên để đếm qua hook. Đây **không** phải benchmark, không đo thời gian.

**Kết quả đối chiếu với ghi chú:**

- SEQ 1,11M / MRF 1,15M / baseline 1,66M: ✔.
- 0,98 / 1,23 GFLOPs: ✔ theo quy ước trên.
- Decoder SEQ nhỏ hơn Piper **33,37%**; tổng tham số SEQ lớn hơn Piper **3,90%**.

**Normalizing flow** (`flow_block.py:14–122`, `synthesizer.py:126–134`):

- `model_g.flow` = 4 cặp {`TransformerCouplingLayer`, `Flip`}.
- Mỗi lớp coupling:
  - `pre` 1×1;
  - **self-attention 1 lớp, 2 head**, FFN 384, kernel 1, cộng residual, đặt **trước** WaveNet;
  - WaveNet 4 lớp, kernel 5, dilation 1;
  - `post` 1×1 khởi tạo zero, xuất **mean-only shift** (logs = 0).
- Khi inference, flow chạy chiều ngược: `flow(z_p, reverse=True)` (`synthesizer.py:314`).
- 8,29M tham số so với 7,09M của Piper. Flow của Piper dùng `ResidualCouplingLayer`, **không có attention**.
- Đây là backbone kiểu VITS2 theo bản cài đặt cộng đồng: **không** phải contribution mới, và **không** phải flow matching.

---

## 8. Bảng claim → nguồn → kết quả kiểm → giới hạn

Bản máy đọc được, có SHA-256 đầy đủ của từng nguồn: [claim-ledger.json](claim-ledger.json), schema `evidence-audit` v1. Trạng thái do AI ghi ngày 04/10; **chưa có xác nhận của tác giả**.

| ID | Assertion | Loại | Nguồn (path · locator) | Trạng thái | Đã kiểm / chưa kiểm | Phạm vi |
|---|---|---|---|---|---|---|
| J-E001 | Harvard-720: the four per-sentence result files (baseline, MRF, SEQ, Piper) contain 720 rows each with identical idx 0..719, text and order as harvard_testset_720.json; 0 non-finite UTMOSv2/WER/RTF, 0 audio <0.2 s, 0 empty ASR text; WAV durations match JSON. | result | `evidence/B_harvard/harvard_integrity_checks.json` · checks.* · `d2657c4c4c1d`<br>`evidence/B_harvard/raw/harvard_testset_720.json` · whole file · `bdaf43e33d3a` | **SUPPORTED** | Checked: IDs/text/order, finiteness, WAV headers vs JSON durations, log has no Traceback. Not checked: audio content quality beyond scores. | PyTorch CPU synthesis, Harvard-720, single draw per sentence. |
| J-E002 | Harvard-720 mean per-sentence RTF (PyTorch CPU): HiFi-GAN ResBlock2 0.04177, Parallel-IR 0.03388, Sequential-IR 0.03081, Piper 0.03916. | result | `evidence/B_harvard/recomputed/harvard_stats_recomputed.json` · per_model.*.mean_rtf_per_sentence · `932161a4295f`<br>`evidence/B_harvard/raw/scripts_original/synth_banhmi.py` · timer around model_g.infer only · `23b2548d2268`<br>`evidence/B_harvard/raw/scripts_original/synth_piper.py` · timer around model(...) only · `4f40fe06e3d6` | **SUPPORTED** | Recomputed from per-sentence JSON. Timer = infer call only (excludes model load, phonemization, WAV write); RTF = time/duration per sentence then mean. Not controlled: no warm-up, one timing pass, torch threads not recorded at run time (default 12 observed now), Piper ran under torch 2.14.1 vs 2.13.0, an idle TensorBoard server was running. | i7-14700, WSL Ubuntu 24.04, PyTorch CPU FP32; single run. |
| J-E003 | Sequential-IR is 9% faster than Parallel-IR on Harvard-720 (p=7e-37). | result | `evidence/B_harvard/recomputed/harvard_stats_recomputed.json` · paired_tests.seq_vs_mrf_rtf · `932161a4295f` | **PARTIAL** | 9.08% = RTF reduction 1-mean(SEQ)/mean(MRF); throughput gain 9.98%; ratio-of-sums reduction 9.34%. Wilcoxon (scipy default, two-sided, zero_method=wilcox) p=7.0e-37 replicates. Variation measured is between sentences only; benchmark repetition and training-run variance unmeasured; no warm-up. | PyTorch CPU only; not ONNX/edge. |
| J-E004 | On Harvard-720, Sequential-IR UTMOSv2 is lower than Parallel-IR by about 0.02 (p about 0.05). | result | `evidence/B_harvard/recomputed/harvard_stats_recomputed.json` · paired_tests.seq_vs_mrf_utmosv2 · `932161a4295f` | **SUPPORTED** | Mean diff -0.0237; Wilcoxon p=0.0471; MWU p=0.039; Holm over 15 tests p=0.094 (exploratory); paired bootstrap 95% [-0.0458, -0.0010] (exploratory). UTMOSv2 is a predicted score with unseeded random crops, one pass. | UTMOSv2 predicted quality, not human MOS. |
| J-E005 | On Harvard-720, the WER difference between Sequential-IR and Parallel-IR is not statistically significant. | result | `evidence/B_harvard/recomputed/harvard_stats_recomputed.json` · paired_tests.seq_vs_mrf_wer · `932161a4295f` | **PARTIAL** | Mean diff +0.0149 (WER fraction; SEQ worse). Wilcoxon p=0.0523 after dropping 323/720 zero differences; MWU p=0.080. Exploratory paired bootstrap CI [+0.0019, +0.0277] excludes 0. 'Not significant' is literally true for the reported test but does not imply equivalence. | Whisper-small macro WER. |
| J-E006 | Sequential-IR and Parallel-IR have comparable/equivalent speech quality. | inference | `evidence/B_harvard/recomputed/harvard_stats_recomputed.json` · paired_tests.seq_vs_mrf_* · `932161a4295f` | **UNSUPPORTED** | No equivalence/non-inferiority margin was pre-specified; both point estimates favour Parallel-IR; WER bootstrap CI excludes 0. | — |
| J-E007 | On Harvard-720, Sequential-IR vs Piper: 21% faster, UTMOSv2 +0.09, WER -0.05, all p<1e-10. | result | `evidence/B_harvard/recomputed/harvard_stats_recomputed.json` · paired_tests.seq_vs_piper_* · `932161a4295f` | **SUPPORTED** | 21.33% RTF reduction (27.1% throughput gain); UTMOSv2 +0.0936; WER -0.0525 as macro WER fraction = 5.25 percentage points absolute; Wilcoxon p = 8.1e-98 / 2.9e-13 / 4.7e-11. System-level comparison: different backbone, decoder, precision, data split, torch build. | PyTorch CPU; Piper published checkpoint. |
| J-E008 | Harvard-720 was not used to select checkpoints or the historical PTQ scope. | inference | `evidence/A_inventory/checkpoint_metadata_now.json` · *.stored_in_checkpoint.callbacks · `c08937674b43`<br>`evidence/B_harvard/harvard_integrity_checks.json` · copied[].source_mtime_utc · `d2657c4c4c1d` | **PARTIAL** | Earliest 'harvard' mention in the inspected transcripts 2026-10-03T18:50:53Z; all Harvard files dated 2026-10-03/04; checkpoint callbacks finished 2026-09-02/09-17/09-29; PTQ graphs dated 2026-09-07..09-30. Not covered: activity outside the inspected sessions. Harvard results DID inform the Q02 claim wording proposed in this conversation. | Inspected records only. |
| J-E009 | Harvard-720 sentences do not overlap the LJSpeech training transcripts. | result | `evidence/C_data_split/data_split_checks_recomputed.json` · exact_sentence_overlap_*, harvard_sentences_sharing_any_*gram, exact_phoneme_id_sequence_overlap · `83f654636cab` | **PARTIAL** | 0 exact matches under two normalizations; 0 shared 6-grams; 0 identical phoneme-ID sequences; 37 sentences share a 4-gram and 7 a 5-gram (common phrases). Only against the 13,100 transcripts in dataset.jsonl; Piper's own training transcripts, audio similarity and scorer training data unchecked. | Text-level only. |
| J-E010 | HiFi-GAN ResBlock2 and Sequential-IR were trained on the canonical 12,500/100/500 partition; Parallel-IR on a different random partition sharing 11,930/12,500 (95.44%) training utterances. | result | `evidence/C_data_split/data_split_checks_recomputed.json` · split_overlap · `83f654636cab`<br>`evidence/C_data_split/prior_round_q03/TENSORBOARD_VAL_HEADS.json` · distinct_tag_sequences · `269a28250a95`<br>`evidence/C_data_split/MRF_SPLIT_RECONSTRUCTED_VALIDATED.json` · checks · `aea18304bb06` | **SUPPORTED** | Canonical corroborated by TensorBoard validation heads (9 phases) and kernel timestamps; MRF split reconstructed and corroborated against logged validation heads and the 2026-09-15 test manifest. Not a contemporaneous full-manifest hash. | Training data membership. |
| J-E011 | Selected checkpoints are epoch 1489 (baseline), 1386 (MRF), 1442 (SEQ), chosen by ModelCheckpoint on val_loss_mel (min); Piper lj-med_1000 is epoch 999 with monitor=None. | result | `evidence/A_inventory/checkpoint_metadata_now.json` · *.stored_in_checkpoint.epoch / callbacks · `c08937674b43` | **SUPPORTED** | Epochs are zero-based metadata. Selection by validation mel L1 (x45), not by UTMOS/WER. | — |
| J-E012 | HiFi-GAN ResBlock2 baseline used gradient clipping 1.0 in its final phases; Parallel-IR and Sequential-IR used none. | result | `evidence/A_inventory/checkpoint_metadata_now.json` · baseline.hyper_parameters.gradient_clip_val · `c08937674b43`<br>`evidence/C_data_split/prior_round_q03/Q03_EVIDENCE.json` · phases · `208a16a60181` | **SUPPORTED** | Baseline v0 null, v1/v2 1.0 (hparams per session, Q03). | — |
| J-E013 | The baseline's SDP Log-flow exp clamp was changed from 30 to 10 (and an inference logw clamp added) before resuming from epoch 870. | method | `evidence/C_data_split/prior_round_q03/baseline_v0_to_baseline_v2_flows.py.diff` · hunk @@ -16,10 · `9ee5e7a0c44b`<br>`evidence/C_data_split/prior_round_q03/TRANSCRIPT_TIMELINE_TRAINING_WINDOWS.txt` · 2026-08-31T21:07-21:08 · `513ae8097dce` | **SUPPORTED** | Recovered editor source + transcript timeline + resume log line; no runtime source hash. | Baseline only. |
| J-E014 | Sequential-IR training code differs from Parallel-IR's: non-finite-gradient update skipping, SDP inference-health gating of best-checkpoint eligibility, duration-predictor edits, canonical split loader. | method | `evidence/C_data_split/prior_round_q03/mrf_v0_to_seq_v0_training.py.diff` · whole diff · `4fa58e1ed224` | **SUPPORTED** | Recovered source diff; SEQ logs contain skip messages (Q03). Confounder for any MRF-vs-SEQ causal attribution. | — |
| J-E015 | Decoder parameters: Sequential-IR 1.11M, Parallel-IR 1.15M, HiFi-GAN ResBlock2 1.66M. | result | `evidence/D_architecture/counts_banhmi_recomputed.json` · *.decoder_params_folded_state_dict · `e65450ba49e1` | **SUPPORTED** | Exact 1,107,968 / 1,154,560 / 1,662,976; weight-norm folded; equals live count after remove_weight_norm. | model_g.dec only. |
| J-E016 | Decoder compute: Sequential-IR 0.98 and Parallel-IR 1.23 GFLOPs per second of audio. | result | `evidence/D_architecture/counts_banhmi_recomputed.json` · *.decoder_gflops_T86 / per_exact_second · `e65450ba49e1` | **SUPPORTED** | Convention: Conv1d/ConvTranspose1d MACs x2 only (no activations/bias); T=86 frames (0.998 s): 0.9846 / 1.2306; per exact second 0.9861 / 1.2325. Baseline and Piper 3.90. | Decoder only, batch 1. |
| J-E017 | The Sequential-IR decoder is 33% smaller than Piper's; total inference parameters are 3.9% larger. | result | `evidence/D_architecture/counts_banhmi_recomputed.json` · SEQ · `e65450ba49e1`<br>`evidence/D_architecture/counts_piper_recomputed.json` · Piper · `e7d1f94740a7` | **SUPPORTED** | 1-1,107,968/1,662,976 = 33.37%; 17,052,464/16,412,464-1 = 3.90%; boundary enc_p+dp+flow+dec, enc_q and discriminators excluded. | — |
| J-E018 | The HiFi-GAN ResBlock2 baseline decoder is identical to Piper's decoder. | result | `evidence/D_architecture/code_excerpts.md` · normalization.py L25-38; piper models.py L364 · `e9d3a7408284`<br>`evidence/D_architecture/counts_piper_recomputed.json` · decoder · `e7d1f94740a7` | **CONTRADICTED** | Same parameter count/shapes and MACs, but the activation before conv_post differs: LeakyReLU slope 0.1 (BanhmiTTS) vs F.leaky_relu default 0.01 (Piper). | Forward semantics. |
| J-E019 | Parallel-IR uses three parallel inverted-residual branches (depthwise kernels 3/5/7) averaged per stage; Sequential-IR uses two serial blocks (kernel 3, expansion 1) per stage; upsampling strides 8/8/4, kernels 16/16/8, channels 128/64/32. | method | `evidence/D_architecture/code_excerpts.md` · generator.py L74-169 · `e9d3a7408284`<br>`evidence/A_inventory/checkpoint_metadata_now.json` · hyper_parameters · `c08937674b43` | **SUPPORTED** | Current code + checkpoint hparams + state shapes (Q03 §3). Current code is not a runtime hash of training source. | — |
| J-E020 | The normalizing flow has 4 TransformerCouplingLayer+Flip pairs, each with a 1-layer 2-head self-attention (residual) before a 4-layer WaveNet (kernel 5), mean-only shift, run in reverse at inference; 8.29M params vs Piper 7.09M (Piper flow has no attention). | method | `evidence/D_architecture/code_excerpts.md` · flow_block.py L14-122; synthesizer.py flow construction/infer · `e9d3a7408284`<br>`evidence/D_architecture/counts_piper_recomputed.json` · params_folded_by_module.model_g.flow · `e7d1f94740a7` | **SUPPORTED** | Backbone description; not a novel contribution and not flow matching. | — |
| J-E021 | Historical ONNX Runtime FP32 RTF: Parallel-IR 0.0416 vs HiFi-GAN ResBlock2 0.0388 (Parallel-IR slower). | result | `evidence/E_onnx_ptq/historical_results/eval_mrf_fp32_onnx_500_results.json` · mean rtf · `ce262b192ff2`<br>`evidence/E_onnx_ptq/scripts_and_logs/eval_mrf_fp32_onnx_500.log` · line 2 · `fdc8c8a33ae1`<br>`evidence/E_onnx_ptq/prior_round_q05/FP32_HANDOFF.json` · historical graph check · `62673445e1af` | **PARTIAL** | Numbers reproduce, but mrf_fp32.onnx does not match checkpoint 1386 (Q05: 0/478 tensors) and MRF used its own 500-sentence manifest (21/500 shared). Cannot be attributed to MRF-1386. | Historical; 2 threads. |
| J-E022 | Sequential-IR's speed advantage holds across PyTorch, ONNX FP32 and INT8. | inference | `evidence/E_onnx_ptq/onnx_inventory_inspected_now.json` · seq_final_int8 vs *_flow_enc_p_dp · `86ae3c7488b3` | **UNSUPPORTED** | No final runs on the Q05 graphs; historical MRF graph source unknown; historical INT8 scopes differ (SEQ adds 40 QLinearMatMul). | — |
| J-E023 | In the inspected INT8 graphs the decoder remains FP32. | result | `evidence/E_onnx_ptq/onnx_inventory_inspected_now.json` · any_quantized_op_under_dec · `86ae3c7488b3` | **SUPPORTED** | No quantized op under /dec/ in baseline/MRF flow_enc_p_dp and SEQ int8 graphs. Does not show INT8 robustness of the decoder. | Graph structure only. |
| J-E024 | The PTQ scope quantizes Conv layers of enc_p, dp and flow. | result | `evidence/E_onnx_ptq/onnx_inventory_inspected_now.json` · quantized_ops_by_prefix · `86ae3c7488b3`<br>`evidence/E_onnx_ptq/scripts_and_logs/ptq_quantize.py` · L64-118 · `392ca92c1b7e` | **PARTIAL** | Baseline/MRF historical: 133 QLinearConv (enc_p 37, dp 32, flow 64), no QLinearMatMul. SEQ final INT8 and the 09-21 rtf_compare graphs also have 40 QLinearMatMul (op-types Conv,ConvTranspose,MatMul). Static QOperator, int8 per-channel weights, uint8 activations. | Historical graphs. |
| J-E025 | The 60 PTQ calibration sentences were drawn from training data only. | result | `evidence/E_onnx_ptq/historical_calibration60_recomputed.json` · membership · `41154000e343` | **CONTRADICTED** | Recomputed from ptq_quantize.py's own rule (Random(42).sample over all dataset.jsonl rows): 3 in canonical/LJ-500-baseline test, 1 in MRF test manifest, 2 in MRF validation, 0 Harvard. | Historical PTQ. |
| J-E026 | The historical PTQ scope was selected with a validated evaluator. | result | `evidence/E_onnx_ptq/prior_round_audits/round7_evaluation_code.md` · Main findings 1-2 · `8757d5e848b1`<br>`evidence/E_onnx_ptq/scripts_and_logs/eval_wer_cheap.py` · line 71 · `29e88c096cfc` | **CONTRADICTED** | Cheap selector passed 22.05 kHz arrays to Whisper (expects 16 kHz), used base.en and different normalization; no rerun of the selection exists. | — |
| J-E027 | Q05 FP32 ONNX graphs match PyTorch within NRMSE<=1e-3 and max-abs<=5e-3 on 10 cases per model. | result | `evidence/E_onnx_ptq/prior_round_q05/FINAL_VERIFICATION.json` · checks · `2c229fc9d249`<br>`evidence/E_onnx_ptq/prior_round_q05/RUN_CONFIG.json` · thresholds · `e1e27d787834` | **PARTIAL** | Evidence produced by a prior round with saved NPZ in WSL; copied here but not re-executed in this round. | ORT 1.23.2 CPU, 2 threads, given inputs. |
| J-E028 | RTF/latency measurements on Intel N150 and Raspberry Pi 5. | result | — | **UNSUPPORTED** | NOT AVAILABLE: only planning documents and desktop i7-14700 measurements exist. | — |
| J-E029 | Human listening test results. | result | — | **UNSUPPORTED** | NOT AVAILABLE: no protocol, responses or approvals found. UTMOSv2 is not a substitute. | — |
| J-E030 | MRF_vs_SEQ_comparison_Q2.md §2 'PyTorch FP32' row for baseline and MRF. | author-note | `evidence/author_notes/MRF_vs_SEQ_comparison_Q2.md` · §2 table · `15b199751bbb`<br>`evidence/C_data_split/lj500_results_historical/eval_baseline_fp32_seeded_500_results.json` · means · `82324daf74dd`<br>`evidence/C_data_split/lj500_results_historical/eval_mrf_final1500_500_results.json` · means · `fa40e92fc8ec` | **CONTRADICTED** | Baseline RTF 0.0434 is PyTorch but UTMOSv2 3.132/WER 0.1230 are the ONNX values (PyTorch: 3.125/0.1230); MRF RTF 0.0370 is PyTorch-1386 but 3.537/0.1043 are from the unverified ONNX graph (PyTorch-1386: 3.651/0.0983) on the MRF manifest. | Historical LJ-500. |
| J-E031 | Sequential-IR has higher UTMOSv2 than Parallel-IR (LJSpeech-500 tables). | result | `evidence/C_data_split/lj500_results_historical/eval_seq111_fp32_seeded_500_results.json` · means · `c2115a4c9d13`<br>`evidence/C_data_split/lj500_results_historical/eval_mrf_final1500_500_results.json` · means · `fa40e92fc8ec`<br>`evidence/B_harvard/recomputed/harvard_stats_recomputed.json` · seq_vs_mrf_utmosv2 · `932161a4295f` | **CONTRADICTED** | LJ-500 rows use different manifests for MRF vs SEQ (21/500 shared); on the shared Harvard-720 set SEQ is lower (-0.024). | — |
| J-E032 | Piper has seen most of the canonical LJ-500 test sentences during training. | inference | `evidence/C_data_split/data_split_checks_recomputed.json` · hparams_from_checkpoints.Piper (validation_split, num_test_examples) · `83f654636cab` | **NOT_CHECKED** | Plausible from validation_split=0.05 and num_test_examples=0, but Piper's split is unknown; not measurable. | — |
| J-E033 | Scorer identity: Whisper small and UTMOSv2 fusion_stage3 fold0 s42 with the stated hashes were used for Harvard scoring. | result | `evidence/B_harvard/raw/scripts_original/score.py` · whisper.load_model('small'); utmosv2.create_model(pretrained=True) · `1bdd3994e689`<br>`evidence/A_inventory/environment_now.json` · main_env.packages · `e566c78fb17e` | **PARTIAL** | Script names the models; weights hashes (Whisper 9ecf7799..., UTMOSv2 c8149d98...) and package versions are current observations (Q04), not recorded at run time. | — |

---

## 9. Mâu thuẫn giữa ghi chú và nguồn gốc

| # | Ghi chú | Nguồn gốc cho thấy | Claim |
|---|---|---|---|
| 1 | `MRF_vs_SEQ_comparison_Q2.md` §2, dòng "PyTorch FP32" | Trộn nguồn: baseline lấy RTF PyTorch nhưng UTMOS/WER từ ONNX; MRF lấy RTF PyTorch-1386 nhưng UTMOS/WER từ graph ONNX không xác định, trên manifest MRF | J-E030 |
| 2 | "SEQ tốt hơn MRF về UTMOSv2" (cùng báo cáo §4–5) | Trên LJ-500 hai model dùng hai manifest khác nhau; trên Harvard chung, SEQ thấp hơn 0,024 | J-E031 |
| 3 | "Decoder baseline giống hệt Piper" (báo cáo Piper/VITS2 §4.2; câu trả lời chat ngày 04/10) | Khác slope LeakyReLU trước `conv_post` (0,1 so với 0,01) | J-E018 |
| 4 | Bảng ONNX/INT8 gán cho MRF-1386 | Graph `mrf_fp32.onnx` có 0/478 tensor khớp ckpt 1386 | J-E021 |
| 5 | "PTQ flow+enc_p+dp" như nhau cho mọi model (MRF_vs_SEQ §3) | SEQ INT8 có thêm 40 QLinearMatMul | J-E024 |
| 6 | Calibration "train-only" | Calibration lịch sử có 3 câu test-500 baseline/SEQ, 1 câu test MRF, 2 câu val MRF | J-E025 |
| 7 | "Ngoài decoder, MRF và SEQ giống hệt" (chat 04/10) | Hparams giống, nhưng code training khác: bỏ update khi gradient không hữu hạn, cổng SDP khi chọn ckpt; split khác | J-E014, J-E010 |
| 8 | "Chất lượng tương đương" (đề xuất Q02 trong chat) | Không có margin; ước lượng điểm nghiêng về MRF; CI bootstrap của WER không chứa 0 | J-E006 |
| 9 | "Lợi thế tốc độ SEQ giữ qua ONNX/INT8" (đề xuất Q02) | Chưa có run final; dữ liệu lịch sử có vấn đề ở mục 4–6 | J-E022 |
| 10 | Báo cáo Piper/VITS2 §5.1 nêu sai số log-mel bf16 0,0017/0,0077 | Phép đo chỉ có trong hội thoại, **không có file nguồn** trong gói | NOT_CHECKED |

---

## 10. Bằng chứng còn thiếu và việc tiếp theo, theo ưu tiên

1. **Đánh giá final trên graph Q05** cho 4 model, dùng protocol Q04 v0.2:
   - Harvard-720 và LJ-500 theo manifest riêng từng model.
   - Ghi số thread, warm-up, số lần lặp và hash WAV; seed cho UTMOS crop.
   - Cần trước bất kỳ claim nào về ONNX hoặc tốc độ triển khai.
2. **Q06: làm lại PTQ:**
   - Calibration v0.2 chỉ từ train (đã có manifest ứng viên); sửa bộ chọn có lỗi sample rate.
   - **Cùng op-types cho mọi model** (chốt có lượng tử MatMul hay không).
   - Tiêu chí chấp nhận định trước, chọn trên validation riêng.
3. **Chốt cách nói về chất lượng SEQ so với MRF:** hoặc định trước margin và làm phân tích non-inferiority trên dữ liệu mới, hoặc viết đúng hướng hiện có ("slightly lower predicted quality").
4. **Benchmark N150 / Pi 5** cho ma trận đã chọn, với cấu hình máy, runtime, thread, warm-up, nhiều lần lặp, peak memory.
5. **Nghe thử** (MOS/MUSHRA), có protocol, ẩn danh, ngẫu nhiên hóa thứ tự, consent.
6. **Q10:**
   - Profiling decoder với **trọng số đã train** theo T, kèm bộ đếm memory bandwidth, trước khi nêu cơ chế.
   - Nếu muốn claim nhân quả cho decoder: cần ablation kiểm soát activation, MRD, recipe và split.
7. **Hồ sơ run:** lưu hash scorer weights, `torch.get_num_threads()`, phiên bản thư viện vào từng file kết quả; đặt seed cho UTMOS.
8. **License và trích dẫn:**
   - License của checkpoint Piper: MODEL_CARD chỉ ghi license **dataset** (public domain).
   - Nguồn và license của Harvard sentences (IEEE 1969; trang cs.columbia.edu).
9. Nếu vẫn báo bảng LJ-500 lịch sử: ghi rõ manifest từng dòng, nhãn historical, và bỏ các dòng MRF ONNX.
10. Lưu một manifest split có hash **ngay khi** bắt đầu mỗi run mới.

---

## 11. Lệnh đã chạy, script và đầu ra

Tất cả script nằm trong [scripts/](scripts/). Các script không sửa file nguồn, chỉ ghi vào gói.

| Lệnh | Mục đích | Đầu ra |
|---|---|---|
| `wsl … <WSL_PROJECT_HOME>/env/bin/python scripts/collect_A_inventory.py evidence/A_inventory` | Đọc môi trường, phần cứng, metadata và hash checkpoint | `environment_now.json`, `checkpoint_metadata_now.json` |
| `… python -W ignore scripts/collect_B_harvard.py <pkg>` | Chép byte Harvard; băm 2.880 WAV; kiểm ID/text/thứ tự | `B_harvard/raw/*`, `wav_inventory.csv`, `harvard_integrity_checks.json` |
| `… scripts/recompute_B_stats.py <pkg>` | Tính lại mean, kiểm định đã báo, các phân tích exploratory | `B_harvard/recomputed/*` |
| `… scripts/collect_C_data_split.py <pkg>` | Chép split, metadata Piper; kiểm trùng Harvard–LJSpeech; tính độ trùng split; so hash Piper; bảng hparams | `C_data_split/*` |
| `cd <WSL_PROJECT_HOME>/BanhmiTTS && env/bin/python scripts/recompute_D_counts.py banhmi …`; `piper_env/bin/python … piper …` | Đếm tham số và FLOPs decoder (nạp ckpt, 1 forward ngẫu nhiên để đếm hook) | `D_architecture/counts_*_recomputed.json` |
| `python scripts/extract_code_excerpts.py <pkg>` (Windows) | Trích code kèm số dòng và hash | `D_architecture/code_excerpts.md`, `code_excerpts_index.json` |
| `… scripts/collect_E_onnx_ptq.py <pkg>` | Kiểm 51 graph ONNX (chỉ load, không chạy); chép script/log; tái lập calibration | `E_onnx_ptq/*` |
| `wsl … python3 scripts/collect_misc.py <pkg>` | Chép kết quả LJ-500, evidence lượt trước, ghi chú, diff, profiling; stat checkpoint | thư mục tương ứng + `A_inventory/misc_copies_and_checkpoint_stat.json` |
| `python scripts/build_ledger_and_manifest.py` | Dựng claim ledger, bảng claim, manifest | `claim-ledger.json`, `evidence/claim_table.md`, `evidence-manifest.csv` |
| Truy vấn chỉ đọc (Bash/Python trực tiếp) | `git rev-parse/status`; `find` ONNX; grep tìm thiết bị/nghe thử; tìm lần đầu "harvard" và lệnh ONNX SEQ trong transcript; so hash ckpt với Q01; so PyTorch/ONNX baseline từng câu; quét secrets | Kết quả ghi trong tài liệu này |

**Không chạy:** train, export, quantize, benchmark thời gian, TTS, ASR, UTMOS, hay bất kỳ dịch vụ ngoài nào. **Không ghi đè** kết quả gốc.

---

## 12. Danh mục bàn giao

| Đường dẫn trong gói | Nội dung |
|---|---|
| `JOURNAL_EVIDENCE_CONTEXT.md` | Tài liệu này |
| `evidence-manifest.csv` | Mọi file: SHA-256, kích thước, đường dẫn gốc, loại nguồn, claim hỗ trợ |
| `claim-ledger.json`, `evidence/claim_table.md` | 33 claim (15 SUPPORTED, 8 PARTIAL, 4 UNSUPPORTED, 5 CONTRADICTED, 1 NOT_CHECKED) |
| `evidence/A_inventory/` | Môi trường, metadata checkpoint, stat |
| `evidence/B_harvard/` | Raw Harvard, script gốc, kiểm toàn vẹn, WAV inventory, thống kê tính lại |
| `evidence/C_data_split/` | Split, metadata Piper, kiểm trùng, hparams, kết quả LJ-500 lịch sử, evidence Q03 lượt trước, diff source |
| `evidence/D_architecture/` | Code kèm số dòng, đếm tham số/FLOPs |
| `evidence/E_onnx_ptq/` | Inventory ONNX, script/log PTQ và eval, kết quả lịch sử, calibration tái lập, evidence Q05 và audit round 5–7 |
| `evidence/F_device_listening/` | Profiling thăm dò (không phải bằng chứng thiết bị) |
| `evidence/author_notes/` | Q03/Q04/Q05, checklist, các báo cáo Markdown (chỉ dùng định hướng) |
| `scripts/` | Mọi script thu thập/tính lại của lượt này |

**Không nằm trong gói** (chỉ có hash và đường dẫn): 4 checkpoint (~3,4 GB), các graph ONNX (~3 GB), 2.880 WAV Harvard (~308 MB) cùng WAV LJ-500, file TensorBoard event (~43 GB).

---

## 13. Bổ sung sau 04/10: gói xác minh MobileNetV2 (bản 1–3, 04–05/10/2026)

Gói riêng: `BanhmiTTS/work/mobilenetv2-verification/`. Trong zip tổng hợp, gói này nằm cạnh gói journal-evidence; xem `README.md` của gói đó. Các kết quả ảnh hưởng tới gói journal-evidence:

| Chủ đề | Kết quả | Ảnh hưởng tới claim ở §8 |
|---|---|---|
| Nguồn gốc block | `ResBlockInverted` được viết mới trong dự án (09/09/2026), theo mẫu Bảng 1 MobileNetV2; tác giả xác nhận MobileNetV2 là nguồn cảm hứng duy nhất | Bổ sung cho J-E019 (mô tả kiến trúc) |
| SnakeBeta / MRD | SnakeBeta có trong code chính thức BigVGAN, không có trong paper, và dùng **không có anti-aliasing**; MRD là của UnivNet, được BigVGAN dùng để thay MSD, còn dự án **thêm** MRD vào | Mô tả confounder (§4.6) phải ghi đúng nguồn |
| Literature | Depthwise-separable trong HiFi-GAN (Tang 2022, DeepGAN 2024; MISRNet 2022 dùng làm baseline và cho MOS thấp hơn) và IRB 1D trong vocoder (FBWave 2020) **đã có** | Không claim "novel/first" ở mức block |
| Expansion t | Run thăm dò dùng (6,6,6) và (4,4,1); chuyển sang t = 1 với lý do chi phí ∝ t × T; số tốc độ đo khi có train song song | Không dùng số tốc độ t > 1 làm kết quả |
| Docx của tác giả (MRF epoch 480 so với baseline) | Eval dùng manifest baseline, trong đó **473/500 câu thuộc train MRF**, nên WER/UTMOS bị rò rỉ | Thêm vào danh sách mâu thuẫn (§9). Không dùng so sánh này |
| Tên gọi | Khuyến nghị giữ "Parallel-IR / Sequential-IR" kèm định nghĩa t = 1 | — |
