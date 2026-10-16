# Piper_ParIR_no_VITS2_cpn: kết quả train và đánh giá chất lượng Harvard-720

*English: [README.md](README.md)*

Câu hỏi (protocol: `docs/training/piper-ir-protocol.md`, chốt ngày 2026-10-11): thay decoder
ResBlock2 của `Piper_no_VITS2_cpn` (Piper-ConfigA) bằng bộ decoder Parallel-IR đã phát
hành thì model có tốt hơn không? **Piper-ParIR** giữ nguyên backbone Config A, dữ liệu,
split và công thức train. Nó chỉ thay ba thứ:

- ba nhánh inverted-residual song song ở mỗi tầng (depthwise kernel 3/5/7, expansion 1);
- activation SnakeBeta;
- thêm discriminator MRD.

Đây là phép so sánh **cả bộ decoder**, không phải chỉ so cấu trúc khối.

Mọi con số dưới đây là **phép đo mới** (2026-10-16), trừ các kết quả đã lưu dùng để so
sánh. **Đợt này chưa đo RTF**; xem mục 6.

## 1. Model được đánh giá

- **Run:** `training_output/piper-parir-no-vits2-cpn` (nằm ngoài Git).
  - Commit `9f0988b` (tree sạch); 2× RTX 4070 Ti, DDP, bf16.
  - 1.500 epoch từ đầu, seed 1234, không gradient clipping.
  - Bắt đầu 2026-10-11 20:30 +07; xong 2026-10-16 13:52 +07.
- **Một lần gián đoạn.** Ngày 2026-10-15, khoảng 13:57 +07, máy Windows bị reset không tắt sạch (Kernel-Power 41, bugcheck 0) khi đang ở epoch 1222.
  - Trước khi resume, `last.ckpt` (epoch 1221, global step 958.048) đã được kiểm: CRC zip đúng, tensor hữu hạn, đọc được file events.
  - Controller resume từ checkpoint đó (`run_sequence.py resume`, có `resume_record_*.json`).
  - Dưới DDP, trạng thái RNG không được khôi phục (theo thiết kế). Vì vậy từ epoch 1222, thứ tự dữ liệu và noise khác với một run không bị gián đoạn.
- **Controller validate:** đạt cả 8 mục (`validation_parir.json`):
  - exit code 0, chạy đủ budget, không có traceback;
  - run record nhất quán;
  - đủ 1.500 điểm `val_loss_mel`, mọi scalar hữu hạn;
  - `last.ckpt` ở epoch 1.499;
  - checkpoint được chọn trùng với điểm thấp nhất.
- **Quy tắc chọn** (chốt trước khi train): checkpoint có `val_loss_mel` thấp nhất.
  - Được chọn: **epoch 1486, `val_loss_mel` 19.0069**.
  - File: `best-epoch=1486-val_loss_mel=19.0069.ckpt`.
  - SHA-256 `d57f6abb589a28db83dc60d4f2b7d0c0c4a5a16db9ae124e12f40870d31c97ee`.
  - Global step 1.165.808.
- **Load:** strict, `weights_only=True`; 917 entry. Hyper-parameter lưu trong checkpoint trùng với `configs/training/piper-parir-no-vits2-cpn.yaml`.
- **Dữ liệu held-out:** Harvard-720 và tập test không được dùng để chọn checkpoint.
  - Có một phiên nghe thử không chính thức ngày 2026-10-14 trên Harvard list 1 (idx 0–9), dùng các checkpoint trung gian (epoch 645/664/717). Phiên này được ghi lại ngoài Git, trong `outputs/Piper_ParIR_no_VITS2_cpn/`.
  - Không có quyết định nào về train hay chọn checkpoint dựa trên phiên nghe đó.

## 2. Đường train so với Piper_no_VITS2_cpn (`val_loss_mel` trên tập validation)

Cùng 100 câu validation, cùng hàm loss. Giá trị lấy từ scalar TensorBoard
(`training_summary.json`, có kèm chuỗi theo từng epoch). Hai file events của run có
resume được ghép theo global step.

| Epoch | Piper-ParIR trung bình (SD) | Piper_no_VITS2_cpn trung bình (SD) | Chênh lệch |
|---|---:|---:|---:|
| 500–599 | 19.925 (0.177) | 20.796 (0.187) | −0.872 |
| 1000–1099 | 19.828 (0.181) | 20.364 (0.151) | −0.535 |
| 1200–1299 | 19.466 (0.164) | 20.224 (0.152) | −0.758 |
| 1300–1399 | 19.404 (0.162) | 20.160 (0.143) | −0.756 |
| 1400–1499 | 19.317 (0.143) | 20.105 (0.152) | −0.788 |
| Best (epoch) | 19.0069 (1486) | 19.7719 (1403) | −0.765 |

Từ epoch 1200 trở đi, mel loss trên validation của ParIR thấp hơn khoảng 0,75–0,79,
tức gấp khoảng năm lần SD giữa các epoch.

Từ khoảng epoch 800, mel loss train và validation của ParIR tăng khoảng 0,3–0,4:
- trung bình 20 epoch đi từ 19,71 (780–799) lên khoảng 20,1 (800–860);
- trung bình 900–999 vẫn là 19,97;
- chỉ xuống dưới mức trước epoch 800 vào khoảng epoch 1100–1200.

Lúc bắt đầu tăng, loss của discriminator MRD giảm (từ ~1,02 xuống ~0,96 trong epoch
800–860), nghĩa là thế cân bằng giữa generator và discriminator đã thay đổi. Run ResBlock2
không có MRD và không có bậc tăng này.

Mel loss chỉ là loss tái tạo. Mục 3 cho thấy khoảng cách này không chuyển thành khác
biệt trên các thước đo chất lượng Harvard.

## 3. Kết quả Harvard-720

Protocol: protocol Harvard lịch sử, không thay đổi (`docs/evaluation.md`), giống hệt lần đánh giá Piper_no_VITS2_cpn.
- PyTorch FP32 trên CPU, 12 thread; scales 0.667/1.0/0.8; seed 1234 trước mỗi câu.
- UTMOSv2 `fusion_stage3` fold 0 (SHA-256 trọng số `c8149d98…`), một lần dự đoán cho mỗi WAV.
- Whisper `small` (`9ecf7799…`).
- Macro WER bằng JiWER.
- Môi trường scorer giống lần 2026-10-10: Python 3.10.20, torch 2.13.0+cu130, NumPy 1.26.4.

Synthesis và chấm điểm chạy trên CPU trong lúc Piper-SeqIR đang train trên GPU. Synthesis
là deterministic và điểm số không phụ thuộc tải máy, nhưng thời gian thì có. **Vì vậy RTF
của phiên này không được báo cáo.**

Năm dòng còn lại là kết quả đã lưu:
- Piper_no_VITS2_cpn từ `results/piper-no-vits2-cpn/` (2026-10-10);
- bốn hệ thống lịch sử từ `results/historical/harvard/`.

| Model | UTMOSv2 (dự đoán) | Macro WER | Corpus WER | Số câu WER = 0 | RTF trung bình |
|---|---:|---:|---:|---:|---:|
| **Piper-ParIR** (mới) | **3.530** | **0.0726** | 0.0707 | 464 | chưa đo |
| Piper_no_VITS2_cpn (ResBlock2, 2026-10-10) | 3.517 | 0.0612 | 0.0596 | 502 | 0.0421 * |
| baseline-resblock2 (có VITS2) | 3.026 | 0.1476 | 0.1437 | 318 | 0.0418 |
| Parallel-IR (MRF, đã phát hành) | 3.568 | 0.1073 | 0.1033 | 381 | 0.0339 |
| Sequential-IR (SEQ, đã phát hành) | 3.544 | 0.1222 | 0.1186 | 351 | 0.0308 |
| Piper (checkpoint đã công bố) | 3.451 | 0.1747 | 0.1696 | 228 | 0.0392 |

\* Đo trong phiên khác với các dòng lịch sử; xem `results/piper-no-vits2-cpn/README.vi.md`, mục 4.

Số lỗi từ trên tổng 5.742 từ tham chiếu (thay thế / mất từ / chèn thêm):
- Piper-ParIR: 321 / 71 / 14;
- Piper_no_VITS2_cpn: 283 / 46 / 13.

So sánh ghép cặp theo từng câu (`comparison.json`). Chênh lệch = ParIR − đối chứng; CI là
bootstrap ghép cặp 95%. Hiệu chỉnh Holm tính trên 10 phép thử này (chỉ UTMOSv2 và WER)
và chỉ mang tính thăm dò.

| ParIR so với | ΔUTMOSv2 [CI 95%] (Holm p) | ΔWER [CI 95%] (Holm p) | Số câu tốt hơn / kém hơn (WER) |
|---|---|---|---|
| **Piper_no_VITS2_cpn** | +0.013 [−0.011, +0.037] (0.43) | **+0.011** [+0.002, +0.021] (0.11) | 108 / 159 |
| baseline-resblock2 | **+0.504** [+0.479, +0.530] | **−0.075** [−0.088, −0.062] | 286 / 99 |
| Parallel-IR (MRF) | −0.038 [−0.060, −0.016] (0.003) | −0.035 [−0.046, −0.024] | 225 / 122 |
| Sequential-IR (SEQ) | −0.014 [−0.037, +0.008] (0.43) | −0.050 [−0.062, −0.037] | 252 / 112 |
| Piper (đã công bố) | +0.079 [+0.056, +0.103] | −0.102 [−0.116, −0.089] | 382 / 84 |

Holm p chỉ được ghi khi lớn hơn 0,001. WER là tỉ lệ từ sai: +0.011 nghĩa là tăng
1,1 điểm phần trăm.

**UTMOSv2 so với Piper_no_VITS2_cpn trong cùng phiên** (bổ sung, `comparison_same_session_utmosv2.json`):
- UTMOSv2 thay đổi giữa các lượt chấm (mục 4), nên các WAV của Piper_no_VITS2_cpn đã được chấm lại trong phiên này.
- So với điểm chấm lại đó, ParIR cao hơn **+0.025 [+0.002, +0.049]** (Wilcoxon p 0.097); cao hơn ở 368 câu, thấp hơn ở 352 câu.
- WER giống hệt giữa hai lần chấm, nên phép so WER ở trên không bị ảnh hưởng.

## 4. Các control

- **Tính tái lập của scorer (`scorer_control.json`).** Các WAV ngày 2026-10-10 của Piper_no_VITS2_cpn được chấm lại bằng cùng `score.py` trong phiên này. `score.py` kiểm SHA-256 của từng WAV trước khi chấm.
  - Whisper cho **720/720 transcript giống hệt**; macro WER 0.0612 trùng giá trị đã lưu.
  - UTMOSv2: trung bình 3.505 so với 3.517 đã lưu (−0.012). Chênh lệch theo từng câu có SD 0.16, lớn nhất 0.62, và r = 0.80, vì bước cắt ngẫu nhiên không được gắn seed.
  - Do đó ΔWER ở trên không phải do scorer bị trôi. Còn ΔUTMOSv2 cỡ ±0,01–0,02 giữa hai phiên nằm trong mức dao động của scorer.
- **Synthesis:** 720/720 WAV đều hữu hạn. SHA-256 của WAV có trong `harvard720_synth_record.json`. Việc chạy lại để xác nhận tính deterministic của checkpoint này được lên lịch cùng phiên đo RTF (mục 6).

## 5. Đọc kết quả

- **So với Piper_no_VITS2_cpn** (cùng backbone, decoder ResBlock2): trên Harvard-720, hai model rất gần nhau.
  - **UTMOSv2:** ParIR nhỉnh hơn một chút. So khác phiên thì chênh +0,013 và CI chứa 0. So cùng phiên thì chênh +0,025 và CI vừa đủ loại trừ 0, nhưng Wilcoxon p là 0,097.
  - **WER:** ParIR kém hơn một chút, 0,073 so với 0,061 (+1,1 điểm). CI loại trừ 0 (Wilcoxon p 0,037), nhưng không còn ý nghĩa sau Holm (0,11). ParIR sai nhiều hơn ở 159 câu và ít hơn ở 108 câu. Phần lỗi tăng thêm chủ yếu là mất từ (71 so với 46).
  - **Mel loss trên validation:** ParIR thấp hơn rõ rệt (−0,77). Ưu thế này không chuyển thành độ dễ hiểu tốt hơn, hay chất lượng dự đoán tốt hơn rõ rệt.
  - **Kích thước:** decoder ParIR có 1,16 triệu tham số so với 1,66 triệu (dạng weight-norm), ít hơn 30%.
  - **Tốc độ:** ParIR có nhanh hơn hay không thì chưa biết (mục 6).
- **So với bốn hệ thống đã lưu:**
  - ParIR tốt hơn rõ rệt so với baseline-resblock2 (có VITS2) và checkpoint Piper đã công bố, trên cả hai thước đo.
  - So với Parallel-IR (MRF) đã phát hành, tức cùng bộ decoder nhưng train bằng công thức baseline VITS2: ParIR có WER thấp hơn (−3,5 điểm) và UTMOSv2 thấp hơn một chút (−0,038).
  - So với Sequential-IR: ParIR có WER thấp hơn, còn UTMOSv2 không phân biệt được.
- **Xếp theo WER:** Piper_no_VITS2_cpn có WER thấp nhất trong sáu hệ thống, Piper-ParIR đứng thứ hai.

Những gì kết quả này **không** chứng minh:
- **Phần nào của bộ decoder tạo ra khác biệt.** Khối decoder, SnakeBeta và MRD thêm vào không tách riêng được trong run này (protocol, mục 4).
- **Kết quả đúng ngoài thiết lập này.** Mỗi model chỉ có một lần train với seed 1234, mỗi câu sinh một lần, WER dùng Whisper-small, và MOS là dự đoán (không phải người chấm). Run này cũng có một lần resume dưới DDP mà không khôi phục RNG.
- **Ý nghĩa qua các seed train khác nhau.** Phép thử ghép cặp chỉ mô tả dao động giữa các câu.

## 6. Chưa làm trong đợt này

- **RTF.** Sẽ đo trên máy rảnh sau khi Piper-SeqIR train xong. ParIR, SeqIR và control Piper_no_VITS2_cpn sẽ được đo liền nhau trong một phiên (`session_control.py`), đồng thời sinh lại WAV của ParIR để xác nhận tính deterministic.
- **Tập test-500 canonical, ONNX export, PTQ và đo thời gian trên thiết bị biên:** chưa làm (cùng lý do như `results/piper-no-vits2-cpn/README.vi.md`, mục 6).

## 7. Các file

| File | Nội dung | Loại |
|---|---|---|
| `selected_checkpoint.json` | Lựa chọn của controller: file, bytes, SHA-256, epoch, global step, `val_loss_mel`, quy tắc | bản ghi run |
| `training_summary.json` | Bản ghi run, hash checkpoint, `val_loss_mel` theo từng epoch của ParIR và Piper_no_VITS2_cpn, thống kê theo cửa sổ | tóm tắt mới từ log run |
| `harvard720_synth_record.json` | Bản ghi synthesis: hash checkpoint/config/manifest, protocol, môi trường, SHA-256 WAV theo từng câu. Các trường thời gian được ghi khi máy đang tải training, không dùng | phép đo mới |
| `harvard_piper_parir_no_vits2_cpn_results.json` | Kết quả theo từng câu, theo schema lịch sử, thêm `wav_sha256`. Trường `rtf` đo khi máy đang tải; không dùng | phép đo mới |
| `score_record.json` | Package scorer, hash trọng số, môi trường, hash kết quả | bản ghi phép đo mới |
| `comparison.json` | Tóm tắt cả sáu hệ thống và 10 phép thử ghép cặp (UTMOSv2, WER) | phân tích mới |
| `comparison_same_session_utmosv2.json` | Phép thử UTMOSv2 bổ sung, có bản chấm lại cùng phiên của Piper_no_VITS2_cpn | phân tích mới |
| `harvard_piper_no_vits2_cpn_rescored_results.json`, `piper_no_vits2_cpn_rescore_record.json` | Chấm lại các WAV ngày 2026-10-10 của Piper_no_VITS2_cpn trong phiên này | control mới |
| `scorer_control.json` | So điểm chấm lại với điểm Piper_no_VITS2_cpn đã lưu | control mới |

720 WAV nằm ngoài Git, trong `training_output/piper-parir-no-vits2-cpn/eval/harvard720/wavs`.
Hash của chúng có trong các bản ghi.

## 8. Tái lập

Chạy từ thư mục gốc của clone training:
- synthesis dùng venv training;
- chấm điểm dùng môi trường scorer đã dùng ngày 2026-10-10 (UTMOSv2, Whisper, JiWER; version trong `score_record.json`);
- phân tích dùng bất kỳ môi trường nào có SciPy.

Các tuỳ chọn `compare.py --extra-reference/--metrics` được thêm ngày 2026-10-16 cho phép so sánh này.
Các lệnh giống hệt mục 8 của [README.md](README.md).

**Kết quả mong đợi:**
- Synthesis là deterministic, nên hash WAV phải trùng.
- Transcript của Whisper đã được tái tạo chính xác trong control.
- Điểm UTMOSv2 theo từng câu thay đổi giữa các lượt chấm.

Pipeline đã chạy các bước này được giữ cùng run (`eval/pipeline.sh`, `eval/pipeline.log`).
