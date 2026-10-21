# Piper_SeqIR_no_VITS2_cpn: kết quả train, chất lượng Harvard-720 và RTF cùng phiên

*English: [README.md](README.md)*

Câu hỏi (protocol: `docs/training/piper-ir-protocol.md`, chốt ngày 2026-10-11): thay decoder
ResBlock2 của `Piper_no_VITS2_cpn` (Piper-ConfigA) bằng bộ decoder Sequential-IR đã phát
hành thì model có tốt hơn không? **Piper-SeqIR** giữ nguyên backbone Config A, dữ liệu,
split và công thức train. Nó chỉ thay ba thứ:

- hai khối inverted-residual nối tiếp ở mỗi tầng (depthwise kernel 3, expansion 1);
- activation SnakeBeta;
- thêm discriminator MRD.

Đây là phép so sánh **cả bộ decoder**, không phải chỉ so cấu trúc khối. So với Piper-ParIR
cũng vậy: ngoài chuyện nối tiếp hay song song, số khối mỗi tầng (2 so với 3) và kernel
depthwise (3 so với 3/5/7) cũng khác nhau.

Mọi con số dưới đây là **phép đo mới** (2026-10-21), trừ các kết quả đã lưu dùng để so
sánh. Đợt này có cả **phiên đo thời gian trên máy rảnh** đã bị hoãn trong báo cáo ParIR:
Piper_no_VITS2_cpn (control), Piper-ParIR và Piper-SeqIR được đo liền nhau (mục 4). Bản ghi
của phiên này nằm trong [`../piper-ir-timing-session/`](../piper-ir-timing-session/).

## 1. Model được đánh giá

- **Run:** `training_output/piper-seqir-no-vits2-cpn` (nằm ngoài Git).
  - Commit `9f0988b` (tree sạch); 2× RTX 4070 Ti, DDP, bf16.
  - 1.500 epoch từ đầu, seed 1234, không gradient clipping.
  - Bắt đầu 2026-10-16 13:55 +07; trainer kết thúc 2026-10-21 07:55 +07, controller đánh dấu hoàn tất lúc 07:58.
- **Không bị gián đoạn.** Trạng thái controller chỉ có một lần start, không có bản ghi resume; run chỉ ghi một file events TensorBoard.
- **Controller validate:** đạt cả 8 mục (`validation_seqir.json`, giữ cùng trạng thái controller):
  - exit code 0, chạy đủ budget, không có traceback;
  - run record nhất quán, không có bản ghi resume;
  - đủ 1.500 điểm `val_loss_mel`, mọi scalar hữu hạn;
  - `last.ckpt` ở epoch 1.499, global step 1.176.000 (đúng như dự kiến);
  - checkpoint được chọn trùng với điểm thấp nhất.
- **Quy tắc chọn** (chốt trước khi train): checkpoint có `val_loss_mel` thấp nhất.
  - Được chọn: **epoch 1455, `val_loss_mel` 19.0332**.
  - File: `best-epoch=1455-val_loss_mel=19.0332.ckpt`.
  - SHA-256 `cc1ba70d20471fe587d0099fc9d1c739bc0d8d3b08541450629eb8bb068c6637`.
  - Global step 1.141.504.
  - Các checkpoint được giữ thêm: epoch 1382 (19.1521), 1491 (19.0770) và `last.ckpt`.
- **Load:** strict, `weights_only=True`; 878 entry. `synthesize.py` sẽ dừng nếu hyper-parameter lưu trong checkpoint khác `configs/training/piper-seqir-no-vits2-cpn.yaml`; ở đây không khác.
- **Dữ liệu held-out:** Harvard-720 và tập test không được dùng để chọn checkpoint.

## 2. Đường train (`val_loss_mel` trên tập validation)

Cả ba run dùng cùng 100 câu validation và cùng hàm loss. Giá trị lấy từ scalar TensorBoard
(`training_summary.json` cho SeqIR và Piper_no_VITS2_cpn;
`../piper-parir-no-vits2-cpn/training_summary.json` cho ParIR).

| Epoch | Piper-SeqIR trung bình (SD) | Piper-ParIR trung bình (SD) | Piper_no_VITS2_cpn trung bình (SD) | SeqIR − no_VITS2 | SeqIR − ParIR |
|---|---:|---:|---:|---:|---:|
| 500–599 | 20.318 (0.172) | 19.925 (0.177) | 20.796 (0.187) | −0.479 | +0.393 |
| 1000–1099 | 19.753 (0.180) | 19.828 (0.181) | 20.364 (0.151) | −0.610 | −0.075 |
| 1200–1299 | 19.553 (0.155) | 19.466 (0.164) | 20.224 (0.152) | −0.671 | +0.087 |
| 1300–1399 | 19.507 (0.145) | 19.404 (0.162) | 20.160 (0.143) | −0.652 | +0.103 |
| 1400–1499 | 19.441 (0.147) | 19.317 (0.143) | 20.105 (0.152) | −0.664 | +0.124 |
| Best (epoch) | 19.0332 (1455) | 19.0069 (1486) | 19.7719 (1403) | −0.739 | +0.026 |

- **So với Piper_no_VITS2_cpn:** từ epoch 1200 trở đi, mel loss trên validation của SeqIR thấp hơn khoảng 0,65–0,67, tức gấp khoảng bốn lần SD giữa các epoch.
- **So với ParIR:** trong 300 epoch cuối, SeqIR cao hơn 0,09–0,12, nhỏ hơn một SD giữa các epoch.
- **Không có bậc tăng quanh epoch 800.** Loss của ParIR tăng khoảng 0,3 từ khoảng epoch 800 (README ParIR, mục 2). SeqIR cũng có MRD nhưng loss vẫn tiếp tục giảm. Trung bình 40 epoch:

  | Epoch | 760–799 | 800–839 | 840–879 | 880–919 |
  |---|---:|---:|---:|---:|
  | SeqIR | 20.03 | 20.02 | 19.98 | 19.89 |
  | ParIR | 19.74 | 20.05 | 20.07 | 20.04 |

  Cửa sổ 1000–1099, nơi SeqIR thấp hơn ParIR, nằm trong đoạn tăng của ParIR. Vì vậy chỉ riêng việc có MRD không giải thích được bậc tăng đó. Nguyên nhân chưa được xác định, và loss discriminator của SeqIR chưa được xem trong báo cáo này.

## 3. Kết quả Harvard-720

Protocol: protocol Harvard lịch sử, không thay đổi (`docs/evaluation.md`), giống hai model
Config A còn lại.
- PyTorch FP32 trên CPU, 12 thread; scales 0.667/1.0/0.8; seed 1234 trước mỗi câu.
- UTMOSv2 `fusion_stage3` fold 0 (SHA-256 trọng số `c8149d98…`), một lần dự đoán cho mỗi WAV.
- Whisper `small` (`9ecf7799…`).
- Macro WER bằng JiWER.
- Môi trường scorer giống các đợt trước: Python 3.10.20, torch 2.13.0+cu130, NumPy 1.26.4.

SeqIR được synthesis trong phiên đo thời gian trên máy rảnh (mục 4), rồi chấm điểm trên CPU
sau đó, lúc không có training nào chạy.

Sáu dòng còn lại là kết quả đã lưu:
- Piper-ParIR từ `results/piper-parir-no-vits2-cpn/` (2026-10-16);
- Piper_no_VITS2_cpn từ `results/piper-no-vits2-cpn/` (2026-10-10);
- bốn hệ thống lịch sử từ `results/historical/harvard/`.

RTF chỉ ghi cho ba model Config A, lấy từ phiên 2026-10-21. RTF lịch sử đo ở các phiên khác
nên không so sánh được (mục 4).

| Model | UTMOSv2 (dự đoán) | Macro WER | Corpus WER | Số câu WER = 0 | RTF trung bình (phiên 2026-10-21) |
|---|---:|---:|---:|---:|---:|
| **Piper-SeqIR** (mới) | **3.493** | **0.0696** | 0.0665 | 476 | **0.0340** |
| Piper-ParIR (2026-10-16) | 3.530 | 0.0726 | 0.0707 | 464 | 0.0392 |
| Piper_no_VITS2_cpn (ResBlock2, 2026-10-10) | 3.517 | 0.0612 | 0.0596 | 502 | 0.0510 |
| baseline-resblock2 (có VITS2) | 3.026 | 0.1476 | 0.1437 | 318 | — |
| Parallel-IR (MRF, đã phát hành) | 3.568 | 0.1073 | 0.1033 | 381 | — |
| Sequential-IR (SEQ, đã phát hành) | 3.544 | 0.1222 | 0.1186 | 351 | — |
| Piper (checkpoint đã công bố) | 3.451 | 0.1747 | 0.1696 | 228 | — |

Số lỗi từ trên tổng 5.742 từ tham chiếu (thay thế / mất từ / chèn thêm):
- Piper-SeqIR: 292 / 76 / 14;
- Piper-ParIR: 321 / 71 / 14;
- Piper_no_VITS2_cpn: 283 / 46 / 13.

So sánh ghép cặp theo từng câu (`comparison.json`). Chênh lệch = SeqIR − đối chứng; CI là
bootstrap ghép cặp 95%. Hiệu chỉnh Holm tính trên 12 phép thử này (chỉ UTMOSv2 và WER)
và chỉ mang tính thăm dò.

| SeqIR so với | ΔUTMOSv2 [CI 95%] (Holm p) | ΔWER [CI 95%] (Holm p) | Số câu tốt hơn / kém hơn (WER) |
|---|---|---|---|
| **Piper_no_VITS2_cpn** | −0.024 [−0.048, −0.00005] (0.12) | +0.008 [−0.0004, +0.018] (0.12) | 116 / 140 |
| **Piper-ParIR** | −0.037 [−0.060, −0.014] (0.021) | −0.003 [−0.013, +0.007] (0.71) | 144 / 128 |
| baseline-resblock2 | **+0.467** [+0.439, +0.495] | **−0.078** [−0.091, −0.065] | 290 / 84 |
| Parallel-IR (MRF) | −0.075 [−0.097, −0.052] | −0.038 [−0.049, −0.026] | 220 / 113 |
| Sequential-IR (SEQ) | −0.051 [−0.075, −0.028] | −0.053 [−0.064, −0.041] | 248 / 96 |
| Piper (đã công bố) | +0.042 [+0.016, +0.068] (0.003) | −0.105 [−0.119, −0.091] | 380 / 80 |

Holm p chỉ được ghi khi lớn hơn 0,001. WER là tỉ lệ từ sai: +0.008 nghĩa là tăng
0,8 điểm phần trăm.

**UTMOSv2 trong cùng phiên** (bổ sung, `comparison_same_session_utmosv2.json`, Holm trên 6 phép thử của file này):
- UTMOSv2 thay đổi giữa các lượt chấm (mục 5). Vì vậy các WAV đã lưu của Piper_no_VITS2_cpn và ParIR được chấm lại trong phiên này, ngay sau SeqIR.
- So với Piper_no_VITS2_cpn chấm lại (3.508): **−0.015 [−0.039, +0.009]**, Wilcoxon p 0.23; cao hơn ở 344 câu, thấp hơn ở 376 câu.
- So với ParIR chấm lại (3.526): **−0.033 [−0.056, −0.010]**, Wilcoxon p 0.013 (Holm 0.025); cao hơn ở 330 câu, thấp hơn ở 390 câu.
- WER giống hệt giữa hai lần chấm, nên các phép so WER ở trên không bị ảnh hưởng.

## 4. RTF cùng phiên (máy rảnh)

**Thiết lập.**
- **Khi nào:** phiên bắt đầu lúc 09:05 +07 ngày 2026-10-21, sau khi controller đã đánh dấu SeqIR hoàn tất và không còn tiến trình trainer nào. Phiên kết thúc lúc 09:09:36.
- **Đo gì:** ba model được synthesis liền nhau, mỗi model trong một tiến trình `synthesize.py` mới, qua cùng harness và code EdgeTTS:
  1. Piper_no_VITS2_cpn (control, epoch 1403);
  2. Piper-ParIR (epoch 1486);
  3. Piper-SeqIR (epoch 1455).
- **Cách đo:** CPU FP32, 12 thread, độ ưu tiên bình thường, ẩn CUDA. Vùng được bấm giờ chỉ là lời gọi forward. Mỗi câu đo một lần, không warm-up.
- **Trạng thái máy trước khi đo** (`machine_state_before.txt`): load average 0.03; GPU dùng 6% và 0%; tiến trình bận nhất là TensorBoard với 0,2% CPU.

| Model | RTF trung bình | RTF trung vị | Δ trung bình so với control [CI 95%] | Tương đối | Nhanh hơn control | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (control) | 0.0510 | 0.0490 | — | — | — | — |
| Piper-ParIR | 0.0392 | 0.0374 | −0.0118 [−0.0126, −0.0110] | **−23,2%** | 653 / 720 | 4e-99 |
| Piper-SeqIR | 0.0340 | 0.0320 | −0.0170 [−0.0178, −0.0162] | **−33,3%** | 690 / 720 | 2e-113 |

Các phép so với control nằm trong `session_control_parir.json` và `session_control_seqir.json`.

**SeqIR so với ParIR** trong cùng phiên: RTF trung bình thấp hơn 13,2% (0.0340 / 0.0392 = 0.868), và SeqIR nhanh hơn ở 561 trên 720 câu. Đây chỉ là phép tính trực tiếp trên hai bản ghi synthesis; không có file phép thử ghép cặp nào cho cặp này.

**Các kiểm tra trong cùng phiên:**
- **Synthesis của control:** 720/720 WAV giống từng byte với WAV ngày 2026-10-10 của Piper_no_VITS2_cpn, và thời lượng audio bằng nhau.
- **Sinh lại ParIR:** 720/720 SHA-256 WAV trùng với lần đánh giá 2026-10-16, và thời lượng giống hệt (`parir_determinism.json`). Vì vậy điểm chất lượng ParIR ngày 2026-10-16 mô tả đúng phần audio đã được bấm giờ ở đây.
- **Synthesis của SeqIR:** chỉ chạy một lần; 720/720 WAV hữu hạn. Tính deterministic chưa được kiểm bằng lần synthesis thứ hai.
- **Độ trôi giữa các phiên:** RTF trung bình của control ở phiên này là 0.0510, so với 0.0421 trong phiên riêng ngày 2026-10-10 (×1,21), dù máy đang rảnh. Nguyên nhân không được ghi lại: trạng thái nguồn và nhiệt của máy Windows, cùng tải bên ngoài WSL, không được log.

**Giới hạn.**
- RTF chỉ so sánh được trong nội bộ phiên này. Các giá trị lịch sử (MRF 0.0339, SEQ 0.0308, baseline 0.0418, Piper 0.0392) và giá trị 0.0421 ngày 2026-10-10 thuộc các phiên khác; không đặt chúng cạnh 0.0340.
- Thứ tự (control → ParIR → SeqIR) cố định, không đảo vòng. Mỗi câu đo một lần, không lặp lại.
- Đây là PyTorch trên CPU dưới WSL của máy trạm. Không phải ONNX, cũng không phải đo trên thiết bị biên.
- Các model sinh audio dài khác nhau (tổng 1.749 s control, 1.798 s ParIR, 1.744 s SeqIR); RTF đã chuẩn hoá theo từng câu.

## 5. Control của scorer

- **Chấm lại WAV của Piper_no_VITS2_cpn (`scorer_control.json`):**
  - Whisper cho **720/720 transcript giống hệt**; macro WER 0.0612 trùng giá trị đã lưu.
  - UTMOSv2: trung bình 3.508 so với 3.517 đã lưu (−0.009). Chênh lệch theo từng câu có SD 0.17, lớn nhất 0.88, và r = 0.79.
- **Chấm lại WAV của ParIR (`scorer_control_parir.json`):**
  - Whisper cho **720/720 transcript giống hệt**; macro WER 0.0726 trùng giá trị đã lưu.
  - UTMOSv2: trung bình 3.526 so với 3.530 đã lưu (−0.004). Chênh lệch theo từng câu có SD 0.15, lớn nhất 0.48, và r = 0.82.
- **Ý nghĩa:**
  - Các chênh lệch WER ở mục 3 không phải do scorer bị trôi.
  - ΔUTMOSv2 cỡ ±0,01–0,02 giữa hai phiên nằm trong mức dao động của scorer, vì bước cắt ngẫu nhiên của UTMOSv2 không được gắn seed.
  - Phép thử cùng phiên loại bỏ ảnh hưởng của phiên lên giá trị trung bình, nhưng không loại bỏ được nhiễu theo từng WAV này.

## 6. Đọc kết quả

- **So với Piper_no_VITS2_cpn** (cùng backbone, decoder ResBlock2):
  - **Tốc độ:** trong cùng phiên, SeqIR có RTF thấp hơn 33% và nhanh hơn ở 690/720 câu.
  - **WER:** SeqIR kém hơn một chút, 0,070 so với 0,061 (+0,8 điểm). CI vừa chứa 0, và Holm p là 0,12. Phần lỗi tăng thêm chủ yếu là mất từ (76 so với 46).
  - **UTMOSv2:** SeqIR thấp hơn một chút. So khác phiên thì chênh −0,024, cận trên CI vừa dưới 0 (Holm 0,12). So cùng phiên thì chênh −0,015 và CI chứa 0 (p 0,23).
  - **Mel loss trên validation:** SeqIR thấp hơn rõ rệt (−0,66 trong 300 epoch cuối). Giống ParIR, ưu thế này không chuyển thành WER hay chất lượng dự đoán tốt hơn.
  - **Kích thước:** decoder SeqIR có 1,11 triệu tham số so với 1,66 triệu (dạng weight-norm), ít hơn 33%.
- **So với Piper-ParIR:**
  - **Tốc độ:** trong cùng phiên, SeqIR có RTF thấp hơn 13%.
  - **WER:** không phân biệt được (−0,003 [−0,013, +0,007]).
  - **UTMOSv2:** SeqIR thấp hơn 0,033–0,037, cả khi so khác phiên lẫn cùng phiên (Holm 0,021 và 0,025).
  - **Mel loss trên validation:** SeqIR cao hơn một chút (+0,09 đến +0,12).
  - **Kích thước:** 1,11 triệu so với 1,16 triệu tham số decoder.
- **So với bốn hệ thống đã lưu:**
  - SeqIR tốt hơn rõ rệt so với baseline-resblock2 (có VITS2) và checkpoint Piper đã công bố, trên cả hai thước đo.
  - So với Parallel-IR và Sequential-IR đã phát hành (công thức baseline VITS2), SeqIR có WER thấp hơn (−3,8 và −5,3 điểm) và UTMOSv2 thấp hơn (−0,075 và −0,051).
- **Xếp theo WER (bảy hệ thống):** Piper_no_VITS2_cpn 0,061, Piper-SeqIR 0,070, Piper-ParIR 0,073, sau đó là MRF 0,107, SEQ 0,122, baseline-resblock2 0,148 và Piper 0,175.
- **Giữa ba model Config A:**
  - **Tốc độ:** SeqIR nhanh nhất, rồi đến ParIR, rồi đến control.
  - **WER:** Piper_no_VITS2_cpn tốt nhất; SeqIR và ParIR gần nhau.
  - **UTMOSv2:** cả ba nằm trong khoảng 0,04; ParIR cao nhất, SeqIR thấp nhất.

Những gì kết quả này **không** chứng minh:
- **Phần nào của bộ decoder tạo ra khác biệt.** Khối decoder, SnakeBeta và MRD thêm vào không tách riêng được trong run này. Giữa ParIR và SeqIR còn khác cả số khối lẫn kernel (protocol, mục 4).
- **Kết quả đúng ngoài thiết lập này.** Mỗi model chỉ có một lần train với seed 1234, mỗi câu sinh một lần, WER dùng Whisper-small, và MOS là dự đoán (không phải người chấm).
- **Ý nghĩa qua các seed train khác nhau.** Phép thử ghép cặp chỉ mô tả dao động giữa các câu.
- **Tương đương.** Không có ý nghĩa thống kê không có nghĩa là tương đương; không có ngưỡng tương đương nào được chốt trước.
- **Tốc độ trên thiết bị biên.** Phép đo thời gian chỉ là một phiên PyTorch CPU có thứ tự cố định trên máy trạm.

## 7. Chưa làm trong đợt này

- **Tập test-500 canonical, ONNX export, PTQ và đo thời gian trên thiết bị biên:** chưa làm (cùng lý do như `results/piper-no-vits2-cpn/README.vi.md`, mục 6).
- **Synthesis SeqIR lần thứ hai** để xác nhận tính deterministic.
- **Đo thời gian lặp lại hoặc đảo thứ tự,** và đo có warm-up.

## 8. Các file

Trong thư mục này:

| File | Nội dung | Loại |
|---|---|---|
| `selected_checkpoint.json` | Lựa chọn của controller: file, bytes, SHA-256, epoch, global step, `val_loss_mel`, quy tắc | bản ghi run |
| `training_summary.json` | Bản ghi run, hash checkpoint, `val_loss_mel` theo từng epoch của SeqIR và Piper_no_VITS2_cpn, thống kê theo cửa sổ | tóm tắt mới từ log run |
| `harvard720_synth_record.json` | Bản ghi synthesis trong phiên đo trên máy rảnh: hash checkpoint/config/manifest, protocol, môi trường, thời gian và SHA-256 WAV theo từng câu | phép đo mới |
| `harvard_piper_seqir_no_vits2_cpn_results.json` | Kết quả theo từng câu, theo schema lịch sử, thêm `wav_sha256`. Trường `rtf` lấy từ phiên máy rảnh | phép đo mới |
| `score_record.json` | Package scorer, hash trọng số, môi trường, hash kết quả | bản ghi phép đo mới |
| `comparison.json` | Tóm tắt cả bảy hệ thống và 12 phép thử ghép cặp (UTMOSv2, WER) | phân tích mới |
| `comparison_same_session_utmosv2.json` | Phép thử UTMOSv2 bổ sung, so với các bản chấm lại cùng phiên | phân tích mới |
| `harvard_piper_no_vits2_cpn_rescored_results.json`, `piper_no_vits2_cpn_rescore_record.json` | Chấm lại các WAV ngày 2026-10-10 của Piper_no_VITS2_cpn trong phiên này. Trường `rtf` là của ngày 2026-10-10 | control mới |
| `harvard_piper_parir_no_vits2_cpn_rescored_results.json`, `parir_rescore_record.json` | Chấm lại các WAV ngày 2026-10-16 của ParIR trong phiên này. Trường `rtf` đo khi máy đang tải ngày 2026-10-16; không dùng | control mới |
| `scorer_control.json`, `scorer_control_parir.json` | So điểm chấm lại với điểm đã lưu của Piper_no_VITS2_cpn và ParIR | control mới |

Trong [`../piper-ir-timing-session/`](../piper-ir-timing-session/):

| File | Nội dung | Loại |
|---|---|---|
| `piper_no_vits2_cpn_ctrl_synth_record.json` | Synthesis control của Piper_no_VITS2_cpn trong phiên, thời gian và SHA-256 WAV theo từng câu | phép đo mới |
| `piper_parir_timing_synth_record.json` | Synthesis ParIR trong phiên, thời gian và SHA-256 WAV theo từng câu | phép đo mới |
| `session_control_parir.json`, `session_control_seqir.json` | Tính tái lập của control, độ trôi giữa các phiên và phép so RTF ghép cặp với control | control mới |
| `parir_determinism.json` | Hash và thời lượng WAV của ParIR so với lần đánh giá 2026-10-16 | control mới |
| `machine_state_before.txt`, `machine_state_after.txt` | Load average, mức dùng GPU và các tiến trình bận nhất trước và sau khi đo | bản ghi phiên |

Bản ghi SeqIR ở trên (`harvard720_synth_record.json`) là lần synthesis thứ ba của cùng phiên đó.

Các WAV nằm ngoài Git:
- SeqIR: `training_output/piper-seqir-no-vits2-cpn/eval/harvard720/wavs`;
- phiên đo thời gian: `training_output/piper-ir-post-eval/timing/{ctrl,parir}/wavs`.

Hash của chúng có trong các bản ghi.

## 9. Tái lập

Chạy từ thư mục gốc của clone training:
- synthesis dùng venv training;
- chấm điểm dùng môi trường scorer (version trong `score_record.json`);
- phân tích dùng bất kỳ môi trường nào có SciPy.

Các lệnh giống hệt mục 9 của [README.md](README.md).

**Kết quả mong đợi:**
- Synthesis là deterministic, nên hash WAV phải trùng.
- Transcript của Whisper đã được tái tạo chính xác trong cả hai control.
- Điểm UTMOSv2 theo từng câu thay đổi giữa các lượt chấm.
- Thời gian thay đổi giữa các phiên; chỉ tỉ lệ trong cùng một phiên mới có ý nghĩa.

Pipeline đã chạy các bước này được giữ cùng các run, ngoài Git (`training_output/piper-ir-post-eval/post_seqir.sh`, `post.log`).
