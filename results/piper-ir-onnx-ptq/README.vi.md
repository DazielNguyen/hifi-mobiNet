# Các model Piper-IR ở dạng ONNX FP32 và INT8 (PTQ tĩnh): export, chất lượng Harvard-720 và tốc độ

*English: [README.md](README.md)*

Ba model Config A được export sang ONNX, lượng tử hoá INT8 bằng công thức PTQ lịch sử, rồi
đánh giá trên Harvard-720:
- Piper_no_VITS2_cpn (ResBlock2);
- Piper-ParIR;
- Piper-SeqIR.

Protocol `docs/evaluation-onnx-ptq.md` đã chốt các ngưỡng, scope PTQ và tập calibration trước
khi có bất kỳ graph nào. Mọi con số là **phép đo mới** ngày 2026-10-21, trừ các kết quả
PyTorch dùng để so sánh. Đợt này chạy từ 16:35 đến 21:47 +07
(`scripts/evaluation/onnx/run_round.sh`), và mọi cổng kiểm đều đạt.

## 1. Đã làm gì

- **Export FP32.** Giao diện giống các graph Q05: `input`, `input_lengths`, `scales` → `output` [batch, 1, samples], opset 18. Hai trở ngại export trong EdgeTTS được sửa tận gốc. Cả hai đều không đổi con số nào; output PyTorch trước và sau khi sửa giống nhau từng bit, và điều này đã được kiểm.
  - `infer()` lấy số frame bằng `.item()`, khiến khi trace thì giá trị này bị đóng cứng. Wrapper export giữ nó ở dạng tensor.
  - `WN` của flow lưu số kênh trong một buffer int32 dùng làm biên của Slice. Kết quả là Slice có `ends` int32 cạnh `starts` int64, điều mà spec ONNX không cho phép. Bốn buffer này được đổi sang int64 (cùng giá trị 192).
- **PTQ.** Công thức của các graph INT8 lịch sử baseline/MRF:
  - `quantize_static`, QOperator, trọng số QInt8 theo kênh, activation QUInt8, calibration MinMax;
  - các node `Conv` dưới `/enc_p/`, `/dp/` và `/flow/` (109 node mỗi model); decoder giữ FP32.
  - Calibration dùng tập 60 câu chỉ lấy từ train, được thiết kế ngày 2026-10-04 và trước đây chưa từng dùng. Cả 60 câu thuộc train canonical; không câu nào thuộc validation hay test.
- **Chất lượng.** Mỗi graph FP32 và INT8 synthesis Harvard-720 qua một bản *matched-noise*: hai node ngẫu nhiên được thay bằng noise PyTorch của từng câu (seed 1234). Nhờ đó PyTorch, FP32-ONNX và INT8 dùng chung cùng các lần rút ngẫu nhiên.
  - Chấm điểm bằng `harvard/score.py` không sửa đổi (UTMOSv2, Whisper-small, JiWER). Cả sáu bộ được chấm trong cùng một phiên.
- **Tốc độ.** Một phiên trên máy rảnh, 16:55–17:24.
  - Các graph triển khai được đo với noise ngẫu nhiên gốc, ORT CPU, intra-op 2 và 12 thread.
  - Mỗi cấu hình thread chạy vòng A, rồi vòng B theo thứ tự ngược lại.
  - Một lần chạy control PyTorch của Piper_no_VITS2_cpn mở đầu phiên.

## 2. Kiểm tra export và PTQ

| | Piper_no_VITS2_cpn | Piper-ParIR | Piper-SeqIR |
|---|---:|---:|---:|
| Graph FP32 (SHA-256) | `9cd2f70a…` | `d8a70911…` | `2c04e820…` |
| Kích thước FP32 | 63,5 MB | 61,6 MB | 61,3 MB |
| Cổng parity 13 ca: NRMSE lớn nhất | 1,07e-4 | 8,90e-5 | 8,59e-5 |
| Parity Harvard-720, FP32: đạt | 720 / 720 | 720 / 720 | 720 / 720 |
| Parity Harvard-720, FP32: NRMSE lớn nhất / sai số tuyệt đối lớn nhất | 3,3e-4 / 8,7e-4 | 5,9e-4 / 1,3e-3 | 9,6e-4 / 2,0e-3 |
| Graph INT8 (SHA-256) | `7d4b35e4…` | `8ed64208…` | `b0a7ffc3…` |
| Kích thước INT8 (tỉ lệ so với FP32) | 43,3 MB (0,681) | 41,3 MB (0,671) | 41,1 MB (0,670) |
| QLinearConv (enc_p / dp / flow) | 37 / 32 / 40 | 37 / 32 / 40 | 37 / 32 / 40 |
| Initializer decoder giống từng byte với FP32 | 45 / 45 | 101 / 101 | 77 / 77 |
| Lượng tử hoá lần hai giống từng byte | có | có | có |

Ngưỡng parity là NRMSE ≤ 1e-3 và sai số tuyệt đối lớn nhất ≤ 5e-3.

Các kiểm tra khác:
- **Tái lập của SeqIR.** Graph FP32 và INT8 của SeqIR có SHA-256 trùng với lần chạy thử bằng cùng các script ba giờ trước đó.
- **Tính deterministic của PyTorch.** Trong mọi lượt synthesis chất lượng, WAV PyTorch tham chiếu giống từng byte với các lần đánh giá PyTorch đã lưu: 720/720 ở mỗi model. Với SeqIR, đây chính là lần synthesis thứ hai còn thiếu.
- **Parity waveform của INT8 không có ý nghĩa.** Duration predictor đã lượng tử hoá làm số frame đổi một chút, nên chỉ 116, 103 và 108 trên 720 output INT8 có cùng độ dài với PyTorch. Chất lượng INT8 được đánh giá bằng điểm số ở mục 3.

## 3. Chất lượng Harvard-720

| Model | Phiên bản | UTMOSv2 (dự đoán) | Macro WER | Thay thế / mất từ / chèn thêm | Số câu WER = 0 |
|---|---|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | PyTorch (đã lưu) | 3.517 | 0.0612 | 283 / 46 / 13 | 502 |
| | ONNX FP32 | 3.508 | 0.0612 | 283 / 46 / 13 | 502 |
| | **ONNX INT8** | **3.503** | **0.0620** | 268 / 59 / 13 | 501 |
| Piper-ParIR | PyTorch (đã lưu) | 3.530 | 0.0726 | 321 / 71 / 14 | 464 |
| | ONNX FP32 | 3.524 | 0.0726 | 321 / 71 / 14 | 464 |
| | **ONNX INT8** | **3.524** | **0.0772** | 327 / 81 / 18 | 456 |
| Piper-SeqIR | PyTorch (đã lưu) | 3.493 | 0.0696 | 292 / 76 / 14 | 476 |
| | ONNX FP32 | 3.495 | 0.0696 | 292 / 76 / 14 | 476 |
| | **ONNX INT8** | **3.459** | **0.0739** | 316 / 78 / 12 | 469 |

So sánh ghép cặp theo từng câu (`*_int8_vs_fp32.json`, `*_fp32_vs_pytorch.json`):
- Chênh lệch = model được xét − mốc so sánh; CI là bootstrap ghép cặp 95%.
- Holm tính trên hai phép thử của mỗi file.
- FP32 và INT8 được chấm trong cùng một phiên. Điểm PyTorch đã lưu đến từ các phiên trước.

| So sánh | ΔUTMOSv2 [CI 95%] (Wilcoxon p) | ΔWER [CI 95%] (Wilcoxon p) | WER tốt hơn / kém hơn |
|---|---|---|---|
| no_VITS2: INT8 vs FP32 | −0,005 [−0,023, +0,013] (0,74) | +0,001 [−0,006, +0,008] (1,00) | 77 / 73 |
| ParIR: INT8 vs FP32 | +0,000 [−0,018, +0,019] (0,51) | +0,005 [−0,003, +0,012] (0,25) | 91 / 94 |
| SeqIR: INT8 vs FP32 | **−0,036 [−0,056, −0,016] (0,0007; Holm 0,0013)** | +0,004 [−0,003, +0,011] (0,33) | 71 / 85 |
| no_VITS2: FP32 vs PyTorch | −0,009 [−0,021, +0,003] (0,15) | bằng 0 ở mọi câu | 0 / 0 |
| ParIR: FP32 vs PyTorch | −0,007 [−0,018, +0,005] (0,34) | bằng 0 ở mọi câu | 0 / 0 |
| SeqIR: FP32 vs PyTorch | +0,002 [−0,010, +0,014] (0,85) | bằng 0 ở mọi câu | 0 / 0 |

`*_int8_vs_historical.json` so từng graph INT8 với bốn hệ thống đã lưu (Holm trên 8 phép thử). Mọi graph INT8 đều có WER thấp hơn cả bốn:
- thấp hơn MRF 3,0 đến 4,5 điểm;
- thấp hơn SEQ 4,5 đến 6,0 điểm;
- thấp hơn baseline 7,0 đến 8,6 điểm;
- thấp hơn Piper 9,7 đến 11,3 điểm.

## 4. Tốc độ (ONNX Runtime CPU, cùng phiên)

RTF trung bình của vòng A, kèm vòng B (thứ tự ngược) trong ngoặc. Cột "INT8 so với FP32" là mức thay đổi của giá trị trung bình vòng A (ghép cặp; mọi CI đều loại trừ 0, xem `timing_summary.json`).

| Model | 2 thread: FP32 | 2 thread: INT8 | INT8 so với FP32 | 12 thread: FP32 | 12 thread: INT8 | INT8 so với FP32 |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | 0.0420 (0.0439) | 0.0376 (0.0394) | −10,5% (−10,4%) | 0.0292 (0.0271) | 0.0277 (0.0247) | −4,9% (−9,1%) |
| Piper-ParIR | 0.0397 (0.0453) | 0.0341 (0.0399) | −13,9% (−12,0%) | 0.0397 (0.0362) | 0.0366 (0.0348) | −7,8% (−4,0%) |
| Piper-SeqIR | 0.0323 (0.0323) | 0.0269 (0.0270) | −16,8% (−16,5%) | 0.0312 (0.0319) | 0.0297 (0.0291) | −4,8% (−8,5%) |

**So với Piper_no_VITS2_cpn** (cùng precision và số thread, vòng A, ghép cặp):

| | 2 thread, FP32 | 2 thread, INT8 | 12 thread, FP32 | 12 thread, INT8 |
|---|---:|---:|---:|---:|
| Piper-ParIR | −5,5% | −9,1% | **+36,1%** | **+31,9%** |
| Piper-SeqIR | **−23,1%** | **−28,5%** | +6,9% | +7,1% |

**Các control:**
- **Control PyTorch:** Piper_no_VITS2_cpn với 12 thread có RTF trung bình 0,0495, so với 0,0510 ở phiên buổi trưa (×0,97). WAV giống từng byte (720/720).
- **ORT so với PyTorch:** trên cùng model và cùng phiên, ORT FP32 12 thread chạy với RTF bằng 0,59 lần PyTorch.
- **Hiệu ứng thứ tự:** tỉ lệ vòng B / vòng A là 1,00–1,05 với đa số graph ở 2 thread, nhưng 1,14–1,17 với ParIR. Ở 12 thread là 0,89–1,02. Vì vậy chênh lệch vài phần trăm giữa các graph nằm trong mức dao động do thứ tự và giữa các lần chạy.
  - Ở vòng B, kết quả của SeqIR và mọi thứ hạng ở 12 thread vẫn như cũ.
  - Mức lợi nhỏ của ParIR ở 2 thread thì không: ở vòng B, ParIR chậm hơn control (FP32 0,0453 so với 0,0439; INT8 0,0399 so với 0,0394).

## 5. Đọc kết quả

- **Export không làm mất gì.** Cả 720 câu Harvard đều đạt ngưỡng parity FP32, WER của Whisper giống hệt ở mọi câu, và UTMOSv2 lệch dưới 0,01. Mức lệch đó nằm trong dao động của scorer giữa các lần chấm (±0,01–0,02).
- **INT8:**
  - **Kích thước:** graph nhỏ hơn 32–33%.
  - **Tốc độ:** INT8 nhanh hơn FP32 ở mọi model, 10–17% với 2 thread và 4–9% với 12 thread. Chỉ enc_p, dp và flow được lượng tử hoá; decoder giữ FP32. Thời gian phân bổ giữa các phần này chưa được đo.
  - **Chất lượng, Piper_no_VITS2_cpn và ParIR:** không phát hiện thay đổi. Cả hai CI đều chứa 0. WER của ParIR tăng 0,5 điểm, không có ý nghĩa thống kê.
  - **Chất lượng, SeqIR:** UTMOSv2 giảm 0,036, có ý nghĩa sau Holm, còn WER không thay đổi đáng kể.
  - **Thứ hạng:** Piper_no_VITS2_cpn INT8 vẫn có WER thấp nhất trong mọi hệ thống đã đánh giá (0,0620).
- **Tốc độ của decoder phụ thuộc runtime và số thread:**
  - **PyTorch, 12 thread** (phiên trước): SeqIR nhanh hơn 33% và ParIR nhanh hơn 23% so với ResBlock2.
  - **ORT, 2 thread:** SeqIR vẫn nhanh nhất rõ rệt (−23% FP32, −28,5% INT8) ở cả hai vòng. Mức lợi 5–9% của ParIR ở vòng A không lặp lại ở vòng B.
  - **ORT, 12 thread:** thứ tự đảo ngược. ResBlock2 nhanh nhất, SeqIR chậm hơn 7%, ParIR chậm hơn 32–36%.
  - Nguyên nhân chưa được tìm hiểu. Một cách giải thích hợp lý nhưng chưa kiểm chứng: activation SnakeBeta và các khối inverted-residual của decoder IR biến thành nhiều op element-wise nhỏ trong ORT, ít được lợi khi tăng thread, trong khi các convolution lớn hơn của ResBlock2 thì được lợi.
  - Vì vậy nhận định "decoder IR nhanh hơn" đúng với PyTorch và với ORT 2 thread, nhưng không đúng với ORT 12 thread trên CPU này.

Những gì kết quả này **không** chứng minh:
- **Tốc độ trên thiết bị biên.** Đây là CPU máy trạm dưới WSL, mỗi câu đo một lần, không warm-up, hai thứ tự.
- **Các scope hoặc phương pháp PTQ khác.** Decoder không được lượng tử hoá, calibration MinMax trên 60 câu, và chưa thử QAT.
- **Mức giảm chất lượng INT8 của SeqIR là do decoder của nó.** Các phần được lượng tử hoá (enc_p, dp, flow) có cùng kiến trúc ở cả ba model nhưng trọng số train khác nhau.
- **Ý nghĩa qua các seed train,** hay MOS do người chấm.

## 6. Các file

Trong mỗi thư mục model (`piper-no-vits2-cpn/`, `piper-parir-no-vits2-cpn/`, `piper-seqir-no-vits2-cpn/`):

| File | Nội dung | Loại |
|---|---|---|
| `export_record.json` | Hash và cấu trúc graph FP32, kiểm wrapper, đổi kiểu int64, 13 ca parity, kiểm noise ngẫu nhiên gốc, hành vi RNG của ORT | bản ghi phép đo mới |
| `quantize_record.json` | Kiểm calibration và danh sách ID, công thức và scope, cấu trúc INT8, kiểm initializer decoder, tái lập, kích thước | bản ghi phép đo mới |
| `harvard720_{fp32,int8}_synth_record.json` | Synthesis matched-noise: hash graph, SHA-256 WAV theo từng câu, tính deterministic của PyTorch và parity. Các trường thời gian không dùng | phép đo mới |
| `harvard_<model>_{fp32,int8}_results.json` | UTMOSv2 / Whisper / WER theo từng câu, theo schema lịch sử. Trường `rtf` không dùng | phép đo mới |
| `{fp32,int8}_score_record.json` | Định danh scorer và môi trường | bản ghi phép đo mới |
| `*_int8_vs_fp32.json`, `*_fp32_vs_pytorch.json`, `*_int8_vs_historical.json` | Các phép so ghép cặp (UTMOSv2, WER) | phân tích mới |

`timing/`:

| File | Nội dung | Loại |
|---|---|---|
| `t{2,12}_{A,B}_<model>_{fp32,int8}.json` (24 file) | Thời gian ORT theo từng câu của các graph triển khai | phép đo mới |
| `pytorch_ctrl_synth_record.json` | Control PyTorch của phiên | control mới |
| `timing_summary.json` | Giá trị trung bình, kiểm thứ tự, phép thử ghép cặp INT8 vs FP32 và model vs control | phân tích mới |
| `machine_state_before.txt`, `machine_state_after.txt` | Tải máy, mức dùng GPU, các tiến trình bận nhất | bản ghi phiên |

Các graph (`*.onnx`), WAV và log của đợt chạy nằm ngoài Git, trong `training_output/onnx-ptq/`. Hash của chúng có trong các bản ghi.

## 7. Tái lập

Chạy từ thư mục gốc của clone training. Cần một môi trường Python có torch, onnx, onnxruntime và các scorer (`SPY`; đợt này dùng torch 2.13.0, onnx 1.22.0, onnxruntime 1.23.2), cùng venv training (`TPY`). Lệnh giống mục 7 của [README.md](README.md).

Script dừng ở cổng kiểm đầu tiên bị lỗi, và chạy tiếp sau stage cuối cùng đã xong. Các bước riêng lẻ là `scripts/evaluation/onnx/{export,quantize,synthesize,time_graphs,summarize_timing}.py`, rồi đến `scripts/evaluation/harvard/{score,compare}.py`. Test nằm trong `tests/test_onnx_ptq.py`.

**Kết quả mong đợi:**
- Export và lượng tử hoá là deterministic, nên hash graph phải trùng.
- Synthesis là deterministic, nên hash WAV phải trùng.
- UTMOSv2 thay đổi giữa các phiên chấm.
- RTF thay đổi giữa các phiên; chỉ tỉ lệ trong cùng một phiên mới có ý nghĩa.
