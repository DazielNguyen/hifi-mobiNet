# ONNX/PTQ đợt 2: đã gộp weight norm của flow

*English: [README.md](README.md)*

Đợt 2 lặp lại việc export ONNX, PTQ INT8 tĩnh và đánh giá Harvard-720 cho ba model Config A, chỉ sửa đúng một lỗi của đợt 1:
- Piper_no_VITS2_cpn (ResBlock2);
- Piper-ParIR;
- Piper-SeqIR.

Đợt 1 (`../piper-ir-onnx-ptq/`, 2026-10-21) chưa gộp weight norm của flow. Trong các graph đó, 32 trên 40 convolution của flow bị lượng tử hoá lúc chạy, per-tensor, dạng UINT8, thay vì INT8 per-channel tĩnh như đã khai báo.

Mọi thứ khác giữ nguyên protocol của đợt 1 (`docs/evaluation-onnx-ptq.md`, plan `docs/plan-onnx-ptq-v2.vi.md`):
- cùng model, checkpoint, công thức PTQ và scope;
- cùng bộ calibration chỉ lấy từ train, cùng cách synthesis với noise khớp PyTorch, cùng scorer;
- cùng thiết kế đo thời gian.

Mọi con số là **phép đo mới** ngày 2026-10-26.
- Đợt chạy 03:06–07:09 +07 ở commit `f495672`, bằng `scripts/evaluation/onnx/run_round.sh`.
- Mọi bước kiểm tra đều đạt.
- Ngoại lệ duy nhất là phép so sánh giữa hai đợt ở mục 4. Đó là một phân tích mới trên kết quả từng câu đã lưu.

## 1. Điều gì đã thay đổi

- **Gộp trước khi export.** `fold_flow_weight_norm` gọi `WN.remove_weight_norm()` trên cả 4 lớp coupling của flow trước khi trace.
  - Gate mới `flow_weight_norm_fold_changes_only_rounding` so output của model trước và sau khi gộp trên 3 câu. Kết quả **không lệch chút nào** (NRMSE 0, sai số tuyệt đối lớn nhất 0).
- **Không còn weight norm trong scope PTQ.** Gate mới `no_weight_norm_ops_in_ptq_scope` yêu cầu không có `ReduceL2` dưới `/enc_p/`, `/dp/` hay `/flow/`. Các graph không có `ReduceL2` ở bất kỳ đâu (`weight_norm_ops_in_graph` rỗng).
- **Chỉ có trọng số hằng.** Gate mới `every_qlinearconv_weight_constant_per_channel_int8` yêu cầu mọi trọng số QLinearConv là initializer INT8 per-channel cố định. Cả 109 QLinearConv của mỗi model đều đạt.
- **Không đổi:**
  - **công thức PTQ:** QOperator, trọng số QInt8 per-channel, activation QUInt8, calibration MinMax;
  - **scope:** `Conv` dưới enc_p, dp và flow, decoder giữ FP32;
  - **calibration:** 60 câu chỉ lấy từ train;
  - **đánh giá:** synthesis với noise khớp PyTorch, scorer, và các vòng đo A và B ở 2 và 12 thread.

## 2. Kiểm tra export và PTQ

| | Piper_no_VITS2_cpn | Piper-ParIR | Piper-SeqIR |
|---|---:|---:|---:|
| Graph FP32 (SHA-256) | `b4a95a82…` | `506e76dc…` | `a8d18a9e…` |
| Kích thước FP32 | 63,5 MB | 61,5 MB | 61,3 MB |
| Số lớp `WN` của flow đã gộp; output thay đổi | 4; không | 4; không | 4; không |
| `ReduceL2` (weight norm) trong graph | 0 | 0 | 0 |
| Gate parity, 13 trường hợp: NRMSE lớn nhất | 8,71e-5 | 8,77e-5 | 7,86e-5 |
| Parity Harvard-720, FP32: số câu đạt | 720 / 720 | 720 / 720 | 720 / 720 |
| Parity Harvard-720, FP32: NRMSE / sai số tuyệt đối lớn nhất | 3,3e-4 / 8,6e-4 | 6,0e-4 / 1,3e-3 | 1,0e-3 / 2,0e-3 |
| Graph INT8 (SHA-256) | `f54bc09c…` | `a94703b7…` | `ac2353c1…` |
| **Kích thước INT8 (tỉ lệ so với FP32)** | **22,4 MB (0,354)** | **20,5 MB (0,333)** | **20,3 MB (0,331)** |
| Kích thước INT8 đợt 1, để đối chiếu | 43,3 MB | 41,3 MB | 41,1 MB |
| QLinearConv (enc_p / dp / flow) | 37 / 32 / 40 | 37 / 32 / 40 | 37 / 32 / 40 |
| trong đó trọng số INT8 per-channel cố định | 109 / 109 | 109 / 109 | 109 / 109 |
| Initializer của decoder giống từng byte với FP32 | 45 / 45 | 101 / 101 | 77 / 77 |
| Lượng tử hoá lần hai giống từng byte | có | có | có |

Ngưỡng parity là NRMSE ≤ 1e-3 và sai số tuyệt đối lớn nhất ≤ 5e-3. NRMSE lớn nhất của SeqIR trên Harvard-720 là 0,000996, vừa nằm trong ngưỡng, giống đợt 1 (0,00096).

Các kiểm tra khác:
- **PyTorch tất định.** Trong mọi lượt synthesis để chấm chất lượng, WAV tham chiếu của PyTorch giống từng byte với các lần đánh giá PyTorch đã lưu: 720/720 cho mỗi model.
- **Parity dạng sóng của INT8 không có ý nghĩa.** Duration predictor đã lượng tử hoá làm số frame lệch đi một chút. Chỉ 116, 103 và 108 trên 720 output INT8 có cùng độ dài với PyTorch, đúng bằng số của đợt 1. Chất lượng của graph INT8 được đánh giá bằng điểm ở mục 3.
- **Kích thước.** Graph INT8 nằm trong khoảng 20–25 MB mà plan dự kiến, vì 27,8 MB trọng số FP32 của flow nay đã thành INT8. Chúng gần bằng các graph INT8 lịch sử (22,0–24,1 MB).

## 3. Chất lượng trên Harvard-720

| Model | Phiên bản | UTMOSv2 (dự đoán) | Macro WER | Thay thế / xoá / chèn | Số câu có WER 0 |
|---|---|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | PyTorch (đã lưu) | 3.517 | 0.0612 | 283 / 46 / 13 | 502 |
| | ONNX FP32 | 3.517 | 0.0612 | 283 / 46 / 13 | 502 |
| | **ONNX INT8** | **3.506** | **0.0584** | 260 / 54 / 11 | 505 |
| Piper-ParIR | PyTorch (đã lưu) | 3.530 | 0.0726 | 321 / 71 / 14 | 464 |
| | ONNX FP32 | 3.526 | 0.0726 | 321 / 71 / 14 | 464 |
| | **ONNX INT8** | **3.520** | **0.0764** | 335 / 69 / 19 | 457 |
| Piper-SeqIR | PyTorch (đã lưu) | 3.493 | 0.0696 | 292 / 76 / 14 | 476 |
| | ONNX FP32 | 3.497 | 0.0696 | 292 / 76 / 14 | 476 |
| | **ONNX INT8** | **3.474** | **0.0687** | 292 / 75 / 11 | 484 |

Ghép cặp theo câu (`*_int8_vs_fp32.json`, `*_fp32_vs_pytorch.json`):
- Chênh lệch là candidate − reference; khoảng tin cậy là bootstrap ghép cặp 95%.
- Holm tính trên hai phép kiểm định của mỗi file.
- FP32 và INT8 được chấm trong cùng một phiên. Điểm PyTorch đã lưu đến từ các phiên trước.

| So sánh | ΔUTMOSv2 [KTC 95%] (Wilcoxon p) | ΔWER [KTC 95%] (Wilcoxon p) | WER tốt hơn / kém hơn |
|---|---|---|---|
| no_VITS2: INT8 so với FP32 | −0,011 [−0,031; +0,008] (0,27) | −0,003 [−0,009; +0,004] (0,19) | 78 / 67 |
| ParIR: INT8 so với FP32 | −0,007 [−0,025; +0,011] (0,50) | +0,004 [−0,004; +0,012] (0,49) | 91 / 99 |
| SeqIR: INT8 so với FP32 | −0,023 [−0,043; −0,002] (0,058; Holm 0,12) | −0,001 [−0,008; +0,006] (0,80) | 83 / 73 |
| no_VITS2: FP32 so với PyTorch | +0,000 [−0,012; +0,012] (0,91) | 0 ở mọi câu | 0 / 0 |
| ParIR: FP32 so với PyTorch | −0,004 [−0,014; +0,007] (0,64) | 0 ở mọi câu | 0 / 0 |
| SeqIR: FP32 so với PyTorch | +0,004 [−0,009; +0,016] (0,47) | 0 ở mọi câu | 0 / 0 |

`*_int8_vs_historical.json` so mỗi graph INT8 với bốn hệ thống đã lưu (Holm trên 8 phép kiểm định). Mọi graph INT8 đều có WER thấp hơn cả bốn:
- thấp hơn MRF 3,1 đến 4,9 điểm;
- thấp hơn SEQ 4,6 đến 6,4 điểm;
- thấp hơn baseline 7,1 đến 8,9 điểm;
- thấp hơn Piper 9,8 đến 11,6 điểm. Hãy đọc các số này cùng lưu ý về frontend ở ghi chú 6 của `../README.vi.md`.

Về UTMOSv2, mọi graph INT8 đều cao hơn baseline (+0,45 đến +0,49) và thấp hơn MRF (−0,05 đến −0,09) và SEQ (−0,03 đến −0,07). Mọi khoảng tin cậy UTMOSv2 này đều loại trừ 0.

## 4. Đợt 2 so với đợt 1 (INT8)

`*_int8_vs_round1_int8.json` ghép từng kết quả INT8 của đợt 2 với kết quả INT8 đã lưu của đợt 1 cho cùng model. File được tạo bằng `harvard/compare.py --no-historical --metrics utmosv2,wer` sau khi đợt chạy xong.
- WER so transcript của hai graph khác nhau.
- UTMOSv2 còn mang theo dao động của scorer giữa các phiên (±0,01–0,02), vì hai đợt được chấm ở hai phiên khác nhau.

| Model | ΔUTMOSv2, đợt 2 − đợt 1 [KTC 95%] (p) | ΔWER [KTC 95%] (p) | WER tốt hơn / kém hơn |
|---|---|---|---|
| Piper_no_VITS2_cpn | +0,002 [−0,013; +0,017] (0,54) | −0,004 [−0,009; +0,001] (0,26) | 44 / 38 |
| Piper-ParIR | −0,004 [−0,019; +0,010] (0,72) | −0,001 [−0,006; +0,004] (0,84) | 50 / 54 |
| Piper-SeqIR | +0,015 [−0,001; +0,031] (0,052) | −0,005 [−0,011; +0,000] (0,13) | 59 / 35 |

Không chênh lệch nào có ý nghĩa thống kê. Cả hai thay đổi của SeqIR đều nghiêng về phía đợt 2 tốt hơn, xấp xỉ mức mà mức giảm INT8 của nó đã thu hẹp (−0,036 → −0,023 so với FP32).

## 5. Tốc độ (ONNX Runtime CPU, cùng một phiên)

**Phiên đo:** máy rảnh, 03:24–03:47; load average 0,94 lúc bắt đầu, không có training, cả hai GPU đều rảnh.
- **Thứ tự:** một lượt đối chứng PyTorch mở đầu phiên. Sau đó là vòng A, rồi vòng B theo thứ tự ngược lại, cho từng mức thread.
- **Cách đọc bảng:** RTF trung bình của vòng A, vòng B trong ngoặc. "INT8 so với FP32" là thay đổi của giá trị trung bình ở vòng A (vòng B).
- **Ý nghĩa thống kê:** mọi khoảng tin cậy ghép cặp INT8 so với FP32 đều loại trừ 0 (`timing_summary.json`).

| Model | 2 thread: FP32 | 2 thread: INT8 | INT8 so với FP32 | 12 thread: FP32 | 12 thread: INT8 | INT8 so với FP32 |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | 0.0323 (0.0321) | 0.0280 (0.0281) | −13,3% (−12,6%) | **0.0253** (0.0253) | **0.0226** (0.0226) | −10,6% (−10,8%) |
| Piper-ParIR | 0.0338 (0.0333) | 0.0306 (0.0289) | −9,5% (−13,2%) | 0.0318 (0.0321) | 0.0290 (0.0292) | −8,6% (−9,1%) |
| Piper-SeqIR | **0.0278** (0.0276) | **0.0238** (0.0232) | −14,5% (−15,8%) | 0.0267 (0.0282) | 0.0243 (0.0244) | −9,0% (−13,3%) |

**So với Piper_no_VITS2_cpn** (cùng độ chính xác và số thread; vòng A, vòng B trong ngoặc; mọi khoảng tin cậy của vòng A đều loại trừ 0):

| | 2 thread, FP32 | 2 thread, INT8 | 12 thread, FP32 | 12 thread, INT8 |
|---|---:|---:|---:|---:|
| Piper-ParIR | +4,6% (+3,6%) | +9,1% (+2,9%) | +25,6% (+27,1%) | +28,4% (+29,4%) |
| Piper-SeqIR | **−14,1% (−14,1%)** | **−15,3% (−17,3%)** | +5,4% (+11,4%) | +7,3% (+8,2%) |

**Đối chứng:**
- **Đối chứng PyTorch:** Piper_no_VITS2_cpn với 12 thread cho RTF trung bình 0,0388, so với 0,0510 ở phiên đo ngày 2026-10-21 (×0,76). WAV giống từng byte (720/720).
  - Tức là máy chạy nhanh hơn khoảng một phần tư trong phiên này.
  - **Chỉ so RTF trong phiên này, không so với đợt 1.**
- **ORT so với PyTorch:** ORT FP32 với 12 thread chạy Piper_no_VITS2_cpn ở mức 0,65× RTF của PyTorch trong cùng phiên (đợt 1 là 0,59×).
- **Ảnh hưởng của thứ tự:** tỉ số vòng B / vòng A là 0,94–1,00 ở 2 thread và 1,00–1,06 ở 12 thread. Chênh lệch vài phần trăm giữa các graph nằm trong mức dao động đó.

**Khác gì so với đợt 1** (chỉ so các tỉ số trong cùng phiên):
- **Lợi ích của INT8:** ở 12 thread, INT8 nay nhanh hơn FP32 9–13%, trong khi đợt 1 chỉ 4–9%. Ở 2 thread mức lợi tương tự (10–16%, so với 10–17%). Graph INT8 đợt 1 phải tính lại weight norm của flow và lượng tử hoá trọng số flow ở mỗi lần chạy.
- **Thứ hạng giữ nguyên:**
  - ở 2 thread, SeqIR nhanh nhất (−14% đến −17% so với ResBlock2);
  - ở 12 thread, ResBlock2 nhanh nhất, SeqIR chậm hơn 5–11% và ParIR chậm hơn 26–29%;
  - nguyên nhân không đổi: `Sin` của SnakeBeta chạy đơn luồng trong ONNX Runtime 1.23, đã profile trong `../piper-ir-onnx-ptq/profile/`. Decoder là FP32 và giống hệt nhau ở cả hai đợt.
- **ParIR ở 2 thread** nay chậm hơn ResBlock2 một chút (+3% đến +9%). Mức nhanh hơn 5–9% của nó ở vòng A đợt 1 vốn đã không lặp lại ở vòng B.

## 6. Cách đọc kết quả

- **Bản sửa hoạt động đúng như dự định:**
  - mọi convolution của flow nay là INT8 per-channel tĩnh;
  - graph INT8 chỉ bằng một phần ba kích thước FP32 (20,3–22,4 MB, so với 41–43 MB ở đợt 1);
  - INT8 nhanh hơn FP32 với mọi model và mọi mức thread.
- **Export vẫn không mất mát.** FP32 đạt parity ở cả 720 câu, WER giống PyTorch ở từng câu, và UTMOSv2 lệch dưới 0,005.
- **Chất lượng INT8:**
  - **WER:** không model nào thay đổi đáng kể so với graph FP32 của nó.
  - **Piper_no_VITS2_cpn INT8** có WER thấp nhất trong mọi hệ thống đã đánh giá (0,0584). Đây không phải là cải thiện phát hiện được so với FP32 (khoảng tin cậy chứa 0).
  - **Piper_no_VITS2_cpn và ParIR:** không phát hiện thay đổi UTMOSv2.
  - **SeqIR:**
    - UTMOSv2 thấp hơn graph FP32 của nó 0,023. Khoảng tin cậy bootstrap loại trừ 0, nhưng kiểm định Wilcoxon không đạt 0,05, kể cả trước và sau Holm. Hãy coi đây là một mức giảm nhỏ có thể có, chưa phải đã xác lập.
    - Ở đợt 1 mức giảm là 0,036 và có ý nghĩa sau Holm. Phép so sánh giữa hai đợt (mục 4) không cho thấy thay đổi có ý nghĩa. Vì vậy trọng số flow per-tensor nhiều nhất chỉ giải thích được một phần mức giảm đó; đợt này chưa xác lập được điều đó.
- **Tốc độ:** thứ hạng decoder vẫn phụ thuộc vào runtime và số thread, như ở đợt 1.

Kết quả này **không** cho thấy:
- **Tốc độ trên thiết bị edge.** Đây là CPU máy trạm chạy WSL, đo một lần mỗi câu, không warm-up, hai thứ tự.
- **Các scope hoặc phương pháp PTQ khác.** Decoder không được lượng tử hoá, calibration MinMax dùng 60 câu, và chưa thử QAT.
- **Rằng thay đổi INT8 còn lại của SeqIR là do decoder của nó.** Các phần được lượng tử hoá (enc_p, dp, flow) có cùng kiến trúc ở cả ba model nhưng trọng số train khác nhau.
- **Ý nghĩa thống kê qua nhiều seed train,** hay MOS do người chấm.

## 7. Các file

Trong mỗi thư mục model (`piper-no-vits2-cpn/`, `piper-parir-no-vits2-cpn/`, `piper-seqir-no-vits2-cpn/`):

| File | Nội dung | Loại |
|---|---|---|
| `export_record.json` | Hash và cấu trúc graph FP32, việc gộp weight norm của flow và kiểm tra của nó, số `ReduceL2`, wrapper bằng `infer()`, ép kiểu int64, 13 trường hợp parity, kiểm tra noise gốc | bản ghi phép đo mới |
| `quantize_record.json` | Kiểm tra calibration và danh sách ID, công thức và scope, cấu trúc INT8 và loại trọng số QLinearConv, kiểm tra initializer của decoder, tính tái lập, kích thước | bản ghi phép đo mới |
| `harvard720_{fp32,int8}_synth_record.json` | Synthesis với noise khớp: hash graph, SHA-256 WAV từng câu, tính tất định và parity của PyTorch. Các trường thời gian không dùng | phép đo mới |
| `harvard_<model>_{fp32,int8}_results.json` | UTMOSv2 / Whisper / WER từng câu, schema lịch sử. Trường `rtf` không dùng | phép đo mới |
| `{fp32,int8}_score_record.json` | Định danh và môi trường của scorer | bản ghi phép đo mới |
| `*_int8_vs_fp32.json`, `*_fp32_vs_pytorch.json`, `*_int8_vs_historical.json` | So sánh ghép cặp (UTMOSv2, WER) | phân tích mới |
| `*_int8_vs_round1_int8.json` | INT8 đợt 2 so với INT8 đợt 1, ghép cặp (mục 4) | phân tích mới trên kết quả đã lưu |

`timing/`:

| File | Nội dung | Loại |
|---|---|---|
| `t{2,12}_{A,B}_<model>_{fp32,int8}.json` (24 file) | Thời gian ORT từng câu của các graph triển khai | phép đo mới |
| `pytorch_ctrl_synth_record.json` | Lượt đối chứng PyTorch của phiên | đối chứng mới |
| `timing_summary.json` | Giá trị trung bình, kiểm tra thứ tự, kiểm định ghép cặp INT8 so với FP32 và model so với đối chứng | phân tích mới |
| `machine_state_before.txt`, `machine_state_after.txt` | Load, mức dùng GPU và các tiến trình nặng nhất | bản ghi phiên |

Các graph (`*.onnx`), WAV và log của đợt nằm ngoài Git trong `training_output/onnx-ptq-v2/`. Hash của chúng có trong các bản ghi.

## 8. Tái lập

Từ thư mục gốc của bản clone dùng để train, với môi trường scorer (`SPY`: torch 2.13.0, onnx 1.22.0, onnxruntime 1.23.2) và venv train (`TPY`):

```sh
SPY=<scorer python> TPY=<training venv python> OUT=training_output/onnx-ptq-v2 \
  RESULTS=<repository>/results/piper-ir-onnx-ptq-v2 \
  bash <repository>/scripts/evaluation/onnx/run_round.sh
```

`OUT` phải là thư mục mới. Nếu thư mục đã tồn tại, script sẽ chạy tiếp đợt đó từ sau bước cuối cùng đã xong. Phép so sánh giữa hai đợt ở mục 4 được chạy sau đó, cho từng model:

```sh
$SPY <repository>/scripts/evaluation/harvard/compare.py \
  --candidate results/piper-ir-onnx-ptq-v2/<model>/harvard_<id>_int8_results.json --label <id>_int8_round2 \
  --extra-reference <id>_int8_round1=results/piper-ir-onnx-ptq/<model>/harvard_<id>_int8_results.json \
  --no-historical --metrics utmosv2,wer --output results/piper-ir-onnx-ptq-v2/<model>/<model>_int8_vs_round1_int8.json
```

Test nằm trong `tests/test_onnx_ptq.py`; cả 13 test đều đạt trước khi chạy đợt này.

**Kết quả mong đợi:**
- Export và lượng tử hoá là tất định, nên hash graph phải trùng.
- Synthesis là tất định, nên hash WAV phải trùng.
- UTMOSv2 dao động giữa các phiên chấm điểm.
- RTF dao động giữa các phiên; chỉ các tỉ số trong cùng phiên mới có ý nghĩa.
