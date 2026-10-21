# Kết quả

*English: [README.md](README.md)*

Thư mục này chứa mọi kết quả đã lưu của hai thế hệ model trong hifi-mobiNet:
- **Thế hệ 1:** các model lịch sử của BanhmiTTS: baseline HiFi-GAN ResBlock2 có thành phần VITS2, Parallel-IR (MRF), Sequential-IR (SEQ) và checkpoint Piper bên ngoài.
- **Thế hệ 2:** các model Config A có kiểm soát của tháng 10/2026: Piper_no_VITS2_cpn (ResBlock2), Piper-ParIR và Piper-SeqIR.

`manifest.json` liệt kê mọi file kèm SHA-256, phân loại và giới hạn.

| Mục | Nội dung |
|---|---|
| [Kết quả lịch sử](#kết-quả-lịch-sử-thế-hệ-1) | `historical/`, `prior_audit/`, `manifests/` |
| [Phép đo mới](#phép-đo-mới-thế-hệ-2-pytorch) | `piper-no-vits2-cpn/`, `piper-parir-no-vits2-cpn/`, `piper-seqir-no-vits2-cpn/`, `piper-ir-timing-session/` |
| [Export ONNX và PTQ INT8](#export-onnx-và-ptq-int8) | `piper-ir-onnx-ptq/` cùng các bản ghi ONNX, PTQ và QAT lịch sử |
| [So sánh bảy model](#so-sánh-bảy-model) | một bảng gồm mọi metric của bảy model trên Harvard-720 |
| [Kích thước model](#kích-thước-model) | `model_parameters.json` |

Bản tổng quan về mọi bằng chứng SEQ/PAR và mức sẵn sàng cho paper journal nằm ở `docs/q2-readiness-seq-par.vi.md`.

## Kết quả lịch sử (thế hệ 1)

**Nguồn gốc các file:**
- **Kết quả:** các file kết quả trong `historical/harvard/`, `historical/lj500/` và `historical/onnx_historical/` giống từng byte với các file trong gói bằng chứng journal. Hash, vị trí nguồn và giới hạn của chúng nằm trong `manifest.json` và `docs/source-map.json`.
- **Các đợt nhập sau:** các thư mục nhập ngày 2026-10-22 được mô tả bên dưới.
- **Không có gì mới:** không có phép đo nghiên cứu mới nào được tạo ra để điền vào các thư mục này.

| Model | RTF trung bình (Harvard) | UTMOSv2 (dự đoán) | Macro WER |
|---|---:|---:|---:|
| HiFi-GAN ResBlock2 | 0.04177 | 3.0260 | 0.1476 |
| Parallel-IR | 0.03388 | 3.5680 | 0.1073 |
| Sequential-IR | 0.03081 | 3.5443 | 0.1222 |
| Checkpoint Piper bên ngoài | 0.03916 | 3.4507 | 0.1747 |

**Giới hạn của các quan sát này:**
- Đây là **các quan sát lịch sử bằng PyTorch trên CPU**, trên 720 câu Harvard chung, đo trên máy phát triển chạy WSL. Đây không phải kết quả trên thiết bị biên hay ONNX.
- UTMOSv2 không phải MOS do người chấm.
- Mỗi model chỉ có một lần train, mỗi câu chỉ sinh một lần, và không có protocol đo thời gian lặp lại.
- Phiên bản runtime lịch sử được suy ra từ các bản ghi gần đó; chúng không được hash trong từng lần chạy.
- **Các bản export Q05:** các bản export FP32 Q05 đã được định danh và qua kiểm tra kỹ thuật, nhưng điểm chất lượng ở trên không gắn với chúng. Demo chỉ là giao diện chạy được; nó không chứng minh code mới tái tạo được bảng trên.

**Các thư mục:**
- **`lj500/`:** giữ cả lần đánh giá MRF epoch 1386 đã chọn (`eval_mrf_final1500_500_results.json`) lẫn một bảng MRF cũ hơn; tên file và giá trị giữ nguyên.
  - Tập test của MRF và tập test LJSpeech canonical chỉ chung 21 ID. Không được coi mọi file 500 dòng là một phép thử ghép cặp chung.
- **`manifests/`:**
  - Gồm split canonical (được ghi nhận về sau) và danh sách ID có thứ tự đã đóng băng của Q04.
  - Điều này không chứng minh mọi manifest đều được ghi lại cùng lúc với lúc train.
- **`onnx_historical/`:** chỉ mang tính thăm dò, được giữ để minh bạch, không phải kết luận ONNX/PTQ cuối cùng (xem [PTQ và QAT lịch sử](#ptq-và-qat-lịch-sử-thế-hệ-1)):
  - graph MRF cũ không gắn với checkpoint đã chọn;
  - dữ liệu calibration lịch sử trùng với dữ liệu test;
  - scope PTQ khác nhau giữa các model.
- **`prior_audit/`:** một bản audit thống kê về sau, giữ nguyên không sửa.
  - Script thống kê mới (chạy được trên mọi máy) tái tạo bản audit đó vào `outputs/`, không đụng vào dữ liệu gốc.
  - Phép thử ghép cặp và khoảng bootstrap chỉ nói về dao động giữa các câu, không nói về dao động giữa các lần train hay giữa các phần cứng.
  - Không có ý nghĩa thống kê không có nghĩa là tương đương.
  - Khác biệt về công thức train và split không cho phép kết luận khác biệt chỉ do decoder.

Nhập ngày 2026-10-22 (`docs/evidence/historical-import-2026-10-22.json`). Byte kết quả giữ nguyên, và script đã được che đường dẫn máy.
- **`lj500_development/`:** thêm các bảng LJ-500 lịch sử.
  - Các checkpoint baseline trước đó.
  - Quá trình train của MRF: epoch 74, 216, 353, 480, 836 và 1170.
  - Các tiền thân của SEQ (`mbconv`); checkpoint của chúng đã bị xoá.
  - Các biến thể ResBlock2 có Snake, MRD và BigVGAN.
  - Script nằm trong `records/`. Log không được Git theo dõi (repository bỏ qua `*.log`); bản ghi nhập liệt kê từng log theo hash.
- **`qat/`:** QAT của baseline và MRF. Các bảng này gồm cả số epoch train thêm và thời gian đo bằng PyTorch fake-quant, nên không phải kết quả của runtime INT8.
- **`ptq_sweep/`:** PTQ sweep lịch sử để chọn scope.
  - Evaluator WER của nó có lỗi sample rate đã biết (J-E026).
  - Các file kết quả là bản dẫn xuất đã bỏ các mục Vocos (`*.without_vocos.json`). Log có chứa run Vocos chỉ được ghi hash.
- **Tập câu:** các bảng LJ-500 dùng ba tập 500 câu khác nhau. Chỉ so các bảng trong cùng một tập.

## Phép đo mới (thế hệ 2, PyTorch)

**`piper-no-vits2-cpn/`** (README tiếng Việt: [piper-no-vits2-cpn/README.vi.md](piper-no-vits2-cpn/README.vi.md), 2026-10-10/11): các phép đo mới đầu tiên trong repository này.
- Đây là đánh giá Harvard-720 của `Piper_no_VITS2_cpn`, tức baseline không có các thành phần VITS2.
- Thư mục gồm README riêng, kết quả theo từng câu, bản ghi của scorer, phép so ghép cặp với các file lịch sử ở trên, và các control chạy trong cùng phiên.
- Các file này bổ sung cho bảng lịch sử, không thay thế nó.
- RTF lịch sử không so sánh được với RTF của phiên đó: trên cùng model baseline đã đo được độ trôi thời gian +8,3%.

**`piper-parir-no-vits2-cpn/`** (README tiếng Việt: [piper-parir-no-vits2-cpn/README.vi.md](piper-parir-no-vits2-cpn/README.vi.md), 2026-10-16): đánh giá chất lượng Harvard-720 của `Piper_ParIR_no_VITS2_cpn`.
- Model dùng cùng backbone, thay bằng bộ decoder Parallel-IR đã phát hành: decoder inverted-residual, SnakeBeta và MRD.
- Model được so ghép cặp theo từng câu với `Piper_no_VITS2_cpn` và bốn hệ thống lịch sử.
- Có control scorer trong cùng phiên, bằng cách chấm lại các WAV của `Piper_no_VITS2_cpn`.
- RTF được đo sau đó, trong phiên ngày 2026-10-21.

**`piper-seqir-no-vits2-cpn/`** (README tiếng Việt: [piper-seqir-no-vits2-cpn/README.vi.md](piper-seqir-no-vits2-cpn/README.vi.md), 2026-10-21): đánh giá `Piper_SeqIR_no_VITS2_cpn`, cùng backbone, thay bằng bộ decoder Sequential-IR đã phát hành (hai khối IR mỗi tầng, SnakeBeta, MRD).
- Model được so ghép cặp theo từng câu với hai model Config A còn lại và bốn hệ thống lịch sử.
- Có bản chấm lại cùng phiên cho các WAV của `Piper_no_VITS2_cpn` và ParIR.
- README của nó cũng báo cáo phiên đo thời gian trên máy rảnh, đo liền nhau ba model Config A với `Piper_no_VITS2_cpn` làm control. Bản ghi nằm trong `piper-ir-timing-session/`.

| Model mới | UTMOSv2 (dự đoán) | Macro WER | RTF trung bình, phiên 2026-10-21 | RTF tương đối |
|---|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2, control) | 3.517 | 0.0612 | 0.0510 | — |
| Piper_ParIR_no_VITS2_cpn | 3.530 | 0.0726 | 0.0392 | −23,2% |
| Piper_SeqIR_no_VITS2_cpn | 3.493 | 0.0696 | 0.0340 | −33,3% |

Cột RTF lấy từ một phiên duy nhất và chỉ so sánh được trong nội bộ phiên đó. Ở phiên này control chậm hơn 1,21 lần so với phiên riêng ngày 2026-10-10 (0.0421), nên không thể đặt giá trị đó, hay bảng RTF lịch sử, cạnh các giá trị này.

## Export ONNX và PTQ INT8

### Thế hệ 2: đợt ngày 2026-10-21

`piper-ir-onnx-ptq/` ([README](piper-ir-onnx-ptq/README.vi.md)) chứa bản export ONNX FP32, PTQ INT8 tĩnh và đánh giá Harvard-720 của ba model Config A. Protocol `docs/evaluation-onnx-ptq.md` được chốt trước khi chạy. Script nằm trong `scripts/evaluation/onnx/`.

**Thiết lập:**
- **Export:**
  - opset 18, giao diện như Q05;
  - wrapper export tái hiện đúng `infer()` của EdgeTTS;
  - cổng parity NRMSE ≤ 1e-3 và sai số tuyệt đối lớn nhất ≤ 5e-3.
- **PTQ** (công thức của các graph lịch sử baseline/MRF):
  - `quantize_static` của ONNX Runtime, QOperator, trọng số QInt8 per-channel, activation QUInt8, calibration MinMax;
  - các node `Conv` dưới `/enc_p/`, `/dp/` và `/flow/`; decoder giữ FP32.
  - **Calibration:** tập 60 câu chỉ lấy từ train (không câu nào thuộc validation hay test).
- **Chất lượng:** mỗi graph synthesis Harvard-720 qua một bản matched-noise dùng noise PyTorch của từng câu. Cả sáu bộ được chấm trong một phiên bằng `harvard/score.py`.
- **Tốc độ:**
  - một phiên trên máy rảnh; ORT CPU, tuần tự, intra-op 2 và 12 thread;
  - vòng A, rồi vòng B theo thứ tự ngược lại;
  - mỗi câu đo một lần, không warm-up.

**Các kiểm tra:**

| | Piper_no_VITS2_cpn | Piper-ParIR | Piper-SeqIR |
|---|---:|---:|---:|
| Cổng parity 13 ca: NRMSE lớn nhất | 1,07e-4 | 8,90e-5 | 8,59e-5 |
| Parity Harvard-720, FP32 | 720 / 720 | 720 / 720 | 720 / 720 |
| QLinearConv (enc_p / dp / flow) | 37 / 32 / 40 | 37 / 32 / 40 | 37 / 32 / 40 |
| Initializer decoder giống từng byte với FP32 | có | có | có |
| Lượng tử hoá lần hai giống từng byte | có | có | có |
| WAV PyTorch tham chiếu = lần đánh giá đã lưu | 720 / 720 | 720 / 720 | 720 / 720 |

**Chất lượng trên Harvard-720:**

| Model | PyTorch (đã lưu) UTMOSv2 / WER | ONNX FP32 UTMOSv2 / WER | ONNX INT8 UTMOSv2 / WER | INT8 − FP32: ΔUTMOSv2 [CI 95%] | INT8 − FP32: ΔWER [CI 95%] |
|---|---|---|---|---|---|
| Piper_no_VITS2_cpn | 3.517 / 0.0612 | 3.508 / 0.0612 | 3.503 / **0.0620** | −0,005 [−0,023, +0,013] | +0,001 [−0,006, +0,008] |
| Piper-ParIR | 3.530 / 0.0726 | 3.524 / 0.0726 | **3.524** / 0.0772 | +0,000 [−0,018, +0,019] | +0,005 [−0,003, +0,012] |
| Piper-SeqIR | 3.493 / 0.0696 | 3.495 / 0.0696 | 3.459 / 0.0739 | **−0,036 [−0,056, −0,016]** (Holm 0,0013) | +0,004 [−0,003, +0,011] |

- **Export:** FP32-ONNX so với PyTorch có WER giống hệt ở mọi câu, và UTMOSv2 lệch dưới 0,01, tức là dao động của scorer.
- **INT8:** không phát hiện thay đổi với Piper_no_VITS2_cpn và ParIR. SeqIR giảm 0,036 UTMOSv2, còn WER không thay đổi đáng kể.

**Tốc độ** (RTF trung bình vòng A; vòng B trong ngoặc; mọi CI của INT8 so với FP32 đều loại trừ 0):

| Model | 2 thread FP32 | 2 thread INT8 | INT8 so với FP32 | 12 thread FP32 | 12 thread INT8 | INT8 so với FP32 |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | 0.0420 (0.0439) | 0.0376 (0.0394) | −10,5% | 0.0292 (0.0271) | 0.0277 (0.0247) | −4,9% |
| Piper-ParIR | 0.0397 (0.0453) | 0.0341 (0.0399) | −13,9% | 0.0397 (0.0362) | 0.0366 (0.0348) | −7,8% |
| Piper-SeqIR | **0.0323** (0.0323) | **0.0269** (0.0270) | −16,8% | 0.0312 (0.0319) | 0.0297 (0.0291) | −4,8% |

- **Phụ thuộc số thread:** ở 2 thread SeqIR nhanh nhất, thấp hơn ResBlock2 23–28%. Ở 12 thread ResBlock2 nhanh nhất, còn ParIR chậm hơn 32–36%.
- **Nguyên nhân, theo profile kernel trong `piper-ir-onnx-ptq/profile/`:** `Sin` của SnakeBeta không dùng thread pool của ONNX Runtime 1.23, và còn chậm đi khi các thread rảnh spinning, trong khi convolution dày của ResBlock2 tăng tốc theo số thread.
- **ORT so với PyTorch:** ở 12 thread, ORT FP32 chạy model ResBlock2 với RTF bằng 0,59 lần PyTorch trong cùng phiên.

**Kích thước graph:**

| Model | ONNX FP32 | ONNX INT8 | INT8 / FP32 |
|---|---:|---:|---:|
| Piper_no_VITS2_cpn | 63,5 MB | 43,3 MB | 0,681 |
| Piper-ParIR | 61,6 MB | 41,3 MB | 0,671 |
| Piper-SeqIR | 61,3 MB | 41,1 MB | 0,670 |

**Lỗi đã biết của đợt này (phát hiện ngày 2026-10-22): weight norm của flow chưa được gộp trước khi export.**
- **Sai ở đâu:**
  - Lúc export chỉ bỏ weight norm của decoder, như `harvard/synthesize.py` làm.
  - Các lớp coupling của flow trong EdgeTTS (`WN`) vẫn giữ weight norm. Vì vậy trong graph, 32 trọng số convolution của chúng được tính lại ở mỗi lần chạy (weight_g · weight_v / ‖weight_v‖, chính là 32 node `ReduceL2`).
  - Bộ lượng tử hoá không coi các trọng số đó là hằng số được. Nó lượng tử hoá chúng **lúc chạy, theo từng tensor, dạng UINT8**, thay vì INT8 per-channel tĩnh như đã khai báo.
  - Chỉ 8 trên 40 convolution của flow đúng công thức đã khai báo.
  - Exporter Q05 gộp weight norm của flow khi module flow có `remove_weight_norm()`. Khối flow của EdgeTTS thì không có; chỉ các lớp `WN` bên trong mới có.
- **Hệ quả:**
  - **Kích thước:** graph INT8 vẫn lưu 27,8 MB trọng số flow dạng FP32. Vì vậy các kích thước ở trên thấp hơn mức mà scope PTQ này có thể đạt; để so sánh, graph INT8 lịch sử chỉ 22–24 MB.
  - **Tốc độ:** RTF của INT8 có tính cả phần tính weight norm và lượng tử hoá trọng số ở mỗi lần chạy.
  - **Chất lượng:** kết quả INT8 đo được là của chính các graph này. Chưa biết mức giảm UTMOSv2 của SeqIR có phải do trọng số flow per-tensor hay không.
  - **Những gì vẫn đúng:** kết quả FP32, kết quả parity và phân tích tốc độ decoder không bị ảnh hưởng. Graph FP32 và INT8 đều mang cùng một flow chưa gộp, và decoder là FP32 ở cả hai.
- **Cách sửa:** export lại, gọi `remove_weight_norm()` trên mọi lớp coupling của flow, rồi lượng tử hoá, đánh giá và đo thời gian lại. Việc này chưa được chạy; nó đã được chuẩn bị trong `docs/plan-onnx-ptq-v2.vi.md`.

### PTQ và QAT lịch sử (thế hệ 1)

Các bản ghi này được giữ để minh bạch. Chúng khác đợt thế hệ 2 ở những điểm khiến không thể so sánh trực tiếp:

| | PTQ lịch sử (09/2026) | Đợt thế hệ 2 (2026-10-21) |
|---|---|---|
| Graph | graph benchmark. Baseline và SEQ gắn được với checkpoint; graph MRF thì **không** (0/478 trọng số cùng tên trùng checkpoint 1386) | export từ các checkpoint đã chọn, có hash trong metadata của graph |
| Calibration | 60 câu ngẫu nhiên, 57 câu train và **3 câu test** (J-E025) | 60 câu chỉ lấy từ train |
| Scope | baseline/MRF: Conv trong enc_p/dp/flow (133 QLinearConv); **SEQ thêm MatMul** (+40 QLinearMatMul) | cùng một scope Conv cho cả ba (109 QLinearConv) |
| Cách chọn scope | sweep với evaluator bị lỗi (J-E026) | chốt trong protocol trước khi chạy |
| Weight norm của flow | không được ghi lại; kích thước INT8 22–24 MB gợi ý trọng số flow được lưu dạng INT8 | **chưa gộp** (xem lỗi ở trên) |
| Tập đánh giá | LJ-500, MRF dùng tập câu riêng | Harvard-720, noise khớp với PyTorch |
| Kích thước file INT8 | baseline 24,1 MB, MRF 22,2 MB, SEQ 22,0 MB | 41,1–43,3 MB |

**Kết quả ONNX lịch sử** (LJ-500, ORT CPU 2 thread, `historical/onnx_historical/`):

| Model (tập) | FP32 RTF / UTMOSv2 / WER | INT8 RTF / UTMOSv2 / WER |
|---|---|---|
| baseline 1489 (tập baseline) | 0.0388 / 3.132 / 0.1230 | 0.0345 / 3.121 / 0.1135 |
| MRF, graph không gắn được (tập MRF) | 0.0416 / 3.537 / 0.1043 | 0.0371 / 3.493 / 0.1030 (chạy lại 3.490 / 0.1124) |
| SEQ 1442, scope INT8 rộng hơn (tập baseline) | 0.0346 / 3.571 / 0.1064 | 0.0289 / 3.586 / 0.1081 |

**QAT lịch sử** (`historical/qat/`, LJ-500):
- baseline QAT ở epoch 144: UTMOSv2 3,549–3,569, WER 0,079–0,082;
- MRF QAT ở epoch 97: UTMOSv2 3,630–3,650, WER 0,098–0,108.
- Không thể quy phần cải thiện này cho lượng tử hoá: QAT train thêm 97–144 epoch, và RTF của nó là PyTorch với op fake-quant. Không có bản ghi QAT nào cho SEQ.

**Scope sweep lịch sử** (`historical/ptq_sweep/`): trên baseline, lượng tử hoá flow + enc_p + dp cho WER thấp nhất trong các scope đã thử (0,1215, 23,0 MB). Evaluator có lỗi sample rate, nên đây chỉ là thứ hạng.

## So sánh bảy model

Cả bảy model trên Harvard-720 (720 câu, 5.742 từ tham chiếu), cùng protocol synthesis và chấm điểm (`docs/evaluation.md`).
- Mọi giá trị được chép từ các bản ghi liệt kê dưới bảng; không đo lại gì cho bảng này.
- **In đậm** là giá trị tốt nhất của mỗi dòng. Với RTF, giá trị tốt nhất chỉ được đánh dấu trong từng nhóm đo (ghi chú 3).

| Metric | Piper_no_VITS2_cpn | Piper-ParIR | Piper-SeqIR | Baseline ResBlock2 | Parallel-IR (MRF) | Sequential-IR (SEQ) | Piper (bên ngoài) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ***Model*** | | | | | | | |
| Decoder | ResBlock2 | Parallel-IR | Sequential-IR | ResBlock2 | Parallel-IR | Sequential-IR | ResBlock2 |
| Backbone / công thức train | Config A, không VITS2 | Config A, không VITS2 | Config A, không VITS2 | baseline VITS2 | baseline VITS2 | baseline VITS2 | Piper đã công bố |
| Epoch được chọn | 1403 | 1486 | 1455 | 1489 | 1386 | 1442 | 999 |
| `val_loss_mel` trên validation ¹ | 19.7719 | **19.0069** | 19.0332 | 19.7882 | (18.7681) | (19.0136) | — |
| ***Chất lượng, PyTorch*** | | | | | | | |
| UTMOSv2, trung bình (dự đoán) ² | 3.517 | 3.530 | 3.493 | 3.026 | **3.568** | 3.544 | 3.451 |
| UTMOSv2, SD | 0.250 | 0.251 | 0.267 | 0.303 | 0.236 | 0.246 | 0.259 |
| Macro WER | **0.0612** | 0.0726 | 0.0696 | 0.1476 | 0.1073 | 0.1222 | 0.1747 |
| Corpus WER | **0.0596** | 0.0707 | 0.0665 | 0.1437 | 0.1033 | 0.1186 | 0.1696 |
| Thay thế / mất từ / chèn thêm | **283** / **46** / **13** | 321 / 71 / 14 | 292 / 76 / 14 | 681 / 101 / 43 | 467 / 109 / 17 | 531 / 111 / 39 | 812 / 112 / 50 |
| Số câu WER = 0 (trên 720) | **502** | 464 | 476 | 318 | 381 | 351 | 228 |
| Phiên chấm điểm | 2026-10-10 | 2026-10-16 | 2026-10-21 | lịch sử | lịch sử | lịch sử | lịch sử |
| ***Tốc độ, PyTorch FP32, CPU, 12 thread*** ³ | | | | | | | |
| RTF trung bình | 0.0510 | 0.0392 | **0.0340** | 0.0418 | 0.0339 | **0.0308** | 0.0392 |
| RTF trung vị | 0.0490 | 0.0374 | **0.0320** | 0.0412 | 0.0329 | **0.0296** | 0.0379 |
| RTF trung bình so với model ResBlock2 cùng nhóm | — | −23,2% | −33,3% | — | −18,9% | −26,2% | (−6,2%) |
| Nhóm đo thời gian | 2026-10-21 | 2026-10-21 | 2026-10-21 | lịch sử | lịch sử | lịch sử | lịch sử |
| ***Triển khai, ONNX Runtime CPU*** ⁴ | | | | | | | |
| ONNX FP32: UTMOSv2 / macro WER | 3.508 / **0.0612** | **3.524** / 0.0726 | 3.495 / 0.0696 | — | — | — | — |
| ONNX INT8: UTMOSv2 / macro WER | 3.503 / **0.0620** | **3.524** / 0.0772 | 3.459 / 0.0739 | — | — | — | — |
| RTF ORT, 2 thread: FP32 / INT8 | 0.0420 / 0.0376 | 0.0397 / 0.0341 | **0.0323 / 0.0269** | — | — | — | — |
| RTF ORT, 12 thread: FP32 / INT8 | **0.0292 / 0.0277** | 0.0397 / 0.0366 | 0.0312 / 0.0297 | — | — | — | — |
| ***Kích thước*** ⁵ | | | | | | | |
| Tham số (định nghĩa Q05) | 16.425.776 | 15.918.032 | **15.870.768** | 17.620.784 | 17.113.040 | 17.065.776 | 16.425.776 |
| Tham số dùng khi inference | 15.663.771 | 15.156.027 | **15.108.763** | — | — | — | — |
| Tham số decoder | 1.664.768 | 1.157.024 | **1.109.760** | — | — | — | — |
| Trọng số FP32 | 65,7 MB | 63,7 MB | **63,5 MB** | 70,5 MB | 68,5 MB | 68,3 MB | 65,7 MB |
| File ONNX FP32 | 63,5 MB | 61,6 MB | **61,3 MB** | 68,5 MB | 66,5 MB | 66,3 MB | 63,5 MB |
| File ONNX INT8 | 43,3 MB | 41,3 MB | 41,1 MB | 24,1 MB | (22,2 MB) | (**22,0 MB**) | — |
| Checkpoint train | 845,9 MB | 843,4 MB | **842,7 MB** | 867,0 MB | 864,5 MB | 863,9 MB | 845,9 MB |

Ghi chú:
1. **Loss validation** chỉ so sánh được giữa ba model Config A và baseline. Bốn model này dùng cùng 100 câu validation canonical.
   - MRF được train trên một split khác (split dựng lại).
   - Validation set của SEQ chưa được xác minh là giống.
   - Giá trị của MRF và SEQ lấy từ tên file checkpoint, đặt trong ngoặc.
   - Checkpoint Piper đã công bố không có bản ghi validation ở đây.
2. **UTMOSv2** là điểm dự đoán, không phải MOS do người chấm. Bước cắt ngẫu nhiên không được gắn seed, nên giá trị trung bình của một model dao động khoảng ±0,01–0,02 giữa các phiên chấm.
   - Trong phiên 2026-10-21, WAV đã lưu của Piper_no_VITS2_cpn và ParIR được chấm lại cạnh SeqIR. Kết quả là 3.508 và 3.526, so với 3.493 của SeqIR.
   - WER tái tạo chính xác giữa các phiên.
   - Phép thử ghép cặp kèm khoảng tin cậy nằm trong README của từng model.
3. **RTF PyTorch** đến từ hai nhóm phép đo không so sánh được với nhau.
   - Ba model Config A được đo liền nhau trong cùng một phiên ngày 2026-10-21.
   - Bốn giá trị lịch sử lấy từ các lần chạy gốc. Chúng không được chứng minh là cùng một phiên, và Piper chạy trên phiên bản PyTorch khác các model nội bộ, nên giá trị của Piper đặt trong ngoặc.
   - Control ngày 2026-10-21 chậm hơn 1,21 lần so với phiên riêng ngày 2026-10-10. Còn trong phiên 2026-10-10, chính baseline chậm hơn 8,3% so với giá trị lịch sử của nó.
   - Vì vậy chỉ so RTF trong cùng một nhóm. Dòng "so với" làm đúng như vậy, lấy model ResBlock2 của cùng nhóm làm mốc.
4. **Các dòng triển khai:** chỉ có ba model Config A, lấy từ `piper-ir-onnx-ptq/`, trên Harvard-720. Thời gian ORT vòng A đo trong một phiên trên máy rảnh.
   - Graph ONNX lịch sử chỉ được đánh giá trên LJ-500 (xem bảng ONNX lịch sử ở trên), nên ô Harvard-720 của chúng để trống.
   - Kết quả INT8 của Config A chịu ảnh hưởng của lỗi weight norm của flow đã nêu ở trên.
5. **Kích thước:** định nghĩa và bản ghi nằm ở [Kích thước model](#kích-thước-model) (`model_parameters.json`; MB = 10^6 byte).
   - File ONNX chỉ chứa graph inference, nên nhỏ hơn trọng số FP32.
   - **File INT8 lịch sử** là các graph benchmark. Graph MRF không gắn được với checkpoint 1386, còn graph SEQ có scope rộng hơn, nên đặt trong ngoặc.
   - Checkpoint train còn chứa posterior encoder, discriminator và trạng thái optimizer.

Nguồn:
- chất lượng: `piper-seqir-no-vits2-cpn/comparison.json`, có tóm tắt của cả bảy file kết quả;
- RTF PyTorch: `piper-ir-timing-session/` và các file kết quả lịch sử;
- triển khai: `piper-ir-onnx-ptq/`;
- kích thước file INT8 lịch sử: `docs/evidence/journal-evidence-package-2026-10-04/evidence/E_onnx_ptq/onnx_inventory_inspected_now.json`;
- epoch và validation: `selected_checkpoint.json` của từng model hoặc `models/manifest.json`;
- kích thước: `model_parameters.json` và các bản ghi export, lượng tử hoá.

Mỗi model chỉ có một lần train, và mỗi câu chỉ sinh một lần.

## Kích thước model

`model_parameters.json` (2026-10-21) ghi số tham số và kích thước của cả bảy model Harvard-720.
- File được tạo bởi `scripts/evaluation/count_parameters.py`. Script đọc ba checkpoint mới bằng `weights_only=True`, không dựng và không chạy model.
- Các dòng lịch sử lấy từ báo cáo Q05 đã lưu và `models/manifest.json`.
- Kích thước file ONNX của các model mới lấy từ `piper-ir-onnx-ptq/*/export_record.json` và `quantize_record.json`.

| Model | Tham số (định nghĩa Q05) | Dùng khi inference | Decoder | Trọng số FP32 | File ONNX FP32 | File ONNX INT8 | Checkpoint train |
|---|---:|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2) | 16.425.776 | 15.663.771 | 1.664.768 | 65,7 MB | 63,5 MB | 43,3 MB | 845,9 MB |
| Piper_ParIR_no_VITS2_cpn | 15.918.032 | 15.156.027 | 1.157.024 | 63,7 MB | 61,6 MB | 41,3 MB | 843,4 MB |
| Piper_SeqIR_no_VITS2_cpn | 15.870.768 | 15.108.763 | 1.109.760 | 63,5 MB | 61,3 MB | 41,1 MB | 842,7 MB |
| HiFi-GAN ResBlock2 (baseline, có VITS2) | 17.620.784 | — | — | 70,5 MB | 68,5 MB | 24,1 MB | 867,0 MB |
| Parallel-IR (MRF) | 17.113.040 | — | — | 68,5 MB | 66,5 MB | (22,2 MB) | 864,5 MB |
| Sequential-IR (SEQ) | 17.065.776 | — | — | 68,3 MB | 66,3 MB | (22,0 MB) | 863,9 MB |
| Checkpoint Piper bên ngoài | 16.425.776 | — | — | 65,7 MB | 63,5 MB | — | 845,9 MB |

Cách đọc các cột (MB = 10^6 byte):
- **Tham số (định nghĩa Q05):**
  - tham số của generator, không tính posterior encoder `enc_q` (chỉ dùng khi train);
  - đếm ở dạng weight-norm, không tính buffer;
  - cùng định nghĩa với các báo cáo export Q05, nên bảy dòng so sánh được với nhau.
  - Con số này vẫn gồm 762.005 tham số của stochastic duration predictor mà inference không dùng: nhánh posterior `dp.post_*` chỉ dùng khi train (640.248) và flow bị bỏ ở chế độ reverse (121.757).
- **Dùng khi inference:** số Q05 trừ đi 762.005 tham số đó. Chỉ tính cho các model mới.
- **Trọng số FP32:** 4 byte × số Q05. Đây là con số tính ra, không phải kích thước file.
- **File ONNX FP32:** kích thước đo được của graph đã export. Graph chỉ chứa đường inference, nên nhỏ hơn cột trọng số FP32.
  - Model mới: bản export ngày 2026-10-21.
  - Model lịch sử: các bản export Q05.
- **File ONNX INT8:**
  - Model mới: đợt PTQ ngày 2026-10-21. Các graph này vẫn lưu flow dạng FP32 vì weight norm của flow chưa được gộp (xem [lỗi đã biết](#export-onnx-và-ptq-int8)).
  - Model lịch sử: các graph INT8 benchmark, với các lưu ý ở ghi chú 5 phía trên.
- **Checkpoint train:** toàn bộ trạng thái Lightning: generator, posterior encoder, discriminator và trạng thái optimizer. Đây không phải kích thước model để triển khai.

Các con số cho thấy:
- **Phần chung:** ở ba model mới, mọi thứ ngoài decoder giống hệt nhau, 21.999.024 tham số (tính cả `enc_q`).
- **Decoder:** decoder của ParIR và SeqIR ít hơn ResBlock2 lần lượt 30,5% và 33,3%. Tính trên toàn model thì chỉ ít hơn 3,1% và 3,4%, vì decoder chỉ là một phần nhỏ của generator.
- **Piper_no_VITS2_cpn và checkpoint Piper bên ngoài:** cả hai có đúng 16.425.776 tham số. Cả hai đều dùng kiến trúc Piper (Config A).
- **Chỉ dùng khi train:** MPD (46.747.132) và, với các model IR, MRD (280.902) là discriminator. Chúng không thuộc model synthesis.

Việc phát hành công khai các bộ câu, bản ghi kết quả và audio sinh ra vẫn do tác giả quyết định. Việc chúng có mặt trong repo không khẳng định các quyền đó.
