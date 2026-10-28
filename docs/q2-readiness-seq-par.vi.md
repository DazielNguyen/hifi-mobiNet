# Decoder Sequential-IR và Parallel-IR: tổng hợp bằng chứng và mức sẵn sàng cho paper Q2

*English: [q2-readiness-seq-par.md](q2-readiness-seq-par.md)*

Tình trạng ngày 2026-10-22. Tài liệu này gom về một chỗ mọi kết quả đã lưu của baseline
HiFi-GAN ResBlock2, decoder Parallel-IR (PAR, nhãn lịch sử MRF) và decoder Sequential-IR
(SEQ). Sau đó tài liệu đánh giá một paper journal về SEQ và PAR hiện có thể khẳng định điều
gì, và còn thiếu gì.

- **Nguồn.** Mọi con số đều lấy từ một bản ghi trong repository này; tên file được ghi dưới mỗi bảng.
- **Không phải bằng chứng.** Phần đánh giá và đề xuất (mục 3–5) là cách đọc các bản ghi đó từ phía tác giả.
- **Không chạy gì mới.** Không có gì được train, chạy hay đo cho tài liệu này.

## 1. Hai thế hệ thí nghiệm

| | Thế hệ 1 (lịch sử, 08–09/2026) | Thế hệ 2 (có kiểm soát, 10/2026) |
|---|---|---|
| Code base | BanhmiTTS, công thức dựa trên VITS2 | EdgeTTS Config A (backbone Piper, tắt các thành phần VITS2) |
| Model | baseline-resblock2 (epoch 1489), Parallel-IR/MRF (1386), Sequential-IR/SEQ (1442), checkpoint Piper bên ngoài | Piper_no_VITS2_cpn (ResBlock2, 1403), Piper-ParIR (1486), Piper-SeqIR (1455) |
| Giữa các model khác nhau ở đâu | Decoder **và** split (MRF dùng split khác, chung 95,4% dữ liệu train), code train (SEQ thêm bỏ qua gradient không hữu hạn và chặn SDP), gradient clipping (baseline 1,0, IR không), activation và MRD | Chỉ bộ decoder: khối IR + SnakeBeta + MRD. Cùng dữ liệu, split, seed, công thức train và trọng số khởi tạo của backbone |
| Bản ghi | `results/historical/`, `results/prior_audit/`, `docs/evidence/journal-evidence-package-2026-10-04/`, `docs/evidence/claim-evidence-2026-10-07/` | `results/piper-no-vits2-cpn/`, `results/piper-parir-no-vits2-cpn/`, `results/piper-seqir-no-vits2-cpn/`, `results/piper-ir-timing-session/`, `results/piper-ir-onnx-ptq/` |
| Sức nặng bằng chứng | So sánh cấp hệ thống bị nhiễu bởi nhiều yếu tố, mỗi model một lần train | So sánh bộ decoder có kiểm soát, mỗi model một lần train |

**Bộ decoder** (cả hai thế hệ): khối inverted-residual (pointwise mở rộng → depthwise conv → pointwise chiếu, expansion 1), activation SnakeBeta và thêm discriminator MRD.
- **PAR:** ba nhánh song song mỗi tầng, kernel depthwise 3/5/7, lấy trung bình.
- **SEQ:** hai khối nối tiếp mỗi tầng, kernel 3.
- **Phép so sánh tách được gì:** bộ decoder thay cùng lúc khối, activation và discriminator. Giữa PAR và SEQ còn khác cả số khối lẫn kernel, nên không phép so sánh nào chỉ đo cấu trúc khối.

## 2. Tổng hợp kết quả

### 2.1 Harvard-720 (720 câu, cùng protocol cho cả bảy model)

Nguồn: `results/README.vi.md`, mục "So sánh bảy model" (danh sách file của từng model ở đó).

| Model | Thế hệ | Tham số decoder | UTMOSv2 (dự đoán) | Macro WER | RTF trung bình (PyTorch CPU) | RTF so với ResBlock2 cùng nhóm |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2) | 2 | 1.664.768 | 3.517 | **0.0612** | 0.0510 | — |
| Piper-ParIR | 2 | 1.157.024 | **3.530** | 0.0726 | 0.0392 | −23,2% |
| Piper-SeqIR | 2 | **1.109.760** | 3.493 | 0.0696 | **0.0340** | **−33,3%** |
| baseline-resblock2 (có VITS2) | 1 | 1.664.768 | 3.026 | 0.1476 | 0.0418 | — |
| Parallel-IR (MRF) | 1 | 1.157.024 | **3.568** | 0.1073 | 0.0339 | −18,9% |
| Sequential-IR (SEQ) | 1 | **1.109.760** | 3.544 | 0.1222 | **0.0308** | **−26,2%** |
| Piper (bên ngoài) | — | 1.664.768 | 3.451 | 0.1747 | 0.0392 | (−6,2%) |

Ghi chú:
- **Tham số decoder.** Mọi số đều ở dạng weight-norm, đếm lại từ checkpoint (`results/model_parameters.json`). Mỗi decoder có cùng kích thước ở cả hai thế hệ. J-E015 cho cùng các decoder này ở dạng đã gộp weight norm (ResBlock2 1.662.976, PAR 1.154.560, SEQ 1.107.968), ít hơn khoảng 0,1%.
- **Nhóm RTF.** Số của thế hệ 2 đến từ một phiên đo trên máy rảnh ngày 2026-10-21. Số của thế hệ 1 là các lần chạy lịch sử. Hai nhóm không so sánh được với nhau.
- **Khối lượng tính toán decoder (thế hệ 1, J-E016).** SEQ 0,98, PAR 1,23, ResBlock2 và Piper 3,90 GFLOPs cho mỗi giây audio.

**So sánh ghép cặp trong từng thế hệ** (theo từng câu; CI = bootstrap ghép cặp 95%):

| So sánh | ΔUTMOSv2 | ΔWER (tỉ lệ) | ΔRTF | Nguồn |
|---|---|---|---|---|
| Thế hệ 2: ParIR − ResBlock2 | +0,013 [−0,011, +0,037]; cùng phiên +0,025 [+0,002, +0,049] | +0,011 [+0,002, +0,021] (Holm 0,11) | −23,2% (cùng phiên) | `results/piper-parir-no-vits2-cpn/`, `results/piper-ir-timing-session/` |
| Thế hệ 2: SeqIR − ResBlock2 | −0,024 [−0,048, −0,00005]; cùng phiên −0,015 [−0,039, +0,009] | +0,008 [−0,0004, +0,018] (Holm 0,12) | −33,3% | `results/piper-seqir-no-vits2-cpn/` |
| Thế hệ 2: SeqIR − ParIR | −0,037 [−0,060, −0,014]; cùng phiên −0,033 [−0,056, −0,010] | −0,003 [−0,013, +0,007] | −13,2% | như trên |
| Thế hệ 1: SEQ − MRF | −0,024 (p 0,047; Holm 0,094) | +0,015 [+0,002, +0,028] | −9,1% (p 7e-37) | `results/prior_audit/`, ledger J-E003–J-E005 |
| Thế hệ 1: SEQ − baseline | +0,518 | −0,026 | −26,2% | `results/prior_audit/` |
| Thế hệ 1: SEQ − Piper | +0,094 | −0,053 | −21,3% | J-E007; Piper được đưa phoneme lệch (en-us), xem `results/frontend-check/` |

Mel loss trên validation (cùng 100 câu, chỉ thế hệ 2 và baseline thế hệ 1): ResBlock2 19,77, ParIR 19,01, SeqIR 19,03, baseline-resblock2 19,79.

### 2.2 Các bảng LJ-500 (lịch sử, thế hệ 1)

Các bảng LJ-500 lịch sử dùng **ba tập 500 câu khác nhau**, nên chỉ so được các bảng trong cùng một tập:
- **Tập baseline/canonical** (`eval_true_testset_500_baseline.json`).
- **Tập MRF** (`eval_true_testset_500_mrf.json`): chung 21 câu với tập baseline.
- **Tập overnight:** chung 17 câu với tập baseline.

Nguồn: `results/historical/{lj500,onnx_historical,lj500_development,qat}/`.

| Tập | Model / bảng | Checkpoint | RTF | UTMOSv2 | WER |
|---|---|---|---:|---:|---:|
| baseline | baseline, PyTorch có seed (`eval_baseline_fp32_seeded_500`) | 1489 | 0.0434 | 3.125 | 0.1230 |
| baseline | baseline, PyTorch không seed (`eval_baseline_best_500`) | 1489 | 0.0431 | 3.139 | 0.1197 |
| baseline | SEQ, PyTorch có seed (`eval_seq111_fp32_seeded_500`) | 1442 | 0.0324 | 3.581 | 0.1033 |
| baseline | Piper bên ngoài (`eval_piper_original_500_full`) | 999 | 0.0386 | 3.534 | 0.1632 |
| MRF | MRF, PyTorch (`eval_mrf_final1500_500`) | 1386 | 0.0370 | 3.651 | 0.0983 |
| baseline | baseline, ONNX FP32 / INT8 (scope lịch sử) | 1489 | 0.0388 / 0.0345 | 3.132 / 3.121 | 0.1230 / 0.1135 |
| MRF | MRF, ONNX FP32 / INT8 (**graph không gắn được với 1386**) | ? | 0.0416 / 0.0371 | 3.537 / 3.493 (chạy lại 3.490) | 0.1043 / 0.1030 (chạy lại 0.1124) |
| baseline | SEQ, ONNX FP32 / INT8 (**scope INT8 rộng hơn**) | 1442 | 0.0346 / 0.0289 | 3.571 / 3.586 | 0.1064 / 0.1081 |

**Quá trình train của MRF** (tập baseline; `results/historical/lj500_development/`):

| Epoch | 74 | 216 | 353 | 480 | 836 | 836, chấm lại |
|---|---:|---:|---:|---:|---:|---:|
| UTMOSv2 | 2.187 | 3.169 | 3.297 | 3.348 | 3.505 | 3.518 |
| WER | 0.2552 | 0.1692 | 0.1013 | 0.1110 | 0.1024 | 0.1074 |

Hai lần đánh giá cùng một checkpoint MRF lệch nhau 0,013 UTMOSv2 và 0,005 WER. Hai lần đánh giá cùng một checkpoint baseline lệch nhau 0,003 WER. Đó là mức nhiễu của các lần đánh giá lịch sử này (một phần không có seed).

**Các tiền thân của SEQ:**
- `mbconv_441` ở epoch 12: WER 0,735 (một cấu hình thử sớm, thất bại).
- `mbconv_mrd` ở epoch 74: WER 0,138.
- Checkpoint của cả hai không còn.

**QAT** (`results/historical/qat/`, LJ-500): fake-quantize flow, enc_p và dp, sau khi train thêm:

| Model | Tập | Số epoch QAT | RTF (PyTorch, fake-quant) | UTMOSv2 | WER |
|---|---|---:|---:|---:|---:|
| baseline QAT | baseline | 144 | 0.0497 / 0.0508 (có seed) | 3.569 / 3.549 | 0.0819 / 0.0786 |
| MRF QAT (từ model 1000 epoch) | baseline | 97 | 0.0456–0.0464 | 3.630–3.634 | 0.0996–0.1081 |
| MRF QAT (từ model 1000 epoch) | MRF | 97 | 0.0461 | 3.650 | 0.0977 |
| MRF QAT (từ model 1500 epoch) | MRF | 97 | 0.0456 | 3.639 | 0.0992 |

Không có bản ghi QAT nào cho SEQ. Không thể quy phần cải thiện của QAT cho lượng tử hoá: QAT train thêm 97–144 epoch, và RTF của nó là PyTorch với op fake-quant, không phải runtime INT8.

**PTQ sweep lịch sử** (`results/historical/ptq_sweep/`; các file kết quả là bản dẫn xuất đã bỏ các mục Vocos):
- Trên baseline, lượng tử hoá flow + enc_p + dp cho WER thấp nhất trong các scope đã thử (0,1215, 23,0 MB).
- **Lưu ý (J-E026):** evaluator của sweep này đưa audio 22,05 kHz vào Whisper (Whisper cần 16 kHz) và dùng cách chuẩn hoá WER khác. Số WER của nó chỉ dùng để xếp hạng scope, và ngay cả thứ hạng đó cũng chưa được kiểm chứng.

### 2.3 ONNX FP32 và INT8, thế hệ 2 (protocol sạch)

Nguồn: `results/piper-ir-onnx-ptq-v2/` (đợt 2, đã gộp weight norm của flow; đợt 1 ở `results/piper-ir-onnx-ptq/`). Harvard-720; noise khớp với PyTorch; calibration chỉ lấy từ train; cùng một scope INT8 cho mọi model.

| Model | Kích thước FP32 → INT8 | ΔUTMOSv2 INT8 − FP32 | ΔWER INT8 − FP32 | RTF ORT 2 thread (FP32 → INT8) | RTF ORT 12 thread (FP32 → INT8) |
|---|---|---|---|---|---|
| ResBlock2 | 63,5 → 22,4 MB | −0,011 [−0,031, +0,008] | −0,003 [−0,009, +0,004] | 0.0323 → 0.0280 | 0.0253 → 0.0226 |
| ParIR | 61,5 → 20,5 MB | −0,007 [−0,025, +0,011] | +0,004 [−0,004, +0,012] | 0.0338 → 0.0306 | 0.0318 → 0.0290 |
| SeqIR | 61,3 → 20,3 MB | −0,023 [−0,043, −0,002] (Wilcoxon p 0,058) | −0,001 [−0,008, +0,006] | **0.0278 → 0.0238** | 0.0267 → 0.0243 |

- **Export FP32 không mất mát:** parity đạt 720/720 câu, WER giống hệt.
- **Thứ hạng tốc độ phụ thuộc runtime và số thread:**
  - ORT 2 thread: SeqIR nhanh nhất (−14% đến −17% so với ResBlock2 ở đợt 2; −23% đến −28% ở đợt 1).
  - ORT 12 thread: ResBlock2 nhanh nhất; ParIR chậm hơn 26–29% (đợt 1: 32–36%).
- **Nguyên nhân (profile kernel):** `Sin` của SnakeBeta không dùng thread pool của ONNX Runtime 1.23, trong khi convolution dày của ResBlock2 thì có.

## 3. Những gì bằng chứng hiện hỗ trợ

| Nhận định | Trạng thái | Cơ sở |
|---|---|---|
| Decoder IR có ít hơn 30–33% tham số decoder và ít hơn 3–4 lần FLOPs decoder so với ResBlock2; toàn model nhỏ hơn khoảng 3% | **Được hỗ trợ** | số tham số, J-E015/J-E016 |
| Trên PyTorch CPU, SeqIR / ParIR giảm RTF 33% / 23% so với ResBlock2 trong thiết lập có kiểm soát | **Được hỗ trợ**, một phiên, mỗi model một seed | phiên đo thời gian của thế hệ 2 |
| SeqIR nhanh hơn ParIR | **Được hỗ trợ** trên PyTorch (thế hệ 1 −9%, thế hệ 2 −13%) và trên ORT 2 thread | cả hai thế hệ |
| Decoder IR giữ được chất lượng của ResBlock2 | **Một phần.** UTMOSv2 lệch trong khoảng 0,04 và WER trong khoảng 1,1 điểm, nhưng WER của cả hai model IR đều kém hơn một chút, và không có ngưỡng tương đương nào được chốt | phép thử ghép cặp thế hệ 2 |
| SEQ có chất lượng tốt hơn PAR | **Bị bác bỏ.** UTMOSv2 của SEQ thấp hơn ở cả thế hệ 1 lẫn thế hệ 2; WER không phân biệt được | J-E004, J-E031, thế hệ 2 |
| Lợi thế tốc độ vẫn giữ sau khi export ONNX và INT8 | **Chỉ ở số thread thấp.** Giữ ở ORT 2 thread và đảo ngược ở 12 thread (đã xác định nguyên nhân) | `results/piper-ir-onnx-ptq-v2/`, `results/piper-ir-onnx-ptq/` |
| PTQ INT8 có chọn lọc (enc_p/dp/flow) không làm giảm chất lượng | **Phần lớn được hỗ trợ** (đợt 2, đã gộp weight norm của flow). Không model nào thay đổi WER đáng kể, ResBlock2 và ParIR giữ nguyên UTMOSv2. SeqIR có thể giảm 0,023 UTMOSv2: khoảng tin cậy bootstrap loại trừ 0, nhưng Wilcoxon p là 0,058. Graph nhỏ hơn 65–67% và nhanh hơn 9–16%. Mỗi model một lần chạy; không có ngưỡng tương đương | PTQ thế hệ 2 đợt 2, `results/piper-ir-onnx-ptq-v2/` |
| Bỏ các thành phần VITS2 giúp model cỡ Piper tốt hơn | **Một phần:** WER −8,6 điểm và UTMOSv2 +0,49, nhưng code base cũng khác nhau | `results/piper-no-vits2-cpn/` |
| Tốc độ trên thiết bị biên (N150, Raspberry Pi 5) | **Chưa có** | J-E028 |
| Chất lượng qua người nghe (MOS) | **Chưa có**; UTMOSv2 chỉ là điểm dự đoán | J-E029 |
| Độ ổn định qua các seed train | **Chưa có**: mỗi model một lần train | — |

## 4. Còn thiếu gì cho một paper journal Q2

Với một paper về hệ thống TTS hay vocoder, các journal ở mức này thường yêu cầu: kết quả nghe thử của người, hơn một lần train hoặc một ước lượng phương sai, ablation có kiểm soát cho cơ chế được khẳng định, so sánh với các hệ thống nhẹ đã có, và đo trên phần cứng đích khi paper nói về triển khai. Cần đối chiếu yêu cầu cụ thể với journal được chọn.

| Ưu tiên | Còn thiếu | Cách bổ sung | Chi phí ước tính |
|---|---|---|---|
| 1 | Chưa có nghe thử | MOS hoặc CMOS kèm khoảng tin cậy trên thế hệ 2 (ResBlock2, ParIR, SeqIR, các bản INT8, ground truth); thiết kế chốt trước, ngẫu nhiên và ẩn danh | 1–2 tuần tổ chức; khoảng 20+ người nghe |
| 1 | Mỗi model một lần train | Thêm hai seed cho mỗi model thế hệ 2 (6 run) | khoảng 5 ngày mỗi run 1500 epoch trên 2× RTX 4070 Ti: khoảng 30 ngày trên một máy, khoảng 15 ngày trên hai máy |
| 1 | Chưa tách được bộ decoder (khối, SnakeBeta, MRD) | Tối thiểu: khối IR với LeakyReLU (không Snake), và ResBlock2 + SnakeBeta + MRD. Run thứ nhất còn kiểm luôn xem IR có giữ được lợi thế tốc độ trên ORT khi không có `Sin` hay không | 2 run, khoảng 10 ngày |
| 1 | Chưa đo trên thiết bị đích | N150 và Pi 5 (hoặc thiết bị được chọn): ORT 1–4 thread, warm-up, lặp lại, bộ nhớ đỉnh, kích thước file | 3–5 ngày khi có thiết bị |
| 2 | Chưa có baseline nhẹ bên ngoài theo cùng protocol | ví dụ giọng Piper "low"/"x_low", hoặc một biến thể VITS dùng iSTFT, chấm bằng cùng pipeline Harvard-720 | vài ngày, nếu checkpoint công khai |
| 2 | Protocol đo thời gian | đo lặp lại, đảo thứ tự, có warm-up; độ trễ với câu ngắn và câu dài | 1–2 ngày |
| 2 | Tập test cùng miền cho thế hệ 2 | tập test-500 LJSpeech canonical, với cùng cách gắn seed theo từng câu như Harvard-720 | khoảng 1 ngày |
| 3 | Cách diễn đạt "chất lượng tương đương" | chốt trước ngưỡng tương đương (TOST), hoặc diễn đạt dạng "lệch trong X, kèm CI" | chỉ phân tích |
| 3 | Vai trò của thế hệ 1 | chỉ làm phần lịch sử bổ sung, nêu rõ các yếu tố gây nhiễu; không đặt nhận định chính nào lên nó | viết |

## 5. Đánh giá

- **Dữ liệu hiện chưa đủ để nộp Q2.** Phần mạnh nhất là thế hệ 2: so sánh có kiểm soát, protocol ONNX/INT8 sạch, kiểm tính deterministic và một cơ chế runtime đã được xác định. Nhưng phần này mới dựa trên một lần train mỗi model, chưa có nghe thử và chưa có số đo trên thiết bị.
- **Thế hệ 1 không nên gánh nhận định nào.** Kết quả MRF/SEQ của nó bị nhiễu bởi split, code train và clipping, còn các bản ghi ONNX/INT8 có lỗi đã biết. Ledger đã đánh dấu nhận định "SEQ chất lượng tốt hơn PAR" và nhận định lịch sử "tốc độ giữ nguyên qua INT8" là bị bác bỏ hoặc không được hỗ trợ.
- **Một câu chuyện paper có thể bảo vệ được:**
  - decoder inverted-residual gọn nhẹ cho TTS cỡ Piper/VITS chạy trên CPU;
  - giảm 30% tham số decoder và giảm 23–33% RTF trên PyTorch, đổi lại khoảng 1 điểm WER;
  - PTQ INT8 có chọn lọc giảm thêm 65–67% kích thước graph và 9–16% RTF trên ORT, không làm WER thay đổi đáng kể (đợt 2);
  - **tốc độ triển khai phụ thuộc vào độ phủ kernel của runtime:** FLOPs và số tham số không dự đoán được tốc độ ORT ở 12 thread, vì `Sin` của SnakeBeta chạy đơn luồng. Phát hiện này cụ thể và đã được đo, nên đáng làm điểm nhấn của paper.
- **Có đủ các mục ưu tiên 1, một paper hệ thống/triển khai ở mức Q2 là khả thi, nhưng không chắc chắn.** Các mục đó là nghe thử, thêm seed, một hoặc hai ablation và đo trên thiết bị: khoảng 4–6 tuần tính toán, cộng với nghiên cứu nghe thử. Tính mới cũng cần được định vị so với các công trình vocoder/TTS nhẹ đã có bằng một bước rà soát tài liệu; tài liệu này không đánh giá điều đó.

## 6. Mọi thứ nằm ở đâu

| Tài liệu | Vị trí |
|---|---|
| Kết quả theo từng câu của thế hệ 1, Harvard-720 và LJ-500 | `results/historical/harvard/`, `results/historical/lj500/` |
| Checkpoint phát triển của thế hệ 1, quá trình train MRF, tiền thân SEQ, các biến thể Snake/MRD | `results/historical/lj500_development/` (script trong `records/`; log được ghi hash trong bản ghi nhập) |
| QAT của thế hệ 1 | `results/historical/qat/` |
| ONNX/INT8 và PTQ sweep của thế hệ 1 | `results/historical/onnx_historical/`, `results/historical/ptq_sweep/` |
| Audit thống kê của thế hệ 1 | `results/prior_audit/` |
| Gói bằng chứng ngày 2026-10-04 của thế hệ 1: claim ledger J-E001–J-E033, ghi chú của tác giả (gồm checklist Q2 trước đây và ghi chú MRF/SEQ), lịch sử train Q03, export Q05 | `docs/evidence/journal-evidence-package-2026-10-04/` |
| Catalog lưu trữ Q01 của thế hệ 1 (27 bảng kết quả, các lỗ hổng nguồn gốc) | `docs/evidence/q01-archive-2026-10-04/` |
| Claim evidence ngày 2026-10-07 (các claim J-C) | `docs/evidence/claim-evidence-2026-10-07/` |
| Bản ghi nhập cho hai thư mục trên và `results/historical/{lj500_development,qat,ptq_sweep}` | `docs/evidence/historical-import-2026-10-22.json` |
| Kết quả, đo thời gian, ONNX/INT8 và profile của thế hệ 2 | `results/piper-*/`, `results/piper-ir-timing-session/`, `results/piper-ir-onnx-ptq-v2/`, `results/piper-ir-onnx-ptq/` |
| Số tham số và kích thước | `results/model_parameters.json` |
| Protocol của thế hệ 2 | `docs/training/piper-ir-protocol.md`, `docs/evaluation-onnx-ptq.md` |

Nằm ngoài phạm vi, vì thuộc các decoder khác:
- **Không được nhập:** các bảng đánh giá Vocos (`eval_true_testset_500`, `eval_last_ckpt_500`, `eval_full_500`) và báo cáo QAT về BanhmiTTS_v1 (`QAT-Training/QAT_Quantization_Report.md`).
- **Không dùng Vocos ở bất kỳ đâu.** Các log PTQ sweep có Vocos chỉ được ghi hash, còn hai file kết quả PTQ sweep là bản dẫn xuất đã bỏ các mục Vocos (`*.without_vocos.json`).
