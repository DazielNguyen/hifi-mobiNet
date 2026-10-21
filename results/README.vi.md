# Kết quả nghiên cứu lịch sử

*English: [README.md](README.md)*

Mọi file trong `historical/` giống từng byte với các file tương ứng trong gói bằng
chứng (evidence package) của bài báo. Hash, vị trí nguồn và giới hạn của chúng nằm
trong `manifest.json` và `docs/source-map.json`. Không có phép đo nghiên cứu mới nào
được tạo ra để điền vào các thư mục này.

| Model | RTF trung bình (Harvard) | UTMOSv2 (dự đoán) | Macro WER |
|---|---:|---:|---:|
| HiFi-GAN ResBlock2 | 0.04177 | 3.0260 | 0.1476 |
| Parallel-IR | 0.03388 | 3.5680 | 0.1073 |
| Sequential-IR | 0.03081 | 3.5443 | 0.1222 |
| Checkpoint Piper bên ngoài | 0.03916 | 3.4507 | 0.1747 |

Đây là **các quan sát lịch sử bằng PyTorch trên CPU**, trên 720 câu Harvard chung,
đo trên máy phát triển chạy WSL.
- Đây không phải kết quả trên thiết bị biên (edge) hay ONNX.
- UTMOSv2 không phải MOS do người chấm.
- Mỗi model chỉ có một lần train, mỗi câu chỉ sinh một lần, và không có protocol đo thời gian lặp lại.
- Version runtime lịch sử được xác định gián tiếp qua các bản ghi gần đó, không được hash trong từng lần chạy.

Các bản export FP32 Q05 về sau đã được định danh và kiểm tra kỹ thuật, nhưng các
điểm chất lượng ở trên không gắn với các bản export đó. Demo chỉ là giao diện chạy
được; nó không chứng minh code mới tái tạo được bảng trên.

`lj500/`:
- Giữ cả lần đánh giá MRF epoch 1386 đã chọn (`eval_mrf_final1500_500_results.json`) lẫn một bảng MRF cũ hơn. Tên file và giá trị giữ nguyên.
- Tập test của MRF và tập test LJSpeech canonical chỉ trùng 21 ID. Không được coi mọi file 500 dòng là một phép thử ghép cặp chung.

`manifests/`:
- Gồm split canonical (được ghi nhận về sau) và danh sách ID có thứ tự đã đóng băng của Q04.
- Điều này không chứng minh rằng mọi manifest đều được ghi lại cùng thời điểm với lúc train.

`onnx_historical/` chỉ mang tính thăm dò:
- Graph MRF cũ không gắn với checkpoint đã chọn.
- Dữ liệu calibration lịch sử trùng với dữ liệu test, và phạm vi PTQ không đồng nhất.
- Các bảng này được giữ để minh bạch, không được nêu như kết luận ONNX/PTQ cuối cùng.

`prior_audit/` chứa một bản audit thống kê về sau, giữ nguyên không sửa.

Script thống kê mới (chạy được trên mọi máy) tái tạo bản audit đó vào `outputs/`,
không đụng vào dữ liệu gốc. Cần lưu ý khi đọc các kết quả thống kê:
- Phép thử ghép cặp và khoảng bootstrap chỉ nói về dao động giữa các câu, không nói về dao động giữa các lần train hay giữa các phần cứng.
- Không có ý nghĩa thống kê không có nghĩa là tương đương.
- Công thức train và split khác nhau, nên không thể kết luận khác biệt chỉ do decoder.

## Phép đo mới

`piper-no-vits2-cpn/` (README tiếng Việt: [piper-no-vits2-cpn/README.vi.md](piper-no-vits2-cpn/README.vi.md))
chứa các phép đo mới đầu tiên trong repository này (2026-10-10/11): đánh giá
Harvard-720 của `Piper_no_VITS2_cpn`, tức baseline không có các thành phần VITS2.
Thư mục gồm:
- README riêng;
- kết quả theo từng câu và bản ghi của scorer;
- phép so ghép cặp với các file lịch sử ở trên;
- các control chạy trong cùng phiên.

Các file này bổ sung cho bảng lịch sử, không thay thế nó. RTF lịch sử không so sánh
được với RTF đo trong phiên đó: trên cùng model baseline, đã đo được độ trôi thời
gian +8,3%.

`piper-parir-no-vits2-cpn/` (README tiếng Việt: [piper-parir-no-vits2-cpn/README.vi.md](piper-parir-no-vits2-cpn/README.vi.md), 2026-10-16)
chứa đánh giá chất lượng Harvard-720 của `Piper_ParIR_no_VITS2_cpn`: cùng backbone, thay
bằng bộ decoder Parallel-IR đã phát hành (decoder inverted-residual, SnakeBeta, MRD).
- Model này được so ghép cặp theo từng câu với `Piper_no_VITS2_cpn` và bốn hệ thống lịch sử.
- Có control scorer trong cùng phiên, bằng cách chấm lại các WAV của `Piper_no_VITS2_cpn`.
- RTF được đo sau đó, trong phiên ngày 2026-10-21 nêu dưới đây.

`piper-seqir-no-vits2-cpn/` (README tiếng Việt: [piper-seqir-no-vits2-cpn/README.vi.md](piper-seqir-no-vits2-cpn/README.vi.md), 2026-10-21)
chứa đánh giá `Piper_SeqIR_no_VITS2_cpn`: cùng backbone, thay bằng bộ decoder Sequential-IR
đã phát hành (hai khối IR mỗi tầng, SnakeBeta, MRD).
- Model này được so ghép cặp theo từng câu với hai model Config A còn lại và bốn hệ thống lịch sử.
- Có bản chấm lại cùng phiên cho các WAV của `Piper_no_VITS2_cpn` và ParIR.
- README của nó cũng báo cáo phiên đo thời gian trên máy rảnh. Phiên này đo liền nhau ba
  model Config A, với `Piper_no_VITS2_cpn` làm control. Bản ghi của phiên nằm trong
  `piper-ir-timing-session/`.

| Model mới | UTMOSv2 (dự đoán) | Macro WER | RTF trung bình, phiên 2026-10-21 | RTF tương đối |
|---|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2, control) | 3.517 | 0.0612 | 0.0510 | — |
| Piper_ParIR_no_VITS2_cpn | 3.530 | 0.0726 | 0.0392 | −23,2% |
| Piper_SeqIR_no_VITS2_cpn | 3.493 | 0.0696 | 0.0340 | −33,3% |

Cột RTF lấy từ một phiên duy nhất và chỉ so sánh được trong nội bộ phiên đó. Control chạy
chậm hơn 1,21 lần so với phiên riêng ngày 2026-10-10 (0.0421). Vì vậy không thể đặt giá trị
đó, hay bảng RTF lịch sử ở trên, cạnh các giá trị này.

`piper-ir-onnx-ptq/` (README tiếng Việt: [piper-ir-onnx-ptq/README.vi.md](piper-ir-onnx-ptq/README.vi.md), 2026-10-21)
chứa bản export ONNX FP32 và PTQ INT8 tĩnh của ba model Config A. Protocol là
`docs/evaluation-onnx-ptq.md`. Thư mục gồm chất lượng Harvard-720 của cả sáu graph và một
phiên đo thời gian bằng ONNX Runtime.
- **Export:** không mất mát. Cả 720 câu đạt parity, và WER giống hệt PyTorch.
- **Kích thước và tốc độ INT8:** graph nhỏ hơn 32–33% và nhanh hơn FP32 4–17%.
- **Chất lượng INT8:** không phát hiện thay đổi với Piper_no_VITS2_cpn và ParIR. SeqIR giảm 0,036 UTMOSv2.
- **Tốc độ decoder:** trên ORT 12 thread, decoder IR không nhanh hơn ResBlock2.

| Model, ONNX INT8 | UTMOSv2 (dự đoán) | Macro WER | Kích thước graph | RTF ORT, 2 thread | RTF ORT, 12 thread |
|---|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn | 3.503 | 0.0620 | 43,3 MB | 0.0376 | 0.0277 |
| Piper_ParIR_no_VITS2_cpn | 3.524 | 0.0772 | 41,3 MB | 0.0341 | 0.0366 |
| Piper_SeqIR_no_VITS2_cpn | 3.459 | 0.0739 | 41,1 MB | 0.0269 | 0.0297 |

## So sánh bảy model

Cả bảy model trên Harvard-720 (720 câu, 5.742 từ tham chiếu), cùng protocol synthesis và
chấm điểm (`docs/evaluation.md`). Mọi giá trị được chép từ các bản ghi liệt kê dưới bảng;
không đo lại gì cho bảng này. **In đậm** là giá trị tốt nhất của mỗi dòng. Với RTF, giá trị
tốt nhất chỉ được đánh dấu trong từng nhóm đo (ghi chú 3).

| Metric | Piper_no_VITS2_cpn | Piper-ParIR | Piper-SeqIR | Baseline ResBlock2 | Parallel-IR (MRF) | Sequential-IR (SEQ) | Piper (bên ngoài) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ***Model*** | | | | | | | |
| Decoder | ResBlock2 | Parallel-IR | Sequential-IR | ResBlock2 | Parallel-IR | Sequential-IR | ResBlock2 |
| Backbone / công thức train | Config A, không VITS2 | Config A, không VITS2 | Config A, không VITS2 | baseline VITS2 | baseline VITS2 | baseline VITS2 | Piper đã công bố |
| Epoch được chọn | 1403 | 1486 | 1455 | 1489 | 1386 | 1442 | 999 |
| `val_loss_mel` trên validation ¹ | 19.7719 | **19.0069** | 19.0332 | 19.7882 | (18.7681) | (19.0136) | — |
| ***Chất lượng*** | | | | | | | |
| UTMOSv2, trung bình (dự đoán) ² | 3.517 | 3.530 | 3.493 | 3.026 | **3.568** | 3.544 | 3.451 |
| UTMOSv2, SD | 0.250 | 0.251 | 0.267 | 0.303 | 0.236 | 0.246 | 0.259 |
| Macro WER | **0.0612** | 0.0726 | 0.0696 | 0.1476 | 0.1073 | 0.1222 | 0.1747 |
| Corpus WER | **0.0596** | 0.0707 | 0.0665 | 0.1437 | 0.1033 | 0.1186 | 0.1696 |
| Thay thế / mất từ / chèn thêm | **283** / **46** / **13** | 321 / 71 / 14 | 292 / 76 / 14 | 681 / 101 / 43 | 467 / 109 / 17 | 531 / 111 / 39 | 812 / 112 / 50 |
| Số câu WER = 0 (trên 720) | **502** | 464 | 476 | 318 | 381 | 351 | 228 |
| Phiên chấm điểm | 2026-10-10 | 2026-10-16 | 2026-10-21 | lịch sử | lịch sử | lịch sử | lịch sử |
| ***Tốc độ (PyTorch FP32, CPU, 12 thread)*** ³ | | | | | | | |
| RTF trung bình | 0.0510 | 0.0392 | **0.0340** | 0.0418 | 0.0339 | **0.0308** | 0.0392 |
| RTF trung vị | 0.0490 | 0.0374 | **0.0320** | 0.0412 | 0.0329 | **0.0296** | 0.0379 |
| RTF trung bình so với model ResBlock2 cùng nhóm | — | −23,2% | −33,3% | — | −18,9% | −26,2% | (−6,2%) |
| Nhóm đo thời gian | 2026-10-21 | 2026-10-21 | 2026-10-21 | lịch sử | lịch sử | lịch sử | lịch sử |
| ***Kích thước*** ⁴ | | | | | | | |
| Tham số (định nghĩa Q05) | 16.425.776 | 15.918.032 | **15.870.768** | 17.620.784 | 17.113.040 | 17.065.776 | 16.425.776 |
| Tham số dùng khi inference | 15.663.771 | 15.156.027 | **15.108.763** | — | — | — | — |
| Tham số decoder | 1.664.768 | 1.157.024 | **1.109.760** | — | — | — | — |
| Trọng số FP32 | 65,7 MB | 63,7 MB | **63,5 MB** | 70,5 MB | 68,5 MB | 68,3 MB | 65,7 MB |
| File ONNX FP32 | chưa export | chưa export | chưa export | 68,5 MB | 66,5 MB | 66,3 MB | **63,5 MB** |
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
3. **RTF** đến từ hai nhóm phép đo không so sánh được với nhau.
   - Ba model Config A được đo liền nhau trong cùng một phiên ngày 2026-10-21.
   - Bốn giá trị lịch sử lấy từ các lần chạy gốc. Chúng không được chứng minh là cùng một phiên, và Piper chạy trên phiên bản PyTorch khác các model nội bộ, nên giá trị của Piper đặt trong ngoặc.
   - Control ngày 2026-10-21 chậm hơn 1,21 lần so với phiên riêng ngày 2026-10-10. Còn trong phiên 2026-10-10, chính baseline chậm hơn 8,3% so với giá trị lịch sử của nó.
   - Vì vậy chỉ so RTF trong cùng một nhóm. Dòng "so với" làm đúng như vậy, lấy model ResBlock2 của cùng nhóm làm mốc.
4. **Kích thước:** định nghĩa và bản ghi nằm ở mục "Kích thước model" bên dưới (`model_parameters.json`; MB = 10^6 byte).
   - File ONNX chỉ chứa graph inference, nên nhỏ hơn trọng số FP32.
   - Checkpoint train còn chứa posterior encoder, discriminator và trạng thái optimizer.

Nguồn:
- chất lượng: `piper-seqir-no-vits2-cpn/comparison.json`, có tóm tắt của cả bảy file kết quả;
- RTF: `piper-ir-timing-session/` và các file kết quả lịch sử;
- epoch và validation: `selected_checkpoint.json` của từng model hoặc `models/manifest.json`;
- kích thước: `model_parameters.json`.

Mỗi model chỉ có một lần train, và mỗi câu chỉ sinh một lần.

## Kích thước model

`model_parameters.json` (2026-10-21) ghi số tham số và kích thước của cả bảy model
Harvard-720. File được tạo bởi `scripts/evaluation/count_parameters.py`. Script đọc ba
checkpoint mới bằng `weights_only=True`, không dựng và không chạy model. Các dòng lịch sử
lấy từ báo cáo Q05 đã lưu và `models/manifest.json`.

| Model | Tham số (định nghĩa Q05) | Dùng khi inference | Decoder | Trọng số FP32 | File ONNX FP32 | Checkpoint train |
|---|---:|---:|---:|---:|---:|---:|
| Piper_no_VITS2_cpn (ResBlock2) | 16.425.776 | 15.663.771 | 1.664.768 | 65,7 MB | chưa export | 845,9 MB |
| Piper_ParIR_no_VITS2_cpn | 15.918.032 | 15.156.027 | 1.157.024 | 63,7 MB | chưa export | 843,4 MB |
| Piper_SeqIR_no_VITS2_cpn | 15.870.768 | 15.108.763 | 1.109.760 | 63,5 MB | chưa export | 842,7 MB |
| HiFi-GAN ResBlock2 (baseline, có VITS2) | 17.620.784 | — | — | 70,5 MB | 68,5 MB | 867,0 MB |
| Parallel-IR (MRF) | 17.113.040 | — | — | 68,5 MB | 66,5 MB | 864,5 MB |
| Sequential-IR (SEQ) | 17.065.776 | — | — | 68,3 MB | 66,3 MB | 863,9 MB |
| Checkpoint Piper bên ngoài | 16.425.776 | — | — | 65,7 MB | 63,5 MB | 845,9 MB |

Cách đọc các cột (MB = 10^6 byte):
- **Tham số (định nghĩa Q05):**
  - tham số của generator, không tính posterior encoder `enc_q` (chỉ dùng khi train);
  - đếm ở dạng weight-norm, không tính buffer;
  - cùng định nghĩa với các báo cáo export Q05, nên bảy dòng so sánh được với nhau.
  - Con số này vẫn gồm 762.005 tham số của stochastic duration predictor mà inference không dùng: nhánh posterior `dp.post_*` chỉ dùng khi train (640.248) và flow bị bỏ ở chế độ reverse (121.757).
- **Dùng khi inference:** số Q05 trừ đi 762.005 tham số đó. Chỉ tính cho các model mới.
- **Trọng số FP32:** 4 byte × số Q05. Đây là con số tính ra, không phải kích thước file.
- **File ONNX FP32:** kích thước đo được của các bản export Q05. Graph ONNX chỉ chứa đường inference, nên nhỏ hơn cột trọng số FP32. Các model mới chưa được export.
- **Checkpoint train:** toàn bộ trạng thái Lightning: generator, posterior encoder, discriminator và trạng thái optimizer. Đây không phải kích thước model để triển khai.

Các con số cho thấy:
- **Phần chung:** ở ba model mới, mọi thứ ngoài decoder giống hệt nhau, 21.999.024 tham số (tính cả `enc_q`).
- **Decoder:** decoder của ParIR và SeqIR ít hơn ResBlock2 lần lượt 30,5% và 33,3%. Tính trên toàn model thì chỉ ít hơn 3,1% và 3,4%, vì decoder chỉ là một phần nhỏ của generator.
- **Piper_no_VITS2_cpn và checkpoint Piper bên ngoài:** cả hai có đúng 16.425.776 tham số. Cả hai đều dùng kiến trúc Piper (Config A).
- **Chỉ dùng khi train:** MPD (46.747.132) và, với các model IR, MRD (280.902) là discriminator. Chúng không thuộc model synthesis.

Việc phát hành công khai các bộ câu, bản ghi kết quả và audio sinh ra vẫn do tác giả
quyết định. Việc chúng có mặt trong repo không khẳng định các quyền đó.
