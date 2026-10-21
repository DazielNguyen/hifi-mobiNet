# Q05 — Export ONNX FP32 và kiểm chứng nguồn

Cập nhật **04/10/2026**. **Q05 hoàn tất cho bốn artifact FP32 mới:** baseline 1489, MRF 1386, SEQ 1442 và Piper gốc `lj-med_1000` (metadata epoch 999). Đã export, xác định checkpoint/source/config bằng hash và kiểm tương thích số học với PyTorch. Chưa thực hiện PTQ mới, chấm WER/UTMOS hoặc benchmark N150/Pi 5.

## 1. Artifact dùng cho các bước tiếp theo

Thư mục cố định trong WSL Ubuntu-24.04:

```text
<WSL_PROJECT_HOME>/paper_evidence/q05_2026-10-04T000830Z/
```

| Model | Checkpoint, epoch đếm từ 0 | ONNX FP32, đường dẫn tương đối với thư mục trên | MiB | SHA-256 rút gọn |
|---|---:|---|---:|---|
| Baseline | 1489 | `baseline/baseline_fp32.onnx` | 65,354 | `fdfda3ac…402ad6f` |
| MRF | 1386 | `mrf/mrf_fp32.onnx` | 63,464 | `14c9f547…ae869e7` |
| SEQ | 1442 | `seq/seq_fp32.onnx` | 63,269 | `4052629a…62a17d3` |
| Piper gốc | 999, checkpoint công bố mốc 1.000 epoch | `piper/piper_fp32.onnx` | 60,590 | `2c87a6c3…7420d7b` |

Hash đầy đủ, đường dẫn checkpoint, graph và báo cáo: [FP32_HANDOFF.json](evidence/q05_2026-10-04/FP32_HANDOFF.json). Đây là kích thước file ONNX, **không phải** peak RAM, số tham số decoder hay thời gian suy luận.

Các graph `*_validation_external_noise.onnx` trong cùng thư mục chỉ phục vụ đối chiếu số học; **không dùng làm artifact triển khai hoặc đầu vào PTQ** trong thiết kế hiện tại.

## 2. Liên kết checkpoint → source → export

- Đọc checkpoint từ bản sao cố định Q01, kiểm SHA-256 trước và sau xử lý. Khởi tạo đúng generator theo hparams; nạp toàn bộ state `model_g.*` bằng `strict=True`, không có missing/unexpected keys.
- Ba model BanhmiTTS dùng source BanhmiTTS đã lưu trong Q01. Piper dùng **source Piper đối chứng đã lưu trong Q01**, không thay bằng backbone BanhmiTTS. Các snapshot được sao chép sang gói Q05 trước khi chạy.
- Không train hoặc cập nhật trọng số. Bỏ weight normalization để export; kiểm PyTorch trước/sau thao tác này với cùng seed, cả bốn model đều đạt.
- Export FP32, opset 18, legacy TorchScript exporter (`dynamo=False`). Giữ hai nguồn nhiễu ngẫu nhiên của SDP và acoustic latent trong graph triển khai. Không ép noise về 0 để tạo artifact.
- Graph không chứa posterior encoder; các discriminator không thuộc generator được export. F0/Vocos không được kích hoạt ở bốn checkpoint này.
- Ghi checkpoint hash, epoch và source-manifest hash vào metadata ONNX. Lưu constructor đầy đủ, hparams checkpoint, phiên bản môi trường, argv, hash runner và log export.

Source là snapshot dùng **ở lần export này**, không được gán thành source runtime của mọi phiên train cũ. WSL source không có commit lịch sử đã xác minh; [SOURCE_MANIFEST.json](evidence/q05_2026-10-04/SOURCE_MANIFEST.json) định danh từng file bằng hash. Module Cython MAS của Piper được build từ source đã lưu trong thư mục cô lập để đáp ứng import; inference không gọi MAS.

Lệnh đã chạy, thay `<model>` lần lượt bằng `baseline`, `mrf`, `seq`, `piper`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
<WSL_PROJECT_HOME>/env/bin/python \
  <WSL_PROJECT_HOME>/paper_evidence/q05_2026-10-04T000830Z/scripts/run_q05_model.py \
  <WSL_PROJECT_HOME>/paper_evidence/q05_2026-10-04T000830Z <model>
```

Đây là lệnh lưu lại của lần chạy đã hoàn tất. Runner chủ động từ chối ghi đè thư mục model đã có; khi tái chạy cần chuẩn bị gói mới. [Runner thực sự đã chạy](evidence/q05_2026-10-04/run_q05_model.executed.py), [cấu hình lần chạy](evidence/q05_2026-10-04/RUN_CONFIG.json) và [script chuẩn bị](evidence/q05_2026-10-04/prepare_q05.py) được lưu để đối chiếu.

## 3. Sửa lỗi hợp đồng đầu ra trong exporter

`SynthesizerTrn.infer(...)[0]` đã trả waveform dạng `[B, 1, N]`. Exporter trước đó thêm `.unsqueeze(1)`, tạo `[B, 1, 1, N]`, rồi khai báo chiều 1 là thời gian. Đã sửa [export_onnx.py](../banhmi_train/export_onnx.py) trong repository Windows và bản BanhmiTTS liên quan trong WSL:

- Bỏ chiều thêm thừa; output là float32 `[batch, 1, samples]`.
- Khai báo chiều **2** là `samples` động. Text length và batch cũng được khai báo động.
- Giữ input `input: int64[B,L]`, `input_lengths: int64[B]`, `scales: float32[3]` theo thứ tự noise/length/noise_w; các model này không có input speaker ID.

Gói mới dùng hợp đồng đầu ra này cho cả bốn model. Bản trước sửa của exporter WSL cùng hash trước/sau được lưu trong [LIVE_EXPORTER_FIX.json](evidence/q05_2026-10-04/LIVE_EXPORTER_FIX.json). Các graph cũ giữ nguyên byte; không sửa đè để làm chúng trông giống graph mới.

## 4. Kiểm số học với cùng nhiễu, không dùng điểm chất lượng để chọn mẫu

Chọn trước **6 câu** ở các mức độ dài của manifest calibration train chung ứng viên, gồm **43, 147, 201, 249, 277 và 359 phoneme IDs**; danh sách cụ thể và hash nguồn nằm trong [EXPORT_CHECK_INPUTS.json](evidence/q05_2026-10-04/EXPORT_CHECK_INPUTS.json). Đây là đầu vào kiểm kỹ thuật, không phải kết quả WER/UTMOS hay chứng nhận độc lập của tập test.

Mỗi model có 10 ca: sáu độ dài trên; một ca đổi seed; hai ca length scale 0,8 và 1,2; một batch 2 có padding và độ dài thực khác nhau. Scales mặc định `[0.667, 1.0, 0.8]`, seed 1234 và ca bổ sung 1235. Các giá trị noise vẫn khác 0.

Quy trình đối chiếu:

1. Chạy PyTorch và lưu chính xác tensor nhiễu ở SDP và acoustic latent.
2. Tạo graph kiểm riêng bằng cách thay **chỉ hai random nodes** của graph triển khai thành hai input nhiễu. Kiểm cấu trúc còn lại giống graph gốc sau khi loại đúng hai node; trọng số không thay đổi.
3. Truyền cùng phoneme IDs, độ dài, scales và hai tensor nhiễu vào graph kiểm. So waveform với PyTorch, yêu cầu cùng shape và tất cả giá trị hữu hạn.
4. Lưu input/noise/waveform hai backend thành NPZ có hash. Tính lại độc lập sai số từ các NPZ khi đóng gói.
5. Chạy thêm ba câu ngắn/vừa/dài qua **graph triển khai còn random nodes**, kiểm shape, waveform hữu hạn và có tín hiệu. Những lần chạy này không phải đánh giá chất lượng tiếng nói.

Ngưỡng đã ghi trong `RUN_CONFIG.json` **trước khi chạy**: `NRMSE ≤ 0.001` và `max_abs_error ≤ 0.005`. NRMSE được tính bằng RMSE của sai khác chia RMS waveform PyTorch. Không thay ngưỡng sau khi xem kết quả.

| Model | Ca cùng nhiễu đạt | Max absolute error lớn nhất | NRMSE lớn nhất | Ca graph ngẫu nhiên gốc đạt |
|---|---:|---:|---:|---:|
| Baseline | 10/10 | 2,73176e-4 | 9,23032e-5 | 3/3 |
| MRF | 10/10 | 5,58168e-4 | 1,13963e-4 | 3/3 |
| SEQ | 10/10 | 1,42181e-4 | 4,41353e-5 | 3/3 |
| Piper | 10/10 | 1,34364e-4 | 7,05622e-5 | 3/3 |

Môi trường: Python 3.10.20; Torch 2.13.0+cu130 chạy **CPU**; ONNX 1.22.0; ONNX Runtime 1.23.2, `CPUExecutionProvider`, intra-op 2/inter-op 1. Kết quả thuộc môi trường này và các đầu vào đã ghi, không phải bảo đảm sai số cho mọi câu hoặc mọi thiết bị.

Log giữ các `TracerWarning` về nhánh Python trong attention/spline/cảnh báo duration và thông báo constant folding. Các ca đã chọn kiểm được nhiều độ dài, batch và scales; không gọi việc vượt qua chúng là chứng minh mọi nhánh dữ liệu có thể gặp đều đã được bao phủ.

## 5. RNG của graph triển khai

Cả bốn model cho cùng kết quả khi tạo **hai ORT session mới**, đặt cùng seed 1234 trước mỗi session và chạy cùng đầu vào kiểm. Hai lần gọi liên tiếp trong **cùng session** cho kết quả khác nhau. Kết luận chỉ áp dụng cho runtime/provider và đầu vào đã kiểm.

Q06/Q08 phải quản lý rõ vòng đời session và random state; không giả định đặt seed Python/Torch sẽ điều khiển noise trong ONNX. Việc dùng cùng số seed qua PyTorch và ORT cũng không tạo cùng tensor nhiễu, nên phép đối chiếu mục 4 mới dùng tensor được lưu trực tiếp. Chưa coi phần RNG thực thi của evaluator Q04 đã hoàn tất chỉ từ phép kiểm này.

## 6. Những graph cũ có nguồn đến đâu?

Đối chiếu initializer còn giữ tên với state generator tương ứng sau bỏ weight normalization:

| Graph FP32 cũ | Tensor cùng tên khớp chính xác | Tensor cùng tên khác | Diễn giải |
|---|---:|---:|---|
| Baseline `ptq_sweep/baseline_fp32.onnx` | 422 | 0 | Các trọng số đối chiếu được phù hợp checkpoint 1489; chưa xác minh hết tensor đã đổi tên/fold, graph semantics hay lệnh export lịch sử. |
| MRF `ptq_sweep/mrf_fp32.onnx` | 0 | 478 | **Không khớp trọng số checkpoint 1386** ở các tensor đã so. Không dùng graph này như FP32 đã xác minh của 1386. Chưa xác định checkpoint nguồn chính xác. |
| SEQ `/tmp/seq_final_onnx/seq_final_fp32.onnx` | 454 | 0 | Có thêm bằng chứng phù hợp checkpoint 1442 ở các trọng số đối chiếu được; chưa khôi phục lệnh export/calibration cũ. |

Q05 giải quyết nhu cầu thực nghiệm tiếp theo bằng **bốn graph mới có nguồn xác định**. Các số cũ vẫn gắn với hash artifact cũ; không chuyển WER/UTMOS/RTF lịch sử sang bảng của các graph Q05. Các graph PTQ cũ chưa được xác minh lại hoặc tạo lại trong lượt này.

## 7. Bằng chứng và điều kiện hoàn tất

- [FP32_HANDOFF.json](evidence/q05_2026-10-04/FP32_HANDOFF.json): artifact chính thức chuyển sang Q06/Q07/Q08, hợp đồng input/output và giới hạn.
- [FINAL_VERIFICATION.json](evidence/q05_2026-10-04/FINAL_VERIFICATION.json): **328 phép kiểm đều đạt**, gồm kiểm hash/source/graph và tính lại 40 ca số học từ NPZ; có 12 ca chạy graph ngẫu nhiên gốc trong báo cáo.
- Báo cáo từng model: [baseline](evidence/q05_2026-10-04/baseline/REPORT.json), [MRF](evidence/q05_2026-10-04/mrf/REPORT.json), [SEQ](evidence/q05_2026-10-04/seq/REPORT.json), [Piper](evidence/q05_2026-10-04/piper/REPORT.json). Mỗi thư mục còn có `export.log` và `checkpoint_hparams.json`.
- [ARCHIVE_LOCATION.json](evidence/q05_2026-10-04/ARCHIVE_LOCATION.json), [FILE_MANIFEST.json](evidence/q05_2026-10-04/FILE_MANIFEST.json), [SHA256SUMS](evidence/q05_2026-10-04/SHA256SUMS): gói WSL 188 file, khoảng 587,8 MB, gồm source, graph, NPZ, log và biên bản. Không bao gồm việc sao chép lại checkpoint Q01.

**Q05: hoàn tất phạm vi FP32 mới và kiểm tương thích.** Q03 vẫn giữ các giới hạn lịch sử; Q04 vẫn chưa khóa toàn bộ điều kiện đánh giá. Việc tiếp theo là **Q06**: chốt protocol calibration/selection và ngưỡng chấp nhận, tạo selective PTQ từ đúng graph Q05; sau đó Q08 chấm chất lượng và Q07 đo trên N150/Pi 5.
