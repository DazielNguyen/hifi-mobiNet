# Đo sáu model trên Intel N100 / Ubuntu / OpenVINO CPU

Đây là bộ scripts và **sáu graph FP32 có sẵn**, dùng nội bộ. Chưa có kết quả
OpenVINO/N100 trong gói. Không cần checkpoint, PyTorch, training dataset hay audio cache.
Không chạy inference/quantization trên PC khi tạo gói. Không có lệnh upload.

| ID | Nhóm | Selected checkpoint epoch |
| --- | --- | ---: |
| baseline-resblock2 | Historical Q05 FP32 / ResBlock2 | 1489 |
| parallel-ir | Historical Q05 FP32 / ParIR | 1386 |
| sequential-ir | Historical Q05 FP32 / SeqIR | 1442 |
| piper-no-vits2-cpn | Config A / Piper, FP32 round 2 | 1403 |
| piper-parir-no-vits2-cpn | Config A / ParIR, FP32 round 2 | 1486 |
| piper-seqir-no-vits2-cpn | Config A / SeqIR, FP32 round 2 | 1455 |

Tất cả dùng 22.050 Hz. `models.json` lưu SHA-256 graph, checkpoint nguồn và config.
Các quality report lịch sử chưa được ràng buộc đầy đủ với ba graph Q05 này;
không ghép điểm quality cũ vào bảng N100 như thể đã đo cùng artifact.
Piper external là model thứ bảy, không thuộc sáu model nội bộ. Low đang train và
High chưa hoàn tất cũng không thuộc bộ này.

## Copy và kiểm tra

Copy ZIP và `.zip.sha256` sang Ubuntu/N100, rồi chạy trong thư mục chứa chúng:

```bash
sha256sum -c hifi-mobiNet-n100-openvino-2026-10-31.zip.sha256
unzip hifi-mobiNet-n100-openvino-2026-10-31.zip
cd hifi-mobiNet-n100-openvino-2026-10-31
sha256sum -c SHA256SUMS
python3 -m unittest discover -s tests -p test_edge_openvino.py
```

## Môi trường trên N100

Dùng Python 3.10–3.12 trong môi trường OpenVINO đã có trên N100. Nếu thiếu thư viện,
cài trong chính môi trường dành cho phép đo, không vào môi trường train trên PC:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > environment-lock.txt
set -o pipefail
python openvino_suite.py doctor 2>&1 | tee doctor.log
```

`requirements.txt` là danh sách đầu vào cho pip, chưa phải lockfile đã kiểm thử trên
N100. Giữ nguyên environment-lock và versions cho cả sáu model. Scripts kiểm tra
Linux/N100, dependencies và hashes; không tự cài, tự train hay tự khởi chạy High.
Không dùng `AUTO`/GPU trong thí nghiệm CPU này. Chạy từ thư mục vừa giải nén.

## Chuẩn bị FP32 và INT8 trên N100

```bash
python openvino_suite.py prepare --work work-n100 2>&1 | tee prepare.log
```

Lệnh này thực sự convert sang IR, calibration/quantize và chạy parity/smoke trên N100.
Nó kiểm tra FP32 giữa ONNX Runtime và OpenVINO trên 13 câu có cùng noise đầu vào
(NRMSE ≤ 0,001; max absolute error ≤ 0,005), rồi lưu IR FP32 không nén FP16.

INT8 là **thí nghiệm NNCF mới, riêng biệt** cho sáu model: Conv trong enc_p/dp/flow,
decoder và các weighted operation khác được loại khỏi phạm vi quantize. Dùng 60
câu train cố định, kiểm membership canonical split và đối chiếu phoneme IDs với
60 dòng lấy từ dataset manifest nguồn; không dùng Harvard hay test set để calibration.
Preset MIXED, CPU, fast bias correction; danh sách layer cụ thể được ghi lại.
Đây là INT8 chọn lọc, còn phần FP32, không phải mọi operation đều INT8.
Graph thực thi được kiểm tra để tránh gắn nhãn INT8 khi backend chỉ chạy FP32.

Không dùng graph INT8 lịch sử hay round 1 có lỗi flow weight norm. Không gộp các số
NNCF mới với kết quả ORT PTQ round 2. Đây là thay đổi backend/quy trình quantization,
không phải phép chuyển nguyên trạng INT8 cũ.

`work-n100/<model>/preparation.json` lưu trạng thái, hashes, phiên bản và calibration;
IR `.xml/.bin`, parity report, runtime nodes và 3 WAV pilot/precision ở cùng thư mục.
WAV pilot được peak-normalize riêng để nghe, không thay đổi waveform dùng đo RTF.
Smoke/parity FP32 **không chứng minh chất lượng INT8**. Listening/WER/UTMOS cho INT8
vẫn pending; cần đánh giá chất lượng riêng trước khi kết luận về trade-off.
Native RNG của graph không bị ép đồng nhất giữa runtime/precision; chỉ parity dùng
noise bên ngoài cố định. Không cam kết INT8 tạo lại có bytes giống hệt.

Scripts dừng nếu có lỗi hoặc phạm vi/precision không xác minh được. Không tự bỏ qua
model lỗi. Mỗi thư mục model phải mới. Khi chuẩn bị từng model, dùng `--model <ID>`;
nếu cần làm lại, giữ thư mục lỗi làm evidence và dùng một `--work` mới cho cả bộ.

## Đo RTF

Đóng ứng dụng nặng trên N100, dùng cùng nguồn điện, cooling và cấu hình nguồn cho
cả bộ. Scripts ghi nhiệt độ/governor nếu đọc được nhưng không tự chỉnh hệ thống.
Chạy một suite tại một thời điểm. Không chạy prepare và run đồng thời.

```bash
python openvino_suite.py run --work work-n100 --threads 4 --rounds 3 --output results-n100-t4 2>&1 | tee run-t4.log
```

Để đo cả 1/2/4 luồng thay cho lệnh trên:

```bash
python openvino_suite.py run --work work-n100 --threads 1 2 4 --rounds 3 --output results-n100-t124 2>&1 | tee run-t124.log
```

Protocol: batch 1, CPU, LATENCY, một stream, synchronous InferRequest. FP32 dùng
precision hint f32 và ACCURACY; INT8 dùng chính IR đã quantize, không đổi hint thành
INT8. 720 câu Harvard/round, scales `[0.667, 1.0, 0.8]`, ba round. Mỗi cell là process
mới; thứ tự model/precision được đảo giữa round, cùng thứ tự câu trong mỗi round.
Warm-up 10 câu trải theo độ dài; không warm-up từng dynamic shape, nên chi phí lần
đầu gặp shape mới sau warm-up vẫn nằm trong phép đo.

Thời gian chỉ bao quanh `InferRequest.infer(share_outputs=True)`, gồm overhead gọi
Python/OpenVINO và trả output views. Load/compile, tạo/gán input, phonemization,
kiểm audio, ghi file và scoring đều nằm ngoài timer. Đây là **RTF model inference**,
không phải text-to-WAV end-to-end, time-to-first-audio hay throughput phục vụ nhiều request.

`RTF = latency_seconds / (output_samples / sample_rate)`. RTF < 1 nghĩa là thời gian
sinh toàn câu ngắn hơn thời lượng audio. Mỗi lần dùng chiều dài output thực tế;
native RNG có thể làm thời lượng khác nhau giữa FP32/INT8. Báo cả latency và duration
để tránh diễn giải mọi khác biệt RTF là tốc độ kernel. Không hứa INT8 sẽ nhanh hơn.

Kết quả: `summary_fp32.csv`, `summary_int8.csv`, `summary.json`; từng cell có raw
`utterances.jsonl`, `result.json`, runtime graph metadata và log. Có mean/median/p95
RTF, duration-weighted RTF = tổng latency/tổng audio, latency, peak process RSS
(bao gồm runtime/input/model, không phải riêng model). Ba round trên cùng 720 câu
không được coi là 2.160 câu độc lập. Full matrix 4 luồng có 36 cell/25.920 lượt timed;
1/2/4 luồng có 108 cell/77.760 lượt timed, chưa tính warm-up/prepare.

Run không ghi đè output có sẵn. Khi fail, giữ raw records/log; summary đầy đủ không
được tạo nếu thiếu cell/câu. Sau khi hoàn tất có thể dựng lại bảng từ raw records:

```bash
python openvino_suite.py summarize --output results-n100-t4
```

## Nội dung và nguồn

`PACKAGE-MANIFEST.json` ghi association, source path, size, hash, loại trừ và phần
chưa chạy. `SHA256SUMS` phủ toàn bộ payload trừ chính file checksum. Gói không chứa
checkpoint, dataset audio, cache, venv hoặc credentials. Các đường dẫn dataset trong
metadata chỉ là khóa đối chiếu, không cần tồn tại trên N100.

API synchronous/output views: [OpenVINO InferRequest](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.InferRequest.html).
Calibration và ignored scopes: [NNCF basic quantization](https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html).
Kiểm soát precision: [OpenVINO precision control](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html).

Gói được kiểm tra hash nguồn/bản copy, CRC, file list, giải nén thử và internal hashes.
Đây là kiểm tra đóng gói; compatibility/parity/quantization/timing trên N100 chỉ được
xác nhận bằng các record mới sinh khi bạn chạy bộ scripts trên thiết bị đó.
