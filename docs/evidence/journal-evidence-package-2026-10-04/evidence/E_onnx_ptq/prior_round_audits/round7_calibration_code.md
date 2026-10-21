# Lượt 7 — đọc code calibration, split và chọn cấu hình PTQ

Ngày 29/09/2026. Chỉ đọc code, JSON, log và metadata thư viện đã cài; không huấn luyện, suy luận, export, quantize hoặc sửa source/checkpoint. Bằng chứng có số dòng và SHA-256 nằm trong `round7_calibration_code.json`. Phạm vi chính là BanhmiTTS và script lịch sử tại `<WSL_PROJECT_HOME>/`; tài liệu BanhmiTTS trong workspace chỉ dùng bổ trợ.

## Kết luận có thể chốt từ code

**Calibration và sweep PTQ đã được chạy trong lịch sử. Tuy nhiên, chưa tìm thấy lần chạy calibration chỉ dùng train và chọn cấu hình chỉ dùng validation với manifest đã lưu.** Có thể hoàn thành việc mô tả đúng triển khai trong luận văn; chưa thể đánh dấu hoàn thành thực nghiệm cuối chỉ bằng việc đọc code.

| Nội dung | Kết quả kiểm tra |
|---|---|
| Calibration lịch sử | 60 mẫu từ toàn bộ `dataset.jsonl`, chọn không hoàn lại bằng `random.Random(42)`; không lọc train/val/test. |
| Đầu vào calibration | `phoneme_ids` kiểu int64, shape `[1,L]`; độ dài kiểu int64 `[1]`; `scales=[0.667,1.0,0.8]` float32. Không dùng waveform tham chiếu, F0 hay nhãn WER để calibration. |
| Phương pháp | PTQ tĩnh, `QOperator`, trọng số `QInt8` theo kênh, activation `QUInt8`; `Conv,ConvTranspose` là danh sách op yêu cầu cho baseline/MRF. |
| Mặc định thư viện hiện cài | ONNX Runtime 1.23.2: `MinMax`, `reduce_range=False`, calibration chạy CPU, không moving average, range calibration không bị ép đối xứng. Script không ghi đè các mặc định này. |
| Scope được dùng trong báo cáo lịch sử | `flow,enc_p,dp`; giữ decoder FP32. Việc ConvTranspose có trong danh sách yêu cầu không chứng minh ConvTranspose thực sự bị lượng tử hóa trong scope này. |
| Sweep | 14 ứng viên mỗi model; baseline có 14 kết quả WER, MRF có 10 kết quả WER và 4 lỗi quantize khi scope chứa decoder. |
| Quy tắc lựa chọn | Code chỉ lưu từng ứng viên, WER và dung lượng. Không có thuật toán `argmin`, ngưỡng chấp nhận hay manifest đánh dấu ứng viên thắng. `flow_enc_p_dp` có WER nhỏ nhất trong các dòng thành công đã lưu: baseline 0.12146555; MRF 0.10777549. Đây là quan sát trên screening lịch sử, không chứng minh cấu hình tối ưu. |
| Seed | Seed 42 cố định danh sách calibration; seed 123 cố định 25 câu screening. Các seed lấy mẫu này không kiểm soát các node ngẫu nhiên trong ONNX. |

Nguồn: `ptq_quantize.py:38–60, 103–115`; `run_ptq_sweep.py:18–37, 46–106, 113–124`; `eval_wer_cheap.py:27–80`; `onnxruntime/quantization/quantize.py:479–495`; `calibrate.py:213–239, 297–320, 484–508, 1176–1202`. Các mô tả QAT/Vocos trong docstring là lịch sử của script, không phải yêu cầu khôi phục QAT/Vocos vào phạm vi luận văn.

## Split hiện có và giới hạn truy xuất lịch sử

`canonical_split.json` hiện có **12.500 train / 100 validation / 500 test**. Đã kiểm tra đủ 13.100 ID: không trùng trong từng tập, không giao giữa ba tập, không thiếu hoặc thừa ID so với JSONL hiện tại. SHA-256 canonical: `e678cf436075a10bb68c9a1b435f33c4a40ff1f095b7b78ac12f546c5f12ae9d`; JSONL: `f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b`.

Loader hiện tại đọc canonical nếu có; nếu thiếu, dùng `torch.Generator().manual_seed(20260101)` riêng để tách split độc lập với RNG khởi tạo kiến trúc (`training.py:275–315`). Đây là cơ chế đã được triển khai. Nhưng `run_ptq_sweep.py` không đi qua loader này: nó chuyển thẳng JSONL đầy đủ cho quantizer và cheap evaluator.

Tái dựng từ source/JSONL hiện có cho thấy 60 calibration gồm **57 canonical-train + 3 canonical-test**; 25 screening gồm **23 canonical-train + 1 canonical-validation + 1 canonical-test**. Đối chiếu hai test manifest lịch sử ở lượt 5 đã xác định calibration giao baseline-test 3 câu, MRF-test 1 câu; screening giao mỗi test-set 1 câu. Kết luận ID lịch sử vẫn có điều kiện vì chưa có hash dữ liệu/source được ghi ngay trong lần chạy.

Không thể gán canonical hiện tại cho checkpoint MRF cũ chỉ vì file nằm cùng thư mục. Tài liệu BanhmiTTS mô tả split cũ phụ thuộc RNG khởi tạo kiến trúc và MRF đã train trước khi có canonical. Các file `/tmp/eval_true_testset_500_baseline.json` và `_mrf.json` là manifest test được tái dựng; sự tồn tại của chúng không phải manifest train/validation được đóng băng cùng checkpoint. Loader hiện tại cũng không chứng minh mọi lần resume trước đây đã dùng cùng split.

Đã quét 442 file text thuộc cây `<WSL_PROJECT_HOME>/` (bỏ environment, cache, WAV, symlink; text tối đa 2 MB), cùng tên các manifest test liên quan trong `/tmp`. Không thu hồi được manifest calibration train-only hoặc tuning validation-only. Giới hạn này là “chưa có bằng chứng đã lưu trong phạm vi đã đọc”, không khẳng định nhóm chưa từng thử ở nơi khác.

## Những thay đổi cụ thể cần áp dụng vào protocol

1. Gắn checkpoint, ONNX FP32, split và source bằng hash trong một bản ghi chạy. MRF cuối vẫn cần giải quyết ONNX khớp checkpoint như lượt 6; canonical không thay thế được provenance đó.
2. Tạo và lưu danh sách calibration từ **train của model đã xác minh**, kiểm tra không giao validation/test; giữ nguyên danh sách này cho tất cả scope đang so sánh. Có thể bắt đầu từ 60 mẫu để kế thừa cấu hình, nhưng code hiện có chưa chứng minh 60 là đủ đại diện mọi độ dài/phoneme; số mẫu cuối phải được quyết định trên validation.
3. Dùng validation đã đóng băng để chọn scope, ghi rõ tiêu chí và kết quả tất cả ứng viên; không dùng test để chọn recipe. 25 câu toàn JSONL hiện nay chưa đáp ứng điều này. Sửa luồng resampling trong cheap WER trước khi dùng lại: audit evaluation riêng đang xác nhận đầu vào 22.05 kHz truyền trực tiếp cho Whisper ndarray vốn yêu cầu 16 kHz.
4. Ghi phiên bản ORT, MinMax, dtype, granularity, scope, op thực tế, inference scales và cơ chế kiểm soát stochastic inference. Seed chọn câu không được ghi thành seed kiểm soát toàn bộ suy luận.
5. Khóa recipe trước khi đánh giá FP32/PTQ cùng test của cùng model; so sánh baseline/MRF cần test chung có provenance hợp lệ hoặc nêu rõ hai test-set khác nhau. Không tự gán lại canonical cho MRF cũ, không tự phát sinh yêu cầu retrain chỉ từ audit này.

## Trạng thái phù hợp cho checklist

- [x] Đọc và xác minh thuật toán calibration, đầu vào, dtype, scope và mặc định MinMax từ thư viện cài thực tế.
- [x] Xác minh canonical hiện tại hợp lệ về cấu trúc; phân biệt loader đã sửa với sweep lịch sử vẫn dùng toàn JSONL.
- [x] Thu hồi số ứng viên sweep và cách lưu kết quả; tránh gọi scope là “tối ưu” hoặc “validation-selected” khi thiếu bằng chứng.
- [ ] Xác minh split gắn với các checkpoint cuối, lưu calibration train-only và tuning validation-only cùng hash.
- [ ] Chạy PTQ và evaluation theo protocol đã sửa, sau khi nguồn export MRF cuối và phạm vi chạy được chốt.

Số liệu lịch sử được giữ nguyên và gắn giới hạn phương pháp; audit không tạo ra số liệu thực nghiệm mới.
