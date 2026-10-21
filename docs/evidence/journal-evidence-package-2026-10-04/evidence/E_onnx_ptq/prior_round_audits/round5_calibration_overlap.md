# Lượt 5 — Kiểm tra calibration và dữ liệu chọn cấu hình PTQ

Ngày 28/09/2026. Chỉ đọc source, JSONL, manifest và log của `<WSL_PROJECT_HOME>/`; không quantize, inference, huấn luyện hoặc sửa artifact gốc. Báo cáo này thuộc các model baseline/MRF của BanhmiTTS, tách khỏi cặp lịch sử 773/1079 ở `<WSL_USER_HOME>/`.

## Kết quả chính

**Protocol PTQ đã lưu không chứng minh calibration chỉ dùng train hay lựa chọn scope chỉ dùng validation.** Lệnh chạy được ghi trong log lấy dữ liệu từ toàn bộ `dataset.jsonl`. Tái dựng trên source/manifest hiện có cho thấy calibration giao tập test 3 câu của baseline và 1 câu của MRF. Tập 25 câu dùng chấm WER để chọn cấu hình PTQ cũng giao 1 câu với mỗi tập test.

Đây là giao nhau ở **đầu vào calibration và lựa chọn cấu hình**; không tự chứng minh gradient training đã dùng các câu test, không chứng minh độ lệch điểm số là bao nhiêu, và không làm các con số đã lưu thành số liệu bịa. Cần giữ chúng là kết quả lịch sử/thăm dò với giới hạn này, rồi tái tạo PTQ bằng calibration train-only và selection validation-only trước đánh giá cuối.

## Bằng chứng lệnh chạy và phép tái dựng

- `<WSL_PROJECT_HOME>/scripts/ptq_quantize.py`: `JsonlCalibReader` đọc toàn bộ JSONL, dùng `random.Random(42).sample(entries, min(n_samples, len(entries)))`, không lọc split. Source hash `392ca92c1b7e1f81f3f4f3c18435b7d6455475500e6e8282e6346fe3281de5ab`.
- `<WSL_PROJECT_HOME>/quantized_models/ptq_sweep/sweep.log`, dòng 40: lệnh tạo `baseline_flow_enc_p_dp.onnx` dùng `--calibration-dataset <WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/dataset.jsonl --calibration-sentences 60`; dòng 42 chấm 25 câu trên cùng JSONL. Log hash `54fbca6e6935fdbdb05ac898550d17cb98ace255f32e59bd60ddec50170c7703`.
- `<WSL_PROJECT_HOME>/scripts/ptq_sweep_mrf.log`, dòng 80/82: lệnh tương tự cho `mrf_flow_enc_p_dp.onnx`; hash `372cf05be81a4e1edf7692ab286750e756185503c780d483bba7e4a62f0be96b`.
- `eval_wer_cheap.py` mặc định seed 123, lấy ngẫu nhiên 25 entries từ toàn JSONL; không kiểm tra validation membership.
- Đã tái kiểm tra thứ tự 60 mẫu bằng chính `<WSL_PROJECT_HOME>/env/bin/python` (Python 3.10.20) được ghi trong log: khớp hoàn toàn bản tái dựng.

JSONL hiện có 13.100 entries, hash `f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b`. Không có hash JSONL/calibration-ID manifest được lưu ngay trong lệnh lịch sử, nên phép xác định ID dưới đây vẫn có điều kiện: source và nội dung/thứ tự JSONL tại thời điểm chạy phải giống bản hiện có.

## Các ID giao nhau

So khớp bằng toàn bộ `text` và `phoneme_ids`, kiểm tra thêm `audio_path`/cache ID. Mỗi test manifest có 500 signatures duy nhất; cả 500 đều khớp JSONL hiện tại.

| Vai trò | Baseline test | MRF test |
|---|---|---|
| Calibration 60, seed 42 | 3: LJ033-0033, LJ040-0043, LJ047-0012 | 1: LJ037-0042 |
| Chọn scope bằng WER 25, seed 123 | 1: LJ033-0033 | 1: LJ003-0103 |

Hai bộ test chỉ giao **21/500** câu, nên không dùng toàn bộ 500 hàng để kiểm định ghép cặp giữa baseline và MRF. Tập 60 calibration và tập 25 tuning giao nhau 2 mẫu; tập tuning không phải bộ validation tách biệt được đóng băng.

Test manifest hashes:

- `/tmp/eval_true_testset_500_baseline.json`: `72916b93d84a9ea7acc020da5af966a1fb6233b2af3594a95e62feff8838756e`.
- `/tmp/eval_true_testset_500_mrf.json`: `01986a8e8b85aa2bcb7ff89f9cc9db33f0cafc13a7496c568ccad1c9da20b3c3`.

## Quan hệ với canonical_split.json hiện có

File `<WSL_PROJECT_HOME>/BanhmiTTS/training_output/ljspeech/medium/canonical_split.json` có 12.500 train / 100 val / 500 test, hash `e678cf436075a10bb68c9a1b435f33c4a40ff1f095b7b78ac12f546c5f12ae9d`.

| Tập đang đối chiếu | Canonical train | Canonical val | Canonical test |
|---|---:|---:|---:|
| Calibration 60 | 57 | 0 | 3 |
| Tuning 25 | 23 | 1 | 1 |
| Baseline test 500 | 0 | 0 | 500 |
| MRF test 500 | 478 | 1 | 21 |

Canonical hiện khớp baseline test. **Không gán canonical này cho quá trình train MRF lịch sử** chỉ từ việc nó tồn tại cùng thư mục. Tương tự, cột “478 train” ở bảng này nói về membership của canonical hiện tại; nó không chứng minh MRF đã train trên 478 câu đó.

## Nội dung nên bổ sung vào luận văn

> Saved PTQ sweep logs calibrate both baseline and MRF exports with 60 inputs sampled from the full dataset JSONL, and select quantization scopes using 25 further sampled inputs. Reconstructing the logged commands with the present source and manifest gives calibration overlaps of three baseline-test utterances and one MRF-test utterance; the scope-selection subset overlaps one test utterance for each model. These records do not establish train-only calibration or validation-only tuning. The historical results remain exploratory, and the final protocol requires frozen training calibration IDs and separate validation selection before held-out evaluation. The reconstruction does not quantify any resulting score bias or prove gradient-training leakage.

## Cần tiếp tục

1. Giữ model/artifact lịch sử kèm hash, ghi rõ giới hạn protocol; không tự nâng thành ma trận thực nghiệm cuối đã hoàn chỉnh.
2. Xác minh split của checkpoint baseline/MRF và chọn tập đánh giá chung hợp lệ. Nếu không có 500 câu chung thực sự held-out, không tự coi hai tập hiện tại là cùng một tập.
3. Đóng băng danh sách ID calibration chỉ từ train; dùng validation riêng để chọn scope. Tái tạo PTQ từ cùng checkpoint/FP32 export, không cần huấn luyện thêm chỉ vì PTQ.
4. Sau khi khóa các điều kiện trên, chấm lại chất lượng và đo tám ô Intel N150/Pi 5 theo protocol cuối.

Máy đọc: `round5_calibration_overlap.json`, gồm 60 ID đầy đủ, chỉ số từng câu giao nhau, log và source hashes. Không thay đổi kết quả hay graph ONNX gốc trong lượt kiểm tra này.
