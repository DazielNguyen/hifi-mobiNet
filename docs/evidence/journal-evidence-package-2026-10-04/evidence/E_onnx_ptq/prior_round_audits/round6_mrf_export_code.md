# Đọc code nguồn export MRF — 29/09/2026

**Đã có code và checkpoint MRF; đã có ONNX/PTQ và kết quả đánh giá lịch sử.** Phần còn mở là xác định checkpoint nguồn của đúng file `mrf_fp32.onnx` đang lưu, hoặc thực hiện một bản export mới có thể đối chiếu với checkpoint đã chốt. Không nên ghi toàn bộ MRF là “team chưa chạy”. Với bước riêng “export lại từ checkpoint 1386 và lưu bằng chứng khớp”, hiện chưa có kết quả lưu đã xác minh.

Đây là kiểm tra tĩnh: chỉ đọc source và đối chiếu kết quả audit trước. Không dựng model, nạp checkpoint mới, export, inference, train, PTQ hay thay đổi source. Báo cáo JSON đi kèm lưu SHA256 đầy đủ và trích đoạn có số dòng của 25 nguồn. Tám file code/config đối chiếu giữa Windows/BanhmiTTS và `<WSL_PROJECT_HOME>/BanhmiTTS` giống nhau sau khi chuẩn hóa xuống dòng.

| Nội dung | Bằng chứng code | Kết luận |
|---|---|---|
| Chọn kiến trúc | `BanhmiTTS/configs/banhmi_mrf.yaml:42`; `vits/training.py:191–196`; `vits/modules/synthesizer.py:107–118` | `resblock=mrf`, expansion `[1,1,1]` được chuyển từ hparams sang generator. Export đọc hparams checkpoint, không tự lấy YAML mới để thay kiến trúc. |
| MobileNet/MRF | `vits/modules/generator.py:98–129,156–158` | Mỗi tầng có ba nhánh `ResBlockInverted` song song, kernel mặc định 3/5/7; lấy trung bình đầu ra. Có implementation thật, khác nhánh `mb` xếp tuần tự. |
| Export chính | `banhmi_train/export_onnx.py:26–44,64–84` | Nhận checkpoint/output cụ thể; nạp checkpoint, chuyển eval, bỏ weight normalization decoder/flow, export ONNX opset18. Không tự chọn “best” theo tên thư mục. |
| Export scratch đã tìm thấy | `<WSL_PROJECT_HOME>/scripts/export_fp32_onnx.py:24–32,52–66` | Cùng cách export; thêm `map_location=cpu`, `use_f0=False` để tương thích checkpoint cũ. Không đổi sang checkpoint khác khi load lỗi. |
| Khôi phục trọng số | `<WSL_PROJECT_HOME>/env/lib/python3.10/site-packages/pytorch_lightning/core/saving.py:58–64,243–259` | API đang cài mặc định `strict=True`, rồi nạp `checkpoint['state_dict']`. `VitsModel` không override load hoặc hook nạp checkpoint. Export FP32 thông thường không truyền `strict=False`. |
| PTQ sweep | `<WSL_PROJECT_HOME>/scripts/run_ptq_sweep.py:46–66,119–123` | Dùng file `quantized_models/ptq_sweep/mrf_fp32.onnx` đã tồn tại. Không export lại FP32, không gắn checkpoint1386, không kiểm tra hash nguồn. Vì vậy sửa/tiếp tục checkpoint không tự cập nhật graph cũ. |
| Phân biệt Piper | `piper/src/python/piper_train/export_onnx.py:8,41`; `piper/src/python/piper_train/vits/models.py:299–318` | Tab Piper đang mở không phải đường export MRF của BanhmiTTS đang kiểm tra. Generator Piper này chọn ResBlock1/2; không thay nó làm bằng chứng cho MobileNet/MRF. |

## Sai khác embedding có phải do export biến đổi trọng số không?

Không tìm thấy cơ chế đó trong đường code đã đọc:

- Bỏ weight normalization chỉ áp dụng decoder/flow. Những trọng số này có thể đổi cách lưu, nên audit trước đã không so trực tiếp các initializer đã gộp theo cách thiếu tương đương.
- `text_encoder.py:33–34,42` dùng `nn.Embedding`; hệ số `sqrt(hidden_channels)` nhân vào **đầu ra tra embedding**, không ghi đè ma trận `enc_p.emb.weight`.
- Không có EMA, lấy trung bình checkpoint, ép FP16/BF16 chủ động, nạp checkpoint thay thế hoặc ghi lại embedding trong export FP32 đã đọc. Khởi tạo ngẫu nhiên khi dựng model được theo sau bởi bước khôi phục state dict; seed export không phải lời giải thích cho embedding khác sau khi load thành công.
- `sitecustomize.py` của môi trường chỉ allowlist `PosixPath` và bỏ kiểm tra API scheduler. Nó giải thích vì sao Python môi trường tự import Lightning, nhưng không sửa trọng số hoặc cách chọn checkpoint. Thu thập bằng chứng vòng này dùng `/usr/bin/python3 -S`, không import patch hay code mô hình.

Do đó, code hiện có **không giải thích** kết quả vòng5: graph MRF khác checkpoint1386 ở416 initializer có thể đối chiếu trực tiếp, gồm toàn bộ embedding49.152 phần tử. Kết quả fingerprint14 checkpoint ở vòng5 chưa xác định được nguồn đúng; vòng này không chạy lại phép so lớn đó. Không suy đoán một epoch khác, và không xem việc đọc code là kiểm tra runtime thành công.

Nhánh lưu trữ `export_fp32_from_qat.py:48–60` có hai lần load vì checkpoint lịch sử đã bỏ weight normalization; lần thứ hai có `assert not missing`. Nhánh này cũng nạp nguyên embedding và không chứng minh graph hiện tại xuất phát từ QAT. Đọc nó chỉ phục vụ truy nguồn; phạm vi nhóm vẫn là PTQ.

## Cách ghi vào checklist/luận văn

- [x] Có implementation MRF và đường export ONNX; đã truy cách chọn hparams và nạp trọng số.
- [x] Có checkpoint MRF và artifact ONNX/PTQ lịch sử theo audit vòng5; không mô tả toàn bộ công việc này là chưa chạy.
- [x] Xác minh code export thông thường không có phép biến đổi embedding giải thích liên kết sai với1386.
- [ ] Bản export được truy nguyên từ checkpoint1386 chưa có kết quả lưu đã xác minh. Nếu đây là bước team chưa chạy thì ghi “chưa thực hiện/chưa có kết quả”, rồi thực hiện sau; không coi đó là lỗi thực nghiệm đã khắc phục chỉ bằng đọc code.

Đoạn đề xuất: “The repository implements the MobileNet/MRF decoder and an ONNX export path, and saved MRF checkpoints and historical ONNX/PTQ artifacts are available. However, the currently stored MRF FP32 graph has not been linked to the selected epoch1386 checkpoint by a matching weight record. A fresh export from the selected checkpoint with source and output hashes remains pending; historical results are not relabeled as results of that checkpoint.”

Nguồn audit đầy đủ: `round6_mrf_export_code.json`. Bằng chứng tensor được dùng lại: `round5_mrf_export_trace.json` và `round5_capstone_artifact_evidence.json`.
