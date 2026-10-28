# Kế hoạch: đợt ONNX/PTQ thứ 2 (gộp weight norm của flow)

*English: [plan-onnx-ptq-v2.md](plan-onnx-ptq-v2.md)*

Trạng thái: **đã chạy ngày 2026-10-26** (03:06–07:09 +07, commit `f495672`); mọi tiêu chí chấp nhận ở mục 5 đều đạt. Kết quả: `results/piper-ir-onnx-ptq-v2/`. Phần plan bên dưới được giữ nguyên như khi viết ngày 2026-10-22.

## 1. Vì sao cần đợt thứ hai

Đợt 1 (2026-10-21, `results/piper-ir-onnx-ptq/`) chưa gộp weight norm của flow khi export, làm
hỏng công thức đã khai báo cho phần lớn flow (protocol `docs/evaluation-onnx-ptq.md`, mục 7):
- 32 trên 40 convolution của flow bị lượng tử hoá trọng số lúc chạy, theo từng tensor, dạng UINT8, thay vì INT8 per-channel tĩnh như đã khai báo.
- Graph INT8 vẫn giữ 27,8 MB trọng số flow dạng FP32. Graph nặng 41–43 MB, trong khi graph INT8 lịch sử chỉ 22–24 MB.
- RTF của INT8 có tính cả phần tính weight norm và lượng tử hoá ở mỗi lần chạy.

Đợt 2 lặp lại toàn bộ đợt chạy, chỉ sửa đúng lỗi đó. Mọi thứ khác giữ nguyên protocol của đợt 1: model, checkpoint, công thức, scope, tập calibration, cách đánh giá chất lượng bằng matched-noise, scorer và thiết kế đo thời gian.

## 2. Những gì đã chuẩn bị sẵn

| Thay đổi | Ở đâu | Kiểm tra |
|---|---|---|
| Weight norm của flow được gộp trước khi export (`fold_flow_weight_norm`: gọi `WN.remove_weight_norm()` trên từng lớp coupling) | `scripts/evaluation/onnx/common.py`, `export.py` (mặc định; `--keep-flow-weight-norm` để tái tạo đợt 1) | cổng mới `flow_weight_norm_fold_changes_only_rounding`: output trước và sau khi gộp nằm trong NRMSE ≤ 1e-5 và sai số tuyệt đối lớn nhất ≤ 1e-4 |
| Không được còn op weight-norm nào trong scope PTQ | `export.py` | cổng mới `no_weight_norm_ops_in_ptq_scope` (không còn `ReduceL2` dưới enc_p, dp hay flow) |
| Mọi QLinearConv phải có trọng số INT8 per-channel hằng số | `common.py: qlinearconv_weight_kinds`, `quantize.py` | cổng mới `every_qlinearconv_weight_constant_per_channel_int8` |
| Pipeline bắt buộc khai báo thư mục output mới | `scripts/evaluation/onnx/run_round.sh` (bắt buộc có `OUT`) | `OUT` đã tồn tại sẽ chạy tiếp đợt đó |
| Unit test | `tests/test_onnx_ptq.py` | gộp weight norm chỉ đổi sai số làm tròn trên một model EdgeTTS nhỏ; phép kiểm loại trọng số phát hiện được trọng số tính lúc chạy |
| Script khởi chạy cục bộ chứa đường dẫn máy, nằm ngoài OneDrive và Git | `<WSL_PROJECT_HOME>/hifimobinet-private/launch_onnx_ptq_v2.sh` | từ chối chạy nếu session hoặc thư mục output đã tồn tại, nếu đang có training, hoặc nếu script chưa được commit |

## 3. Trước khi chạy

1. Commit các thay đổi đã chuẩn bị. Script khởi chạy từ chối chạy khi `scripts/evaluation`, `tests` hoặc protocol còn thay đổi chưa commit, để các bản ghi ghi đúng một commit sạch.
2. Bảo đảm máy rảnh:
   - không có training nào đang chạy;
   - không mở ứng dụng nặng trên Windows;
   - chế độ ngủ của Windows đặt là never;
   - có tiến trình giữ WSL chạy (keep-alive).

   WSL đã tự khởi động lại ngày 2026-10-22 và mất mọi session tmux. Nếu không có keep-alive, WSL tắt giữa chừng sẽ làm hỏng đợt chạy.
3. Kiểm tra dung lượng đĩa: đợt 1 dùng 1,2 GB trong `training_output/onnx-ptq/`.

## 4. Chạy

Trong WSL:

```sh
bash <WSL_PROJECT_HOME>/hifimobinet-private/launch_onnx_ptq_v2.sh
```

Script khởi chạy mở session tmux `onnx-ptq-v2`, chạy `scripts/evaluation/onnx/run_round.sh` với:
- `OUT=training_output/onnx-ptq-v2` trong clone training;
- `RESULTS=<repository>/results/piper-ir-onnx-ptq-v2`.

Thời gian dự kiến, theo đợt 1: khoảng 5 giờ 15 phút.

| Stage | Thời gian ở đợt 1 |
|---|---|
| export và PTQ, 3 model | khoảng 4 phút |
| synthesis chất lượng, 6 graph | khoảng 16 phút |
| phiên đo thời gian (máy rảnh) | khoảng 30 phút |
| chấm điểm, 6 bộ WAV | khoảng 4 giờ 20 phút |
| phân tích và sao chép | khoảng 1 phút |

**Theo dõi đợt chạy:**
- `tail -f <clone training>/training_output/onnx-ptq-v2/round.log` cho biết tiến độ.
- Một popup Windows (`msg.exe`) báo DONE hoặc FAILED.
- Nếu bị gián đoạn, chạy lại `run_round.sh` với cùng `OUT`: nó sẽ tiếp tục sau stage cuối cùng đã xong. Đừng dùng script khởi chạy cho việc này, vì nó từ chối thư mục đã tồn tại.

## 5. Tiêu chí chấp nhận

Đợt chạy chỉ được tính khi **mọi** cổng đều đạt. Các cổng của đợt 1 nằm trong protocol, mục 2–3; ba cổng mới nằm ở mục 2 phía trên. Ngoài ra, kiểm tra thêm các mục sau trước khi viết bất kỳ kết quả nào:

| Kiểm tra | Mong đợi |
|---|---|
| `quantize_record.json → int8_structure.qlinearconv_weight_kinds` | chỉ có `constant int8 per-channel`, 109 mỗi model (enc_p 37, dp 32, flow 40) |
| `export_record.json → weight_norm_ops_in_graph` | không có mục nào cho enc_p, dp hay flow |
| Parity FP32 trên Harvard-720 | 720 / 720 mỗi model, như đợt 1 |
| WAV PyTorch tham chiếu | 720 / 720 trùng với lần đánh giá đã lưu |
| Kích thước file INT8 | dự kiến khoảng 20–25 MB (27,8 MB trọng số flow FP32 chuyển thành INT8). Đây là kỳ vọng, không phải cổng: báo đúng giá trị đo được, dù là bao nhiêu |

## 6. Sau khi chạy xong

1. **Đọc số liệu, không giả định.** So đợt 2 với đợt 1 cho từng model:
   - kích thước INT8;
   - RTF INT8 ở 2 và 12 thread;
   - ΔUTMOSv2 và ΔWER của INT8 so với FP32.
   - Đặc biệt kiểm tra xem mức giảm UTMOSv2 của SeqIR (−0,036 ở đợt 1) có còn không.
2. **Viết `results/piper-ir-onnx-ptq-v2/README.md` và `README.vi.md`** theo cấu trúc README của đợt 1:
   - mục 1 nói rõ điều đã thay đổi (chỉ có việc gộp flow);
   - thêm một bảng ngắn so sánh đợt 1 với đợt 2.
3. **Thêm các mục vào `results/manifest.json`** cho các bản ghi mới.
4. **Cập nhật `results/README.md` và `README.vi.md`:**
   - "Export ONNX và PTQ INT8": đợt 2 thành kết quả INT8 chính. Đợt 1 vẫn giữ, ghi là đã bị thay thế, kèm ghi chú lỗi.
   - "So sánh bảy model": các dòng triển khai và dòng kích thước file ONNX INT8.
   - "Kích thước model": cột ONNX INT8.
5. **Cập nhật các tài liệu khác:**
   - `docs/q2-readiness-seq-par.md` và `.vi.md`: dòng nhận định về PTQ ở mục 3 và ý về câu chuyện paper ở mục 5.
   - `docs/evaluation-onnx-ptq.md` mục 7: ghi lại rằng đợt 2 đã áp dụng cách sửa.
6. **Chỉ profile lại khi tốc độ INT8 thay đổi đáng kể.** Chạy `profile_graphs.py` trên các graph của đợt 2; decoder không đổi, nên phát hiện về `Sin` vẫn phải đứng vững.
7. **Commit** theo từng phạm vi riêng (results; docs), không có attribution. Tác giả tự push.

## 7. Prompt cho phiên sẽ chạy đợt này

> Chạy đợt ONNX/PTQ thứ 2 theo kế hoạch trong `hifi-mobiNet/docs/plan-onnx-ptq-v2.md`.
> - Trước tiên kiểm tra mục 3 (script đã commit, máy rảnh, WSL keep-alive). Sau đó khởi chạy `<WSL_PROJECT_HOME>/hifimobinet-private/launch_onnx_ptq_v2.sh` và theo dõi `round.log`.
> - Không thay đổi công thức, scope, calibration hay scorer.
> - Nếu một cổng thất bại thì dừng và báo lại; không được nới lỏng cổng.
> - Khi log ghi ALL DONE, kiểm tra tiêu chí chấp nhận ở mục 5, rồi làm các bước ở mục 6.
> - Chỉ commit khi tôi yêu cầu, không có attribution. Trao đổi bằng tiếng Việt; tài liệu viết bằng tiếng Anh, kèm README.vi.md tiếng Việt.
