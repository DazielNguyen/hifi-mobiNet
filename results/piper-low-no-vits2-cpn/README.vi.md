# Piper Low — kết quả Harvard-720

*English: [README.md](README.md)*

**Đã hoàn tất đánh giá PyTorch FP32 và chấm điểm ONNX FP32 trên 720 câu.**
Phiên kết thúc lúc 00:51 ngày 03/11/2026 (UTC+7). ONNX còn một câu không đạt
ngưỡng parity số học, nên điểm và timing ONNX bên dưới chỉ là kết quả chẩn đoán.
Trạng thái đầy đủ: `completed_with_onnx_parity_failure` trong
[status.json](fp32-evaluation/status.json); bản tóm tắt máy tạo:
[RESULTS.txt](fp32-evaluation/RESULTS.txt).

## Chất lượng

| Model / runtime | Sample rate | Số câu | UTMOSv2 dự đoán ↑ | Macro WER ↓ | Corpus WER ↓ | Macro CER ↓ | Corpus CER ↓ | Câu WER = 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Low / PyTorch FP32** | 16.000 Hz | 720 | **3,5509** | **7,2147%** | 7,0185% | **3,2423%** | 3,1857% | 467 |
| Low / ONNX FP32 — chẩn đoán | 16.000 Hz | 720 | 3,5535 | 7,2147% | 7,0185% | 3,2456% | 3,1893% | 467 |
| Medium / PyTorch FP32 — control cùng phiên | 22.050 Hz | 720 | 3,5102 | 6,1194% | 5,9561% | 2,7518% | 2,6850% | 502 |

Low có điểm UTMOSv2 nhỉnh hơn Medium **0,0408**, nhưng macro WER cao hơn
**1,0954 điểm phần trăm** và macro CER cao hơn **0,4905 điểm phần trăm**.
Vì vậy chưa thể kết luận Low tốt hơn về chất lượng tổng thể. UTMOSv2 là điểm
dự đoán; WER/CER dùng ASR và không thay thế việc nghe hoặc MOS do người chấm.

So sánh ghép cặp trên cùng 720 câu: khoảng bootstrap 95% của chênh lệch
UTMOSv2 là [0,0181; 0,0638], của macro WER là [+0,1546; +2,0691] điểm phần trăm.
Các khoảng này mô tả biến thiên giữa câu, không phải giữa các lần train.
Chi tiết và giới hạn phép kiểm: [low-vs-medium.json](fp32-evaluation/low-vs-medium.json).

Nguồn từng câu và scorer: [Low PyTorch](fp32-evaluation/pytorch/results.json),
[Low ONNX](fp32-evaluation/onnx-fp32/results.json),
[Medium control](fp32-evaluation/medium-control/results.json).
Mỗi thư mục có `score_record.json`, `score_progress.jsonl`, `synth.json` và
`validation.json`; log chấm điểm được giữ trong [logs/](fp32-evaluation/logs/).

## Tốc độ trên PC / WSL

Phần cứng: **Intel Core i7-14700, WSL2, 24 logical CPU** theo
[host.json](fp32-evaluation/host.json). Đây chưa phải phép đo Low trên N100.
RTF = thời gian inference / thời lượng audio; nhỏ hơn là nhanh hơn.

PyTorch CPU, 12 luồng, phiên sinh audio chất lượng: Low **0,03291**;
Medium **0,04138**. Đây là một lượt sinh mỗi model.

ONNX Runtime CPU, 720 câu/lượt, intra-op 2 hoặc 12 luồng, inter-op 1:

| Luồng | Lượt / thứ tự | Low FP32 — chẩn đoán | Medium FP32 |
|---|---|---:|---:|
| 2 | A: Low → Medium | 0,02450 | 0,03376 |
| 2 | B: Medium → Low | 0,02555 | 0,03357 |
| 12 | A: Low → Medium | 0,01988 | 0,02590 |
| 12 | B: Medium → Low | 0,02024 | 0,02613 |

Các giá trị là trung bình RTF từng câu, không phải thời gian nạp model hoặc
latency toàn bộ ứng dụng. Giữ nguyên native RNG, seed 1234, không warmup;
đảo thứ tự A/B để quan sát độ trôi. Record và timing từng câu ở
[timing/](fp32-evaluation/timing/). Không dùng thời gian của vòng kiểm parity
matched-noise làm benchmark triển khai, hoặc suy ra tốc độ N100 từ bảng này.

## Checkpoint, protocol và kích thước

- Train đủ **1.500 epoch**, chọn minimum validation mel loss trong run:
  `best-epoch=1351-val_loss_mel=19.1691.ckpt` (epoch đánh số từ 0).
- SHA-256 checkpoint: `7d34322f637e03eb1186fe8dd5681e1c5c0d81666d21398467f00ca441103e55`.
- Full checkpoint **845.907.083 byte**, gồm trạng thái training; không phải
  kích thước model triển khai. Inference **15.663.771 tham số** theo định nghĩa
  loại posterior encoder và nhánh SDP chỉ dùng khi train; decoder 1.664.768.
  Định nghĩa, số đếm từng module: [parameters.json](fp32-evaluation/parameters.json).
- ONNX FP32: **63.469.360 byte**, opset 18;
  SHA-256 `1afe4889d36086c1e39f1f0b6b1d1a4bc0b077404c00b3b6444cfc2568205a94`.
  Xem [export_record.json](graphs/export_record.json).
- Low dùng ResBlock2 và cấu trúc tham số như Medium, giảm sample rate về 16 kHz;
  đây không phải cấu hình x-low thu nhỏ mạng. Không xếp hạng hai model bằng
  raw validation mel loss vì khác lưới tần số/thời gian.
- Cùng Harvard IDs, text/phoneme, seed 1234 mỗi câu, scales 0,667 / 1 / 0,8.
  Giữ WAV Low 16 kHz và Medium 22,05 kHz. Scorer nhận đúng sample rate;
  mọi resampling cần cho scorer được ghi trong `score_record.json`.
- UTMOSv2 fusion-stage-3 fold 0 và Whisper small dùng cache đã xác minh.
  WAV control Medium khớp byte với phiên lịch sử **720/720**; xem
  [medium-reproducibility.json](fp32-evaluation/medium-reproducibility.json).

## Giới hạn ONNX phải giữ khi trích dẫn

Export qua 13/13 ca kiểm ban đầu. Khi kiểm toàn bộ Harvard-720, output hữu hạn,
cùng độ dài và PyTorch reference tái tạo đúng 720/720, nhưng chỉ **719/720 đạt parity**.
Câu `idx=719`, “When you hear the bell, come quickly.” có NRMSE
**0,0010694584 > 0,001**; max absolute error 0,0010539144 vẫn dưới 0,005.

[Probe](fp32-evaluation/parity_probe.json) cho thấy riêng PyTorch 2 so với 12
luồng cũng có NRMSE 0,0014113898; fold weight norm không đổi output trong
phép kiểm câu này. Đây là bằng chứng độ nhạy số học, chưa phải kết luận mọi
nguyên nhân đã được giải quyết. Không nới ngưỡng hoặc đổi nhãn thành pass.
**`deployment_approved: false`** vẫn được giữ.

## Phiên và evidence

Lượt đầu ở thư mục này dừng tại parity. Phiên tiếp tục nằm trong
`fp32-evaluation/`, sử dụng lại WAV/graph sau kiểm hash, hoàn tất scoring,
timing và comparison với nhãn chẩn đoán. [SESSION-INDEX.txt](SESSION-INDEX.txt)
là ghi chú lúc phiên tiếp tục được thiết lập; trạng thái cuối xem README này
và `fp32-evaluation/status.json`. Không ghi đè record thất bại của lượt đầu.

Hai danh sách hash có phạm vi riêng:

- [SHA256SUMS](SHA256SUMS): 41 file của lượt đầu.
- [fp32-evaluation/SHA256SUMS](fp32-evaluation/SHA256SUMS): 77 file của phiên tiếp tục.

Đã đọc lại và kiểm **118/118 file khớp** khi tổng hợp ngày 03/11/2026.
README được thêm sau phép đo, không sửa các manifest/hash hay evidence gốc.
Kiểm lại trên Linux bằng `sha256sum -c SHA256SUMS` trong từng thư mục tương ứng.

Raw WAV/graph và checkpoint nằm trong training clone Ubuntu WSL:

```text
/home/capstone/hifimobinet-train/hifi-mobiNet/training_output/
  piper-low-no-vits2-cpn-2026-10-30/
  piper-low-eval-2026-11-02/
  piper-low-eval-2026-11-02-continued/
```

Chưa có Low INT8/PTQ, benchmark N100 hoặc human MOS trong phiên này. Chỉ có
một training seed; chưa xác lập biến thiên giữa các lần train. Các số đo trên
không thay thế hoặc trộn vào evidence N100 sáu model trước đó.
