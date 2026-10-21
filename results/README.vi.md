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

Việc phát hành công khai các bộ câu, bản ghi kết quả và audio sinh ra vẫn do tác giả
quyết định. Việc chúng có mặt trong repo không khẳng định các quyền đó.
