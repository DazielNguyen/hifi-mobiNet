# Tích hợp common97 vào PTQ và demo N100

Ngày 05/11/2026. Recipe: **`piper-common97-v1`**. Phạm vi triển khai là ba
model Piper_no_VITS2_cpn (ResBlock2), Piper_ParIR_no_VITS2_cpn và
Piper_SeqIR_no_VITS2_cpn. Code chuẩn bị INT8 và demo đã chuyển sang common97;
thiết bị N100 cần nhận gói source mới và chạy bước chuẩn bị để dùng graph mới.

## 1. Lỗi cũ và thay đổi được chốt

[Code lịch sử](../../../results/edge-n100-openvino/2026-11-01/evidence/openvino_suite.py)
tạo danh sách Conv dự kiến được quantize nhưng chỉ truyền `ignored_scope`
cho NNCF. Các operation không có trọng số ngoài phạm vi dự kiến vẫn có thể
nhận activation quantizer. Bản cũ có **448 FakeQuantize**, gồm đường duration
và alignment, dù runtime vẫn chỉ đếm được **97 weighted kernel INT8**.

Trong phân tích ngày 04/11, `/CumSum/fq_output_0` kẹp thời lượng cộng dồn ở
khoảng 558–569 frame. Sau ngưỡng này alignment mất frame gán âm vị; decoder
vẫn tạo waveform nhưng phần cuối mất nội dung rõ chữ. Giữ decoder FP32
không sửa được latent đã mất điều kiện ngôn ngữ. Đây không phải giới hạn cố
định 8–10 giây của kiến trúc. Tích hợp này áp dụng phạm vi common97 đã thử
nghiệm và được người dùng nghe thử, không sửa bằng cắt hoặc làm im đuôi WAV.

| Thành phần | common97 | Cách ràng buộc |
|---|---|---|
| Text encoder `/enc_p/` | 37 Convolution INT8 | Đúng tên node trong policy, đúng loại Convolution |
| Stochastic duration predictor `/dp/` | 20 Convolution INT8 | Chỉ Conv thường, không GroupConvolution/depthwise |
| Main flow `/flow/` | 40 Convolution INT8 | Đúng tập tên node đã kiểm |
| Duration postprocessing, CumSum, mask, alignment | Giữ độ chính xác gốc, không PTQ | Loại mọi computation ngoài 97 Conv |
| Spline, phi tuyến, MatMul, decoder `/dec/` | Không PTQ; phép toán số thực giữ FP32 | Không có FQ ngoài cạnh activation của Conv được chọn |
| Random, scales, shape/index | Giữ nguyên semantics và kiểu dữ liệu | Graph triển khai vẫn 3 input, 1 output |

INT8 ở đây là **precision hỗn hợp**, không phải toàn bộ mạng dùng INT8.
96 activation FQ và 97 weighted kernel là hai đại lượng khác nhau; một
activation quantizer có thể phục vụ nhiều Conv. Kiểm tra mới xét từng cạnh
FQ và từng kernel, không dùng riêng số 97 để kết luận graph hợp lệ.

## 2. Các lớp kiểm tra trong code mới

1. [Policy](../../../configs/quantization/piper-common97.json) khóa SHA-256 của
   ba ONNX FP32, tên 97 node/model và phiên bản runtime. Model khác hoặc graph
   đã quantize không được tự dùng recipe này.
2. [common97.py](../../../src/hifimobinet/common97.py) tạo `ignored_scope`
   bằng **toàn bộ computation trừ các Conv đã chọn**. Constant/Parameter/Result
   không nằm trong danh sách operation cần loại. Đây là thay đổi trực tiếp
   ngăn quantizer lan vào duration/alignment.
3. Dùng đúng 60 dòng calibration thuộc train split; không lấy regression text
   hay Harvard test làm calibration. MIXED, CPU, fast bias correction;
   `compress_to_fp16=False` khi lưu IR.
4. [Numeric guard](../../../src/hifimobinet/ptq_numeric_guard.py) giữ nguyên
   phép tính/dtype của NumPy, bắt overflow/invalid/divide trong mean, variance,
   standard deviation và các RuntimeWarning số học tương ứng. Khi có lỗi,
   bước chuẩn bị dừng và lưu chẩn đoán. Hook chỉ tồn tại trong tiến trình
   chuẩn bị và được khôi phục sau lời gọi. Không sửa thư viện đã cài, không
   che warning hoặc tự ép dtype để báo pass; range FQ và waveform còn được
   kiểm finite riêng.
5. Đọc lại IR đã lưu. Mọi FQ phải có range hữu hạn hợp lệ, 256 mức và mọi
   consumer đều là input activation của Conv trong tập đã chọn. Đúng 96 FQ.
6. Clone graph để quan sát CumSum/alignment trên 6 text × length scale 1,0/1,3.
   Mọi frame phải được gán đúng một âm vị; waveform hữu hạn, không im lặng,
   số mẫu = số frame × 256; có mẫu dài hơn 10 giây. Graph dùng cho demo vẫn
   giữ nguyên I/O, không thêm debug outputs.
7. Kiểm đúng 97 runtime INT8 kernel, phủ đúng 37+20+40 node, không decoder.
   FP32 tiếp tục có 13 ca parity OpenVINO/ORT dùng cùng input nhiễu.
8. Chỉ khi cả ba model đạt, [migration](../../../scripts/n100_demo/prepare_common97.py)
   mới tạo catalog local mới. Thiếu model, sai hash, sai recipe/checkpoint,
   thiếu ca regression hoặc lỗi frontend đều không kích hoạt catalog.
9. Demo kiểm hash, policy, phiên bản guard/regression và audit graph/runtime
   trước khi trả audio INT8. JSON từng lần sinh ghi `quantization_recipe`.
   Các graph INT8 cũ không được nạp thay thế khi common97 chưa sẵn sàng.

Các file training, checkpoint, FP32 nguồn và evidence N100 01/11 giữ nguyên.
Sáu model vẫn có FP32 trong demo; chỉ ba Piper có INT8 common97. Ba model
Q05 khác và Piper Low/High chưa được gán recipe common97.

## 3. Kiểm thử thực tế sau tích hợp

Đã chạy **chính hàm `openvino_suite.prepare` của code mới** trên CPU PC/WSL,
dùng môi trường đã có. Wrapper chỉ cho phép host kiểm thử không phải N100;
không thay thuật toán PTQ, calibration, graph hoặc tiêu chí pass. Có ghi rõ
`validation_host_only=true`, `not_n100_benchmark=true` trong environment.
CPU ISA được giới hạn AVX2 trong lần kiểm này; điều đó không biến PC thành N100.

| Model | FP32 parity | FQ mới / kernel INT8 | Alignment | Độ dài WAV thực tế |
|---|---:|---:|---:|---:|
| Piper ResBlock2 | 13/13 | 96 / 97 | 12/12, 0 frame mất | 3,622–21,409 s |
| Piper ParIR | 13/13 | 96 / 97 | 12/12, 0 frame mất | 3,181–22,129 s |
| Piper SeqIR | 13/13 | 96 / 97 | 12/12, 0 frame mất | 3,541–20,584 s |

Các WAV dùng RNG native nên không phải phép so chất lượng FP32/INT8 với cùng
nhiễu. Kết quả này kiểm lỗi mất alignment ở đuôi; **không chứng nhận WER/MOS
trên tập độc lập, không phải số đo RTF N100**. Không gán lại số đo 01/11 cho
graph common97. N100 sẽ chạy lại các điều kiện chuẩn bị trước khi demo.

Audit trực tiếp ba IR cũ đều bị bộ kiểm mới từ chối vì FQ ngoài cạnh Conv
đã chọn; ba IR mới đều đạt. Dữ liệu này nằm trong
[legacy-and-new-audit.json](evidence/legacy-and-new-audit.json).

46 kiểm thử tự động đạt, bao gồm chặn FQ bị đổi tên trên đường duration,
kernel ngoài phạm vi, nguồn sai hash, numeric guard, catalog không được kích
hoạt từng phần, WAV/API/demo và protocol benchmark. Lệnh kiểm:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_common97.py tests/test_common97_activation.py tests/test_n100_demo.py tests/test_edge_openvino.py -q -p no:cacheprovider
```

Evidence mới: [integration-summary.json](evidence/integration-summary.json),
preparation/parity/runtime/alignment JSON của từng model, log đầy đủ
[prepare.stdout.txt](evidence/prepare.stdout.txt), wrapper kiểm trên PC và
[SHA256SUMS](evidence/SHA256SUMS). Hash nguồn ONNX, IR đã sinh và mọi WAV
alignment đã được tính lại và đối chiếu. Model weights/WAV nằm local tại
`.local/common97-integration-20261105/work/`, không đưa vào Git hoặc gói source.

## 4. Đưa sang N100 đang thực nghiệm

Gói mới: `hifi-mobiNet-n100-common97-2026-11-05.zip` cùng `.zip.sha256`.
Gói chỉ chứa code/config/frontend; dùng lại bộ FP32 và môi trường
`~/venvs/openvino-n100` trên N100, không tạo venv mới.

Làm theo [LENH-COMMON97-N100.txt](../../../demo/LENH-COMMON97-N100.txt):

1. Copy ZIP và checksum, kiểm SHA-256 rồi giải nén vào `~/benchmarks/` thành
   thư mục mới; không trộn với source demo cũ.
2. Dừng server cũ bằng Ctrl+C. Chạy `04-prepare-common97.sh` để tạo IR,
   WAV regression, audit và catalog mới.
3. Chạy `02-check.sh`, sau đó `03-start.sh`; PC mở địa chỉ N100 được script in.
4. Chọn một trong ba Piper và INT8 common97, thử câu dài trong file lệnh,
   nghe đến cuối, tải WAV/JSON để lưu cùng phiên thực nghiệm.
5. Khi cần số đo chính thức, dừng server rồi chạy `05-benchmark-common97.sh`.
   Phép đo mới gồm 3 Piper × 2 precision × 3 vòng × 720 câu, 4 threads.

Nếu bước chuẩn bị lỗi, log và work mới được giữ; không sửa hoặc ghi đè graph
cũ. Mỗi lần sinh common97 dùng work mới. Catalog chỉ được tạo sau khi mọi
điều kiện đạt. Không có kết nối SSH hoặc triển khai N100 tự động trong lần
tích hợp này; trạng thái triển khai thực tế phụ thuộc các bước chạy tại N100.
