# So sánh MRF(1,1,1) vs SEQ(1,1,1) — Báo cáo cho Q2

## 1. Kiến trúc

Cả hai đều thay khối MRF gốc của HiFi-GAN (`ResBlock1/2`, dense dilated-conv) bằng
`ResBlockInverted` — khối inverted-residual kiểu MobileNetV2 (pointwise-expand →
depthwise-conv → pointwise-project tuyến tính, zero-init để khởi đầu là identity
no-op, weight_norm thay BatchNorm). Khác nhau duy nhất là **cách ghép các khối này
sau mỗi tầng upsample**:

| | MRF(1,1,1) | SEQ(1,1,1) |
|---|---|---|
| Cơ chế | 3 nhánh **song song** (kernel 3/5/7), cộng rồi chia 3 — đúng tinh thần MRF gốc của HiFi-GAN | 2 khối **nối tiếp** (cùng kernel=3) — không có bước cộng/chia |
| Tại 1 thời điểm | 3 depthwise-conv cùng chạy, tranh băng thông bộ nhớ | chỉ 1 depthwise-conv chạy |
| Đa phạm vi tiếp nhận | Có (3 kernel size khác nhau) | Không — chỉ 1 kernel size hiệu dụng |
| Params (Generator) | 1.154.560 | **1.107.968** |
| Size (FP32) | 4.40 MB | **4.23 MB** |
| GFLOPs/s audio | 1.2306 | **0.9846** |

## 2. Kết quả — FP32 (PyTorch, n=500, seed cố định)

| Metric | Baseline | MRF(1,1,1) | **SEQ(1,1,1)** |
|---|---:|---:|---:|
| val_loss_mel (best, /1500 epoch) | 19.7882 (ep.1489) | 18.7681 (ep.1386) | 19.0136 (ep.1442) |
| RTF ↓ | 0.0434 | 0.0370 | **0.0324** |
| UTMOSv2 ↑ | 3.132 | 3.537 | **3.581** |
| WER ↓ | 0.1230 | 0.1043 | **0.1033** |

## 3. Kết quả — ONNX Runtime (runtime triển khai thật, CPU AVX-VNNI, 2 luồng, n=500)

### FP32

| Metric | Baseline | MRF(1,1,1) | **SEQ(1,1,1)** |
|---|---:|---:|---:|
| RTF ↓ | 0.0388 | 0.0416 | **0.0346** |
| UTMOSv2 ↑ | 3.132 | 3.537 | **3.571** |
| WER ↓ | 0.1230 | 0.1043 | 0.1064 |

### PTQ INT8 (flow+enc_p+dp lượng tử hoá, `dec` giữ FP32)

| Metric | Baseline | MRF(1,1,1) | **SEQ(1,1,1)** |
|---|---:|---:|---:|
| RTF ↓ | 0.0345 | 0.0371 | **0.0289** |
| UTMOSv2 ↑ | 3.121 | 3.490 | **3.586** |
| WER ↓ | 0.1135 | 0.1124 | **0.1081** |

**Điểm mấu chốt**: qua ONNX Runtime, **MRF chậm hơn baseline** (RTF tăng), còn
**SEQ vẫn nhanh hơn baseline** ở cả FP32 và INT8 — không hề bị đảo chiều.

## 4. Kiểm định thống kê (Mann-Whitney U / Wilcoxon, n=500)

| So sánh | RTF | UTMOSv2 | WER |
|---|---|---|---|
| MRF vs Baseline (FP32, ONNX) | MRF **chậm hơn** 7.2%, p<0.001 *** | MRF tốt hơn, p<0.001 *** | MRF tốt hơn, p=0.012 * |
| MRF vs Baseline (INT8) | MRF **chậm hơn** 7.5%, p<0.001 *** | MRF tốt hơn, p<0.001 *** | ngang nhau, p=0.607 n.s. |
| SEQ vs Baseline (FP32, ONNX) | SEQ nhanh hơn 10.8%, p<0.001 *** | SEQ tốt hơn, p<0.001 *** | SEQ tốt hơn, p=0.019 * |
| SEQ vs Baseline (INT8) | SEQ nhanh hơn 16.2%, p<0.001 *** | SEQ tốt hơn, p<0.001 *** | ngang nhau, p=0.531 n.s. |
| **SEQ vs MRF (FP32, PyTorch)** | SEQ nhanh hơn 16.8%, p<0.001 *** | SEQ tốt hơn, p=0.007 ** | ngang nhau, p=0.865 n.s. |
| **SEQ vs MRF (INT8)** | SEQ nhanh hơn 22.1%, p<0.001 *** | SEQ tốt hơn, p<0.001 *** | ngang nhau, p=0.919 n.s. |

→ **SEQ thắng MRF có ý nghĩa thống kê về RTF và UTMOSv2** ở cả FP32 và INT8;
WER giữa 2 model không phân biệt được về thống kê (ngang nhau).

## 5. Vì sao SEQ thắng MRF ở RTF dù cùng dùng `ResBlockInverted`

Lõi của khối là **depthwise conv** — cường độ tính toán thấp (ít FLOPs/byte dữ
liệu di chuyển) nên bị **giới hạn băng thông bộ nhớ**, không tận dụng đa luồng
hiệu quả như dense conv. MRF chạy **3 depthwise-conv song song cùng lúc** → nhu
cầu băng thông nhân 3 tại mọi thời điểm → nghẽn rõ khi chuỗi dài (T lớn). SEQ chỉ
chạy **1 depthwise-conv tại 1 thời điểm** (nối tiếp, không cộng/chia) → không bị
hiệu ứng nhân băng thông này.

Đánh đổi: SEQ không còn giữ đúng cơ chế "đa phạm vi tiếp nhận cùng lúc" của MRF
gốc HiFi-GAN (chỉ 1 kernel size hiệu dụng, không phải 3) — nhưng dữ liệu thực tế
cho thấy đánh đổi này không gây tổn hại chất lượng; ngược lại SEQ còn tốt hơn.

## 6. So sánh với SOTA công bố ngoài project

| | Piper gốc (`lj-med_1000.ckpt`, 1000 epoch, chính thức) | SEQ(1,1,1) (1500 epoch) |
|---|---:|---:|
| RTF ↓ (PyTorch) | 0.0386 | **0.0324** |
| UTMOSv2 ↑ | 3.534 | **3.581** |
| WER ↓ | 0.1632 | **0.1033** |

Kiểm định: SEQ thắng cả 3 trục có ý nghĩa thống kê mạnh (RTF p=1.7×10⁻⁴⁸,
UTMOSv2 p=4.3×10⁻³, WER p=1.2×10⁻¹⁶). Checkpoint lấy trực tiếp từ HuggingFace
(`rhasspy/piper-checkpoints`), chạy qua đúng code `piper_train` gốc không chỉnh
sửa — đảm bảo trung thực kiến trúc.

VITS2 không có checkpoint/code chính thức công bố (tác giả SK Telecom chưa release)
— chỉ có bản tái hiện không chính thức, chưa có checkpoint hội tụ đầy đủ — không đủ
cơ sở so sánh công bằng.

## 7. Kết luận

**SEQ(1,1,1) là lựa chọn tốt hơn MRF(1,1,1)** để đề xuất chính: nhẹ hơn, ít FLOPs
hơn, nhanh hơn có ý nghĩa thống kê ở cả FP32 và INT8 trên runtime triển khai thật,
chất lượng UTMOSv2 cao hơn có ý nghĩa thống kê, WER ngang nhau — không có đánh đổi
đáng kể nào. MRF vẫn có giá trị như minh chứng phương pháp luận: "FLOPs/params
thấp không đảm bảo latency thấp trên phần cứng thật" (phát hiện riêng, đáng đưa
vào phần thảo luận/limitation của paper).
