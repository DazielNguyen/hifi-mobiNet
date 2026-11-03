# Kiểm kê PTQ của sáu model và cấu hình chung đề xuất — 02-11-2026

Đã đọc trực tiếp **6 ONNX FP32**, XML của các IR tương ứng và execution metadata
N100 được lưu từ phiên 01-11-2026. Tổng cộng **42.211 ONNX nodes**, trong đó
**1.086 Conv / ConvTranspose / MatMul** được liệt kê từng lớp. SHA-256 của nguồn
được đối chiếu với manifest/catalog và đọc lại sau kiểm tra; sáu graph đều qua
`onnx.checker`. Đây là **kiểm tĩnh mới**, không phải một đợt PTQ, inference,
scoring hoặc benchmark mới.

**Lựa chọn chung để thử đầu tiên: 40 Conv thuộc phần coupling/WaveNet của flow;
giữ encoder, toàn bộ duration predictor và decoder ở FP32.** Các lớp này có cùng
tên và kích thước trọng số trên cả sáu model. Cấu hình này chưa được lượng tử hóa
hoặc đo chất lượng/tốc độ. Chưa có căn cứ để bảo đảm nó hết lỗi cắt chữ hoặc nhanh
hơn FP32. Không thay model đang chạy trong demo bằng cấu hình chưa được kiểm chứng.

## File để xem và tái kiểm

| File | Nội dung |
|---|---|
| [Cấu hình chung](../../../configs/quantization/n100-common-flow-conv-int8.json) | Phạm vi 40 lớp, recipe, điều kiện nghiệm thu; trạng thái candidate |
| [inventory.json](evidence/inventory.json) | Hash nguồn, checkpoint, tổng số phép tính, tên và shape của 40 lớp ở từng model |
| [weighted-operations.csv](evidence/weighted-operations.csv) | Đủ 1.086 phép Conv/ConvTranspose/MatMul; module, group, shape, phạm vi cũ/mới |
| [operation-counts.csv](evidence/operation-counts.csv) | Tất cả loại operator của cả sáu graph, kể cả phép phi tuyến/shape/index |
| [previous-runtime-weighted-operations.csv](evidence/previous-runtime-weighted-operations.csv) | Precision ghi nhận trên N100 của các weighted operators trong artifact **cũ** |
| [SHA256SUMS](evidence/SHA256SUMS) | Hash bốn file kiểm kê sinh tự động |
| [Script kiểm tĩnh](../../../scripts/evaluation/edge/audit_quantization_scope.py) | Có thể chạy lại vào thư mục mới; không gọi runtime model |

Các hash/script này kiểm chứng cấu trúc và nguồn. Chúng không thay thế kiểm chất
lượng của một artifact INT8 mới.

## 1. Sáu model đúng phạm vi

| Model ID | Selected epoch | Decoder | Flow trong ONNX |
|---|---:|---|---|
| `baseline-resblock2` | 1489 | ResBlock2, LeakyReLU | Coupling/WaveNet + attention |
| `parallel-ir` | 1386 | Parallel IR, SnakeBeta | Coupling/WaveNet + attention |
| `sequential-ir` | 1442 | Sequential IR, SnakeBeta | Coupling/WaveNet + attention |
| `piper-no-vits2-cpn` | 1403 | ResBlock2, LeakyReLU | Coupling/WaveNet |
| `piper-parir-no-vits2-cpn` | 1486 | Parallel IR, SnakeBeta | Coupling/WaveNet |
| `piper-seqir-no-vits2-cpn` | 1455 | Sequential IR, SnakeBeta | Coupling/WaveNet |

Piper external, Low và High không thuộc sáu model này. Trạng thái training hiện
tại của Low/High không được kiểm hoặc thay đổi trong lượt này. Các cấu hình
checkpoint lịch sử lấy từ metadata đã lưu, không suy ngược từ file recipe của
một lần train mới. Danh tính đầy đủ ở [manifest gốc](../../../results/edge-n100-openvino/2026-11-01/evidence/models.json).

## 2. “Có thể quantize” có ba mức khác nhau

1. **Ứng viên ở mức toán tử:** có phép Conv/MatMul hoặc tensor trọng số phù hợp
   để thử PTQ. Không đồng nghĩa backend sẽ chạy kernel INT8.
2. **Đã có execution metadata INT8:** phiên N100 trước ghi nhận kernel INT8
   cho đúng phép tính, trong đúng artifact trước. Thay scope vẫn phải kiểm lại.
3. **Đạt chất lượng và tốc độ:** cần waveform, WER/nghe và timing của artifact
   mới. Chưa có mức chứng nhận này cho cấu hình đề xuất.

OpenVINO phân biệt graph có lượng tử hóa với cách weighted operations thực thi
ở CPU; cần xem execution graph. Tham khảo [IR INT8](https://docs.openvino.ai/2024/documentation/openvino-ir-format/intermediate-representation-int8-inference.html).

## 3. Kiểm kê các thành phần có thể cân nhắc PTQ

Số trong bảng là số operator của **mỗi model** trong nhóm, không phải số module
Python và không phải phần trăm thời gian chạy.

| Thành phần | Ba model lịch sử | Ba model Config A | Đánh giá khả năng và lựa chọn lần này |
|---|---:|---:|---|
| Encoder Q/K/V/O projections | 24 Conv | 24 Conv | Đã ghi nhận INT8; giữ FP32 để bảo vệ đầu vào duration predictor |
| Encoder feed-forward | 12 Conv | 12 Conv | Đã ghi nhận INT8; giữ FP32 cùng encoder |
| Encoder projection ra mean/log-scale | 1 Conv | 1 Conv | Đã ghi nhận INT8; giữ FP32 để hạn chế đổi phân bố latent |
| Encoder attention products, kể cả relative attention | 24 MatMul | 24 MatMul | Có thể thử activation quantization tùy graph/backend; chưa có bằng chứng INT8 cho các phép này trong phiên N100; giữ FP32 |
| Phoneme embedding và relative-position embeddings | Có | Có | Có thể nghiên cứu nén trọng số/lookup riêng; không đồng nghĩa có kernel INT8 nhanh hơn. Không chọn ở vòng đầu |
| Duration predictor: pre/proj và pointwise Conv của DDS/ConvFlow | 20 Conv | 20 Conv | Đã ghi nhận INT8; loại khỏi cấu hình mới vì tham gia dự đoán thời lượng |
| Duration predictor: depthwise DDS Conv | 12 Conv có group | 12 Conv có group | Có trong allowlist trước nhưng execution metadata ghi **f32 cả 12**; không gọi là đã chạy INT8. Giữ FP32 |
| Flow coupling: `pre`, `post` | 8 Conv | 8 Conv | Đã ghi nhận INT8; chọn |
| Flow coupling: WaveNet `in_layers`, `res_skip_layers` | 32 Conv | 32 Conv | Đã ghi nhận INT8; chọn |
| Attention projections + FFN bên trong flow | 24 Conv | Không có | Đã ghi nhận INT8 trên nhóm lịch sử; giữ FP32 để không đưa thêm thành phần chỉ có ở một nhóm vào recipe chung |
| Attention products bên trong flow | 16 MatMul | Không có | Ứng viên thử nghiệm riêng; không thuộc recipe chung |
| Decoder `conv_pre`, `conv_post` | 2 Conv | 2 Conv | Có trọng số FP32 cố định, ứng viên PTQ; chưa thử INT8 trong phiên N100. Giữ FP32 |
| Decoder ResBlock2 | 18 Conv ở baseline | 18 Conv ở Piper | Có thể thử riêng từng stage; chưa có bằng chứng chất lượng INT8 decoder. Giữ FP32 |
| Decoder Parallel IR | 18 pointwise + 9 depthwise | 18 pointwise + 9 depthwise | Có thể thử PTQ pointwise/depthwise riêng; lợi ích phụ thuộc kernel và overhead. Giữ FP32 |
| Decoder Sequential IR | 12 pointwise + 6 depthwise | 12 pointwise + 6 depthwise | Tương tự Parallel IR; giữ FP32 |
| Decoder upsampling | 3 ConvTranspose | 3 ConvTranspose | Ứng viên cần kiểm hỗ trợ chuyển đổi và kernel INT8 cho đúng ConvTranspose1D/dynamic shape. Không khẳng định đã hỗ trợ/chạy nhanh từ việc import graph; giữ FP32 |
| Mở rộng mean/log-scale theo alignment | 2 MatMul | 2 MatMul | Không phải lớp Linear với trọng số học cố định; liên quan alignment/latent. Không chọn |

**Các phần không phải mục tiêu PTQ trọng số của recipe này:**

- LayerNorm/MVN, Softmax, sigmoid/tanh, GELU, SnakeBeta (`Sin`, `Pow`, `Mul`,
  `Add`), spline (`search`, `gather`, chia, căn, log...), `exp/ceil/clamp` tính
  thời lượng: giữ precision thích hợp, không ép toàn bộ thành INT8. Một số phép
  có thể được backend fuse; việc fusion không chứng minh từng phép có kernel INT8
  độc lập. Giữ một activation FP32 cũng không loại sai số từ Conv đứng trước nó.
- Mask, chỉ số phoneme, shape, `Reshape`, `Transpose`, `Gather`, `Slice`,
  `Concat`, `Range`, `NonZero`, so sánh, các phép trên chỉ số: giữ dtype/ngữ nghĩa
  vốn có. “Giữ FP32” không có nghĩa đổi `int64` IDs hoặc shape thành float.
- RNG và các scale điều khiển inference: giữ nguyên semantics để đối chiếu được.
- Phonemizer/eSpeak, text normalization, ghi WAV và trình phát: không phải
  các lớp neural network để áp dụng NNCF INT8.
- Posterior encoder `enc_q`, nhánh posterior của SDP dùng lúc train,
  MPD/MRD/discriminator khác, MAS phục vụ train, loss và optimizer: không nằm
  trong đường inference của sáu ONNX triển khai; lượng tử hóa chúng không tăng
  tốc demo N100. F0 predictor/speaker conditioning không có trong sáu graph này.

Mọi Conv/ConvTranspose đã kiểm đều lấy trọng số từ initializer FP32 trực tiếp.
Không thấy `ReduceL2` hoặc Q/DQ/QLinearConv trong sáu graph nguồn FP32. Không
load checkpoint, không fold hay sửa lại graph trong quá trình audit.

## 4. Số lượng đối chiếu giữa sáu graph

| Model | Encoder Conv | SDP Conv, gồm depthwise | Flow Conv | Decoder Conv, gồm depthwise | Decoder ConvTranspose | Tổng MatMul |
|---|---:|---:|---:|---:|---:|---:|
| baseline-resblock2 | 37 | 32 | 64 | 20 | 3 | 42 |
| parallel-ir | 37 | 32 | 64 | 29 | 3 | 42 |
| sequential-ir | 37 | 32 | 64 | 20 | 3 | 42 |
| piper-no-vits2-cpn | 37 | 32 | 40 | 20 | 3 | 26 |
| piper-parir-no-vits2-cpn | 37 | 32 | 40 | 29 | 3 | 26 |
| piper-seqir-no-vits2-cpn | 37 | 32 | 40 | 20 | 3 | 26 |

Recipe trước cho phép 133 Conv ở mỗi model lịch sử và 109 ở mỗi model Config A.
Execution metadata chỉ ghi 121 và 97 weighted kernels INT8 tương ứng:
**37 encoder + 20 SDP + 64/40 flow**. Chênh lệch 12 là các depthwise SDP được
ghi f32. Đây là lý do phải kiểm runtime, không lấy số trong allowlist làm số
kernel INT8 thực chạy.

Một chi tiết khi chọn scope: OpenVINO đổi tên ConvTranspose thành
`ConvolutionBackpropData_<số>`; tên `/dec/ups.N/ConvTranspose` có thể thuộc phép
Add của bias. Vì vậy chỉ exclude tên chứa `/dec/` là chưa đủ cho một recipe
mở rộng. Recipe mới cần allowlist chặt, loại mọi weighted op không được chọn
và kiểm precision của cả Deconvolution.

## 5. Cấu hình chung được chọn để thử

ID: **`n100-common-flow-conv-int8-v1`**. File JSON là protocol khai báo; suite cũ
chưa có tham số đọc file này. Không đưa JSON vào suite cũ rồi kỳ vọng nó tự đổi
scope. Audit hiện tại chỉ phân giải và kiểm danh sách.

Mỗi coupling block có 10 Conv, với `i ∈ {0, 2, 4, 6}`, `j ∈ {0, 1, 2, 3}`:

```text
/flow/flows.i/pre/Conv                 1 mỗi block
/flow/flows.i/enc/in_layers.j/Conv      4 mỗi block
/flow/flows.i/enc/res_skip_layers.j/Conv 4 mỗi block
/flow/flows.i/post/Conv                1 mỗi block
                                      4 × 10 = 40 Conv/model
```

Danh sách tên và shape cụ thể nằm trong `inventory.json` dưới
`models[].candidate.nodes`. Có **240/240 ánh xạ một-một** từ các lớp được chọn
sang kernel INT8 trong execution metadata cũ của sáu model. Điều này cho thấy
chúng đã chạy INT8 trong scope cũ, không chứng nhận chất lượng của scope mới.

Giữ nguyên checkpoint và ONNX FP32 cha. Lần thử đầu giữ calibration 60 câu
train-only đã xác minh, `MIXED`, `target_device=CPU`, `fast_bias_correction=true`
và scales `[0.667, 1.0, 0.8]` để thay một yếu tố chính là scope. `MIXED` nghĩa là
weights symmetric / activations asymmetric; không có nghĩa tự chọn chỗ FP32.
Sau đó mới cân nhắc tăng calibration hoặc đổi bias correction trong đợt riêng.
Tham khảo [NNCF basic PTQ](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html).

Pin môi trường theo phiên trước: OpenVINO 2026.4.1, NNCF 3.4.0, NumPy 2.4.6,
ONNX 1.23.2, ONNX Runtime 1.31.1. Python đọc ONNX để audit trên PC/WSL có thể
khác, được ghi riêng trong inventory; nó không thực thi model hoặc tạo artifact
OpenVINO mới. Khi lưu IR mới phải `compress_to_fp16=False`. Weighted operations
ngoài allowlist phải giữ f32; không chỉ dựa vào tên artifact `int8.xml`.

**Vì sao không chọn ngay encoder + flow?** Encoder tạo cả đầu vào SDP và các
tham số prior. Chỉ bỏ quantization trong `/dp/` vẫn chưa bảo vệ đường duration
khi encoder bị thay đổi. Với phản ánh audio cắt chữ, giữ cả encoder và SDP là
biến thể chẩn đoán ít tác động lên duration hơn. Dù vậy flow INT8 vẫn có thể làm
âm thanh mất chi tiết/phát âm kém; không thể hứa hết lỗi. FP32 gốc cũng cần nghe
đúng câu đó để loại lỗi vốn có của model/frontend.

**Vì sao không chọn toàn bộ `/flow/`?** Nhóm lịch sử có thêm 24 Conv và 16 MatMul
attention. Chọn 40 Conv chung làm cho phép thay precision tác động cùng tập
vai trò/tên/shape. Cách này vẫn không biến sáu model khác kiến trúc/training
thành một thí nghiệm chỉ khác decoder.

## 6. Lợi ích có thể kỳ vọng và điều chưa biết

40 Conv chứa **7.077.888 weight elements/model**: 27 MiB ở FP32, tương đương
6,75 MiB nếu mỗi weight được lưu bằng một byte. Chênh 20,25 MiB chỉ là phép tính
lý thuyết trên các tensor này, chưa tính scale, zero-point, metadata, fusion hay
bản sao FP32 mà backend có thể giữ. Không phải dung lượng artifact đo được.

| Model | Tỷ lệ weight elements được chọn / toàn bộ FP32 initializers |
|---|---:|
| baseline-resblock2 | 42,02% |
| parallel-ir | 43,33% |
| sequential-ir | 43,45% |
| piper-no-vits2-cpn | 45,22% |
| piper-parir-no-vits2-cpn | 46,74% |
| piper-seqir-no-vits2-cpn | 46,89% |

Đây là tỷ lệ số phần tử lưu trong initializer của **graph inference**, không
phải FLOPs, thời gian, RSS hoặc toàn bộ tham số checkpoint training. Không suy
ra tốc độ từ số lớp, số weights hoặc kích thước file. Q/DQ và chuyển precision
có thể làm mất lợi ích, nhất là graph nhỏ/dynamic shape.

Phiên INT8 cũ đã có ví dụ cần thận trọng: baseline có duration-weighted RTF
thấp hơn nhưng mean latency cao hơn FP32, vì audio sinh ra dài hơn. Không dùng
chênh lệch đó dự báo tốc độ của cấu hình mới. Xem [giới hạn phiên N100](../../../results/edge-n100-openvino/2026-11-01/README.vi.md).

## 7. Điều kiện để nói “không giảm hiệu suất quá nhiều”

Trong báo cáo này, hiểu yêu cầu là giữ cả **chất lượng đọc, latency và bộ nhớ**.
Các ngưỡng sau là đề xuất kỹ thuật để chốt trước thử nghiệm, chưa phải kết quả:

| Tiêu chí, kiểm riêng từng model | Ngưỡng đề xuất so với FP32 cùng phiên |
|---|---|
| Corpus WER và macro WER | Mỗi chỉ số tăng không quá 0,01, tức 1 điểm phần trăm |
| UTMOS trung bình | Giảm không quá 0,10; đây là predicted score, không phải MOS người nghe |
| Nghe cùng câu | Không xuất hiện lỗi cắt âm/mất từ mới lặp lại ổn định |
| Duration với cùng explicit noise | Cùng số output samples; khác phải điều tra trước khi đo chính thức |
| Mean latency và P95 latency | Mỗi chỉ số không quá 1,05× FP32 cùng phiên |
| Peak RSS | Không quá 1,10× FP32 cùng phiên |
| Runtime | Kernel INT8 chỉ trong allowlist, không có FP16/BF16 hoặc INT8 ngoài dự kiến |
| Numeric checks | Không overflow/NaN/Inf, không audio rỗng/im lặng; không che warning |

Phải báo khoảng tin cậy ghép cặp và các mẫu thất bại. Nếu sai số thống kê khiến
kết luận còn mơ hồ, không gọi là đã chứng minh tương đương. Ngưỡng latency này
chỉ giới hạn mức chậm đi so với FP32; một bản không nhanh hơn FP32 không được
quảng bá là tăng tốc. Báo riêng chênh lệch so với recipe INT8 cũ.

Quy trình nghiệm thu cần làm tiếp:

1. Khóa câu validation, split, scorer/text normalizer và ba seed trước khi sinh
   audio. Calibration chỉ lấy train; không chọn scope bằng Harvard-720 hoặc test.
2. Đối chiếu FP32 và candidate bằng cùng explicit duration/acoustic noise.
   Đặt Python/NumPy seed không tự khóa RNG bên trong OpenVINO. Với câu demo lỗi,
   giữ nguyên text, model, scales và WAV/JSON của cả hai precision để chẩn đoán.
3. Kiểm cả graph deployment native RNG; lưu audio thực sinh, per-phoneme
   durations nếu có diagnostic output, latency, hash IR và runtime metadata.
   Không crop/normalize rồi coi waveform đã biến đổi là mẫu dùng đo thời gian.
4. Chấm WER/nghe/UTMOS trước. Có thể sinh audio ở N100 và chạy scorer trên PC.
5. Chỉ với candidate đạt quality mới đo latency/RTF ba vòng theo cùng threads,
   power, load và nhiệt độ; giữ outlier. Với scope này, nếu cùng noise mà duration
   đổi thì phải tìm rò quantization/fusion hoặc lỗi đối chiếu, không đổi mẫu số RTF
   để che vấn đề.
6. Cả sáu model đều phải đạt. Không lấy trung bình sáu model che một model lỗi.
   Nếu cần thu hẹp các lớp nhạy, dùng cùng tập loại trừ cho cả sáu rồi kiểm lại.
   Nếu không có scope chung đạt thì báo đúng, dùng FP32 cho demo.

Nếu cần thêm tốc độ sau khi cấu hình 40 lớp đạt, thử một nhánh riêng mở thêm
37 Conv encoder, thành **77 Conv/model**, vẫn bỏ SDP/decoder và attention flow
chỉ có ở nhóm lịch sử. Nhánh này phải kiểm lại duration và chất lượng; không
tự coi là an toàn hơn vì SDP vẫn FP32. Decoder PTQ cần một đợt nghiên cứu riêng.

NNCF có [accuracy-control PTQ](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/quantizing-with-accuracy-control.html)
để giữ lại các lớp gây suy giảm lớn, nhưng cần validation function thực sự.
Nếu dùng WER, phải đổi chiều thành metric cao hơn là tốt hơn, chẳng hạn
`1 - corpus_WER`; không đưa raw WER vào callback accuracy rồi đảo nghĩa kết quả.
Scope tự chọn riêng từng model cũng không còn là một scope cố định chung; muốn
giữ protocol chung phải thống nhất tập loại trừ rồi đánh giá lại cả sáu.

## 8. Tái chạy kiểm tĩnh

Từ repo, dùng một môi trường đã có ONNX; chọn thư mục output chưa tồn tại:

```bash
python -B scripts/evaluation/edge/audit_quantization_scope.py \
  --bundle results/edge-n100-openvino/2026-11-01/.local/originals/final \
  --out /tmp/hifimobinet-six-model-audit-new
```

Script từ chối thiếu file, hash sai, graph có external tensors/subgraph chưa
kiểm, Conv không có trọng số cố định, thiếu ánh xạ IR/runtime, hoặc danh sách
40 lớp không đồng nhất. Nó không tạo chứng nhận chất lượng/tốc độ: hai trường
`new_candidate_quality_pass` và `new_candidate_speed_pass` luôn là `null`.

Không sửa evidence cũ, checkpoint, graph, cấu hình train hoặc demo catalog.
