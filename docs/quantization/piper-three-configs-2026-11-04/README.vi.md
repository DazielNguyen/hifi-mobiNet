# Khả năng lượng tử hóa INT8 của ba cấu hình Piper Config A

Ngày kiểm tra: **04/11/2026**. Mục tiêu triển khai đang xét: **OpenVINO CPU trên Intel N100**.

**Cả ba model đều có các thành phần có thể đưa vào phạm vi lượng tử hóa, bao gồm decoder.** Danh sách không chỉ gồm 40 Conv của flow: còn có Conv của text encoder, stochastic duration predictor (SDP), decoder, các phép nhân ma trận có điều kiện và các bảng embedding có thể xem xét nén trọng số. Tuy nhiên, đọc code và graph chỉ xác định được ứng viên; không chứng minh rằng lượng tử hóa tất cả các phần này sẽ giữ nguyên chất lượng hoặc tăng tốc trên N100.

Đây là **báo cáo kiểm tra tĩnh**, không phải cấu hình PTQ mới hay báo cáo chất lượng INT8 mới. Trong lần kiểm tra này không load checkpoint, không export/convert/compile model, không calibration, quantization, inference, scoring, benchmark hoặc training. Không sửa cấu hình model, demo, evidence cũ hay tiến trình đang chạy.

## 1. Đúng ba cấu hình và đúng đường code

| Tên trong project | ID graph | Decoder thực tế |
|---|---|---|
| `Piper_no_VITS2_cpn` | `piper-no-vits2-cpn` | HiFi-GAN ResBlock2: 3 nhánh song song ở mỗi stage |
| `Piper_ParIR_no_VITS2_cpn` | `piper-parir-no-vits2-cpn` | `mrf`: 3 inverted residual block song song/stage, kernel 3/5/7 |
| `Piper_SeqIR_no_VITS2_cpn` | `piper-seqir-no-vits2-cpn` | `mb`: 2 inverted residual block nối tiếp/stage, kernel 3 |

Đối chiếu [registry](../../../src/hifimobinet/training/registry.py#L43), [PiperNoVits2Module](../../../src/hifimobinet/training/piper_module.py), [PiperIRNoVits2Module](../../../src/hifimobinet/training/piper_ir_module.py#L64) và [cơ chế import vendor](../../../src/hifimobinet/training/vendor.py). Backbone thực tế là `vendor/edgetts/vits/`. Với ParIR/SeqIR, harness thay `model_g.dec` bằng `IRDecoder` lấy từ **`src/hifimobinet/architecture/vits/modules/generator.py`** và thêm MRD. Không lấy một lớp có tên tương tự ở vendor khác để suy ra kiến trúc.

Ba file cấu hình đã đọc: [ResBlock2](../../../configs/training/piper-no-vits2-cpn.yaml), [ParIR](../../../configs/training/piper-parir-no-vits2-cpn.yaml), [SeqIR](../../../configs/training/piper-seqir-no-vits2-cpn.yaml).

Các điểm chung đã xác nhận:

- Hidden/inter channels 192; FFN 768; encoder 6 lớp, 2 attention heads; dùng SDP.
- `use_vits2=false`, `use_bigvgan=false`, `use_f0=false`: không có Transformer trong main flow, duration discriminator hay F0 conditioning hoạt động trong inference này.
- Upsampling gồm 3 stage, tỷ lệ `[8, 8, 4]`, kernel `[16, 16, 8]`; decoder channels `256 → 128 → 64 → 32`; đầu ra 22.050 Hz.
- ParIR/SeqIR dùng expansion `[1,1,1]` và SnakeBeta trong decoder. `use_bigvgan=false` không loại bỏ SnakeBeta do IRDecoder được thay riêng trong harness.
- Cùng cấu trúc backbone **không có nghĩa là cùng trọng số**: đây là ba checkpoint được train riêng.

Graph được đọc lại trực tiếp từ bundle FP32 round 2 đã lưu local, không tái sử dụng số đếm của báo cáo cũ:

| Model | Checkpoint theo manifest; không mở checkpoint | Node ONNX | Kích thước ONNX (byte) |
|---|---|---:|---:|
| ResBlock2 | `best-epoch=1403-val_loss_mel=19.7719.ckpt` | 6.101 | 63.469.330 |
| ParIR | `best-epoch=1486-val_loss_mel=19.0069.ckpt` | 6.465 | 61.487.365 |
| SeqIR | `best-epoch=1455-val_loss_mel=19.0332.ckpt` | 6.333 | 61.283.108 |

Cả ba hash ONNX khớp manifest, cả ba config hiện tại khớp byte với config đóng gói cùng graph, và cả ba qua `onnx.checker`. Các kiểm tra này xác nhận danh tính/cấu trúc, không xác nhận chất lượng âm thanh. SHA-256 đầy đủ, phiên bản nguồn và thời điểm kiểm tra nằm trong [inspection.json](evidence/inspection.json).

## 2. Cách hiểu “có thể quantization”

| Mức kết luận | Ý nghĩa trong báo cáo |
|---|---|
| **Ứng viên W8A8** | Có lớp tính toán phù hợp để khảo sát weight 8-bit và activation 8-bit; vẫn cần kiểm tra graph sau PTQ, kernel thực thi và chất lượng |
| **Ứng viên có điều kiện** | Depthwise, ConvTranspose hoặc MatMul có cấu trúc cần đối chiếu pattern/backend cụ thể; không mặc định NNCF sẽ lượng tử hóa hoặc N100 sẽ chạy INT8 |
| **Nén trọng số** | Có thể khảo sát lưu trọng số 8-bit rồi giải lượng tử khi tính toán; khác với bảo đảm kernel W8A8 |
| **Có evidence thực thi cũ** | Metadata của phiên N100 đã lưu ghi kernel INT8; không phải phép chạy mới và không chứng minh chất lượng đạt yêu cầu |

OpenVINO phân biệt precision trong graph, pattern lượng tử hóa và khả năng phần cứng. Chỉ có tên file `int8.xml`, `FakeQuantize` hoặc thông báo `prepared` chưa đủ để kết luận mọi lớp chạy INT8. [Tài liệu CPU precision](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html#quantized-data-types-specifics), [Low Precision IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format/intermediate-representation-int8-inference.html).

Ở đây “INT8” bao gồm trường hợp weight signed INT8 và activation UINT8 theo backend. Bias, accumulator, scale, chỉ số và các vùng float không bắt buộc cùng xuống 8-bit. Ép kiểu tensor sang `int8` trực tiếp không thay thế PTQ với scale/zero point phù hợp.

## 3. Bảng kiểm kê đầy đủ các nhóm tính toán chính

Số lượng dưới đây là **node trong graph inference FP32**, không phải số kernel sau fusion, số tham số hay tỷ lệ thời gian chạy.

| Thành phần | ResBlock2 | ParIR | SeqIR | Khả năng INT8 |
|---|---:|---:|---:|---|
| Encoder: Q/K/V/output projections, Conv 1×1 | 24 | 24 | 24 | Ứng viên W8A8; có evidence kernel cũ |
| Encoder: FFN, Conv kernel 3 | 12 | 12 | 12 | Ứng viên W8A8; có evidence kernel cũ |
| Encoder: prior statistics projection | 1 | 1 | 1 | Ứng viên W8A8; cần lưu ý đầu ra mean/log-scale |
| SDP: pointwise/pre/projection Conv | 20 | 20 | 20 | Ứng viên W8A8; có evidence kernel cũ, nhạy với duration |
| SDP: depthwise Conv | 12 | 12 | 12 | Ứng viên có điều kiện; evidence cũ vẫn FP32 |
| Main flow: pre/post Conv | 8 | 8 | 8 | Ứng viên W8A8; có evidence kernel cũ |
| Main flow: WaveNet `in_layers` Conv | 16 | 16 | 16 | Ứng viên W8A8; có evidence kernel cũ |
| Main flow: WaveNet `res_skip_layers` Conv | 16 | 16 | 16 | Ứng viên W8A8; có evidence kernel cũ |
| Decoder: `conv_pre` + `conv_post` | 2 | 2 | 2 | Ứng viên W8A8; cũ giữ FP32 |
| Decoder: Conv trong ResBlock2 | 18 | 0 | 0 | Ứng viên W8A8; cũ giữ FP32 |
| Decoder: IR `pw_expand` + `pw_project` | 0 | 18 | 12 | Ứng viên W8A8; cũ giữ FP32 |
| Decoder: IR depthwise Conv | 0 | 9 | 6 | Ứng viên có điều kiện; cũ giữ FP32 |
| **Tổng Conv** | **129** | **138** | **129** | Bao gồm depthwise |
| Decoder: ConvTranspose upsampling | 3 | 3 | 3 | Ứng viên có điều kiện; cần xác nhận kernel INT8 cho hình dạng này |
| **Tổng Conv + ConvTranspose** | **132** | **141** | **132** | Không phải số lớp đã chứng minh an toàn chất lượng |
| Encoder: attention MatMul | 24 | 24 | 24 | Có thể khảo sát lượng tử hóa toán hạng; không phải 24 Linear có weight hằng |
| Alignment: MatMul mở rộng mean/log-scale | 2 | 2 | 2 | Khả năng về mặt phép toán; ít ưu tiên, chưa có bằng chứng lợi ích |
| **Tổng node Conv/ConvTranspose/MatMul được liệt kê** | **158** | **167** | **158** | **483 node cho ba graph** |

[weighted-operators.csv](evidence/weighted-operators.csv) liệt kê **từng node trong 483 node**, tên chính xác, module, loại phép toán, group, kernel, dilation, shape/dtype của weight và trạng thái ứng viên. Tên file là quy ước kiểm kê; 78 MatMul trong đó không có weight initializer trực tiếp. [operation-counts.csv](evidence/operation-counts.csv) kiểm kê thêm toàn bộ loại phép toán còn lại, không chỉ Conv/MatMul.

Ngoài bảng trên, **mỗi model** có một phoneme embedding `[256,192]`, 12 bảng relative-position embedding `[1,9,96]`, LayerNorm và các tham số affine nhỏ. Chúng được phân tích riêng ở dưới, không bỏ sót vì không xuất hiện dưới dạng Conv.

Phần chung về cấu trúc gồm **109 Conv ở backbone** (`37 encoder + 32 SDP + 40 flow`), thêm `conv_pre`, `conv_post` và 3 ConvTranspose ở decoder. Đây là giao nhau của danh sách ứng viên, không phải đề xuất lượng tử hóa đồng thời tất cả các lớp chung.

## 4. Text encoder — `model_g.enc_p`

Nguồn: [TextEncoder](../../../vendor/edgetts/vits/models.py#L206), [Encoder](../../../vendor/edgetts/vits/attentions.py#L12), [MultiHeadAttention](../../../vendor/edgetts/vits/attentions.py#L161), [FFN](../../../vendor/edgetts/vits/attentions.py#L362).

| Lớp/nhóm | Số lượng mỗi model | Chi tiết và đánh giá |
|---|---:|---|
| `enc_p.emb` | 1 | Bảng phoneme embedding 256×192. Có thể khảo sát **nén weight INT8**, giải lượng tử khi dùng; không suy ra tốc độ tăng tương ứng vì đây là lookup/Gather |
| `encoder.attn_layers.i.conv_q/k/v/o`, `i=0..5` | 24 Conv | Mỗi Conv `192→192`, kernel 1. Đây là projections tương tự Linear được cài bằng Conv1d; ứng viên W8A8 rõ ràng |
| `encoder.ffn_layers.i.conv_1` | 6 Conv | `192→768`, kernel 3; ứng viên W8A8 |
| `encoder.ffn_layers.i.conv_2` | 6 Conv | `768→192`, kernel 3; ứng viên W8A8 |
| `enc_p.proj` | 1 Conv | `192→384`, kernel 1; chia thành `m_p` và `logs_p`. Có thể quantize nhưng sai số ảnh hưởng prior lấy mẫu latent |
| `emb_rel_k`, `emb_rel_v` của 6 lớp | 12 bảng | Mỗi bảng `[1,9,96]`. Có thể khảo sát nén hằng trọng số; cần nhận diện chuỗi slice/pad/reshape của relative attention, không mặc định NNCF coi chúng như `nn.Embedding` |
| QKᵀ, attention×V và 2 relative-position MatMul/lớp | 24 MatMul | 12 phép attention chính và 12 phép relative attention. Ứng viên có điều kiện vì toán hạng động hoặc được suy ra từ embedding; quantize Conv Q/K/V không đồng nghĩa đã quantize các MatMul này |

NNCF có cơ chế weight compression cho Linear/Convolution/Embedding. Điều đó hỗ trợ việc đưa embedding vào danh sách khảo sát, nhưng chưa xác nhận pattern cụ thể của ba graph này được phiên bản NNCF đang triển khai xử lý tự động. [Hướng dẫn weight compression của NNCF](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md) là tài liệu nhánh phát triển, không phải bằng chứng chạy của phiên bản đã lưu trong repo.

LayerNorm, Softmax, mask và residual Add nằm giữa các lớp trên. Có thể dùng các Conv INT8 cùng vùng chuẩn hóa/Softmax float; không cần biến toàn bộ encoder thành một khối integer. Code FFN của cấu hình này dùng **ReLU**; GELU xuất hiện ở DDSConv của SDP, không nên gộp hai phần này.

## 5. Stochastic duration predictor — `model_g.dp`

Nguồn: [StochasticDurationPredictor](../../../vendor/edgetts/vits/models.py#L14), [nhánh inference reverse](../../../vendor/edgetts/vits/models.py#L108), [DDSConv](../../../vendor/edgetts/vits/modules.py#L101), [ConvFlow](../../../vendor/edgetts/vits/modules.py#L500), [spline transform](../../../vendor/edgetts/vits/transforms.py).

**Có thể khảo sát quantize các Conv của SDP**, nhưng không được đồng nhất điều này với việc spline và phép suy ra duration đều hoạt động tốt ở INT8.

| Nhánh inference | Pointwise/pre/projection Conv | Depthwise Conv | Tổng |
|---|---:|---:|---:|
| Context: `dp.pre`, `dp.convs`, `dp.proj` | 5 | 3 | 8 |
| Mỗi `dp.flows.j`, `j∈{3,5,7}` | 5 | 3 | 8 |
| Ba ConvFlow đang dùng ở inference | 15 | 9 | 24 |
| **Tổng SDP mỗi model** | **20** | **12** | **32** |

Tên lớp cụ thể:

- `dp.pre`, `dp.proj`: Conv 192→192, kernel 1.
- `dp.convs.convs_1x1.i` và `dp.flows.j.convs.convs_1x1.i`: Conv 192→192, kernel 1, `i=0..2`.
- `dp.convs.convs_sep.i` và `dp.flows.j.convs.convs_sep.i`: depthwise 192 channels, group 192, kernel 3, dilation lần lượt 1/3/9. Có thể đưa vào phạm vi PTQ về cấu trúc; việc thực thi INT8 của pattern depthwise này còn phải xác nhận.
- `dp.flows.j.pre`: Conv 1→192, kernel 1; `dp.flows.j.proj`: Conv 192→29, kernel 1, tạo tham số của spline 10 bins. Đây là điểm đầu ra nhạy cảm dù operator vẫn là Conv hợp lệ.
- `dp.flows.0`: `ElementwiseAffine`, có `m` và `logs` nhỏ. Có thể biểu diễn/nén tham số với độ chính xác thấp hoặc gộp affine khi backend cho phép; không phải lớp Conv/Linear để mặc định áp dụng W8A8. Trong graph một số biểu thức đã được constant-fold, nên tên initializer không nhất thiết giữ nguyên mọi tên parameter Python.

Code có 4 ConvFlow khi khởi tạo, nhưng nhánh reverse loại một vflow không cần dùng; graph đang xét chỉ có **3 ConvFlow**. Các `post_pre`, `post_proj`, `post_convs`, `post_flows` dùng cho học phân phối posterior duration không chạy khi sinh tiếng nói. Đếm toàn bộ `state_dict` sẽ làm phình phạm vi cần quantize.

Phần không nên gọi là “đã quantize được” chỉ từ các Conv: rational-quadratic spline, Softmax/Softplus tạo bins/derivatives, phép chia, căn bậc hai, tìm bin và inverse spline. Đây là chuỗi biến đổi số học nhạy cảm, khác một lớp tích chập.

Ở đường sinh âm thanh, `logw → exp → ceil → duration → alignment`. Sai số nhỏ trước `ceil` có thể thay đổi số frame của phoneme; lỗi lớn có thể làm mất/kéo dài âm. Code còn `nan_to_num` và clamp, nên có WAV đầu ra không tự chứng minh duration đúng. Đây là **nguy cơ suy ra từ code**, chưa phải kết luận nguyên nhân của audio demo bị cắt. [ExportWrapper giữ đường duration này](../../../scripts/evaluation/onnx/common.py#L159).

## 6. Main flow — `model_g.flow`

Nguồn: [ResidualCouplingBlock](../../../vendor/edgetts/vits/models.py#L333), [ResidualCouplingLayer](../../../vendor/edgetts/vits/modules.py#L443), [WN](../../../vendor/edgetts/vits/modules.py#L152).

Đây là flow biến latent prior sang latent đưa vào decoder, **khác** các spline flow trong `dp.flows`. Cả ba model dùng 4 coupling block `flow.flows.j`, `j∈{0,2,4,6}`, xen kẽ Flip. Không có Transformer-conditioned flow ở ba cấu hình này.

| Lớp trong mỗi coupling block | Shape weight | Số mỗi block | Số mỗi model |
|---|---|---:|---:|
| `pre` | `[192,96,1]` | 1 | 4 |
| `enc.in_layers.i`, `i=0..3` | `[384,192,5]` | 4 | 16 |
| `enc.res_skip_layers.i`, `i=0..2` | `[384,192,1]` | 3 | 12 |
| `enc.res_skip_layers.3` | `[192,192,1]` | 1 | 4 |
| `post` | `[96,192,1]` | 1 | 4 |
| **Tổng** | | **10** | **40** |

Tất cả 40 Conv là ứng viên W8A8; evidence N100 cũ cũng ghi nhận 40 kernel INT8 tại đây. Tuy nhiên WaveNet còn cổng `tanh × sigmoid`, mask, residual/skip Add; việc Conv chạy INT8 không làm các phép này tự động trở thành INT8.

Coupling được tạo với **`mean_only=True`**, tức dự đoán phần dịch chuyển và dùng log-scale bằng 0 trong coupling. Không nên mô tả main flow này như spline hoặc một affine flow dự đoán cả scale động. `exp(logs_p)` của **prior** ở bên ngoài main flow vẫn tồn tại và là một nguồn nhạy cảm khác.

Các Conv có weight norm phải được đánh giá trên weight hiệu dụng đã fold đúng. [Helper round 2](../../../scripts/evaluation/onnx/common.py#L101) xử lý 4 WN tương ứng 32 Conv bên trong flow; 8 pre/post không thuộc 32 Conv này. Kiểm tra mới xác nhận cả ba graph có weight Conv hằng FP32 và không có `ReduceL2`. Không trộn graph round 1 có lỗi flow weight norm vào kết luận này.

## 7. Decoder — cả ba đều có lớp có thể lượng tử hóa

Nguồn ResBlock2: [EdgeTTS Generator](../../../vendor/edgetts/vits/models.py#L437), [ResBlock2](../../../vendor/edgetts/vits/modules.py#L348). Nguồn ParIR/SeqIR: [Generator thực tế](../../../src/hifimobinet/architecture/vits/modules/generator.py#L25), [ResBlockInverted](../../../src/hifimobinet/architecture/vits/utils/resblocks.py#L107), [SnakeBeta](../../../src/hifimobinet/architecture/vits/utils/normalization.py#L40).

### Phần chung

| Lớp | Cấu trúc | Khả năng |
|---|---|---|
| `dec.conv_pre` | Conv 192→256, kernel 7 | Ứng viên W8A8 |
| `dec.ups.0` | ConvTranspose 256→128, kernel 16, stride 8 | Ứng viên có điều kiện |
| `dec.ups.1` | ConvTranspose 128→64, kernel 16, stride 8 | Ứng viên có điều kiện |
| `dec.ups.2` | ConvTranspose 64→32, kernel 8, stride 4 | Ứng viên có điều kiện |
| `dec.conv_post` | Conv 32→1, kernel 7, không bias | Ứng viên W8A8; trực tiếp ảnh hưởng waveform |

ConvTranspose không bị loại chỉ vì là lớp upsampling. Tuy vậy, chưa có kiểm tra mới rằng NNCF/OpenVINO trên N100 sẽ thực thi **các ConvTranspose1d cụ thể này** bằng INT8. Cần phân biệt khả năng biểu diễn trọng số 8-bit với kernel integer thực tế. Trong IR đã lưu, chúng có thể mang tên `ConvolutionBackpropData_*`, còn tên ONNX `/dec/ups.N/ConvTranspose` gắn với Add bias; lọc đơn thuần theo tên có thể nhầm operator.

### ResBlock2

Ba stage, mỗi stage có ba nhánh kernel 3/5/7. Mỗi nhánh là một ResBlock2 với hai Conv có dilation tương ứng `[1,2]`, `[2,6]`, `[3,12]`. Tổng cộng **9 block × 2 = 18 Conv**, đều là ứng viên W8A8. Tên: `dec.resblocks.b.convs.c`, `b=0..8`, `c=0..1`.

Cộng `conv_pre`/`conv_post`: **20 Conv + 3 ConvTranspose**. Activation là LeakyReLU, đầu ra qua Tanh. Cộng residual và trung bình ba nhánh cần scale tương thích nếu backend giữ ở miền integer; vẫn có thể có chuyển đổi precision giữa các vùng.

### ParIR

Ba stage, mỗi stage có ba inverted residual block **song song** kernel 3/5/7, rồi lấy trung bình. Mỗi block:

`pw_expand → SnakeBeta → depthwise → SnakeBeta → pw_project → residual Add`.

`dec.mb_blocks.s.b`, `s=0..2`, `b=0..2`: **9 `pw_expand` + 9 `pw_project` + 9 depthwise = 27 Conv**. Hai nhóm pointwise là ứng viên W8A8; depthwise là ứng viên có điều kiện. Cộng pre/post thành **29 Conv + 3 ConvTranspose**.

### SeqIR

Ba stage, mỗi stage có hai inverted residual block **nối tiếp**, kernel 3. `dec.mb_blocks.s.b`, `s=0..2`, `b=0..1`: **6 `pw_expand` + 6 `pw_project` + 6 depthwise = 18 Conv**. Cộng pre/post thành **20 Conv + 3 ConvTranspose**.

Ở cả hai IR, expansion=1 làm hidden channels bằng input channels, nhưng code **vẫn tạo hai pointwise Conv**, không bỏ chúng. Các block thực tế không có BatchNorm, squeeze-and-excitation hay ReLU6 để giả định fusion kiểu một MobileNetV2 khác.

### SnakeBeta và độ chính xác hỗn hợp

ParIR có **22** SnakeBeta, SeqIR có **16**, được đối chiếu với 22/16 node `Sin` trong graph. Công thức dùng alpha/beta học được theo kênh, lấy trị tuyệt đối và clamp, rồi tính dạng `x + sin(alpha*x)^2 / beta`.

Các Conv xung quanh SnakeBeta **vẫn là ứng viên INT8**; có thể có thiết kế INT8 Conv kết hợp SnakeBeta float. Alpha/beta có thể khảo sát nén tham số nhưng nhỏ và nhạy, không phải một mục W8A8 giống Conv. Chưa có bằng chứng cho phép coi nguyên block SnakeBeta là integer hoặc cam kết lợi ích sau chi phí quantize/dequantize.

Decoder nằm sau khi latent đã được mở rộng theo thời gian, nên có thể là vùng đáng khảo sát về tính toán. Chỉ số lượng Conv không cho biết tỷ trọng latency: depthwise, pointwise, ConvTranspose và các stage có chi phí khác nhau. Báo cáo này không xếp hạng tốc độ hoặc khẳng định decoder INT8 sẽ nhanh hơn.

## 8. Các phần còn lại: có thể hạ precision đến đâu?

| Phần | Trạng thái trong ba model | Cách phân loại |
|---|---|---|
| LayerNorm gamma/beta | 12 LayerNorm ở encoder + 24 trong DDSConv = 36/model | Tham số float nhỏ có thể biểu diễn độ chính xác thấp, nhưng phép chuẩn hóa gồm reductions/variance; không xem như ứng viên Conv W8A8 hoặc tự khẳng định kernel INT8 |
| Bias của Conv, scale/zero point và accumulator | Đi cùng các lớp tính toán | Thường cần precision lớn hơn 8-bit hoặc xử lý riêng của backend; không ép toàn bộ bias xuống INT8 |
| ReLU/LeakyReLU, residual Add, một số Mul | Encoder, SDP, flow, decoder | Có thể fusion hoặc truyền tensor lượng tử nếu pattern/scale phù hợp; không phải lớp có weight để đếm riêng trong tổng Conv |
| Reshape/Transpose/Squeeze/Split/Concat/Slice/Pad | Nhiều vị trí trong graph | Có thể vận chuyển dữ liệu low precision khi hợp lệ; không mang lại phép nhân INT8 mới. Chỉ số/shape giữ đúng kiểu và ngữ nghĩa |
| Softmax/Softplus/GELU/Tanh/Sigmoid/Sin, Exp/Log/Sqrt/Div | Attention, SDP, flow, decoder, prior | Có thể tồn tại cùng Conv INT8 qua vùng float/fusion phù hợp. Chưa xác nhận backend cho phép cả chuỗi chạy integer; không tuyên bố bất khả thi về lý thuyết |
| Hai MatMul alignment | Mở rộng `m_p`, `logs_p` từ phoneme sang frame | Có thể khảo sát quantization của toán hạng ở mức phép toán; không có weight học được trực tiếp. Giữ đúng đường alignment là yêu cầu riêng, lợi ích INT8 chưa được chứng minh |
| `exp/ceil/clamp`, tạo duration/alignment | Sau SDP | Quyết định số frame và ánh xạ phoneme; không phải mục W8A8 thông thường. Không đổi precision/cắt ngưỡng chỉ để tăng tỷ lệ node INT8 |
| Phoneme ID, độ dài, mask, so sánh, index, shape | Đầu vào và điều khiển graph | Không phải “weight FP32 cần quantize”; phải giữ miền giá trị/ngữ nghĩa. INT64/BOOL không phải thiếu sót của một model mixed precision |
| RNG và lấy mẫu latent/duration | Hai node `RandomNormalLike` mỗi graph | Không phải trọng số để nén. Cần tách biến động do sampling khỏi sai số PTQ khi đánh giá sau này |
| Text normalization/phonemizer, WAV encoding, phát audio | Ngoài mạng neural đã export | Không thuộc PTQ của ba graph. Đổi WAV sang PCM 8/16-bit không phải quantize model |

## 9. Thành phần có trong code/checkpoint nhưng không thuộc inference đang triển khai

| Thành phần | Có ở đâu? | Kết luận về phạm vi |
|---|---|---|
| Posterior encoder `enc_q` | Cả ba model khi train, nhận spectrogram thật | Có Conv có thể khảo sát lượng tử hóa nếu triển khai riêng mạng này, nhưng không có trong text→audio graph đang xét; không tăng tốc demo bằng cách quantize nó |
| SDP posterior `post_*` và vflow bị bỏ ở reverse | Nhánh train của SDP | Có các lớp Conv nhưng không thuộc 32 Conv inference đã đếm |
| `model_d` / MultiPeriodDiscriminator | Cả ba cấu hình khi train | Bản code này chứa 1 DiscriminatorS và 5 DiscriminatorP. Các Conv1d/Conv2d có thể khảo sát riêng, nhưng discriminator không chạy khi sinh audio |
| `model_d_mrd` | ParIR và SeqIR; ResBlock2 Config A không dùng | Có Conv2d trong discriminator đa độ phân giải; không xuất hiện trong inference graph. MRD và MPD là các module discriminator riêng trong training |
| Duration discriminator, Transformer flow, F0 predictor/conditioning | Các cờ tương ứng tắt | Không thêm các lớp của cấu hình VITS2/F0 khác vào danh sách ba model này |
| Speaker embedding/conditioning dùng cho nhiều speaker | Inference đang xét single-speaker, `g=None` | Không có nhánh speaker-conditioning hoạt động trong graph này |

Việc một Conv trong discriminator có thể lượng tử hóa về cấu trúc không phải đề xuất lượng tử hóa mạng đang training. Báo cáo không thay đổi training precision `bf16`, optimizer state hay checkpoint.

## 10. Đối chiếu với INT8 N100 đã có: phạm vi cũ không phải giới hạn kiến trúc

Đã đọc lại JSON runtime và preparation lưu trong [evidence của phiên N100](../../../results/edge-n100-openvino/2026-11-01/evidence/). Preparation được kiểm hash theo [catalog](../../../demo/n100_models.json) và kiểm quan hệ với đúng hash FP32. Không compile hoặc chạy lại INT8. Bản trích các kernel ở [historical-runtime-kernels.csv](evidence/historical-runtime-kernels.csv) được đánh dấu `historical_only=True`.

| Nhóm | ResBlock2 | ParIR | SeqIR | Precision được ghi trong metadata cũ |
|---|---:|---:|---:|---|
| Encoder Conv | 37 | 37 | 37 | INT8/UINT8 |
| SDP pointwise/pre/projection | 20 | 20 | 20 | INT8/UINT8 |
| Main flow Conv | 40 | 40 | 40 | INT8/UINT8 |
| **Tổng kernel INT8 nhóm có trọng số** | **97** | **97** | **97** | Không phải toàn bộ model INT8 |
| SDP depthwise | 12 | 12 | 12 | FP32 |
| Decoder Conv | 20 | 29 | 20 | FP32 |
| Decoder upsampling | 3 | 3 | 3 | FP32; tên runtime có thể nằm ngoài namespace `/dec/` |
| Attention + alignment MatMul | 26 | 26 | 26 | FP32 |

[Suite cũ](../../../results/edge-n100-openvino/2026-11-01/evidence/openvino_suite.py#L261) chủ động cho phép Conv/GroupConvolution trong `enc_p`, `dp`, `flow` và loại decoder cùng các weighted op ngoài phạm vi. **Decoder FP32 là quyết định của scope đó**, không phải kết luận rằng mọi lớp decoder không lượng tử hóa được. NNCF cũng cung cấp `ignored_scope` để chủ động loại những vùng cần bảo toàn độ chính xác. [Basic Quantization Flow](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html).

Trong mỗi graph, scope cũ có 109 Conv ứng viên (`37+32+40`), nhưng metadata ghi 97 kernel INT8; 12 depthwise SDP vẫn FP32. Đây là ví dụ trực tiếp cho việc **nằm trong allowlist không đồng nghĩa thực thi INT8**.

Báo cáo [audit sáu model trước đây](../six-model-audit-2026-11-02/README.vi.md) và [candidate flow-only](../../../configs/quantization/n100-common-flow-conv-int8.json) có mục đích chọn phạm vi thử nghiệm chung. Candidate 40 Conv không phải danh sách đầy đủ những phần có thể quantize. Lần này không thay hoặc áp dụng candidate đó.

## 11. Danh sách kết luận để sử dụng trong nghiên cứu

Với **cả ba** cấu hình, có thể đưa vào danh sách khảo sát: **phoneme/relative embedding theo hướng nén trọng số; 37 Conv encoder; 32 Conv SDP; 40 Conv main flow; các Conv đầu/cuối và bên trong decoder; 3 ConvTranspose; các MatMul attention với điều kiện backend/pattern phù hợp**. Hai MatMul alignment có khả năng xét ở mức toán hạng nhưng chưa có bằng chứng lợi ích và không phải trọng số Linear.

Khác biệt decoder cần ghi chính xác: **ResBlock2 có 20 Conv; ParIR có 29 Conv; SeqIR có 20 Conv**, mỗi loại thêm 3 ConvTranspose. ParIR/SeqIR có SnakeBeta và depthwise nên không thể lấy kết luận về một decoder Conv/LeakyReLU áp nguyên cho chúng.

Chưa thể chốt từ kiểm tra tĩnh rằng lượng tử hóa đồng thời các nhóm trên sẽ giữ WER/chất lượng nghe/RTF tốt. Có kernel INT8, có audio hợp lệ và có chất lượng tiếng nói tốt là ba kết luận khác nhau. Các triệu chứng audio demo bị cắt chưa được quy nguyên nhân chỉ từ báo cáo này; duration, frontend, sampling, giới hạn độ dài và graph chuyển đổi đều cần được phân biệt khi có một đợt kiểm chứng được yêu cầu riêng.

**Trạng thái hiện tại: đã liệt kê và kiểm tra cấu trúc; chưa chọn phạm vi PTQ mới, chưa thực hiện lượng tử hóa hay đánh giá mới.**

## 12. Tệp bằng chứng và khả năng đối chiếu

| Tệp | Nội dung |
|---|---|
| [inspection.json](evidence/inspection.json) | SHA-256 nguồn/graph/config/record, git HEAD, môi trường đọc ONNX, kiểm tra cấu trúc và tổng hợp metadata cũ |
| [weighted-operators.csv](evidence/weighted-operators.csv) | 483 node Conv/ConvTranspose/MatMul; tên và shape để truy đúng từng lớp |
| [initializers.csv](evidence/initializers.csv) | 1.141 initializer của ba graph, dtype, shape và consumer trực tiếp; gồm weight, bias, embedding và hằng được fold, không phải 1.141 parameter trainable |
| [operation-counts.csv](evidence/operation-counts.csv) | Toàn bộ loại operator được đếm theo module/model |
| [historical-runtime-kernels.csv](evidence/historical-runtime-kernels.csv) | Metadata thực thi cũ trích lại để phân biệt khả năng với precision từng ghi nhận |
| [inspect_graphs.py](inspect_graphs.py) | Công cụ chỉ đọc protobuf ONNX/JSON và ghi bảng; không import model project, không load checkpoint hay gọi OpenVINO/NNCF |
| [SHA256SUMS](SHA256SUMS) | Hash các tệp của báo cáo này |

Nguồn local được đọc tại `results/edge-n100-openvino/2026-11-01/.local/originals/final/`. Không sao chép graph/checkpoint lớn vào báo cáo. Đường dẫn `.local` là dữ liệu nội bộ không được bảo đảm hiện diện trong một clone mới; các CSV/JSON đã lưu đủ tên/hash để đối chiếu khi có bundle gốc.

Kiểm tra tĩnh ngày 04/11 dùng Python/ONNX có sẵn; `onnx.checker` kiểm cấu trúc, không chạy phép toán của mạng. Không có subgraph hoặc external initializer cần bỏ qua. Mọi Conv/ConvTranspose đều có weight initializer FP32 trực tiếp; không có `ReduceL2`, `QuantizeLinear`, `DequantizeLinear`, `QLinearConv` trong ba graph FP32 này. Các file nguồn đã đọc được hash lại sau kiểm tra và không thay đổi. File `docs/plan-onnx-ptq-v2.vi.md` đã có thay đổi từ trước và được giữ nguyên.
