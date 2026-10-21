# Q03 — Cấu hình và lịch sử huấn luyện dùng cho paper

Cập nhật sau xử lý tiếp: **04/10/2026**. **Đã hoàn tất phần đối chiếu và khôi phục khả thi trong phạm vi lưu trữ đã kiểm; Q03 chưa đóng toàn bộ lịch sử.** Ngoài bốn checkpoint, chín bộ hparams và bằng chứng TensorBoard, đã khôi phục source từ lịch sử sửa file, ghép kết quả công cụ và lưu phạm vi tìm checkpoint còn thiếu. Xem mục 11. Dữ liệu lịch sử không thu hồi được được ghi thành giới hạn, không biến thành yêu cầu mặc định phải train lại.

Nguồn chính là [gói Q01](evidence/q01_2026-10-04/BAO_CAO_Q01.md), metadata/hash của ba `last.ckpt` và gói bổ sung ở mục 10. Lượt audit ban đầu và lượt rà soát này không dựng model, train, inference, export hay chấm điểm. Gói bổ sung có **kết quả tái dựng MRF bằng script dựng model**; lượt rà soát kiểm lại manifest và bằng chứng đi kèm, không chạy lại script đó.

**Bổ sung cùng ngày (xem [mục 10](#q03-bo-sung)):** năm câu validation được log của baseline/SEQ khớp phần đầu canonical ở cả ba phiên mỗi model; năm câu MRF khớp phần đầu manifest tái dựng, đồng thời 500 test ID tái dựng khớp manifest MRF cũ. Đây là bằng chứng đối chiếu mạnh. Các kết luận về **toàn bộ** split, tính bất biến ở mức byte và lịch sử Piper vẫn phải giữ giới hạn tương ứng.

## 1. Cách đọc mức bằng chứng

| Ký hiệu | Ý nghĩa |
|---|---|
| **K** | Đọc trực tiếp từ checkpoint đã xác định hash: hparams, tensor keys/shapes, optimizer, scheduler, callback. |
| **L** | Có log train/resume hoặc hparams của phiên tương ứng. |
| **C** | Mã nguồn/config trong snapshot Q01 giải thích cơ chế. Source hiện tại không tự chứng minh code đã bất biến trong toàn bộ lịch sử train. |
| **M** | Phép kiểm trên manifest đang lưu: số ID, giao tập, thứ tự, hash. Không tự chứng minh manifest đã dùng ở một run cũ. |
| **T** | Ghi **trong lúc train** vào file TensorBoard event của đúng phiên đó (có wall-time), không phải tái dựng về sau. |
| **F** | Metadata filesystem đang quan sát: birth/ctime/mtime. Hỗ trợ đối chiếu thời gian; không phải hash nội dung được ghi lúc train hoặc lịch sử đầy đủ của file. |
| **H** | Bản trích các yêu cầu gọi công cụ từ transcript Claude Code: lệnh khởi chạy/resume, sửa/sync được bộ lọc nhận diện. Không bao phủ mọi thao tác hoặc tự xác nhận lệnh thành công. |
| **H2** | Source khôi phục từ nội dung trước sửa/bản backup/lệnh sửa có kết quả đi kèm; SHA-256 của văn bản khôi phục và một số kết quả sync đã ghép. Mạnh hơn bản trích H, vẫn không phải hash source đo trong process train Linux. |
| **R** | Manifest tái dựng về sau, đã kiểm số lượng/giao tập và đối chiếu với các bằng chứng cũ. Kết luận lịch sử phụ thuộc giả định về source, dữ liệu và trạng thái RNG của thủ tục tái dựng. |
| **Chưa xác minh** | Chưa có đủ bằng chứng để khẳng định lịch sử thực thi. Không thay bằng giả định thuận lợi cho kết luận. |

Chi tiết máy đọc được: [Q03_EVIDENCE.json](evidence/q03_2026-10-04/Q03_EVIDENCE.json). Trường `sources` chứa đường dẫn nguồn, đường dẫn trong gói Q01, SHA-256, kích thước và mtime; `logs`/`source_excerpts` chứa đoạn trích kèm số dòng. Các số dòng log dùng `splitlines()`, có thể khác cách editor hiển thị carriage return.

## 2. Checkpoint và quy tắc chọn

Các epoch dưới đây là **giá trị metadata, đếm từ 0**. Tên Piper `lj-med_1000` tương ứng metadata epoch 999; không gọi đó là epoch 1000 theo cùng cách đánh số với các checkpoint còn lại.

| Nội dung | Baseline BanhmiTTS | MRF(1,1,1) | SEQ(1,1,1) | Piper gốc |
|---|---|---|---|---|
| Checkpoint đang đối chiếu | `best-epoch=1489-val_loss_mel=19.7882.ckpt` | `best-epoch=1386-val_loss_mel=18.7681.ckpt` | `best-epoch=1442-val_loss_mel=19.0136.ckpt` | `lj-med_1000.ckpt` |
| Epoch trong checkpoint **K** | 1489 | 1386 | 1442 | 999 |
| `global_step` trong checkpoint **K** | 1.168.160 | 1.087.408 | 1.131.312 | 1.114.000 |
| Callback monitor **K** | `val_loss_mel`, `mode=min` | `val_loss_mel`, `mode=min` | `val_loss_mel`, `mode=min` | `monitor=None`, không có best score theo validation |
| Best score lưu trong callback **K** | 19,7881698608 | 18,7680931091 | 19,0136032104 | Không ghi |
| Cuối run hiện còn `last.ckpt` **K/L** | Epoch 1499, callback vẫn chọn 1489 | Epoch 1499, callback vẫn chọn 1386 | Epoch 1499, callback vẫn chọn 1442 | Chưa có hồ sơ `last.ckpt`/log gốc riêng |
| Cách mô tả hợp lệ | Checkpoint được callback cuối run giữ là tốt nhất theo validation mel loss | Tương tự | Tương tự | Checkpoint Piper công bố ở mốc 1.000 epoch; không gọi là best-validation checkpoint |

Ba `last.ckpt` BanhmiTTS cùng lưu `global_step=1.176.000`. Đây là bộ đếm của framework, không phải thời gian train, số mẫu duy nhất hoặc bảo đảm từng lần đếm đều cập nhật trọng số thành công. Log SEQ có các bước bị bỏ qua do gradient không hữu hạn.

Code đã lưu cấu hình kiểm tra checkpoint mỗi epoch, giữ `save_top_k=3`, `save_last=True` và thư mục checkpoint cố định. Callback trong các checkpoint đã chọn và `last.ckpt` cuối đều phù hợp với quy tắc này. Có thể mô tả quy tắc chọn theo validation mel loss; không mở rộng thành “tốt nhất trong mọi thử nghiệm đã từng chạy” hoặc “tốt nhất theo UTMOS/WER”.

`val_loss_mel` trong code đã lưu là mel L1 có nhân `c_mel=45`. Không đồng nhất số loss này với điểm chất lượng tiếng nói hoặc một phép đo trực tiếp độ tự nhiên. Code hiện tại còn kiểm tra SDP: nếu sinh audio suy biến, giá trị dùng để log có thể được đặt thành `inf`. Timeline bổ sung ghi nhận việc thêm nhánh này trước SEQ ngày 23/09, sau các run baseline/MRF. Vì vậy, **cùng tên monitor không bảo đảm toàn bộ cơ chế chọn checkpoint giống nhau** giữa ba model; xem mục 10.4.

Bằng chứng bổ sung cuối run: [FINAL_RUN_CALLBACKS.json](evidence/q03_2026-10-04/FINAL_RUN_CALLBACKS.json). File này ghi hash/thời điểm quan sát của `last.ckpt` **hiện tại**; không khôi phục được nội dung `last.ckpt` tại các lần resume cũ.

## 3. Kiến trúc: phần dùng chung và phần khác

| Thành phần | Baseline BanhmiTTS | MRF(1,1,1) | SEQ(1,1,1) | Piper gốc | Bằng chứng |
|---|---|---|---|---|---|
| Text encoder | Embedding 256 × 192; Transformer 6 lớp, 2 heads, FFN 768 | Cùng cấu trúc tham số | Cùng cấu trúc tham số | State-key/shape khớp text encoder baseline | **K/C** |
| Posterior encoder | Linear spectrogram 513 bins; hidden/inter 192; WN 16 lớp, kernel 5 | Cùng cấu trúc tham số | Cùng cấu trúc tham số | State-key/shape khớp posterior baseline | **K/C** |
| Acoustic flow | 4 coupling blocks có Transformer trong conditioner | Cùng cấu trúc tham số | Cùng cấu trúc tham số | Coupling flow không có các tensor attention bổ sung | **K/C** |
| Duration predictor | Flow-based SDP | Cùng cấu trúc tham số | Cùng cấu trúc tham số | State-key/shape khớp SDP baseline | **K/C** |
| MAS | Có hparams noise ban đầu 0,01, decay 2e-6 | Cùng hparams | Cùng hparams | Code Piper không có nhánh noise bổ sung đó | **K/C** |
| Duration discriminator | Có | Có | Có | Không có trong state dict | **K/C** |
| Decoder `resblock` | `"2"` | `"mrf"` | `"mb"` | `"2"` | **K** |
| Tổ chức residual ở mỗi tầng | Ba ResBlock2 song song, trung bình đầu ra | Ba inverted-residual block song song, trung bình đầu ra | Hai inverted-residual block nối tiếp | Cùng họ ResBlock2/fusion với baseline | **K/C** |
| Kernel residual/depthwise | Dense kernel 3/5/7 | Depthwise kernel 3/5/7 | Hai depthwise kernel đều 3 | Dense kernel 3/5/7 | **K/C** |
| Dilation residual | (1,2), (2,6), (3,12) theo ba kernel | Depthwise dilation 1 trong code đã lưu | Depthwise dilation 1 trong code đã lưu | (1,2), (2,6), (3,12) | **K/C** |
| Expansion theo tầng | Không áp dụng | (1,1,1) | (1,1,1) | Không áp dụng | **K/C** |
| Activation decoder | LeakyReLU | SnakeBeta | SnakeBeta | LeakyReLU | **K/C** |
| Activation ngay trước `conv_post` | LeakyReLU slope 0,1 | SnakeBeta | SnakeBeta | `F.leaky_relu(x)` dùng mặc định 0,01 | **C**; slope không được mã hóa thành tensor trainable |
| Discriminator waveform | Một scale + năm period discriminators | Như baseline, thêm MRD | Như baseline, thêm MRD | Một scale + năm period discriminators | **K/C** |
| F0 predictor / Vocos / speaker embedding trong cấu hình được chọn | Không có nhánh tương ứng trong checkpoint/code path đã đối chiếu | Không dùng | Không dùng | Không có trong checkpoint này | **K/C** |

**Bằng chứng dùng chung backbone:** state-key/shape của `enc_p`, `enc_q`, `flow`, `dp` khớp giữa ba checkpoint BanhmiTTS. Số state entries tương ứng là **111 / 100 / 184 / 284**. Piper có cùng schema `enc_p`, `enc_q`, `dp`; flow Piper có **112** entries. Equality về schema không phải equality về trọng số, forward semantics hoặc recipe train.

**Bằng chứng decoder thật:** checkpoint MRF có 9 depthwise-convolution weight tensors, gồm ba kernel 3/5/7 ở mỗi tầng; SEQ có 6 tensors, hai kernel 3 ở mỗi tầng. Các pointwise expansion shapes xác nhận tỷ lệ 1, không phải chỉ suy từ tên `(1,1,1)`. Hparams vẫn chứa các trường dense-resblock kế thừa; không dùng những trường đó để mô tả kernel/dilation đang thực thi của MRF/SEQ.

Ba decoder BanhmiTTS và Piper medium dùng chung thông số upsampling tổng thể: input latent 192 channels; initial channel 256; ba tầng stride **8/8/4**, kernel **16/16/8**, channel **128/64/32**; đầu ra waveform một kênh. Baseline/Piper có khác biệt activation cuối nên không gọi hai decoder là hoàn toàn giống nhau.

## 4. Cấu hình huấn luyện lưu trong checkpoint

| Thông số | Baseline BanhmiTTS | MRF | SEQ | Piper gốc |
|---|---|---|---|---|
| PyTorch Lightning ghi trong checkpoint | 1.7.7 | 1.7.7 | 1.7.7 | 1.9.5 |
| Training precision | `bf16`; log xác nhận bfloat16 AMP | `bf16`; log xác nhận AMP | `bf16`; log xác nhận AMP | `32` |
| Accelerator / devices / strategy | GPU / 2 / DDP | GPU / 2 / DDP | GPU / 2 / DDP | GPU / 1 / strategy không ghi cụ thể |
| Batch size cấu hình | 16 mỗi process | 16 mỗi process | 16 mỗi process | 22 |
| Batch danh định gộp hai GPU | 32 nếu không tích lũy gradient | 32 với cùng điều kiện | 32 với cùng điều kiện | 22 trên một device |
| `accumulate_grad_batches` trong hparams | `null` | `null` | `null` | `null` |
| Seed cấu hình | 1234 | 1234 | 1234 | 1234 |
| `num_workers` | 8 | 1 | 1 | 1 |
| `gradient_clip_val` ở checkpoint được chọn | **1,0** | `null` | `null` | `null` |
| Giới hạn epoch cuối của run | 1500 | 1500, nâng từ 500 rồi 1000 | 1500 | 1000 |
| `max_phoneme_ids` | `null` | `null` | `null` | **400** |
| Hparams về chia dữ liệu | Val 100, test 500 | Val 100, test 500 | Val 100, test 500 | `validation_split=0.05`, `num_test_examples=0` |
| Sample rate / STFT | 22.050 Hz; FFT/window 1024, hop 256 | Như baseline | Như baseline | Như baseline |
| Segment training | 8192 samples = 32 latent frames | Như baseline | Như baseline | Như baseline |

`null` được báo đúng theo checkpoint; không tự đổi thành một con số đã được CLI chỉ định. Batch 32 là cách quy đổi danh định từ batch 16 × hai process, không bảo đảm mọi batch đều chứa 32 ví dụ khác nhau vì padding/sampling và ranh giới batch. Precision huấn luyện không phải nhãn precision suy luận FP32/PTQ.

Các optimizer/scheduler được xác nhận trực tiếp từ checkpoint, kết hợp code **K/C**:

| Thông số | Cả bốn checkpoint |
|---|---|
| Optimizer | Hai AdamW: generator và các discriminator tương ứng |
| Learning rate khởi đầu | 2e-4 |
| Betas / epsilon | (0,8; 0,99) / 1e-9 |
| Weight decay lưu trong optimizer | 0,01 |
| Scheduler | ExponentialLR, gamma 0,999875 |
| Mel reconstruction coefficient / KL coefficient | 45 / 1 |

Optimizer BanhmiTTS lưu `fused=True`; Piper lưu `fused=None`. Không dùng khác biệt này để kết luận tác động chất lượng khi chưa có thực nghiệm kiểm soát.

**Loss không hoàn toàn giống nhau:** cả bốn có waveform adversarial, feature matching, mel reconstruction, KL và SDP duration likelihood theo code tương ứng. Ba BanhmiTTS thêm duration adversarial loss. MRF/SEQ còn thêm adversarial và feature-matching loss từ MRD; hệ số MRD ở đường HiFi-GAN-family hiện tại là **1,0**, không phải trường `vocos_mrd_loss_coeff=0.1` kế thừa trong hparams. `use_vocos=False` nên trường đó không quyết định nhánh MRF/SEQ này.

## 5. Lịch train/resume đã xác minh

Ngày dưới đây lấy từ **mtime của hparams trong snapshot**, không phải bằng chứng độc lập về thời điểm bắt đầu/kết thúc run. Giữ UTC để đối chiếu với manifest. Quy tắc khôi phục có bằng chứng từ log và hparams; hash checkpoint trung gian tại thời điểm resume còn thiếu.

| Model / phiên | Mtime hparams UTC | Nguồn khởi động/khôi phục | Clip | Giới hạn epoch |
|---|---|---|---|---:|
| Baseline `version_0` | 26/08/2026 | Hparams `resume_from_checkpoint=null` | `null` | 1500 |
| Baseline `version_1` | 27/08/2026 | Log/hparams chỉ định `best-epoch=89-val_loss_mel=22.4389.ckpt` | 1,0 | 1500 |
| Baseline `version_2` | 31/08/2026 | Log/hparams chỉ định `best-epoch=870-val_loss_mel=20.1505.ckpt` | 1,0 | 1500 |
| MRF `version_0` | 10/09/2026 | Hparams không chỉ định resume | `null` | 500 |
| MRF `version_1` | 12/09/2026 | `last.ckpt`; log xác nhận khôi phục trạng thái | `null` | 1000 |
| MRF `version_2` | 16/09/2026 | `last.ckpt`; log xác nhận khôi phục trạng thái | `null` | 1500 |
| SEQ `version_0` | 23/09/2026 | Hparams không chỉ định resume; log đọc canonical split | `null` | 1500 |
| SEQ `version_1` | 28/09/2026 | `last.ckpt`; log khôi phục và đọc canonical split | `null` | 1500 |
| SEQ `version_2` | 29/09/2026 | `last.ckpt`; log khôi phục và đọc canonical split | `null` | 1500 |
| Piper | Chưa có hparams/log phiên gốc độc lập | Checkpoint có `resume_from_checkpoint=null`; callback path chứa `version_1` | `null` | 1000 |

Log baseline/MRF/SEQ đều có thông báo đạt `max_epochs=1500`; ba `last.ckpt` hiện tại đều là epoch 1499. Vì `last.ckpt` là tên file có thể bị ghi đè, không suy được hash của nội dung mà từng lần resume đã đọc chỉ từ đường dẫn hiện tại.

Tên thư mục **`baseline_v2_noclip` không mô tả đúng toàn bộ run**: clip đã bật từ phiên resume dùng checkpoint 89 và vẫn là 1,0 ở checkpoint cuối được chọn. Không viết “cả ba model đều train không gradient clipping”.

Log SEQ có **34 dòng** và **10 dòng** thông báo bỏ qua optimizer step do gradient không hữu hạn ở hai file đầu; các dòng có thể lặp theo hai rank DDP. Chúng xác nhận cơ chế bỏ qua update đã được thực thi, không phải số lượng batch độc lập nếu chưa khử trùng. Không có thông báo tương ứng trong các log baseline/MRF được đọc không chứng minh các run đó chưa từng gặp lỗi số học. Phần bổ sung đọc header/tag của TensorBoard, không phân tích toàn bộ audio/scalar hoặc khôi phục mọi thử nghiệm thất bại trước run cuối.

## 6. Manifest: phân biệt bản đang có với bản dùng khi train

| Tài liệu hiện có | Số lượng / kết quả kiểm tra | Phạm vi chứng minh |
|---|---|---|
| `dataset.jsonl` | 13.100 dòng, 13.100 `audio_norm_path` khác nhau | Danh mục dữ liệu trong snapshot; không tự chứng minh byte nội dung bất biến ở mọi run. |
| `canonical_split.json` | Train 12.500 / val 100 / test 500; không giao **ID/path**, không thiếu hoặc thừa path so với dataset hiện có | Nhất quán về ID. Q04 phát hiện một nội dung trùng train–validation; không viết các tập hoàn toàn không trùng nội dung. |
| Test manifest baseline | 500 path khác nhau; membership khớp toàn bộ canonical test, **thứ tự khác canonical test** | Cùng tập ID hiện tại, không phải cùng thứ tự với file canonical. |
| Test manifest MRF | 500 path khác nhau; giao test baseline **21/500** | Hai manifest test cũ khác nhau. |
| MRF test đối chiếu canonical hiện tại | 478 thuộc train, 1 thuộc val, 21 thuộc test | Membership trong **canonical hiện tại**. Chiều ngược lại theo **manifest MRF tái dựng**: 473/500 canonical test thuộc train, 6 thuộc val, 21 thuộc test; không đổi nhãn thành bản ghi train-time đầy đủ. |

Trong các phép kiểm trên, ID là toàn bộ `audio_norm_path` được lưu trong manifest. Lượt rà soát này chỉ kiểm lại kết quả tái dựng đã cung cấp, không tạo lại split bằng model/seed.

| Model | Bằng chứng về split của run | Trạng thái lịch sử |
|---|---|---|
| Baseline | Test manifest tái dựng; cả ba phiên TensorBoard có cùng năm tag văn bản, khớp `val[0..4]` canonical. | **T + M:** xác nhận phần validation được log; hỗ trợ mạnh giả thuyết cùng split. Chưa xác minh trực tiếp 95 validation ID còn lại và toàn bộ train/test từng phiên. |
| MRF | Manifest cũ được tái dựng từ hparams checkpoint 836. Manifest bổ sung dùng hparams 1386, khớp năm tag validation cả ba phiên và đúng thứ tự 500 test ID cũ. | **T + M + R:** bản tái dựng được đối chiếu tốt, xem [manifest](evidence/q03_bosung_2026-10-04/MRF_SPLIT_RECONSTRUCTED_VALIDATED.json). Chưa có danh sách train/val/test đầy đủ được ghi lúc từng phiên chạy; không gọi toàn bộ là đã xác nhận trực tiếp. |
| SEQ | Ba log xác nhận đọc canonical; metadata file có từ trước run; năm tag validation cả ba phiên khớp canonical. | **L + F + T:** bằng chứng mạnh hơn việc chỉ có file hiện tại. Chưa có hash/full ID manifest ghi trong log lúc mỗi lần đọc; hash hiện tại không tự biến thành hash lịch sử. |
| Piper | Hparams có validation ratio 0,05, test count 0 và cap phoneme IDs 400. | **Chưa thu hồi** manifest/log train gốc trong phạm vi đã kiểm. Không xác định được overlap lịch sử từ các trường này; Harvard bổ sung đánh giá ngoài LJSpeech, không xóa giới hạn provenance. |

Hash các manifest hiện có:

| File | SHA-256 |
|---|---|
| `dataset.jsonl` | `f4a72ae0ed238dfdb3ca995b6678dfe33fe3b59297331d6551a4d6960102295b` |
| `canonical_split.json` | `e678cf436075a10bb68c9a1b435f33c4a40ff1f095b7b78ac12f546c5f12ae9d` |
| `eval_true_testset_500_baseline.json` | `72916b93d84a9ea7acc020da5af966a1fb6233b2af3594a95e62feff8838756e` |
| `eval_true_testset_500_mrf.json` | `01986a8e8b85aa2bcb7ff89f9cc9db33f0cafc13a7496c568ccad1c9da20b3c3` |

Baseline/MRF/SEQ đều ghi seed 1234 nhưng điều đó không đủ để khẳng định cùng split. Code hiện tại ưu tiên canonical manifest hoặc dùng generator riêng cho split; không gán cơ chế hiện tại cho mọi run cũ. Các comment khẳng định “REAL split” hay “guaranteed” trong script tái dựng là lời mô tả của script, không thay bằng chứng lịch sử.

## 7. Biến được giữ chung và giới hạn so sánh

| Nhóm yếu tố | Đã xác minh | Hệ quả cho cách viết paper |
|---|---|---|
| Backbone của ba BanhmiTTS | Cùng hparams chính và schema `enc_p/enc_q/flow/dp` | Có thể viết “shared backbone architecture”; không viết “identical weights”. |
| Thiết kế decoder | Baseline dense ResBlock2; MRF ba nhánh inverted; SEQ hai block nối tiếp | Có thể mô tả khác biệt cấu trúc chính xác từ code và checkpoint. |
| Activation/discriminator/loss | Baseline khác MRF/SEQ ở SnakeBeta và MRD | Baseline–MRF/SEQ chưa phải ablation chỉ thay residual topology. |
| Lịch clipping và ổn định số | Baseline có clip ở các phase sau; SEQ có log bỏ qua update không hữu hạn | Không gọi toàn bộ training recipe giống nhau. |
| Dữ liệu huấn luyện/validation lịch sử | Năm validation tag baseline/SEQ khớp canonical, MRF khác; bản tái dựng MRF có train giao canonical 11.930/12.500 (95,44%). | Báo đây là đối chiếu/tái dựng có bằng chứng hỗ trợ. Chưa gọi toàn bộ lịch sử data-controlled hoặc byte dữ liệu mọi run giống hệt nhau. |
| Piper đối chứng | Khác flow, duration discriminator, precision, batch size, epoch budget, cap độ dài và checkpoint selection | Dùng làm đối chứng cấp hệ thống/checkpoint công bố, không coi là một nhánh ablation nội bộ. |
| Nguồn code tại thời điểm run | WSL BanhmiTTS không có Git; có snapshot hiện tại và timeline công cụ được lọc. Timeline ghi sửa clamp baseline trước resume 870 và sửa xử lý gradient/SDP trước SEQ. | Không tự gán commit Windows cho run Linux. Trình bày các thay đổi đã ghi nhận và giới hạn bộ lọc; không suy source/forward semantics bất biến chỉ từ schema hoặc log resume. |
| Độ lặp lại qua các lần train | Chưa có bộ nhiều seed độc lập được xác minh cho từng cấu hình | Không dùng p-value trên nhiều câu để thay đánh giá độ biến thiên qua các lần train. |

## 8. Phần còn thiếu và việc tiếp tục

| Còn thiếu | Cần bằng chứng nào | Trạng thái sau bổ sung §10 |
|---|---|---|
| Train/val/test IDs thực sự dùng ở từng phase baseline/MRF, và hash split khi SEQ train/resume | Manifest hoặc log có đủ ID/hash tại thời điểm chạy; hoặc tái dựng lịch sử với giả định/phạm vi được khai báo. | **Một phần, bằng chứng đã mạnh hơn.** Đã kiểm phần validation được log và các phép khớp của manifest MRF tái dựng. Chưa đóng toàn bộ lịch sử bằng năm tag hoặc metadata file. |
| Split/history gốc của Piper | Manifest/log của tác giả hoặc hồ sơ tương đương; nếu không có, giữ giới hạn. | **Chưa thu hồi.** Harvard không trùng LJSpeech hiện có là kết quả kiểm dữ liệu hữu ích, chưa trung hòa mọi nguy cơ về dữ liệu/selection lịch sử. |
| Code/commit và thay đổi recipe tại từng giai đoạn | Snapshot/source hash hoặc lịch sử thay đổi gắn với run. | **Đã khôi phục phần có trong bốn transcript (H2 + L), mục 11.** Có hash các phiên bản văn bản và kết quả sync; chưa có hash runtime hoặc bảo đảm bao phủ mọi thay đổi ngoài transcript. |
| Hash checkpoint trung gian dùng khi resume | Bản file còn giữ đúng thời điểm hoặc hash đã ghi trong log. | **Đã kết thúc tìm trong phạm vi ghi ở mục 11; chưa thu hồi.** Nhóm cho biết có thể đã bị ghi đè, chưa xác nhận chắc chắn. Giữ tên checkpoint/epoch từ log, để hash lịch sử là unavailable; không thay bằng hash `last.ckpt` cuối run. |
| Nguồn MRF ONNX, hồ sơ export/calibration SEQ | Liên kết checkpoint → export → PTQ có bằng chứng. | **Q05 đã tạo bốn FP32 mới có nguồn và kiểm tương thích**, xem [hồ sơ Q05](Q05_EXPORT_ONNX_VA_KIEM_CHUNG.md). Graph MRF cũ khác trọng số checkpoint 1386; nguồn chính xác graph cũ và calibration/PTQ lịch sử vẫn chưa được khôi phục. Q06 xử lý PTQ mới. |

**Không mặc định yêu cầu train lại.** Hồ sơ này xác định rõ phần đã kiểm soát và phần chưa kiểm soát để Q02 chọn mức kết luận phù hợp, Q04 khóa đánh giá và Q10 quyết định có cần ablation bổ sung hay không. Q02 vẫn chưa chốt.

## 9. File bằng chứng và tái kiểm tra

- [Q03_EVIDENCE.json](evidence/q03_2026-10-04/Q03_EVIDENCE.json): hparams, callback, optimizer/scheduler, state schema, chín phiên train, trích log/code và kiểm manifest.
- [FINAL_RUN_CALLBACKS.json](evidence/q03_2026-10-04/FINAL_RUN_CALLBACKS.json): metadata/hash của ba `last.ckpt` quan sát thêm trong Q03.
- [audit_q03.py](evidence/q03_2026-10-04/audit_q03.py): quy trình đối chiếu từ gói Q01, không chạy model.
- [inspect_last_checkpoints.py](evidence/q03_2026-10-04/inspect_last_checkpoints.py): đọc checkpoint cuối run hiện có, không sửa archive Q01.
- [Báo cáo Q01](evidence/q01_2026-10-04/BAO_CAO_Q01.md): vị trí raw source/checkpoint và checksum gốc.
- [Gói bổ sung q03_bosung_2026-10-04](evidence/q03_bosung_2026-10-04/SHA256SUMS): bằng chứng cho §10, có checksum.
- [Biên bản rà soát bổ sung](evidence/q03_review_2026-10-04/REVIEW_CHECKS.json): kiểm checksum gói bổ sung và tính lại các phép giao tập; giữ nguyên các file bằng chứng gốc.

**Trạng thái tổng thể Q03: hoàn thành một phần.** Bảng cấu hình/checkpoint đã đối chiếu; lịch sử dữ liệu/source có thêm bằng chứng nhưng chưa được xác minh đầy đủ.

<a id="q03-bo-sung"></a>

## 10. Bổ sung 04/10/2026: bằng chứng mới và giới hạn sau rà soát

Gói bổ sung gồm TensorBoard (**T**), metadata filesystem (**F**), timeline công cụ (**H**) và manifest MRF tái dựng (**R**). **Cả bảy file trong SHA256SUMS khớp checksum.** Lượt rà soát tính lại giao tập và đọc lại header/tag TensorBoard; không dựng model hoặc chạy lại thực nghiệm.

Các checksum này xác nhận byte của **gói bổ sung hiện có**, không chứng minh tự thân rằng mọi kết luận lịch sử trong phần mô tả của script đều đúng. Các file gốc được giữ nguyên; biên bản rà soát ghi riêng các giới hạn diễn giải.

### 10.1. Baseline: năm câu validation được log khớp canonical

Code đã lưu chọn `Subset(val_dataset, range(num_audio_samples))`, với hparams `num_audio_samples=5`; `_log_audio_samples` dùng `utt.text` làm tag. Trong `validation_step`, hàm này được gọi theo **batch validation**, không phải chỉ một lần mỗi epoch. Vì vậy số nhóm tag dưới đây không được đọc thành số epoch hay số bộ validation độc lập.

| Phiên | Số nhóm năm tag parser trích được | Thời gian tag (UTC) | Kết quả đối chiếu |
|---|---:|---|---|
| `version_0` | 695 | 26/08 17:44 → 27/08 01:31 | Một chuỗi duy nhất, khớp canonical `val[0..4]` |
| `version_1` | 5.469 | 27/08 03:59 → 29/08 15:36 | Cùng chuỗi |
| `version_2` | 4.405 | 31/08 21:08 → 02/09 21:31 | Cùng chuỗi |

Năm tag ánh xạ duy nhất trong JSONL hiện có tới `LJ046-0183`, `LJ037-0232`, `LJ015-0080`, `LJ011-0255`, `LJ034-0041`. Đây là **bằng chứng trực tiếp cho phần validation được log** và hỗ trợ mạnh giả thuyết canonical tái hiện split baseline.

Tuy nhiên, **hai hoán vị có thể trùng năm vị trí mà khác các vị trí còn lại**. Không dùng lập luận “khớp năm câu nên toàn bộ hoán vị giống nhau”, hoặc xác suất của hai hoán vị độc lập để khẳng định chắc lịch sử này: các bản tái dựng cùng phụ thuộc seed/source, chưa có giả định độc lập được xác lập. Việc năm tag ổn định qua các phiên cũng chưa ghi lại 95 validation ID còn lại hay toàn bộ train/test từng phiên.

**Giới hạn parser:** `tb_audio_tags.py` bỏ record nhỏ hơn 4.096 byte, đọc tối đa 2.048 byte đầu record lớn, không kiểm CRC, và nhóm tag bằng thứ tự xuất hiện/lặp tag; không dùng `global_step` để nhận dạng lần validation. Kết quả xác nhận các tag trích được, không phải kiểm toàn bộ nội dung event hoặc mọi đầu ra audio. Giữ giá trị bằng chứng này nhưng không mô tả thành manifest đầy đủ.

### 10.2. MRF: manifest tái dựng khớp validation head và test manifest cũ

Ba phiên MRF mỗi phiên có 3.502 nhóm năm tag trích được, cùng một chuỗi: `LJ019-0109`, `LJ046-0041`, `LJ037-0063`, `LJ010-0040`, `LJ005-0063`. Các ID này thuộc train của canonical hiện tại, cho thấy phần validation được log khác baseline/SEQ.

Script bổ sung dùng hparams checkpoint 1386, `manual_seed(1234)`, **code model hiện tại**, rồi gọi `random_split` theo RNG chung nhằm mô phỏng thủ tục trước 19/09. Kết quả đã lưu có các phép khớp sau, được tính lại khi rà soát:

| Kiểm tra trên bản tái dựng | Kết quả |
|---|---|
| `val[0..4]` khớp năm tag TensorBoard MRF, đúng thứ tự | **Đúng** |
| Test khớp `eval_true_testset_500_mrf.json`, đúng thứ tự 500 ID | **Đúng** |
| Kích thước 12.500 / 100 / 500; không trùng ID trong tập; không giao giữa tập; cùng toàn bộ ID với dataset hiện tại | **Đúng** |
| Train tái dựng ∩ train canonical | 11.930 / 12.500 (**95,44%**) |
| Canonical test nằm trong train / val / test tái dựng | **473 / 6 / 21** |
| Hash trường `split`, theo `json.dumps(split, sort_keys=True)` | `f002d251b9a1be866d876da12bbac987651c7d12c39e0773d83369d7e38d4d69` |
| Hash toàn file chứa procedure/checks/split | `aea18304bb060613dadfaf7154ec4ecc57037fda148798ea07f3290757c8d277` |

Đây là một bản tái dựng có đối chiếu mạnh. Nhưng hai phép khớp có **nguồn bằng chứng khác mức**: TensorBoard được ghi trong lúc train chỉ cho thấy năm mẫu validation; manifest test cũ cũng là kết quả tái dựng theo hparams checkpoint 836. Không gọi chúng là hai bản ghi train-time đầy đủ và độc lập. Source/thư viện/RNG lịch sử vẫn là giả định cần công khai.

Kiểm script còn cho thấy các boolean `checks` chỉ được ghi ra JSON; script **không tự dừng khi một phép khớp trả về false**. Ngoài ra nó chọn `distinct_tag_sequences[0]` khi so các phiên, không tự kiểm hết mọi sequence. Gói hiện có thực sự chỉ ghi một sequence mỗi phiên và các phép khớp hiện có đều đúng; điều đó không biến script thành bộ kiểm có cơ chế chặn mọi kết quả sai.

Hệ quả: không dùng canonical-500 làm **held-out test đã xác minh cho MRF**. Theo bản tái dựng, 473 câu thuộc train MRF. Vẫn có thể báo cáo một phép chấm trên tập này như kết quả thăm dò với nhãn phù hợp; không diễn giải như đo tổng quát hóa trên dữ liệu chưa train. Các bảng dùng hai manifest 500 khác nhau không phải so sánh ghép cặp toàn bộ 500 câu.

**Ảnh hưởng trực tiếp tới Q04:** các danh sách ứng viên **v0.1** cho calibration-60 gồm **57 train / 3 val / 0 test MRF**, selection-98 gồm **96 train / 2 val / 0 test MRF**. Đã xử lý bằng phiên bản v0.2: 60 calibration thuộc train của cả hai split đang đối chiếu; selection riêng 98 câu cho baseline/SEQ và 94 câu cho MRF. Validation giao nhau chỉ có 2 ID, nên không gọi đây là một tập validation chung. Xem [Q04 mục 3](Q04_PROTOCOL_DU_LIEU_VA_DANH_GIA.md). Giữ nguyên file v0.1 và mọi run cũ; v0.2 vẫn phụ thuộc giới hạn provenance của canonical/bản tái dựng.

### 10.3. SEQ và dữ liệu: metadata hỗ trợ lịch sử, không thay hash lúc chạy

| File | Birth (UTC, hiển thị đến giây) | Ctime (UTC) | Mtime (UTC) |
|---|---|---|---|
| `canonical_split.json` | 19/09 21:55:06 | 19/09 21:55:06 | 19/09 21:55:06 |
| `dataset.jsonl` | 28/07 13:43:07 | 28/07 13:43:07 | 28/07 13:42:46 |
| 26.200 file cache `.pt`/`.spec.pt` | — | 28/07 13:43 → 13:44 | — |

File metadata ghi birth ở độ phân giải giây, còn ctime/mtime có phần lẻ; không nói ba timestamp bằng nhau tuyệt đối. Log SEQ của cả ba phiên ghi đọc canonical, và TensorBoard khớp năm validation tag canonical. Các dữ kiện này **phù hợp với việc cùng manifest đã được dùng xuyên suốt SEQ** và mạnh hơn chỉ nhìn file hiện tại.

Tuy vậy, không có hash manifest được ghi ngay ở ba lần đọc. Giữ hash hiện tại là **hash quan sát hiện tại**, không ghi thành hash đo lúc train. Metadata cache chỉ có số file và khoảng ctime, không có kiểm hash từng file ở mỗi run. Do đó thay câu “byte dữ liệu/tiền xử lý giống hệt ở mọi run” bằng “các run tham chiếu cùng dataset/cache; metadata hiện có phù hợp với việc tái sử dụng dữ liệu đã chuẩn bị trước khi train”.

Q04 còn phát hiện canonical validation `LJ036-0001` trùng nội dung/phoneme IDs với 11 ID train. Việc không giao ID giữa các partition không loại trừ trùng nội dung. Phát hiện này phải được giữ trong mô tả dữ liệu dù cuối cùng chấp nhận giả thuyết provenance canonical.

### 10.4. Timeline code và checkpoint resume: bằng chứng có phạm vi

Nguồn: [TRANSCRIPT_TIMELINE_TRAINING_WINDOWS.txt](evidence/q03_bosung_2026-10-04/TRANSCRIPT_TIMELINE_TRAINING_WINDOWS.txt), đọc cùng bộ lọc `scan_transcripts.py`.

| Khoảng/mốc | Thao tác được timeline ghi nhận | Diễn giải phù hợp |
|---|---|---|
| Baseline v0 → v1 | Không thấy edit code liên quan trong bản trích | Không phải chứng minh source không đổi bằng hash. Gradient clipping đổi theo hparams đã xác minh. |
| Trước baseline v2, 31/08 | Edit `flows.py` về clamp 10; edit `synthesizer.py` thêm chặn `logw` trước resume checkpoint 870 | Phù hợp thay đổi ổn định số; không thêm tensor tham số vẫn có thể đổi forward/training behavior. |
| Các phiên MRF, 10–17/09 | Có edit/sync `export_onnx.py` ngày 14/09; không thấy edit training khác trong bản trích | Giới hạn theo các yêu cầu công cụ được bộ lọc giữ lại. |
| 19/09 | Edit loader sang generator cố định/canonical | Sau các run baseline/MRF và trước SEQ; không gán loader mới ngược cho run cũ. |
| Trước SEQ, 23/09 16:47–16:48 | Edit `duration_predictor.py`, nhánh bỏ qua gradient không hữu hạn, kiểm SDP và cách log `val_loss_mel` | Đây là khác biệt recipe/ổn định số cần công khai. Log SEQ có thông báo bỏ qua update, củng cố bằng chứng nhánh đó đã được dùng. |
| Các lần resume SEQ | Không thấy edit training thêm trong bản trích | Không khẳng định source bất biến ngoài phạm vi transcript. |

**Giới hạn bộ lọc:** script đọc `tool_use`, không ghép `tool_result` để kiểm thành công; snippet bị cắt 160/200/260 ký tự. Trường hợp SYNC còn có lệnh `cat` đọc script, không phải mọi dòng SYNC đều ghi file. Bộ lọc không phải bộ dò tổng quát cho mọi `sed -i`, `tee`, ghi file bằng Python hoặc toàn bộ sửa đổi bên ngoài transcript. Vì vậy không giữ câu “đã kiểm mọi lần sửa/sync và không có chỉnh sửa trực tiếp khác” chỉ từ gói này.

**Đã bổ sung cách kiểm tốt hơn ở mục 11:** ghép yêu cầu với kết quả công cụ và khôi phục source đầy đủ cho các thao tác được nhận diện. Gói timeline cũ vẫn giữ nguyên để truy nguồn; giới hạn bao phủ mọi thay đổi bên ngoài vẫn còn.

`FILE_TIMESTAMPS.json` ghi một phép tìm bằng `find / -xdev`: phạm vi không bao gồm mọi filesystem gắn ngoài, ổ Windows hay bản sao ở máy khác. Hiện **chưa thu hồi** hash/file checkpoint 89/870 và byte của `last.ckpt` ở thời điểm resume. Log `Restored all states` hỗ trợ việc resume thành công nhưng không chứng minh forward semantics, RNG hoặc split không đổi; thay đổi clamp là ví dụ cụ thể không nhất thiết đổi state-key/shape.

### 10.5. Piper và Harvard: giữ đúng phạm vi kết luận

Chưa thu hồi manifest train/validation gốc của checkpoint Piper. Harvard-720 đã được chấm trên bốn model và không trùng câu với 13.100 transcript LJSpeech **đang lưu**, theo các phép chuẩn hóa đã ghi trong Q04. Script build Harvard còn có kiểm giao cụm sáu từ theo chuẩn hóa riêng; đây vẫn là phép kiểm trên corpus được cung cấp, không phải hồ sơ toàn bộ dữ liệu train.

Không suy từ kết quả đó rằng **mọi model chắc chắn chưa từng thấy Harvard**, hoặc nguy cơ bias đã được trung hòa. Còn thiếu lịch sử đầy đủ của Piper và việc điểm Harvard đã ảnh hưởng quyết định chọn cấu hình hay chưa. Theo yêu cầu của nhóm, giữ cả LJSpeech-500 và Harvard-720 trong paper, thành hai khối kết quả có giới hạn rõ; không loại bỏ yêu cầu báo cáo bộ 500.

### 10.6. Cách viết phù hợp sau rà soát

- Cấu hình và schema backbone dùng chung đã được xác minh; activation/MRD/clipping và các sửa đổi ổn định số phải được khai báo riêng.
- Log validation hỗ trợ canonical cho baseline/SEQ. Split MRF được tái dựng, khớp phần validation được log và test manifest cũ; giữ nhãn **reconstructed and corroborated**, không gọi là đầy đủ manifest đã khóa tại thời điểm train.
- Chỉ báo các số **11.930** và **473/6/21** kèm nguồn “the reconstructed MRF split”. Không đổi chúng thành một phép đo trực tiếp toàn lịch sử train.
- Không khẳng định mọi run có dữ liệu giống byte, source bất biến, hoặc Harvard hoàn toàn độc lập khi thiếu bằng chứng tương ứng.
- Chưa đủ để đóng toàn bộ Q03; vẫn có thể viết các phần cấu hình/kiến trúc và báo kết quả cấp hệ thống với giới hạn minh bạch. Những thiếu hụt này không tự dẫn tới yêu cầu train lại.

## 11. Hoàn tất phần khôi phục khả thi ngày 04/10/2026

### 11.1. Source đã khôi phục và đối chiếu

Đọc bốn transcript cấp chính của dự án cùng backup liên quan, **không thực thi lại lệnh hoặc source tìm thấy**. Trong 153 thao tác sửa/ghi Python được nhận diện trước 02/10, 152 thao tác có kết quả phù hợp và khôi phục được văn bản sau sửa; một thao tác còn lại thiếu kết quả hoặc báo lỗi, không được coi là đã sửa thành công. Không có thao tác thành công trong phạm vi này còn thiếu văn bản sau sửa.

- Đã kiểm hash và parse cú pháp **218 đối tượng source** trước/sau sửa hoặc backup; không gặp lỗi cú pháp. Đây không phải 218 file runtime riêng biệt hoặc kiểm functional correctness.
- Không có bất đồng ở các điểm so sánh văn bản được replay với `originalFile`/backup kế tiếp. Không diễn giải điều đó thành chứng minh mọi sửa đổi đã được ghi nhận.
- **207 yêu cầu shell có kết quả đi kèm** được giữ để đọc lại. Chúng gồm cả lệnh đọc; không phải 207 lần sync thành công. Các kết quả sync tiêu biểu được tách riêng trong biên bản.
- Có danh mục trạng thái editor quan sát gần nhất trước chín mốc launch/resume và diff các thay đổi chính. Đây là **source candidate theo lịch sử editor**, không gán Git HEAD Windows hoặc hash văn bản đó thành hash runtime Linux.

Nguồn: [SOURCE_RECOVERY_SUMMARY.json](evidence/q03_q04_completion_2026-10-04/SOURCE_RECOVERY_SUMMARY.json), [index thao tác và source](evidence/q03_q04_completion_2026-10-04/SOURCE_HISTORY_RECOVERY.json), [script khôi phục](evidence/q03_q04_completion_2026-10-04/recover_source_history.py), [biên nhận kiểm tra cuối lượt](evidence/q03_q04_completion_2026-10-04/COMPLETION_VERIFICATION.json).

| Mốc / file | Source khôi phục và điều được đối chiếu |
|---|---|
| Baseline `training.py` trước ba phiên | Hash UTF-8 `c06b0f7d…2f460`; nhánh `random_split` dùng RNG chung. |
| Baseline v2 `flows.py`, `synthesizer.py` | Hash lần lượt `1f3ac3e8…94c01`, `e643fa10…583d7`; diff xác nhận thay clamp và chặn `logw`. Kết quả lệnh sync 31/08 có các dòng kiểm clamp tương ứng. |
| MRF `training.py` trước ba phiên | Hash `986f7835…1309c5`; vẫn dùng `random_split` theo RNG chung. Không gán cơ chế canonical thêm sau đó cho run MRF này. |
| Loader sửa ngày 19/09 | Kết quả `cp training.py && echo OK` được ghép thành công. Diff có nhánh đọc canonical và generator seed riêng khi không có canonical. |
| SEQ `training.py` trước ba phiên | Hash `633cf472…43fce`; có canonical loader, bỏ update khi gradient không hữu hạn và kiểm sức khỏe inference trước log monitor. Lệnh sync 23/09 có output kiểm cú pháp `SYNTAX OK`. |

Hash đầy đủ và đường dẫn source tương ứng có trong JSON; hash trên là **hash văn bản khôi phục UTF-8**, không phải chứng nhận byte/line ending lúc train. Một số lệnh gồm nhiều `cp` rồi `echo OK` không dùng cơ chế dừng khi một lệnh trước thất bại; không lấy riêng `OK` để xác minh từng phép copy. Các checkpoint/log đã đối chiếu vẫn là nguồn chính để xác nhận cấu hình thực tế.

### 11.2. Checkpoint trung gian: đã tìm, chưa thu hồi được

Đã lưu [phạm vi tìm Windows](evidence/q03_q04_completion_2026-10-04/WINDOWS_CHECKPOINT_SEARCH.json) và [phạm vi tìm WSL](evidence/q03_q04_completion_2026-10-04/WSL_CHECKPOINT_SEARCH.json), có thư mục gốc, phần loại trừ và lỗi truy cập. Không tìm thấy file có tên checkpoint epoch 89/870 trong phạm vi đó. Các `last.ckpt` hiện có là nội dung quan sát hiện tại; ba file cuối của run chính đã được đối chiếu ở Q03 trước đó. Archive `banhmi_head.tar` tìm thấy có 73 mục, không có `.ckpt`; đã đọc danh mục và lưu source liên quan, không giải nén/chạy dự án từ archive.

Ngày 04/10 nhóm trả lời: “hầu như chắc đã được ghi đè vào last của nó rồi thì phải”. Ghi nhận đây là **khả năng theo trí nhớ của nhóm**, không phải xác nhận đã mất mọi bản sao. `last.ckpt` thường là tên được cập nhật; checkpoint tên `best-epoch=89…`/`870…` là các file riêng, nên chưa kết luận chính xác chúng bị ghi đè hay bị callback dọn đi.

**Cách đóng phần việc này:** giữ checkpoint/epoch resume và việc khôi phục trạng thái theo log; ghi hash nội dung trung gian là **not recoverable from the inspected records**. Không tiếp tục tìm vô hạn trên cùng nguồn hoặc bịa hash từ file cuối. Nếu sau này có bản sao mới thì bổ sung, không coi tìm lại file đã mất là điều kiện để viết phần cấu hình hiện có.

### 11.3. Phần đã xử lý và phần không thể xác nhận thay nhóm

| Hạng mục | Kết quả sau lượt này |
|---|---|
| Khôi phục source, kiểm hash/cú pháp và ghép kết quả công cụ trong phạm vi đã chọn | **Hoàn tất.** Có index, source, diff và biên bản. |
| Tìm checkpoint trung gian còn lưu tại các nơi đã kiểm | **Hoàn tất lượt tìm.** Hash lịch sử vẫn không thu hồi được; ghi rõ giới hạn thay vì đánh dấu đã xác minh. |
| Sửa ứng viên calibration/selection Q04 theo split MRF tái dựng | **Hoàn tất tạo và kiểm v0.2.** 60 train chung; validation theo model 98/94. Chưa chạy PTQ. |
| Full manifest/hash lúc từng phase chạy; split Piper gốc | **Chưa có đủ bản ghi lịch sử.** Giữ mức bằng chứng hiện có và giới hạn trong paper. |
| Lịch sử dùng Harvard để ra quyết định | **Chưa xác nhận.** Không suy từ mong muốn báo cáo cả hai tập. |
| Đóng toàn bộ Q03 | Phụ thuộc Q02 chốt phạm vi **so sánh hệ thống** với các giới hạn lịch sử đã khai báo, hoặc có thêm bằng chứng nếu muốn tuyên bố mạnh hơn. Không gọi đây là decoder-only ablation. |

**Câu tiếng Anh có thể dùng trong Experimental Setup:**

> Training configurations and selected checkpoints were audited using checkpoint metadata, training logs, TensorBoard records, and recovered source-edit history. The reconstructed MRF partition was corroborated against logged validation examples and the previously retained test manifest. Complete contemporaneous split hashes and intermediate resume-checkpoint hashes were not recoverable from the inspected records. Differences in training data, activation functions, discriminator configurations, and numerical-stability procedures are therefore disclosed, and the comparison is interpreted at the system level rather than as an isolated decoder ablation.
