# Checklist tiến độ sửa luận văn

Cập nhật: **29/09/2026 — lượt 7**. Bản hiện hành: [AIP491_GFA26AI18_Luan_van_PTQ_v7_2026-09-29.docx](AIP491_GFA26AI18_Luan_van_PTQ_v7_2026-09-29.docx). Đã mở bằng Microsoft Word và cập nhật mục lục: **54 trang, 28 bảng**.

**Chưa hoàn tất toàn bộ luận văn.** Nhóm đã quyết định dừng phát triển F0 và Vocos, giữ MobileNet/MRF và PTQ. Ma trận hiện hành là baseline D0 và MobileNet/MRF D1, mỗi kiến trúc có suy luận FP32 và PTQ; dự kiến benchmark trên **Intel N150 và Raspberry Pi 5**. Đây là **4 điều kiện mô hình × 2 thiết bị = 8 ô benchmark**, chưa phải kết quả đã đo. Nguồn Linux đã xác minh ở lượt 3 vẫn là bằng chứng lịch sử; không được đổi nhãn thành kết quả của ma trận mới.

## 1. Đã sửa đến phiên bản nào?

| Phiên bản | Việc đã thực hiện | Trạng thái |
|---|---|---|
| Bản gốc trong Downloads | Đọc và đối chiếu chủ yếu với BanhmiTTS | Giữ nguyên |
| Bản PTQ — lượt 1 | Thống nhất FP32 → PTQ; bỏ QAT khỏi phương pháp, bảng và kế hoạch | Đã tạo file Word |
| Bản PTQ v2 — lượt 2 | Sửa kiến trúc, dữ liệu, huấn luyện, diễn giải kết quả và nội dung bản thảo; thêm phụ lục bằng chứng | Đã tạo và kiểm tra Word: 43 trang, 24 bảng |
| Bản PTQ v3 — lượt 3 | Đối chiếu WSL: checkpoint, CSV, outlier, split tái dựng và F0 | Đã kiểm tra Word: 44 trang, 24 bảng |
| Bản PTQ v4 — lượt 4 | Dừng F0/Vocos; giữ baseline/MRF × FP32/PTQ; kế hoạch N150/Pi 5 | Đã kiểm tra Word: 50 trang, 27 bảng |
| Bản PTQ v5 — lượt 5 | Truy checkpoint/export và kết quả BanhmiTTS; phát hiện sai nguồn MRF, giao calibration/test; kiểm cache/loss F0 lịch sử và tài liệu tham khảo | Đã kiểm tra Word: 53 trang, 28 bảng |
| Bản PTQ v6 — lượt 6 | Tách code MRF, checkpoint/PyTorch đã chạy với bản ONNX khớp checkpoint còn chờ; bổ sung E.7 | Đã kiểm tra Word: 53 trang, 28 bảng |
| Bản PTQ v7 — lượt 7 | Đọc calibration/evaluation; xác định lỗi sample rate ở chọn scope, khác scorer và thiếu split/RNG; bổ sung E.8 và protocol | Đã kiểm tra Word: 54 trang, 28 bảng |

Lượt 2 chỉnh **111 đoạn**; lượt 3 cập nhật **43 đoạn**; lượt 4 cập nhật **151 đoạn** và phụ lục F. Lượt 5 thay **45 đoạn**, bổ sung bảng VI.4 và phụ lục E.5–E.6. Đây là khối lượng sửa tài liệu, không phải tỷ lệ hoàn thành nghiên cứu. Các bản trước, bản gốc và mã nguồn được giữ nguyên.

## 2. Checklist phần đã sửa trong tài liệu

Dấu **[x]** dưới đây chỉ xác nhận thao tác chỉnh nội dung đã hoàn thành. Mã F01–F16 đối chiếu với báo cáo đánh giá ban đầu; công việc thực nghiệm hoặc xác minh còn lại nằm ở mục 3.

- [x] **F01 — Thống nhất PTQ.** Đã bỏ QAT ở lượt 1. Theo phạm vi lượt 4, chọn checkpoint có hồ sơ huấn luyện, thiết lập suy luận FP32 rồi export và PTQ không huấn luyện thêm. **FP32 trong ma trận là precision suy luận, không phải yêu cầu huấn luyện lại từ đầu bằng FP32.** Precision `bf16-mixed` của run lịch sử được giữ đúng như nguồn. Word đã đồng bộ cách hiểu này từ lượt 4.
- [x] **F02 — Backbone và VITS2.** Phân biệt BanhmiTTS Windows có sẵn VITS2 với nhánh Linux lịch sử có cờ bật/tắt; baseline lịch sử vẫn có Snake trong residual block. Ma trận bốn kiến trúc D0/D1/D2/Complete ở phiên bản trước đã được thay thế theo quyết định lượt 4. Cần chốt các thành phần backbone/activation/loss dùng chung cho baseline D0 và MRF D1 ở C03; không mặc định yêu cầu triển khai mọi cờ “Off” của kế hoạch cũ.
- [x] **F03 — So sánh decoder.** Đã ghi rõ Vocos thay cả discriminator và loss; các config sẵn có không chứng minh phép thử chỉ đổi decoder. **Vocos: dừng theo phạm vi lượt 4**, giữ mô tả lịch sử cần thiết. So sánh decoder còn thực hiện là baseline với MobileNet/MRF, thuộc C03/C05.
- [x] **F04 — Đường F0 vào decoder.** Nhánh BanhmiTTS Windows nối conditioning vào Vocos, chưa nối MobileNet/MRF; nhánh Linux lịch sử có conditioning vào decoder tích chập và cộng loss F0. Không gán bằng chứng lịch sử sang decoder mới. Vị trí: IV.4, V.1–2, phụ lục B/E.4. C04 đã hoàn thành phạm vi xác minh lịch sử ở lượt 5; giới hạn nguồn/split còn tại C01/C02.
- [x] **F05 — Trạng thái cache F0.** Phân biệt snapshot Windows không có F0 với manifest Linux có 13.100 trường F0 và đủ 13.100 file tồn tại. Sự tồn tại file chưa chứng minh tensor hợp lệ hoặc mọi lần train đã dùng đúng cache. Vị trí: IV.3.3, V.2.1, phụ lục A/E.4. C04 đã hoàn thành phạm vi xác minh lịch sử ở lượt 5; giới hạn nguồn/split còn tại C01/C02.
- [x] **F06 — VAD.** Sửa mô tả pipeline hiện hành: không có bước cắt im lặng bằng Silero VAD. Vị trí: IV.2.4, phụ lục A. Không mặc định yêu cầu bổ sung VAD.
- [x] **F07 — Hz và log-F0.** Sửa đơn vị cache, mục tiêu log(mean Hz) theo phoneme và khác biệt training/inference. Phân biệt training của decoder Linux dùng F0 ground-truth theo frame với nhánh Windows đã mô tả trước đó. Nội dung này chỉ phục vụ giải thích cấu hình lịch sử nếu còn được báo cáo; không phải cam kết phát triển hoặc đánh giá nhánh F0 mới. C04 đã hoàn thành phạm vi xác minh lịch sử đã giới hạn ở lượt 5.
- [x] **F08 — Test set và kiểm định.** Lượt 3 xác nhận hai CSV lịch sử ghép cặp đúng cùng 500 văn bản; tính lại Wilcoxon khớp p-value. Split tái dựng không trùng train/validation nhưng chưa xác minh toàn bộ lịch sử code và các lần resume. Vị trí: VI.1.3–1.4, phụ lục D/E.4. Còn: C01, C02.
- [x] **F09 — Kết quả lịch sử.** Thay nhãn “chưa tìm được nguồn” của v2 bằng nguồn checkpoint và CSV Linux đã đối chiếu; trung bình WER/UTMOS/RTF khớp số báo cáo. Giữ giới hạn về scorer, điều kiện đo và lịch sử chạy. Vị trí: bảng VI.2–VI.3, chương VI–VII, phụ lục E.4. Còn: C01.
- [x] **F10 — Số tham số.** Làm rõ 18M/25,2M dùng khác phạm vi. Generator baseline/full: 23,666480M/25,208817M; ước tính đường suy luận: 15,664667M/17,207004M sau các loại trừ đã nêu. Không coi ước tính đường suy luận là kích thước ONNX hoặc bộ nhớ runtime. Vị trí: V.3.3, phụ lục B.3/E.4. Còn: C01, C03, C06.
- [x] **F11 — Chuyển đổi và triển khai.** Theo quyết định mới của nhóm, nội dung chuyển sang PTQ; tách export khỏi bằng chứng thực thi INT8, nêu scope và FP32 fallback. Vị trí: IV.4, V.2–3, phụ lục C–D. Còn: C06.
- [x] **F12 — Cấu hình D1/D2 ở kế hoạch cũ.** Đã phân biệt MRF song song với MB tuần tự và Vocos-Small lịch sử với YAML 512/1536/8. Theo lượt 4, giữ D1 MobileNet/MRF; **D2 Vocos: dừng theo phạm vi**. Chỉ còn chốt cấu hình baseline/MRF thực sự được dùng ở C03, không yêu cầu hoàn thiện D2.
- [x] **F13 — Training, cap và checkpoint.** Phân biệt cấu hình Windows với hai run Linux: Linux dùng cap 400, precision `bf16-mixed`, checkpoint mỗi epoch và `grad_clip=1.0`; các giá trị này không tự áp dụng cho ma trận mới. Giữ giới hạn kiểm tra cache và chính sách clipping. Vị trí: IV.2.5, IV.4, phụ lục A/C/E.4. Còn: C05.
- [x] **F14 — Dữ liệu và outlier.** Xác minh 13.100 mẫu, khoảng 23,92 giờ và 203 outlier theo Tukey 1,5×IQR của số ký tự nguyên văn/thời lượng toàn clip. Cả 203 vẫn có trong manifest Linux và Windows đã kiểm tra; không mô tả là đã loại. Vị trí: I.3, IV.2, phụ lục A/E.4. Còn: C02 về lịch sử split từng run.
- [x] **F15 — Thử nghiệm cũ và ma trận mới.** Gắn cặp 773/1079 với nhánh Linux convolutional và training `bf16-mixed`; tách lần đo lại 1079 khỏi bảng số lịch sử. Không gọi đó là kết quả Vocos/MobileNet hay ma trận FP32/PTQ mới. Vị trí: IV.6, VI.1–2, phụ lục D/E.4. Còn: C01, C05, C06.
- [x] **F16 — Hoàn thiện bản thảo.** Thống nhất tên đề tài, thay lời cảm ơn mẫu, bỏ mục tùy chọn trống, đồng bộ lịch 12 tuần; ghi thông tin Jira/hardware chưa đủ bằng chứng. C09 đã hoàn thành rà soát tài liệu tham khảo ở lượt 5; còn xác minh quản lý tại C08.
- [x] **Kiểm tra file v2.** Word mở không cần sửa lỗi file; mục lục được cập nhật; đã kiểm tra các trang chứa bảng kiến trúc, kết quả và phụ lục bằng chứng. Giữ nguyên các ô số liệu lịch sử và hình ảnh.

## 3. Checklist xác minh và thực nghiệm — trạng thái hiện hành

Các mục này không đồng nghĩa câu chữ hiện tại vẫn sai. Chúng là điều kiện để thay trạng thái “dự kiến/chưa xác minh” bằng mô tả hoặc kết luận có bằng chứng. **“Dừng theo phạm vi” khác với “hoàn thành thực nghiệm”**: không đánh dấu [x] cho một nhánh chỉ vì nhóm quyết định dừng phát triển.

- [ ] **C01 — Truy nguồn kết quả lịch sử — hoàn thành một phần.** Đã xác minh checkpoint/hash, số liệu CSV, cùng 500 câu, toàn bộ tên/shape state entries (825 baseline, 994 enhanced), log resume version_0 → version_2 và ranh giới timer. Callback best-score reset khi đổi thư mục nên không gọi checkpoint 1079 là tốt nhất toàn bộ lịch sử. **Còn:** scorer/CLI/package và điều kiện đo của đúng cặp 773/1079; source tại thời điểm chạy. Không gán log của lần đo khác hoặc môi trường hiện nay thành bằng chứng của lần cũ. Nguồn: `round5_history_provenance.md/.json`.
- [ ] **C02 — Split và 203 outlier — hoàn thành một phần.** Giữ kết quả lượt 3: 203 ID được giữ; split hiện tại tái dựng không trùng, 500 output ghép cặp đúng. Log train/resume bổ sung seed và số lượng nhưng không có source/manifest hash hoặc split-ID từng giai đoạn. Với baseline/MRF ở `<WSL_PROJECT_HOME>/`, hai test manifest chỉ giao 21/500; canonical hiện có khớp baseline, chưa chứng minh split MRF lịch sử. **Còn:** provenance đủ cho tuyên bố held-out của phép so sánh cuối. Không suy việc MRF đã train trên một mẫu từ membership của canonical khác. Lượt 7 kiểm thêm canonical 12.500/100/500 không trùng/giao/thiếu ID; loader đã hỗ trợ canonical, nhưng sweep PTQ bỏ qua cơ chế đó. Tính hợp lệ của manifest hiện tại không đóng phần provenance MRF lịch sử.
- [ ] **C03 — Chốt cấu hình và export baseline/MRF — đã có code, checkpoint và kết quả lịch sử.** Lượt 6 xác nhận implementation MRF, đường export nạp checkpoint chỉ định và log đánh giá PyTorch checkpoint 1386 hoàn tất 500 câu. Không còn coi phần code/huấn luyện/đánh giá checkpoint này là chưa chạy. ONNX MRF hiện có vẫn khác checkpoint 1386; script PTQ chỉ dùng lại ONNX sẵn có, không tự export hoặc kiểm nguồn checkpoint. **Còn:** bản export từ checkpoint cuối được đối chiếu và lưu hash; sau đó calibration/evaluation đúng protocol. Trạng thái bước export khớp 1386 là **chưa có kết quả được xác minh/còn chờ**, không tự chẩn đoán bug exporter hoặc yêu cầu train lại. Những khác biệt Snake/MRD/clipping/split của phép so sánh vẫn phải được xử lý hoặc diễn giải đúng. Nguồn: `round6_mrf_export_code.md/.json`, `round6_mrf_execution_evidence.json`.
- [x] **C04 — Xác minh F0 lịch sử — hoàn thành trong phạm vi tuyên bố còn giữ.** Đã đọc đủ **13.100 tensor**, 7.399.984 giá trị hữu hạn dương; độ dài F0/spec/waveform phù hợp hop 256. Kiểm tra mask padding trong source Linux và **9.323 bản ghi loss_f0 hữu hạn dương** ở version_2, nối với log checkpoint 1079. Không coi đây là số bước duy nhất, độ chính xác pitch, lợi ích nhân quả hoặc bảo đảm cache bất biến mọi thời điểm; phần lịch sử nguồn/split còn ở C01/C02. **Không khôi phục phát triển F0, không đánh dấu nhánh mới là đã hoàn thành.** Nguồn: `round5_f0_audit.md/.json`.
- [ ] **C05 — Đánh giá bốn điều kiện baseline/MRF × FP32/PTQ — đã hoàn tất đọc code, chưa hoàn tất thực nghiệm theo protocol sửa.** Giữ xác minh bốn bộ 500 hàng và 2.000 WER/WAV ở lượt 5. Lượt 7 xác nhận calibration chọn 60 mẫu toàn JSONL, selection 25 mẫu toàn JSONL; chưa thu hồi manifest train-only/validation-only. Tái dựng giao test vẫn là 3 baseline/1 MRF ở calibration và 1 mỗi model ở selection. Phát hiện `eval_wer_cheap.py` đưa ndarray 22.050 Hz vào Whisper không resample về 16.000 Hz, dùng scorer/normalization khác script cuối và gán inference lỗi thành WER 1.0. **Còn:** sửa bước chọn scope, khóa artifact/split hợp lệ, tạo lại PTQ với calibration chỉ từ train, lựa chọn trên validation và đánh giá cuối. Không gọi recipe là tối ưu; không suy mức bias hoặc gradient-training leakage. Không cần mặc định train lại.
- [ ] **C06 — Đánh giá triển khai trên Intel N150 và Raspberry Pi 5 — đã kiểm graph, chưa đo thiết bị.** Bốn ONNX đã được hash và kiểm schema. Mỗi graph PTQ có **133 QLinearConv = enc_p 37 + dp 32 + flow 64**, không có QLinearConv decoder; các initializer decoder đối chiếu giữ nguyên FP32. Bằng chứng graph không chứng minh kernel INT8 thực chạy trên thiết bị. **Còn:** cấu hình máy/RAM/OS/nguồn điện/tản nhiệt, runtime/provider/phiên bản, luồng và tải nền; kiểm thực thi rồi đo đủ **4 điều kiện × 2 thiết bị = 8 ô** với warm-up, số lần lặp, latency/RTF, kích thước và peak memory. Không suy kết quả N150/Pi 5 từ RTF lịch sử.
- [ ] **C07 — Đánh giá chất lượng và khả năng tái lập — đã đối chiếu đường scorer, còn kết quả chạy theo protocol cuối.** Script 500 câu dùng Whisper small/CPU/English, đọc WAV nên có resample đúng; WER là mean theo câu. Không gán lỗi sample rate của cheap evaluator sang script cuối. Đã phân biệt PCM peak-normalized dùng cho ASR với float waveform dùng cho UTMOSv2; đọc cấu hình/weights và random preprocessing hiện có. Python seed lấy mẫu không kiểm soát ONNX RNG; seed chọn checkpoint UTMOS không thay seed NumPy khi predict. **Còn:** thống nhất scorer/normalization/sample rate, lưu version/weight hash/audio thực được chấm và chính sách randomness đã kiểm chứng; chạy lại đánh giá cuối. Hash hiện tại không chứng minh hash của lần chạy cũ. Không yêu cầu thử F0/Vocos mới hoặc tự đổi điểm tự động thành MOS người nghe.
- [ ] **C08 — Xác nhận thông tin quản lý — chưa có nguồn độc lập.** Đã tìm trong các thư mục mã nguồn/tài liệu và kiểm 31 hyperlink Jira (22 URL duy nhất) của Word; chưa có issue export độc lập, chỉ có bản thảo và bản trích dẫn lại. **Còn:** xác nhận phân công, 87 issue, sprint và trạng thái bằng snapshot/export đúng thời điểm hoặc nhóm xác nhận. Không coi link tồn tại là bằng chứng trạng thái. Nguồn: `round5_management_evidence.json`.
- [x] **C09 — Kiểm tra tài liệu tham khảo — hoàn thành phạm vi đối chiếu hiện tại.** Rà 23 mục theo bài gốc, hồ sơ publisher và tài liệu chính thức; tích hợp các sửa định danh/ngữ cảnh và URL. Tukey 1977 được kiểm thông tin xuất bản, công thức 1,5 IQR đối chiếu NIST; không nhận đã đọc toàn bộ sách. Bỏ pagination FastPitch chưa đối chiếu được, dùng bản tác giả xác nhận ICASSP 2021. Citation phần mềm không thay thế version/weight của từng run, còn thuộc C01/C07. Nguồn: `round5_references.md/.json`.

## 4. Lượt 3 — đã xác minh đến đâu?

Các dấu dưới đây là công việc con đã có bằng chứng, giúp theo dõi tiến độ mà không đánh dấu hoàn thành quá sớm toàn bộ C01/C02/C04.

- [x] **L3-01 — Đọc checkpoint gốc Linux.** Xác nhận epoch 773/1079, mel loss 20,2158/19,2943 trong callback, hparams, hash và tensor; mel loss đã nhân `c_mel=45`.
- [x] **L3-02 — Đối chiếu 500 câu và tính lại thống kê.** Hai CSV cùng thứ tự manifest; trung bình khớp; Wilcoxon WER p=0,1146036946 và UTMOS p=1,6496717403×10⁻⁷⁹ khớp chính xác. WER chưa đạt ngưỡng ý nghĩa 0,05.
- [x] **L3-03 — Làm rõ số tham số và kiến trúc lịch sử.** Đếm theo phạm vi nhất quán; phát hiện baseline vẫn có Snake trong residual block, nên không gọi là VITS vanilla nguyên bản hoặc hoàn toàn không Snake.
- [x] **L3-04 — Xác minh 203 outlier.** Tính lại trên toàn bộ WAV, lưu ID; cả 203 được giữ trong manifest 13.100 mẫu. Đây là phép EDA trên toàn clip, không phải bộ lọc VAD.
- [x] **L3-05 — Tái dựng split hiện có.** Test tái dựng khớp đúng 500 ID đánh giá; 0 ID trùng train, 0 ID trùng validation trong phép tái dựng. Chưa suy rộng thành bảo đảm cho mọi lần train/resume lịch sử.
- [x] **L3-06 — Phân biệt F0 Linux và Windows.** Linux có 13.100 file F0 cùng đường conditioning/loss; Windows chưa có F0 cache. Không gán một snapshot cho toàn bộ dự án.
- [x] **L3-07 — Tách các lần đo.** Giữ riêng bộ lịch sử 773/1079, lần đo lại 1079 và kết quả ground-truth; không ghép hàng số liệu từ những lần đo khác nhau rồi giữ p-value cũ.
- [ ] **L3-08 — Chốt nguồn của đúng lần đánh giá — hoàn thành một phần.** Đã so cấu trúc checkpoint và nối log train/resume; còn exact-run scorer/CLI/package và điều kiện timing. Không nhập log của lần đo khác vào cặp lịch sử. Thuộc C01.
- [ ] **L3-09 — Chốt lịch sử split của mọi giai đoạn.** Cần bằng chứng code/manifest lúc bắt đầu và khi resume; phép tái dựng hiện tại chưa đủ. Thuộc C02.
- [x] **L3-10 — Kiểm tra nội dung cache và loss F0 lịch sử.** Hoàn thành theo phạm vi C04 đã giới hạn: toàn bộ tensor F0 hiện có, mask trong source, kích thước và scalar loss của phiên liên quan. Không xác minh chất lượng/gradient hoặc khôi phục phát triển F0 mới.

Nguồn để mở và kiểm tra lại:

| Nội dung | Hồ sơ trong cùng thư mục |
|---|---|
| Checkpoint, hash, hparams, mel loss, tensor | [linux_checkpoint_evidence.json](linux_checkpoint_evidence.json) |
| CSV, trung bình và Wilcoxon tính lại | [history_archived_pair.json](history_archived_pair.json) |
| Dữ liệu Linux, 203 ID và cache F0 | [linux_data_history_audit.json](linux_data_history_audit.json) |
| Phép tái dựng split và giới hạn lịch sử | [linux_split_reconstruction.json](linux_split_reconstruction.json) |
| Kiến trúc lịch sử, Snake, F0 và phạm vi tham số | [linux_architecture_history.md](linux_architecture_history.md) |
| Phân biệt các bộ kết quả và kiểm tra outlier Windows | [history_numeric_sources.md](history_numeric_sources.md), [history_numeric_sources.json](history_numeric_sources.json) |

**C03, C05, C06, C07 và C08 vẫn mở; C01/C02 còn một phần. C04 và C09 đã đóng trong phạm vi ghi rõ ở mục 3.** Bằng chứng lịch sử không hoàn thành thay các thí nghiệm thiết bị.

## 5. Lượt 4 — phạm vi mới và kế hoạch thiết bị

- [x] **L4-01 — Ghi nhận quyết định của nhóm.** Dừng phát triển F0 và Vocos; giữ MobileNet/MRF cùng PTQ. Luận văn ghi nhận các nhánh đã thử rồi dừng; phần nghiên cứu chính tập trung vào baseline/MRF × FP32/PTQ. Chưa có căn cứ để kết luận nguyên nhân kỹ thuật của lỗi; chỉ ghi vấn đề nhóm báo cáo chưa giải quyết được trong phạm vi dự án.
- [x] **L4-02 — Ghi nhận thiết bị dự kiến.** Intel N150 và Raspberry Pi 5 là hai mục tiêu benchmark do nhóm cung cấp. Cấu hình máy cụ thể và kết quả đo còn mở ở C06.
- [x] **L4-03 — Áp dụng phạm vi vào toàn bộ Word v4.** Đồng bộ mục tiêu, phương pháp, bảng cấu hình, kế hoạch, kết luận và phụ lục; giữ nhãn lịch sử của checkpoint 773/1079, không gán số liệu đó cho MRF không F0. Chuyển F0/Vocos sang quá trình phát triển/hạn chế; ghi kế hoạch 8 ô benchmark và các cấu hình thiết bị chưa chốt.
- [x] **L4-04 — Kiểm tra file Word v4 và checklist Word.** Cập nhật mục lục, mở/kiểm tra trình bày, kiểm tra bảng ma trận và các nhãn kết quả lịch sử; đồng bộ checklist Word với file này.

| Điều kiện hiện hành | Intel N150 | Raspberry Pi 5 |
|---|---|---|
| D0 baseline — suy luận FP32 | Dự kiến, chưa đo | Dự kiến, chưa đo |
| D0 baseline — PTQ | Dự kiến, chưa đo | Dự kiến, chưa đo |
| D1 MobileNet/MRF — suy luận FP32 | Dự kiến, chưa đo | Dự kiến, chưa đo |
| D1 MobileNet/MRF — PTQ | Dự kiến, chưa đo | Dự kiến, chưa đo |

Ma trận này thay thế kế hoạch bốn kiến trúc D0/D1/D2/Complete trong các bản trước. Hai trạng thái suy luận của một kiến trúc phải được truy về cùng checkpoint tương ứng để đo tác động của PTQ. Nếu chỉ lượng tử hóa một số module và giữ decoder FP32, phải ghi rõ phạm vi đó trong bảng kết quả.

## 6. Lượt 5 — công việc có bằng chứng đã hoàn thành

**Đóng 2 mục C04/C09 trong phạm vi đã nêu; 7 mục C01/C02/C03/C05/C06/C07/C08 còn mở hoặc hoàn thành một phần.** Không tính việc dừng một nhánh là hoàn thành thực nghiệm.

- [x] **L5-01:** Kiểm toàn bộ tên/shape checkpoint 773/1079; không thiếu/thừa hoặc lệch kích thước.
- [x] **L5-02:** Nối log resume và checkpoint 1079; làm rõ reset metadata chọn checkpoint và timer lịch sử.
- [x] **L5-03:** Đọc 13.100 F0 tensor, xác minh kích thước/mask và 9.323 scalar loss; đóng C04 có giới hạn.
- [x] **L5-04:** Tìm và hash checkpoint baseline1489/MRF1386, bốn ONNX và hồ sơ đánh giá tại `<WSL_PROJECT_HOME>/`.
- [x] **L5-05:** Đối chiếu checkpoint–ONNX; ghi rõ sai nguồn MRF, chưa coi lỗi này đã được khắc phục.
- [x] **L5-06:** Kiểm bốn bộ kết quả lưu, 2.000 WER và WAV header; xác định đúng MRF-PTQ rerun.
- [x] **L5-07:** Đếm operator PTQ và kiểm decoder giữ FP32; chưa thay thế benchmark runtime.
- [x] **L5-08:** Tái dựng calibration/tuning từ lệnh log, ghi các ID giao test và giới hạn lịch sử.
- [x] **L5-09:** Rà 23 tài liệu tham khảo, sửa ngữ cảnh và nguồn dẫn; đóng C09.
- [x] **L5-10:** Tìm nguồn Jira trong phạm vi tài liệu dự án; ghi chưa có export độc lập, C08 vẫn mở.
- [x] **L5-11:** Tích hợp Word v5, cập nhật mục lục, kiểm bảng lịch sử và đồng bộ checklist Word.

| Bằng chứng lượt 5 | File |
|---|---|
| Checkpoint 773/1079, resume, scorer và timer | [round5_history_provenance.md](round5_history_provenance.md) |
| F0 cache/loss lịch sử | [round5_f0_audit.md](round5_f0_audit.md) |
| Baseline/MRF, ONNX, kết quả và lỗi nguồn MRF | [round5_model_artifacts.md](round5_model_artifacts.md) |
| Calibration/tuning giao test | [round5_calibration_overlap.md](round5_calibration_overlap.md) |
| Tài liệu tham khảo | [round5_references.md](round5_references.md) |
| Phạm vi tìm nguồn quản lý | [round5_management_evidence.json](round5_management_evidence.json) |

## 7. Lượt 6 — đọc code và xác nhận trạng thái export MRF

- [x] **L6-01 — Code MRF/export đã có.** Kiểm đường chọn `resblock="mrf"`, nạp checkpoint và export ONNX; không có EMA/đổi checkpoint dự phòng hay ghi lại embedding trong đường FP32 thông thường. Đây là kiểm source, không phải chạy export mới.
- [x] **L6-02 — Checkpoint 1386 và đánh giá PyTorch đã chạy.** Có checkpoint, script/log hoàn tất 500 câu, JSON 500 hàng và 500 WAV. Không gán kết quả PyTorch đó cho ONNX hoặc N150/Pi 5.
- [x] **L6-03 — Phân biệt export cũ với bản xuất cuối.** Có ONNX MRF, log PTQ và đánh giá ONNX lịch sử. `run_ptq_sweep.py` đọc file FP32 có sẵn và không kiểm checkpoint nguồn. Đã rà 410 file code/log/tài liệu trong phạm vi dự án, chưa tìm được bản ghi liên kết ONNX hiện có với 1386; không kết luận toàn bộ MRF chưa chạy.
- [ ] **L6-04 — Export truy nguyên được từ checkpoint MRF đã chốt.** Chưa có kết quả đối chiếu khớp trong artifact đã tìm. Giữ là công việc còn chờ của C03; chưa tự chạy export/train/benchmark. Nếu nhóm chưa chạy bước này thì đây là việc thực nghiệm tiếp theo, không phải lỗi thiếu implementation.

| Thành phần | Trạng thái hiện có |
|---|---|
| Code MRF và exporter | Đã có, đã đọc luồng nạp/export |
| Checkpoint MRF 1386 | Đã có và đã kiểm |
| Đánh giá PyTorch 1386 | Có log hoàn tất 500 câu và kết quả lưu |
| ONNX/PTQ MRF lịch sử | Đã có và có log đánh giá; nguồn checkpoint của graph hiện tại chưa khớp |
| ONNX khớp checkpoint cuối | Chưa có kết quả đối chiếu được xác minh; còn chờ |

Nguồn: [round6_mrf_export_code.md](round6_mrf_export_code.md), [round6_mrf_execution_evidence.json](round6_mrf_execution_evidence.json), [round6_mrf_source_search.json](round6_mrf_source_search.json). **C04/C09 vẫn đóng; các mục C còn lại không tự đóng chỉ vì có code hoặc checkpoint.**

## 8. Lượt 7 — calibration/evaluation: đã xử lý tài liệu, còn sửa và chạy protocol

- [x] **L7-01 — Xác minh calibration bằng source thực.** Đầu vào phoneme IDs/lengths/scales; static QOperator, QInt8 theo kênh, QUInt8 activation; MinMax là mặc định của ORT hiện cài. Không nhận đã pin môi trường lịch sử.
- [x] **L7-02 — Đọc split và sweep.** Canonical hiện tại hợp lệ; script PTQ vẫn đọc full JSONL. Rà 442 file text liên quan, chưa có manifest calibration train-only/selection validation-only được lưu. Không gán canonical hiện tại cho MRF cũ.
- [x] **L7-03 — Xác định lỗi ở chọn scope.** Cheap evaluator thiếu resample, dùng base.en/regex khác small/jiwer của đánh giá cuối; inference lỗi bị ghi WER 1.0. Chưa có thứ hạng cấu hình sau khi sửa. Script chỉ ghi các ứng viên, không có đầy đủ tiêu chí chấp nhận và biên bản chọn.
- [x] **L7-04 — Xác minh scorer, audio và timing.** Script 500 câu đọc WAV đúng sample rate; chốt mean WER theo câu, ASR/UTMOS dùng audio representation khác nhau và các giới hạn RNG. Timer lịch sử chỉ quanh `sess.run`; không gộp với text-to-waveform hoặc log microbenchmark khác.
- [x] **L7-05 — Cập nhật luận văn và hướng dẫn sửa.** Đã bổ sung E.8, sửa phương pháp, diễn giải kết quả, bảng protocol và đồng bộ Word/checklist. Giữ nguyên các bảng số liệu lịch sử; không tự tạo số liệu mới.
- [ ] **L7-06 — Sửa pipeline và đóng băng manifest chạy cuối.** Source thực nghiệm chưa sửa ở lượt này. Cần sửa resample/normalization/failure status, kiểm membership các tập, lưu hash/seed/weights và tiêu chí chọn; phụ thuộc source export và split hợp lệ ở C02/C03.
- [ ] **L7-07 — Calibration/PTQ/evaluation sau sửa.** Chưa chạy lại quantization, synthesis, ASR hoặc UTMOS; chưa đo N150/Pi 5. C05/C07 chưa đóng bằng việc hoàn thành audit.

Nguồn và thứ tự triển khai:

- [Code calibration/split](round7_calibration_code.md)
- [Code evaluation/scorer](round7_evaluation_code.md)
- [Hướng dẫn xử lý calibration/evaluation](round7_calibration_evaluation_plan.md)

**C04/C09 vẫn đóng; C01/C02/C03/C05/C06/C07/C08 còn mở hoặc hoàn thành một phần.** Lỗi mới được định vị ở bước screening; chưa có căn cứ coi toàn bộ số liệu 500 câu là sai vì cùng lỗi sample rate.

## 9. Cách theo dõi các lượt tiếp theo

Sau mỗi lượt sửa, cập nhật checklist này với phiên bản Word tương ứng, mã mục vừa xử lý và mục còn mở. Chỉ đánh dấu C01–C09 hoàn thành khi đáp ứng điều kiện nêu trên; không đánh dấu chỉ vì đã thêm câu “chưa xác minh”.

**Ưu tiên tiếp theo:** xác minh export MRF cuối và split; sửa resample/scorer ở bước chọn scope, đóng băng calibration train-only và validation riêng, rồi tạo lại PTQ/đánh giá. N150/Pi 5 vẫn chờ đủ 8 ô đo. Không dùng việc xóa vài dòng test để nhận rằng đã sửa ảnh hưởng của calibration/selection. C01/C02 còn provenance, C08 cần nguồn quản lý độc lập; không khôi phục F0/Vocos.

Hồ sơ đối chiếu hiện hành: `round7_revision_changes.json`, `round7_word_validation.json`, audit code calibration/evaluation lượt 7 và hướng dẫn protocol. Các hồ sơ lượt 1–6 được giữ để truy lịch sử. Quyết định cuối vẫn là **chỉ PTQ**.
