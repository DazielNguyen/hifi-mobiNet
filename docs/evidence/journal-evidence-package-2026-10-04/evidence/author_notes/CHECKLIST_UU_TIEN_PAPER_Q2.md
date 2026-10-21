# Checklist ưu tiên hoàn thiện paper Q2

Cập nhật **04/10/2026 — Q01 hoàn tất lưu giữ; Q05 hoàn tất export FP32 và kiểm tương thích cho bốn model. Q03/Q04 vẫn có giới hạn chưa đóng.** Q03 có source/hash khôi phục và lịch sử đã đối chiếu; Q04 có ứng viên v0.2. Q05 đã tạo graph mới, kiểm 40 ca số học cùng nhiễu và 12 ca chạy graph ngẫu nhiên gốc; chưa có PTQ, điểm chất lượng hoặc benchmark thiết bị mới. Xem [Q01](evidence/q01_2026-10-04/BAO_CAO_Q01.md), [Q03](Q03_CAU_HINH_VA_LICH_SU_HUAN_LUYEN.md), [Q04](Q04_PROTOCOL_DU_LIEU_VA_DANH_GIA.md) và [Q05](Q05_EXPORT_ONNX_VA_KIEM_CHUNG.md). Q02 vẫn chưa chốt.

**Định hướng đề xuất:** đánh giá thiết kế decoder nhẹ và selective PTQ cho triển khai TTS trên CPU, xét đồng thời tốc độ, bộ nhớ và chất lượng. Baseline là đối chứng nội bộ; Piper gốc là đối chứng bên ngoài. SEQ là ứng viên ưu tiên tốc độ; MRF là phương án so sánh về chất lượng. Đây là hướng viết đề xuất, không phải kết luận đã chốt về hiệu quả trên thiết bị edge.

**Cách sử dụng:** làm theo thứ tự ưu tiên, đồng thời viết các phần đã đủ bằng chứng. Chỉ đổi trạng thái thành **Hoàn tất** khi có đầu ra đáp ứng điều kiện ở cột cuối. Các mã Q01–Q11 dành cho paper, độc lập với mã C01–C09 trong checklist luận văn.

## 1. Bảng công việc theo thứ tự ưu tiên

| Ưu tiên | Mã | Công việc | Trạng thái hiện tại | Việc cần làm và điều kiện hoàn thành |
|---|---|---|---|---|
| 1 — Đã hoàn tất | Q01 | Lưu cố định bằng chứng và artifact | **Hoàn tất phạm vi lưu giữ, 04/10.** Gói WSL đã lưu 9.360 file, khoảng 6,85 GB: 4 checkpoint chính, 35 ONNX, 27 bảng kết quả và 8.931 WAV. | Đã kiểm hash nguồn/bản sao, kiểm độc lập 9.371 mục checksum; 16 bộ đánh giá chính có đủ 8.880 WAV, script và log. Có manifest/hash, metadata checkpoint, source/môi trường hiện tại và hồ sơ từng bảng; nguồn lịch sử còn thiếu được ghi rõ, chuyển tiếp Q03–Q09. **Bằng chứng:** [Báo cáo Q01](evidence/q01_2026-10-04/BAO_CAO_Q01.md), [biên nhận kiểm tra](evidence/q01_2026-10-04/FINAL_VERIFICATION.json). |
| 2 — Trước khi chốt đóng góp | Q02 | Chốt câu hỏi nghiên cứu, phạm vi kết luận và tạp chí mục tiêu | **Đã có hướng đề xuất, chưa chốt.** Kết quả Harvard thể hiện đánh đổi giữa các model. | Chốt vai trò baseline/Piper/MRF/SEQ, đóng góp của decoder và PTQ; chọn tạp chí phù hợp rồi kiểm tra phạm vi, độ mới, yêu cầu thực nghiệm và định dạng. Không gọi baseline là cải thiện toàn diện của Piper; không quy toàn bộ cải thiện chất lượng cho riêng decoder. **Hoàn tất khi:** có câu hỏi nghiên cứu, danh sách đóng góp và giới hạn tuyên bố nhất quán. |
| 3 — Trước khi chốt Experimental Setup | Q03 | Hoàn thiện bảng cấu hình và lịch sử huấn luyện | **Đã hoàn tất audit và phần khôi phục khả thi; lịch sử vẫn một phần.** Bốn checkpoint/chín phiên đã đối chiếu; khôi phục văn bản sau 152 thao tác sửa, kiểm 218 source objects, ghép 207 kết quả shell. TensorBoard và MRF tái dựng có đối chiếu. Đã kết thúc lượt tìm checkpoint trung gian trong phạm vi Windows/WSL ghi nhận. | [Q03 mục 11](Q03_CAU_HINH_VA_LICH_SU_HUAN_LUYEN.md) có source/hash, diff, giới hạn và đoạn dùng trong paper. **Không thu hồi được:** hash checkpoint resume cũ; nhóm cho rằng có thể đã bị ghi đè. **Chưa đủ:** full manifest/hash từng phase, hash source runtime và split Piper. Chốt ở Q02 phạm vi so sánh hệ thống với các giới hạn này; không tiếp tục coi tìm lại file cũ là việc phải làm vô hạn, không gọi decoder-only ablation. |
| 4 — Trước khi lựa chọn cấu hình và đánh giá tiếp | Q04 | Khóa protocol dữ liệu và đánh giá | **Một phần; đã sửa ứng viên v0.2 ngày 04/10.** [Protocol](Q04_PROTOCOL_DU_LIEU_VA_DANH_GIA.md) giữ LJSpeech-500/Harvard-720 riêng. 60 calibration thuộc train chung theo canonical/MRF tái dựng; selection riêng 98 baseline/SEQ và 94 MRF. Hai validation chỉ giao 2 ID. Giữ nguyên v0.1 và mọi số lịch sử. | **Đã xử lý lỗi giao dữ liệu đã biết của ứng viên.** Còn giới hạn provenance và cần chốt thiết kế validation theo model; lịch sử dùng Harvard; ngưỡng/quy tắc PTQ; evaluator/RNG đã kiểm thực thi và artifact đúng nguồn. **Hoàn tất khi:** protocol thực thi được khóa trước run cuối với giới hạn được công khai. Không gọi 98/94 là cùng validation hoặc dùng việc báo cáo hai bộ để tự tuyên bố hết bias. |
| 5 — Đã hoàn tất phạm vi FP32 | Q05 | Xác minh hoặc tạo lại ONNX FP32 | **Hoàn tất, 04/10.** Bốn graph FP32 mới từ baseline 1489, MRF 1386, SEQ 1442 và Piper metadata epoch 999. Có hash checkpoint/source/ONNX, hparams, argv và log. **40/40** ca cùng nhiễu, **12/12** ca graph ngẫu nhiên gốc đạt; 328 phép kiểm đóng gói đạt. | [Hồ sơ Q05](Q05_EXPORT_ONNX_VA_KIEM_CHUNG.md), [artifact chuyển tiếp](evidence/q05_2026-10-04/FP32_HANDOFF.json). Đã sửa output thành `[B,1,N]`, trục thời gian 2. MRF FP32 cũ khác checkpoint 1386 ở 478 tensor cùng tên; giữ nhãn lịch sử/chưa rõ nguồn. Dùng graph mới cho Q06–Q08; không chuyển điểm cũ sang hash mới. Chưa có PTQ hoặc kết luận chất lượng/edge từ kiểm kỹ thuật này. |
| 6 — Trước khi chốt kết quả PTQ | Q06 | Hoàn thiện selective PTQ với calibration/selection hợp lệ | **Một phần, còn vấn đề phương pháp.** Đã có PTQ lịch sử; chưa đóng các mục giao dữ liệu và lỗi evaluator chọn scope. | Xử lý lỗi sample rate từng phát hiện ở evaluator chọn scope; calibration chỉ lấy từ train đã xác minh, chọn scope trên validation. Chốt tiêu chí chấp nhận chất lượng trước khi chọn cấu hình. Ghi scope INT8/FP32, định dạng graph, phương pháp calibration và manifest/hash; tạo lại các artifact bị ảnh hưởng. **Hoàn tất khi:** recipe tái lập được và có kết quả đánh giá theo protocol cuối. |
| 7 — Bắt buộc để kết luận về N150/Pi 5 | Q07 | Benchmark trên Intel N150 và Raspberry Pi 5 | **Chưa có kết quả thiết bị được xác minh.** RTF hiện có không thay thế phép đo hai thiết bị này. | Chuẩn hóa runtime/provider, số luồng, warm-up, số lần lặp, tải nền, nguồn điện/tản nhiệt và ranh giới timer. Đo RTF, latency, kích thước file, peak memory; tách suy luận model với toàn pipeline. Kiểm tra thực thi INT8 bằng profiling. **Hoàn tất khi:** có log/raw measurements, cấu hình máy và kết quả đủ ma trận đã chốt. |
| 8 — Chuẩn bị song song với export/benchmark | Q08 | Đánh giá chất lượng artifact cuối và tổ chức nghe thử | **Một phần.** Có WER/UTMOSv2 Harvard của bốn checkpoint PyTorch; chưa xác nhận đánh giá hoàn chỉnh artifact triển khai cuối hoặc nghiên cứu người nghe. | Chấm các ONNX cuối trên cùng tập câu bằng scorer đã khóa. Thiết kế nghe thử ẩn danh model, ngẫu nhiên thứ tự, tiêu chí chọn câu/người nghe và phương pháp phân tích trước khi thu thập. Không ghi UTMOSv2 thành MOS người nghe. **Hoàn tất khi:** có kết quả chất lượng của đúng artifact triển khai và bằng chứng nghe thử phù hợp với mức kết luận của paper. |
| 9 — Sau khi có các lần đánh giá cuối | Q09 | Hoàn thiện thống kê, bảng và hình kết quả | **Một phần.** Đã tính lại trung bình và một số Wilcoxon Harvard; chưa hoàn thiện phân tích của toàn bộ ma trận. | Báo cáo độ chênh lệch, khoảng tin cậy, kiểm định ghép cặp phù hợp và hiệu chỉnh nhiều phép so sánh. Phân biệt biến thiên theo câu, lần đo và lần huấn luyện. Tách tham số decoder, tham số suy luận, kích thước file và bộ nhớ runtime. Không suy tương đương từ p-value không có ý nghĩa. **Hoàn tất khi:** bảng/hình tái tạo được từ dữ liệu gốc, đúng phạm vi và đúng đơn vị. |
| 10 — Theo mức đóng góp đã chọn ở Q02 | Q10 | Phân tích cơ chế và quyết định ablation | **Còn thiếu bằng chứng nhân quả.** Có giả thuyết giải thích khác biệt tốc độ nhưng chưa đủ profiling để chốt nguyên nhân. | Profile decoder theo độ dài chuỗi và runtime; kiểm tra giả thuyết về chi phí tính toán/bộ nhớ. Nếu muốn khẳng định tác động riêng của decoder, thiết kế ablation kiểm soát activation, discriminator, dữ liệu và recipe huấn luyện; có thể cần train thêm. Nếu giữ so sánh cấp hệ thống, nêu rõ giới hạn. **Hoàn tất khi:** giải thích có bằng chứng hoặc tuyên bố được thu hẹp tương ứng. |
| 11 — Bắt đầu phần phương pháp ngay, hoàn tất sau kết quả | Q11 | Viết, đồng bộ và kiểm tra bản thảo | **Có thể bắt đầu ngay.** Dataset/preprocessing và kiến trúc baseline đã có cơ sở từ code; chưa có bản paper hoàn chỉnh được kiểm tra. | Viết phương pháp và protocol trước, Results/Discussion theo kết quả cuối; chốt Abstract/Contributions/Conclusion sau cùng. Đồng bộ thuật ngữ, hình, bảng, citation, caption, nguồn số và giới hạn; rà yêu cầu tạp chí. **Hoàn tất khi:** bản thảo không còn kết luận vượt bằng chứng, số liệu nhất quán và có gói tái lập phù hợp. |

## 2. Những gì đã có — không cần làm lại từ đầu

| Hạng mục | Bằng chứng đã có | Giới hạn cần giữ |
|---|---|---|
| Baseline hiện tại | Đã đọc code, kiểm checkpoint epoch 1489 và dựng hình training/inference/decoder. | Baseline là hệ thống phát triển từ Piper; không phải Piper nguyên bản hoặc bản tái hiện đầy đủ VITS2. |
| Checkpoint MRF và SEQ | Có MRF epoch 1386 và SEQ epoch 1442; Q05 đã export graph FP32 mới có nguồn và kiểm tương thích. | Kết quả PyTorch/ONNX lịch sử không tự chuyển thành điểm của các graph mới. MRF ONNX cũ chưa xác định checkpoint nguồn. |
| Harvard 720 câu | Bốn file kết quả baseline/MRF/SEQ/Piper có đủ 720 hàng; cùng ID, văn bản và thứ tự; trung bình tính lại khớp bảng. | Đây là một lần đánh giá PyTorch CPU. Chưa thay thế benchmark ONNX/PTQ, edge hoặc nhiều lần đo. |
| Kiểm tra trùng văn bản | Không tìm thấy câu trùng hoàn toàn sau chuẩn hóa giữa Harvard và 13.100 transcript LJSpeech hiện có. | Không tự chứng minh toàn bộ lịch sử dữ liệu/huấn luyện hoặc tính độc lập với mọi lần chọn cấu hình. |
| Hồ sơ Q04 | Manifest có ID/thứ tự/hash, map 27 bảng, source/weight identity scorer hiện tại; kiểm WER từ transcript không lệch. v0.2 có 60 calibration chung và selection 98/94 theo model; quy định dùng cùng WAV cho WER/UTMOS, metric, RNG và trạng thái lỗi. | Danh sách vẫn là ứng viên dựa trên các split đã đối chiếu, không phải một validation chung. Cần chốt thiết kế và kiểm runner; không hồi tố phương pháp mới cho bảng cũ. |
| Chất lượng baseline so với Piper | Trong lần Harvard đã kiểm: baseline có WER thấp hơn, UTMOSv2 thấp hơn và RTF cao hơn Piper. | Không gọi baseline là cải thiện toàn diện; chưa xác định nguyên nhân của chênh lệch chất lượng. |
| Kết quả MRF/SEQ trên Harvard | SEQ có RTF trung bình thấp nhất; MRF có UTMOSv2 trung bình cao nhất và WER trung bình thấp nhất. | Thứ hạng trung bình không tự chứng minh khác biệt có ý nghĩa hoặc nguyên nhân riêng từ decoder. |

Nguồn Harvard đã đọc trong WSL Ubuntu-24.04: `/tmp/harvard_{baseline,mrf,seq,piper}_results.json`, `/tmp/harvard_run_all.log`, `/tmp/harvard_testset_720.json` và `<WSL_PROJECT_HOME>/scripts/harvard/`. **Đã có bản sao cố định** trong `<WSL_PROJECT_HOME>/paper_evidence/q01_2026-10-03T220747Z/raw/`, giữ cấu trúc đường dẫn nguồn. Timestamp tên gói là UTC, tương ứng sáng 04/10 theo UTC+7. Xem [chỉ mục từng bảng kết quả](evidence/q01_2026-10-04/RESULT_CATALOG.json); việc lưu bản sao không đóng các vấn đề phương pháp/provenance còn lại. **Graph FP32 mới của Q05** lưu riêng tại `<WSL_PROJECT_HOME>/paper_evidence/q05_2026-10-04T000830Z/`, không ghi đè Q01 hoặc artifact cũ.

## 3. Ma trận triển khai đề xuất

Ma trận này cần được chốt ở Q02/Q04. Không đánh dấu đã đo chỉ vì đã có kết quả trên máy phát triển.

| Model | FP32 ONNX | Selective-PTQ ONNX | Intel N150 | Raspberry Pi 5 |
|---|---|---|---|---|
| Baseline | Có trong kế hoạch | Có trong kế hoạch | Đo cả hai precision | Đo cả hai precision |
| MRF | Có trong kế hoạch | Có trong kế hoạch | Đo cả hai precision | Đo cả hai precision |
| SEQ | Có trong kế hoạch | Có trong kế hoạch | Đo cả hai precision | Đo cả hai precision |
| Piper gốc | Có trong kế hoạch | Chưa đưa vào phạm vi đề xuất | Đo FP32 | Đo FP32 |

Tổng cộng: **7 cấu hình model/precision × 2 thiết bị = 14 ô benchmark**, chưa tính warm-up và các lần đo lặp. Trước khi so sánh, phải xác minh export Piper đúng checkpoint và sử dụng điều kiện runtime có thể đối chiếu được.

## 4. Thứ tự viết có thể tiến hành song song

| Phần paper | Có thể làm ngay | Điều kiện trước khi chốt |
|---|---|---|
| Dataset and Preprocessing | Viết từ pipeline đã đối chiếu code. | Sửa attribution còn sai nếu có; mô tả split theo bằng chứng Q03/Q04. |
| Baseline Architecture | Viết và chỉnh hình từ code/checkpoint đã kiểm. | Gọi đúng Piper-derived baseline; thể hiện đúng training/inference và các thành phần VITS2-inspired. |
| Decoder Designs | Mô tả baseline/MRF/SEQ và sơ đồ các khối. | Công khai activation, discriminator và phạm vi tham số; chưa gán tác động nhân quả. |
| Related Work / Introduction | Viết nháp vấn đề, bối cảnh và đối chiếu các nghiên cứu liên quan. | Rà tài liệu gốc và độ mới theo tạp chí mục tiêu; hoàn thiện đóng góp sau kết quả. |
| Experimental Setup | Viết protocol và bảng cấu hình; để rõ các ô đang chờ đo. | Q03–Q08 có hồ sơ đủ cho phần được báo cáo; chuyển “dự kiến” thành “đã làm” khi có bằng chứng. |
| Results / Discussion | Dựng khung bảng và câu hỏi phân tích. | Dùng số cuối từ Q09; giữ giới hạn Q10. |
| Abstract / Contributions / Conclusion | Có thể phác thảo. | Chốt cuối cùng theo kết quả thực, không cam kết trước model thắng hoặc lợi ích edge. |

## 5. Khi nào mới cân nhắc train lại?

| Tình huống | Hướng xử lý ưu tiên |
|---|---|
| Baseline có UTMOSv2 thấp hơn Piper | Kiểm chất lượng, điều kiện so sánh và vai trò baseline; riêng quan sát này chưa đủ để yêu cầu train lại. |
| ONNX không rõ nguồn hoặc export sai | Export và đánh giá lại từ checkpoint đã xác minh. |
| Calibration/selection dùng dữ liệu không hợp lệ | Sửa protocol, tạo lại PTQ và đánh giá; không tự suy ra phải train lại model gốc. |
| Muốn chứng minh lợi ích riêng của decoder | Cần ablation có kiểm soát; có thể cần huấn luyện bổ sung. |
| Phát hiện checkpoint hỏng hoặc dữ liệu huấn luyện không đáp ứng tuyên bố bắt buộc | Xác định khả năng thu hẹp tuyên bố, chọn checkpoint/tập đánh giá hợp lệ hoặc train lại; quyết định theo bằng chứng cụ thể. |

## 6. Tài liệu liên quan

- [Checklist luận văn v7](../../paper_revisions/thesis_ptq/CHECKLIST_SUA_LUAN_VAN.md): snapshot ngày 29/09, chưa bao quát SEQ và Harvard mới.
- [Báo cáo khác biệt Piper/VITS2/BanhmiTTS](Piper_VITS2_BanhmiTTS_differences.md) và [bản đối chiếu ngày 04/10](figures/Piper_VITS2_BanhmiTTS_review_2026-10-04.txt): đọc kèm nhau, không coi mọi kết luận của báo cáo ban đầu đã được xác nhận.
- [Báo cáo MRF/SEQ](MRF_vs_SEQ_comparison_Q2.md): chứa kết quả và lập luận lịch sử, cần đồng bộ theo nguồn và giới hạn mới.
- [Kiến trúc baseline và ghi chú](figures/baseline_architecture_complete.pdf).
- [Protocol dữ liệu/đánh giá Q04](Q04_PROTOCOL_DU_LIEU_VA_DANH_GIA.md) và [bản đồ nguồn 27 bảng](evidence/q04_2026-10-04/RESULT_DATASET_MAP.csv): giữ cả LJSpeech-500/Harvard-720, công khai manifest và giới hạn; không gộp trung bình hai bộ.
- [Export và kiểm chứng Q05](Q05_EXPORT_ONNX_VA_KIEM_CHUNG.md): bốn FP32 mới đã xác định nguồn và đạt kiểm kỹ thuật; dùng cho PTQ/đánh giá tiếp theo.
- [Tài liệu ONNX Runtime về quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): tham khảo calibration, biểu diễn graph và debugging suy giảm chất lượng.

**Tiến độ:** Q01 hoàn tất lưu giữ; **Q05 hoàn tất export FP32 và kiểm tương thích**. **Q03/Q04 vẫn một phần** do giới hạn lịch sử và điều kiện thực thi. **Còn mở:** Q02–Q04 và Q06–Q11. Bước kỹ thuật tiếp theo là **Q06**, sử dụng các graph đã định danh trong Q05 và chốt calibration/selection, ngưỡng chấp nhận cùng evaluator. Chưa mặc định phải train lại.
