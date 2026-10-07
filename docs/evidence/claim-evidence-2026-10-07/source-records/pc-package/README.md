# Bằng chứng bổ sung cho 11 claim còn mở (07/10/2026)

Phạm vi: rà lại toàn bộ thư mục trên Windows và WSL. Chỉ đọc: không train, export, quantize, benchmark; không unpickle checkpoint (tên khoá đọc tĩnh bằng `pickletools`); parity được tính lại từ output **đã lưu**, không chạy model. Script: [scripts/collect_claim_evidence.py](scripts/collect_claim_evidence.py) → các file trong [evidence/](evidence/). Metadata công khai của Piper đọc từ Hugging Face (chỉ metadata, không tải dữ liệu/trọng số; [evidence/public_metadata_retrieval.json](evidence/public_metadata_retrieval.json)).

## Tóm tắt

| Claim | Trước | Sau lượt này | Bằng chứng mới chính |
|---|---|---|---|
| J-C001 | Hẹp | **Hỗ trợ được (bản thu hẹp)** | Checkpoint baseline có 876/876 tên tham số trùng code BanhmiTTS hiện tại; khác EdgeTTS và Piper |
| J-C010 | Chưa | **Không đúng với PTQ đã chạy**; chỉ có thiết kế | Tập calibration candidate 60/60 thuộc train, nhưng chưa từng được dùng |
| J-C011 | Hẹp | **Hỗ trợ** (cho graph Q05) | Hash checkpoint trong metadata ONNX khớp; parity tính lại đạt ngưỡng; graph kiểm và graph triển khai cùng trọng số |
| J-C012 | Chưa | **Không có** | Evaluator chưa sửa; không có lần chạy lại |
| J-C013 | Chưa | **Không có** → future work (tác giả: đo khi có thiết bị) | Benchmark VNNI duy nhất chạy trên máy dev, model khác |
| J-C015 | Chưa | **Không có** | Chỉ có kế hoạch/checklist |
| J-C019 | Chưa | **Bị dữ liệu phản bác ở WER** | CI bootstrap của chênh WER SEQ − MRF không chứa 0 |
| J-C023 | Chưa | **Hỗ trợ một phần** (SEQ vs baseline) | Cùng 500 câu, graph đã định danh: SEQ nhanh hơn ở 495/500 (FP32) và 498/500 (INT8) |
| J-C024 | Chưa | **Hỗ trợ một phần** (PTQ lịch sử) | Graph INT8 xác nhận QOperator, INT8 per-channel, UINT8, phạm vi enc_p/dp/flow, decoder FP32 |
| J-C025 | Rộng | **Hỗ trợ** (thu hẹp về artifact cụ thể) | Flow trong graph: 8.285.568 tham số, 64 Conv + 4 Softmax; Piper 7.090.560, không attention |
| J-C026 | Chưa | **Hỗ trợ ở mức văn bản cho cả 4 model** | Model card + hash công khai: Piper train từ đầu trên LJ Speech; Harvard không trùng văn bản LJSpeech |

## Chi tiết theo claim

### J-C001 — Baseline phát triển từ Piper, không phải Piper nguyên bản hay VITS2 đầy đủ
Bằng chứng ([J-C001_checkpoint_key_structure.json](evidence/J-C001_checkpoint_key_structure.json)):
- Checkpoint baseline (epoch 1489) có **876 tên tham số, trùng 100%** với model dựng từ code BanhmiTTS hiện tại. Trong đó có 72 khoá attention trong flow và 20 khoá duration discriminator (thành phần VITS2).
- Checkpoint Piper công bố có 784 khoá, không có attention trong flow, không có duration discriminator.
- Code EdgeTTS (bật VITS2) sinh thêm 5 buffer `_n_channels` mà checkpoint baseline không có ⇒ baseline **không** được train bằng code EdgeTTS.
- Khác VITS2 gốc (Kong et al., Interspeech 2023, §3): VITS2 dùng mel 80 band làm đầu vào posterior encoder và bỏ blank token; baseline dùng phổ tuyến tính 513 bin và chuỗi ID có xen blank 0. Speaker-conditioned text encoder không áp dụng (một người nói).
- Lời tác giả (05/10): viết lại theo ý tưởng Piper, kết hợp một phần VITS2, không fork. Git: commit đầu không có upstream; một số file tự ghi "Ported from upstream Piper/VITS".

Giới hạn: trùng tên tham số là trùng cấu trúc, không chứng minh từng byte code đã chạy.

> *Suggested wording:* "The baseline is an independent VITS implementation that follows Piper's parameterization and training recipe and adds three VITS2 components (Transformer-conditioned flow, duration discriminator, noise-scaled MAS). It is neither the original Piper code nor a complete VITS2 reproduction: the posterior encoder takes linear spectrograms and blank tokens are retained. The selected checkpoint's parameter structure matches the released code exactly."

### J-C010 — Calibration từ train, chọn phạm vi bằng validation riêng
- PTQ **đã chạy** (lịch sử) lấy 60 câu ngẫu nhiên trên toàn bộ 13.100 dòng: 57 train / **3 test** (gói 04/10). Bộ chọn phạm vi có lỗi sample rate.
- Thiết kế Q04/Q05: tập candidate 60 câu, **60/60 thuộc train** (SHA-256 `e4de2ad4…`), cùng danh sách validation riêng cho từng model ([J-C010_calibration.json](evidence/J-C010_calibration.json)). Nhưng ngưỡng và quy tắc chọn chưa khai báo, và **không có file INT8 nào tạo sau 04/10**.

> Không viết claim này cho kết quả hiện có. Nếu giữ: "A train-only calibration set and per-model validation lists have been defined for future PTQ runs."

### J-C011 — ONNX FP32 export đúng checkpoint, đạt parity
Bằng chứng ([J-C011_onnx_export_parity.json](evidence/J-C011_onnx_export_parity.json)), cho cả 4 graph Q05:
- `checkpoint_sha256` trong metadata ONNX **khớp** hash checkpoint hiện tại; nạp ở chế độ strict.
- Parity **tính lại** từ output PyTorch/ONNX đã lưu (10 ca/model): NRMSE tối đa 9,2e-5 / 1,14e-4 / 4,4e-5 / 7,1e-5 (baseline/MRF/SEQ/Piper), max-abs ≤ 5,6e-4; ngưỡng định trước 1e-3 và 5e-3.
- Graph kiểm (nhiễu đưa từ ngoài) và graph triển khai có **toàn bộ initializer giống hệt**; chỉ khác 2 node `RandomNormalLike`.
- Các graph này là bản được công bố ở kho inference công khai (cùng SHA-256; xem `hifi-mobiNet/docs/huggingface-inference-release.json`).

Giới hạn: 10 đầu vào, ONNX Runtime 1.23.2 CPU.

### J-C012 — PTQ cuối đã sửa evaluator và tái lập
Chỉ có `eval_wer_cheap.py` bản chưa sửa; không có lần chọn/đánh giá lại. **Không hỗ trợ** → future work.

### J-C013 — Runtime/memory trên Intel N150, Raspberry Pi 5
Không có log. Gói `vnni_test_package` (23/08) là model Piper thời EdgeTTS, chạy trên chính máy dev i7-14700. Tác giả xác nhận sẽ đo khi có thiết bị → future work.

### J-C015 — Human listening study
Không có protocol, người tham gia hay phân tích. Trang nghe thử công khai là demo, không phải nghiên cứu. → giới hạn/future work.

### J-C019 — SEQ tương đương Parallel-IR
Số đã audit (Harvard-720, gói 04/10): UTMOSv2 SEQ − MRF = −0,024, CI bootstrap [−0,046; −0,001]; WER +0,015, CI [+0,0019; +0,0277] **không chứa 0**; không có margin tương đương. Dữ liệu nghiêng về **SEQ kém hơn nhẹ ở WER**. Giữ cách viết "runtime–quality trade-off", bỏ claim tương đương.

### J-C023 — Lợi thế tốc độ SEQ qua PyTorch, ONNX FP32, INT8
([J-C023_onnx_speed_history.json](evidence/J-C023_onnx_speed_history.json))
- PyTorch (Harvard-720): SEQ có RTF thấp nhất trong 4 model (gói 04/10).
- ONNX lịch sử, **cùng 500 câu**, graph khớp trọng số checkpoint (baseline 1489, SEQ 1442), ORT CPU 2 thread: SEQ nhanh hơn baseline ở **495/500** câu với FP32 (tỉ lệ trung vị 0,89) và **498/500** sau PTQ (0,84).
- Không so được với MRF: graph ONNX của MRF không định danh được và đo trên bộ câu khác (chỉ 21 câu chung).
- Giới hạn: đo khác ngày (19/09 và 30/09), không warm-up, một lần đo/câu; phạm vi INT8 của SEQ rộng hơn (thêm 40 QLinearMatMul); graph Q05 cuối chưa được đo tốc độ.

> *Suggested wording:* "In historical ONNX Runtime CPU measurements on the same 500 LJSpeech sentences, Sequential-IR was faster than the ResBlock2 baseline in FP32 (495/500 sentences) and after backbone PTQ (498/500); an identified Parallel-IR graph was not available for this comparison."

### J-C024 — Recipe PTQ thực tế
([J-C024_J-C025_graph_structure.json](evidence/J-C024_J-C025_graph_structure.json)) Graph INT8 lịch sử của baseline-1489 và SEQ-1442:
- QOperator (`QLinearConv`); trọng số **INT8 per-channel** (133/133); activation **UINT8**.
- Phạm vi: enc_p 37 + dp 32 + flow 64 lớp; **0** op lượng tử dưới `/dec/` (decoder FP32).
- SEQ khác: thêm 40 `QLinearMatMul` (attention của enc_p và flow).
- 60 câu calibration: tái lập từ quy tắc trong script (có 3 câu test).

> Mô tả được recipe của **PTQ lịch sử** cho baseline và SEQ, kèm khác biệt phạm vi của SEQ; không gọi là "PTQ cuối".

### J-C025 — Mô tả flow, số tham số, phạm vi `/flow/`
- Graph Q05: flow của 3 model nội bộ có **8.285.568** tham số, 64 Conv, 4 Softmax, 16 MatMul, 8 LayerNorm (attention trong mỗi coupling). Piper: 7.090.560 (sau khi trừ `weight_g`), 40 Conv, không attention.
- PTQ lịch sử: 64/64 Conv của flow được lượng tử; MatMul attention của flow chỉ được lượng tử ở SEQ.

### J-C026 — Harvard-720 chưa model nào nhìn thấy
([J-C026_harvard_exposure.json](evidence/J-C026_harvard_exposure.json))
- 3 model nội bộ: chỉ train trên dataset LJSpeech của dự án (13.100 dòng); Harvard không trùng câu và không chung 6-gram nào với văn bản LJSpeech (gói 04/10).
- Piper công bố: SHA-256 checkpoint `dcf2449b…` **trùng** hash LFS trong kho công khai `rhasspy/piper-checkpoints`; model card: *"Trained from scratch for 1000 epochs … using the LJ Speech dataset"*; hparams: `dataset training/lj-med`, `resume_from_checkpoint: None`, `max_epochs 1000`.
- Phụ: bảng phoneme ID của dự án chứa đủ 157 ký hiệu của Piper với **cùng ID**; không câu Harvard nào dùng ID ngoài bảng của Piper.
- Giới hạn: không có file dữ liệu train thật của Piper (dựa vào model card của tác giả Piper); chỉ ở mức văn bản; không xét dữ liệu train của Whisper/UTMOSv2.

> *Suggested wording:* "None of the four systems was trained on Harvard sentence text: all were trained on LJ Speech (the published Piper checkpoint per its model card), and no Harvard sentence appears in, or shares a 6-gram with, the LJ Speech transcripts."

## Tệp

| Tệp | Nội dung |
|---|---|
| `evidence/J-C001_checkpoint_key_structure.json` | So sánh tên tham số checkpoint ↔ code |
| `evidence/J-C010_calibration.json` | Thành phần tập calibration candidate; INT8 tạo sau 04/10 |
| `evidence/J-C011_onnx_export_parity.json` | Metadata ONNX, parity tính lại, graph kiểm ↔ graph triển khai |
| `evidence/J-C023_onnx_speed_history.json` | So sánh RTF ONNX theo cặp câu |
| `evidence/J-C024_J-C025_graph_structure.json` | Flow trong graph Q05; định dạng và phạm vi PTQ lịch sử |
| `evidence/J-C026_harvard_exposure.json` | Hparams Piper, bảng phoneme ID, dataset nội bộ |
| `evidence/public_*` | Model card, danh mục file (hash LFS), config công khai của Piper lj-medium |
| `evidence-index.json` | SHA-256 mọi tệp trong gói |
