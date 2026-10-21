# Q04 — Protocol dữ liệu và đánh giá cho paper

Cập nhật **04/10/2026, phiên bản ứng viên v0.2**. **Đã sửa giao dữ liệu của các danh sách ứng viên và lưu manifest/hash mới; chưa đủ điều kiện đóng Q04 hoặc chạy ma trận cuối.** Calibration gồm 60 câu thuộc train chung; chọn PTQ dùng validation riêng 98 câu cho baseline/SEQ và 94 câu cho MRF, theo các split đang đối chiếu. Giới hạn lịch sử và các điều kiện còn thiếu ở mục 8. Đây là thiết kế cho lần chạy tiếp theo, không phải lời khẳng định các run cũ đã tuân thủ.

Theo yêu cầu của nhóm, paper sẽ báo cáo **cả LJSpeech 500 câu và Harvard 720 câu**, thành hai khối kết quả. Việc thêm một bộ đánh giá không tự loại bỏ bias hoặc khôi phục tính độc lập của tập đã được dùng để lựa chọn. Nhóm chưa xác nhận lịch sử dùng điểm Harvard để chọn checkpoint/decoder/PTQ/tham số suy luận; trường này vẫn là **chưa rõ**.

## 1. Những gì đã kiểm tra và lưu

- Kiểm SHA-256 của **43 nguồn** trong gói Q01; đối chiếu **27 bảng kết quả** với ID, văn bản và thứ tự của manifest.
- Tính lại WER từ **14.380 cặp transcript tham chiếu/ASR đã lưu** bằng JiWER hiện có: không có giá trị từng câu lệch quá `1e-12`. Đây là kiểm lại phép tính, **không phải chạy lại ASR**, cũng không xác minh nguồn model/độ độc lập của dữ liệu.
- Lưu manifest LJSpeech baseline 500, MRF 500 và Harvard 720, gồm ID ổn định, thứ tự, văn bản, phoneme IDs và hash. Giữ riêng bản byte gốc và bản dẫn xuất để tránh lẫn.
- Kiểm trùng ID, nội dung chuẩn hóa và chuỗi phoneme IDs. **v0.2** thay ứng viên calibration cũ bằng **60 train chung**, đồng thời tách selection **98 baseline/SEQ / 94 MRF**. Các phép kiểm membership đúng với canonical và split MRF tái dựng, chưa biến hai split này thành bản ghi lịch sử đầy đủ.
- Lưu snapshot source scorer, phiên bản thư viện, hash weights đang có. Không tải, load hoặc chạy scorer/model; không sửa checkpoint, graph, split gốc hoặc bảng điểm lịch sử.

Bằng chứng: [Q04_DATA_AUDIT.json](evidence/q04_2026-10-04/Q04_DATA_AUDIT.json), [bản đồ 27 bảng kết quả](evidence/q04_2026-10-04/RESULT_DATASET_MAP.csv), [danh tính scorer hiện tại](evidence/q04_2026-10-04/CURRENT_SCORER_IDENTITY.json).

## 2. Cách đưa hai bộ đánh giá vào paper

| Khối kết quả | Dữ liệu thực tế | Cách báo cáo hợp lệ | Giới hạn bắt buộc |
|---|---|---|---|
| LJSpeech-500 | Baseline/SEQ và Piper trong các bảng đã đối chiếu dùng manifest baseline; MRF ONNX và MRF checkpoint 1386 dùng manifest MRF. Một số run MRF cũ lại dùng manifest baseline. | Mỗi dòng phải ghi `test_manifest`, runtime, checkpoint/export và trạng thái historical/final. Khi so FP32–PTQ cùng model, chỉ ghép cặp nếu ID/văn bản thực sự khớp. | Hai manifest baseline/MRF chỉ giao **21/500 ID**. Không gọi tất cả các dòng là cùng một held-out test set, không ghép 500 hàng theo số thứ tự giữa hai manifest. Chưa chứng minh LJSpeech-500 held-out cho Piper gốc. |
| Harvard-720 | Bốn bảng baseline/MRF/SEQ/Piper khớp **720/720 ID, văn bản và thứ tự**. | Dùng cho so sánh trên đầu vào chung; ghi rõ đây là tập câu tiếng Anh bên ngoài LJSpeech. Có thể phân tích ghép cặp theo câu. | Những kết quả đã xem không phải một test mới chưa từng quan sát. Chưa xác nhận lịch sử chọn cấu hình từ điểm số này; chưa chứng minh mọi nguồn huấn luyện lịch sử của Piper. |

**Không gộp 500 + 720 thành một điểm trung bình chung.** Phân bố độ dài khác rõ: số phoneme IDs trung bình của canonical test-500 khoảng **217,32**, Harvard-720 khoảng **86,01**. RTF/latency còn phụ thuộc độ dài và runtime; không coi chênh lệch giữa hai bộ là hiệu ứng riêng của kiến trúc.

Đặt hai bảng cạnh nhau trong Results hoặc đưa bảng LJSpeech lịch sử vào phụ lục nếu không đủ chỗ; luôn giữ giới hạn ngay ở caption/Experimental Setup. Không chỉ giữ bộ hoặc model có điểm thuận lợi. Không đổi các số cũ thành số của checkpoint/ONNX mới mà chưa chạy tương ứng.

**Gợi ý ghi chú tiếng Anh cho bảng lịch sử:**

> Historical LJSpeech results use model-specific 500-utterance manifests; the baseline and MRF manifests share 21 utterances. These results are reported separately from the shared 720-sentence Harvard evaluation. Historical split provenance and PTQ selection limitations are disclosed, and cross-model differences on the LJSpeech manifests are not interpreted as paired, data-controlled effects.

Đoạn này không tuyên bố Harvard là test độc lập với mọi quyết định thiết kế. Nếu nhóm đã dùng Harvard để lựa chọn, phải công khai vai trò đó; nếu cần kết luận xác nhận trên tập chưa quan sát, chọn một tập mới và khóa trước khi xem điểm. Tách lại Harvard sau khi đã xem toàn bộ kết quả không tự tạo ra test mới độc lập.

## 3. Kiểm dữ liệu và manifest dùng tiếp

| Hạng mục | Kết quả |
|---|---|
| Canonical hiện tại | 12.500 train / 100 validation / 500 test, không giao ID, phủ đủ 13.100 bản ghi. Đây là snapshot hiện tại, không phải bằng chứng đầy đủ về mọi lần train/resume. |
| Trùng nội dung train–validation | Có **một nội dung**, ở validation `LJ036-0001` và 11 ID train, gồm `LJ045-0001`, `LJ048-0001`, …; chuỗi phoneme IDs cũng trùng. Chi tiết đầy đủ trong JSON. Không phải giao ID hay bằng chứng audio giống byte. |
| Trùng nội dung train–test / validation–test | Không tìm thấy theo phép chuẩn hóa WER đã ghi và theo chuỗi phoneme IDs trong canonical hiện tại. Không phải kiểm near-duplicate hoặc fingerprint audio. |
| MRF test giao canonical validation | `LJ046-0154`. Đây là membership của các manifest đang lưu, chưa xác lập lịch sử train MRF. |
| Harvard so với toàn bộ LJSpeech đang lưu | Không trùng câu sau chuẩn hóa WER; không trùng chuỗi phoneme IDs. Không có trùng nội dung trong chính Harvard-720. Không suy ra mọi dạng tương đồng/ngữ liệu huấn luyện khác đều đã bị loại. |

| File dẫn xuất | Vai trò và giới hạn |
|---|---|
| [harvard720.frozen.jsonl](evidence/q04_2026-10-04/manifests/harvard720.frozen.jsonl) | 720 đầu vào chung đã quan sát, giữ nguyên text/phoneme IDs/thứ tự từ bản gốc. ID `harvard-0000` đến `harvard-0719`; không chạy lại phonemizer. |
| [lj500_baseline.frozen.jsonl](evidence/q04_2026-10-04/manifests/lj500_baseline.frozen.jsonl) | 500 câu theo manifest baseline; các bảng SEQ/Piper đã kiểm cũng khớp thứ tự văn bản này. |
| [lj500_mrf.frozen.jsonl](evidence/q04_2026-10-04/manifests/lj500_mrf.frozen.jsonl) | 500 câu theo manifest MRF. Không đổi ID để ghép giả với baseline. |
| [canonical_split.current.json](evidence/q04_2026-10-04/manifests/canonical_split.current.json) | Bản byte hiện có, không sửa hoặc gán ngược cho mọi checkpoint. |
| [canonical_validation100.candidate.jsonl](evidence/q04_2026-10-04/manifests/canonical_validation100.candidate.jsonl) | Giữ đủ 100 bản ghi để truy nguồn; chứa hai mục không dùng trong tập chọn PTQ đề xuất bên dưới. Không phải tập selection cuối đã sạch. |
| [ptq_selection.candidate.jsonl — v0.1](evidence/q04_2026-10-04/manifests/ptq_selection.candidate.jsonl) | Giữ làm hồ sơ cũ, **không dùng như validation chung**: 96/98 câu thuộc train MRF tái dựng. |
| [calibration60.candidate.jsonl — v0.1](evidence/q04_2026-10-04/manifests/calibration60.candidate.jsonl) | Giữ làm hồ sơ cũ, **đã được thay bằng ứng viên v0.2**: 3/60 câu thuộc validation MRF tái dựng. |
| [calibration60.common_train.candidate.jsonl — v0.2](evidence/q03_q04_completion_2026-10-04/manifests/calibration60.common_train.candidate.jsonl) | **60 câu**, đều thuộc train canonical và train MRF tái dựng; không giao validation của hai split, hai manifest test-500 hoặc Harvard theo ID/nội dung chuẩn hóa/phoneme IDs. Dùng chung cho ba model BanhmiTTS nếu chấp nhận giả định provenance đã khai báo. |
| [ptq_selection.canonical.candidate.jsonl — v0.2](evidence/q03_q04_completion_2026-10-04/manifests/ptq_selection.canonical.candidate.jsonl) | **98 câu cho baseline/SEQ**: thuộc validation canonical, không trùng train canonical, calibration và các tập báo cáo. Cùng danh sách ID với 98 câu v0.1, nay chỉ định rõ model áp dụng. |
| [ptq_selection.mrf_reconstructed.candidate.jsonl — v0.2](evidence/q03_q04_completion_2026-10-04/manifests/ptq_selection.mrf_reconstructed.candidate.jsonl) | **94 câu cho MRF**: thuộc validation MRF tái dựng; loại 6 câu nằm trong canonical test-500. Không trùng train MRF tái dựng, calibration và các tập báo cáo theo ba khóa kiểm. |

**Cách tạo calibration v0.2:** giao hai train có 11.930 ID; loại các nội dung/phoneme IDs giao validation hoặc các tập báo cáo, còn **11.919** ứng viên. Sắp theo utterance ID rồi `random.Random(42).sample(...,60)`; lưu thứ tự cụ thể, không yêu cầu reader lấy mẫu lại. Độ dài min **43**, mean **216,43**, max **359** phoneme IDs. 60 vẫn là cấu hình khởi đầu, chưa chứng minh tối ưu hoặc đủ bao phủ.

**Validation không dùng chung:** hai validation chỉ giao **2 ID**, `LJ008-0142` và `LJ025-0060`. 96/98 câu selection baseline/SEQ thuộc train MRF; 92/94 câu selection MRF thuộc train canonical. Vì vậy mỗi model chỉ dùng danh sách dành cho nó, không so trực tiếp điểm validation giữa hai danh sách, không gộp thành tập chọn scope chung. Cùng quy tắc chọn scope/ngưỡng phải được khai báo trước trong Q06. So sánh trên Harvard chung sau đó vẫn cần công khai lịch sử selection chưa rõ.

**Điều kiện sử dụng:** v0.2 là phương án ứng viên dùng validation theo model cho so sánh cấp hệ thống, chưa được thực thi. Các kiểm tra đã loại giao nhau nêu trên **theo canonical và MRF tái dựng**; không phục hồi được full manifest/hash của mọi phase, không sửa leakage đã có trong graph PTQ cũ. Không thay validation 100 của các run train đã xong bằng 98/94 trong mô tả lịch sử. Nếu cần cùng tập phát triển ngoài train của cả ba model, phải có tập mới với nguồn rõ, tách khỏi các bộ test đã báo cáo; 2 câu giao không đủ cơ sở để mặc định thành tập chọn PTQ đại diện.

Nguồn phiên bản mới: [CANDIDATE_AUDIT_v0.2.json](evidence/q03_q04_completion_2026-10-04/CANDIDATE_AUDIT_v0.2.json), [script tạo](evidence/q03_q04_completion_2026-10-04/build_candidates_v02.py), [kiểm tra độc lập](evidence/q03_q04_completion_2026-10-04/COMPLETION_VERIFICATION.json). Giữ nguyên các manifest/protocol v0.1, checkpoint, split gốc và kết quả lịch sử.

## 4. Scorer và biểu diễn âm thanh

| Mục | Protocol v0.2, kế thừa recipe scorer v0.1 |
|---|---|
| Âm thanh chung để chấm | Lưu một WAV mono PCM16, 22.050 Hz cho từng `(artifact, utterance, synthesis_repeat)`. Dùng cùng WAV đó làm nguồn cho UTMOSv2 và Whisper; hash WAV được ghi trong kết quả. |
| Chuẩn hóa biên độ | Giữ quy tắc đang dùng trong BanhmiTTS: `clip(x * 32767 / max(max(abs(x)),0.01), -32767,32767).astype(int16)`. Lưu peak/RMS trước chuyển đổi; không dùng việc chuẩn hóa để che đầu ra suy biến. Không thêm cắt silence ở pipeline ngoài. |
| Whisper | `small` đa ngôn ngữ, CPU, `language='en'`, `task='transcribe'`, `fp16=False`. Truyền đường dẫn WAV để nhánh đọc thực hiện resample 16 kHz; không truyền ndarray 22.05 kHz trực tiếp. |
| Decode | Khóa rõ temperature fallback `[0,.2,.4,.6,.8,1]`, compression ratio 2,4; log-prob −1; no-speech 0,6; `condition_on_previous_text=True`; không prompt; không beam/best-of override. Các lựa chọn còn lại gắn với source Whisper đã lưu. Reset RNG trước mỗi câu theo chính sách mục 5. |
| WER normalization | `ToLowerCase → RemovePunctuation → RemoveMultipleSpaces → Strip → ReduceToListOfListOfWords`. Lưu cả reference/hypothesis nguyên gốc, bản chuẩn hóa và số S/D/I/H. Không tự thêm chuẩn hóa số hoặc bỏ từ khó sau khi xem điểm. |
| UTMOSv2 | `fusion_stage3`, fold 0, checkpoint seed 42, CPU, `predict_dataset='sarulab'`; batch 1, workers 0. Đọc WAV bằng SoundFile float32, kiểm sr=22.050 rồi truyền `data=wave,sr=22050`. Source hiện tại resample về 16 kHz nội bộ. |
| Silence trong UTMOS | Đặt rõ `remove_silent_section=False`, đồng thời xác nhận config dataset không còn bật cắt silence trước khi predict. Không suy hành vi từ mặc định `True`: code hiện tại khôi phục config trước khi DataLoader đọc nên cờ mặc định có vấn đề cần kiểm tra. |
| Crop UTMOS | Có crop/mixing ngẫu nhiên trong đường prediction. Mỗi WAV chấm **3 lượt seed** và lưu từng điểm, lấy trung bình thành điểm của WAV. Gọi `num_repetitions=1` mỗi lượt để quản lý seed riêng; không nhầm `s42` hoặc `inference.num_tta=5` với số lượt thực tế. |

Hash weights đang có, đã đọc lại trong Q04:

| Scorer | SHA-256 |
|---|---|
| Whisper `small.pt` | `9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794` |
| UTMOSv2 `fold0_s42_best_model.pth` | `c8149d988e4bbf3f347e6966b5d769de347a5f8c59ffca1dc4bd4bf5b8585e57` |

Phiên bản đang có: Whisper `20250625`, UTMOSv2 `1.3.1.dev0`, JiWER `4.0.0`, NumPy `1.26.4`, ORT `1.23.2`. Danh tính này dùng để chuẩn bị môi trường mới, **không chứng minh phiên bản/hash ở mọi run cũ**. Source và metadata đã lưu; chưa chạy kiểm tra tương thích hoặc repeatability của toàn scorer stack.

**Không trộn điểm từ hai protocol:** Harvard cũ dùng WAV sau PCM/chuẩn hóa biên độ, còn ONNX-500 cũ cho UTMOS waveform float trực tiếp. Cheap WER cũ còn dùng `base.en`, chuẩn hóa khác và thiếu resample ndarray. Các bảng cũ giữ số nguyên bản kèm chú thích; bảng cuối phải chấm thống nhất. Không áp lỗi sample rate của cheap WER cho Whisper của toàn bộ các bảng 500/720 truyền WAV path.

## 5. Suy luận, randomness và phép lặp

- Batch 1, phoneme IDs cố định từ manifest; scales **[0.667, 1.0, 0.8]**. Không đổi scale riêng để cải thiện một model sau khi xem test.
- Protocol dự kiến **3 lượt sinh** mỗi câu/model với run seeds **1234, 1235, 1236**. Seed theo ID = 32 bit đầu của SHA-256 UTF-8 `q04-v0.2|<purpose>|<utterance_id>|<run_seed>`; không dùng `hash()` của Python. Cùng ID/lượt dùng chung seed qua các model; không đồng nghĩa latent noise giống nhau. Namespace đổi theo phiên bản; chưa có run v0.1 cần so lại.
- Reset Python, NumPy và Torch seed theo từng câu khi runtime dùng chúng. Với ONNX, phải kiểm random nodes và hành vi reset seed/session bằng thử lặp trước khi nhận ready. Chỉ `ort.set_seed` hoặc seed lấy mẫu calibration không đủ để hứa output bit-identical. Q05/Q06 xác định cơ chế thực tế; không đặt noise về 0 chỉ để né vấn đề.
- UTMOS dùng mục đích `utmos-crop-0/1/2`, ASR dùng `asr`, synthesis dùng `synthesis`; seed phụ thuộc ID và run seed, không phụ thuộc nhãn model. Lưu cả seed, hash audio và điểm từng lượt. Không giả định crop khác độ dài giữa các model là cùng một đoạn tuyệt đối.
- Ba lượt sinh đo biến thiên suy luận; ba lượt crop đo biến thiên scorer. Đơn vị thống kê ghép cặp vẫn là câu, không biến 720 × 3 × 3 thành số câu độc lập. Các lượt này cũng không thay thế nhiều seed huấn luyện.
- Đây là số lượt **đề xuất trước lần chạy mới**, chưa được thực thi. Nếu thay đổi vì nguồn lực, cập nhật version/hash protocol trước khi mở điểm mới, áp dụng đồng nhất cho ma trận đã chọn và công khai thay đổi.

## 6. Metric, lỗi và timing

| Mục | Quy tắc |
|---|---|
| WER chính | Macro WER: trung bình WER từng câu, phù hợp định nghĩa các bảng lịch sử. Với nhiều lượt sinh, trung bình theo lượt của từng câu trước, sau đó theo câu. Ghi rõ tỷ lệ hoặc phần trăm. |
| WER bổ sung | Corpus WER = tổng `(S+D+I)` / tổng số từ tham chiếu; lưu tổng theo từng lượt và quy tắc gộp. Không thay macro bằng corpus mà giữ nguyên nhãn/số. WER có thể >1, không chặn trên ở 1. |
| UTMOSv2 | Điểm dự đoán; lấy trung bình crop trong từng WAV, rồi lượt sinh trong từng câu, rồi câu. Không gọi là MOS người nghe. |
| Inference/audio lỗi | Lưu trạng thái riêng: exception, waveform rỗng, NaN/Inf, sai shape/sample rate, file hỏng. Lưu peak/RMS để đánh dấu im lặng; không biến lỗi thành WER=1. Không âm thầm bỏ câu khó. |
| Scorer lỗi | Ghi lỗi riêng, giữ WAV. Có thể retry lỗi hạ tầng với cùng WAV/seed và ghi log; không sinh lại đến khi được audio có điểm đẹp. |
| Báo cáo lỗi | Luôn ghi `n_expected/n_success/n_failed` từng model. Nếu có lỗi, không công bố trung bình của tập thành công như thể đủ mọi câu; báo tập giao để ghép cặp và cảnh báo nguy cơ chọn lọc. Loại ứng viên PTQ có lỗi kỹ thuật trước khi so chất lượng trên validation. |
| RTF | `time_i / duration_i`, dùng thời lượng của đúng output ở lần đo đó. Tách mean sentence RTF và `sum(time)/sum(duration)`; không tráo hai đại lượng. |
| Timing | Tách process/vòng sinh khỏi ASR/UTMOS; timer model-only bao quanh infer hoặc `sess.run`, không gồm tải model, phonemization, ghi WAV hay scorer. Nếu đo text-to-waveform thì có tên/cột riêng. |
| Q07 | Giữ kế hoạch CPU batch 1, intra-op 2/inter-op 1, 5 warm-up và 3 lượt đo cho protocol khởi đầu; cần khóa thiết bị, provider, runtime, nguồn điện, tải nền và thứ tự đo trước benchmark. Seed synthesis/crop khác với số lượt timing. Chưa có benchmark N150/Pi 5. |

Không cắt câu, thời lượng hoặc lựa chọn ID theo điểm của model. Mọi loại bỏ trước khi chạy phải dựa trên lỗi đầu vào và ghi lý do ở manifest. Q09 báo khoảng tin cậy/kiểm định đúng đơn vị câu và hiệu chỉnh các phép so sánh; không suy nhân quả decoder từ p-value khi recipe/data chưa kiểm soát.

## 7. Hồ sơ bắt buộc trước một run mới

Xem [PROTOCOL_Q04_v0.2.json](evidence/q03_q04_completion_2026-10-04/PROTOCOL_Q04_v0.2.json) và [mẫu hồ sơ run v0.2](evidence/q03_q04_completion_2026-10-04/RUN_RECORD_TEMPLATE_v0.2.json). File ghi `ready_for_final_evaluation=false` cho đến khi các điều kiện chưa rõ được giải quyết. Việc có hash protocol không đồng nghĩa run đã thực hiện hoặc được đăng ký trước các kết quả lịch sử. [v0.1](evidence/q04_2026-10-04/PROTOCOL_Q04_v0.1.json) được giữ để truy nguồn.

Trước chạy, lưu: protocol version/hash; lệnh đầy đủ; timestamp; source/commit thực dùng; checkpoint/ONNX hash; manifest hash; scorer source/weights/config/version; OS/hardware/runtime/provider/thread; seed/repetition; scopes PTQ và calibration/selection IDs. Sau chạy, lưu từng ID, WAV hash, thời lượng, timing, ASR transcript, S/D/I/H, UTMOS từng lượt và trạng thái lỗi.

Đầu ra phải có thư mục cố định riêng cho run. Không ghi đè gói Q01, các bảng lịch sử hoặc file trong `/tmp` đang được dùng làm bằng chứng. Không tự gán Git HEAD Windows cho code Linux đã train trước đó.

## 8. Điều kiện còn mở và trạng thái checklist

| Điều kiện | Trạng thái sau Q04 này | Việc tiếp tục |
|---|---|---|
| Giữ cả LJSpeech-500 và Harvard-720 | **Đã xác nhận với nhóm và ghi protocol.** | Hai khối/bảng riêng; có nhãn manifest và giới hạn. |
| ID/thứ tự/hash và map bảng kết quả | **Đã hoàn tất kiểm trên file đang lưu.** | Dùng các file manifest đã lưu, không tự tái lấy mẫu. |
| Lỗi giao dữ liệu ở danh sách calibration/selection v0.1 | **Đã xử lý ở ứng viên v0.2.** | 60 train chung; selection 98/94 theo model. Đây không phải chứng nhận lịch sử hoặc cùng validation cho mọi model. |
| Membership train/validation lịch sử, đặc biệt MRF/Piper | **Chưa đủ bằng chứng.** | Bổ sung nguồn lịch sử hoặc dùng thiết kế phát triển/đánh giá thay thế có nguồn rõ; không mặc định gán canonical. |
| Harvard từng dùng chọn cấu hình hay chưa | **Chưa xác nhận.** | Nhóm cung cấp nhật ký/quyết định. Mong muốn báo cáo hai bộ không trả lời thay lịch sử này. |
| Ngưỡng chất lượng để chấp nhận/chọn PTQ | **Chưa chốt.** | Q06 chốt ngưỡng WER/UTMOS và tiêu chí hiệu năng trước chọn scope; không tự đặt ngưỡng theo test đã xem. |
| Scorer thống nhất, xử lý lỗi, RNG thực thi | **Đã có recipe/source/hash; chưa có runner mới được kiểm thực thi.** | Q06/Q08 sửa selector, thử trên tập phát triển, kiểm repeatability và hash WAV; không dùng test để debug/chọn recipe. |
| Model/export và nguồn PTQ | **Q05 đã hoàn tất FP32 mới; còn Q06 cho PTQ.** | Dùng [bốn graph Q05 đã định danh](Q05_EXPORT_ONNX_VA_KIEM_CHUNG.md) để tạo PTQ. Không gán điểm lịch sử cho graph mới. Kiểm RNG Q05 còn giới hạn, không tự đóng kiểm runner Q04. |
| Benchmark/thống kê cuối | **Chưa chạy.** | Q07–Q09 sau khi đủ điều kiện. |

**Q04 vẫn “Một phần”: hồ sơ kiểm tra và protocol đã có, các manifest thực thi cuối chưa được xác nhận đủ.** Q05 đã hoàn tất export FP32; bước tiếp theo là chốt các điều kiện dữ liệu/selection và triển khai Q06. Protocol v0.2 đã lưu vẫn giữ nguyên, chưa đổi thành ready chỉ vì kiểm export đạt. Không cần tự động train lại để hoàn thành các phần này.

## 9. Tái kiểm và nguồn

- [audit_q04.py](evidence/q04_2026-10-04/audit_q04.py): tạo evidence/manifest từ Q01, kiểm SHA-256, so transcript bằng JiWER; không chạy TTS/ASR/UTMOS.
- [Q04_DATA_AUDIT.json](evidence/q04_2026-10-04/Q04_DATA_AUDIT.json): nguồn/hash, trùng nội dung, tập ứng viên và kiểm 27 bảng.
- [CURRENT_SCORER_IDENTITY.json](evidence/q04_2026-10-04/CURRENT_SCORER_IDENTITY.json): source/package/weight identity quan sát hiện tại.
- [Hồ sơ Q03](Q03_CAU_HINH_VA_LICH_SU_HUAN_LUYEN.md), [audit calibration trước đây](../../paper_revisions/thesis_ptq/round7_calibration_code.md), [audit evaluator trước đây](../../paper_revisions/thesis_ptq/round7_evaluation_code.md).
