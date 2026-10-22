# Kiểm tra frontend: Piper external và phoneme en-us của Harvard-720

*English: [README.md](README.md)*

**Phân tích thăm dò trên kết quả đã lưu** (2026-10-23). Không synthesis hay chấm điểm gì mới. Phép kiểm định nhân quả đã được chuẩn bị trong `docs/plan-frontend-causal-test.vi.md` và chưa chạy.

## Vì sao làm phân tích này

Trên Harvard-720, Piper_no_VITS2_cpn có macro WER 6,12%, so với 17,47% của checkpoint Piper external. Phần lớn khoảng cách là lỗi thay thế (529 trên 632 lỗi từ ít hơn).

**Thiết lập có thể giải thích điều đó:**
- Harvard-720 đưa phoneme espeak **en-us** vào mọi hệ thống (`results/manifests/harvard720.json`).
- Config phát hành của checkpoint Piper ghi giọng espeak **"en"** (`docs/evidence/claim-evidence-2026-10-07/source-records/public/public_piper_lj_med_config.json`). Giọng "en" của espeak-ng là tiếng Anh–Anh.
- Model card mô tả đây là giọng Anh–Mỹ, và mọi model của hifi-mobiNet đều được train bằng en-us (`config.json` của dữ liệu train).
- Bảng phoneme-ID của Piper và của project trùng nhau ở cả 157 ký hiệu chung. Vì vậy input vẫn hợp lệ với Piper, chỉ là được phiên âm theo một phương ngữ khác với phương ngữ mà config của nó ghi.

## Phương pháp

`scripts/evaluation/harvard/dialect_check.py` tạo ra `dialect_check.json`:
- **Phiên âm lại:** mọi câu Harvard được phiên âm lại bằng chính frontend đã dựng manifest (`banhmi_phonemize`). Kết quả en-us trùng manifest ở 720/720 câu; kết quả en dùng để so sánh.
- **Cấp câu:** khoảng cách chỉnh sửa giữa phoneme en và en-us, bỏ dấu trọng âm và chuẩn hoá theo độ dài, được so với WER của từng hệ thống.
- **Cấp từ:**
  - Mỗi từ tham chiếu được tính là lỗi nếu phép căn chỉnh JiWER (cùng phép biến đổi của scorer) thay thế hoặc xoá nó.
  - Một từ được gọi là "khác" nếu phoneme en và en-us của nó khác nhau.
  - Chỉ dùng 551 câu mà các từ phoneme ghép được một-một với từ tham chiếu (4.246 từ).
- **Đối chứng:** Piper_no_VITS2_cpn làm đối chứng cho độ khó của từ. Difference-in-differences có khoảng bootstrap theo câu (5.000 lần lấy mẫu lại).

## Kết quả

| Nhóm khoảng cách (n) | WER Piper | WER Piper_no_VITS2_cpn | Chênh |
|---|---:|---:|---:|
| thấp nhất (238) | 10,42% | 4,72% | +5,70 |
| giữa (220) | 17,24% | 6,74% | +10,51 |
| cao nhất (262) | **24,06%** | 6,87% | **+17,19** |

Tương quan Spearman giữa khoảng cách và WER: Piper +0,360 (p 2e-23); Piper_no_VITS2_cpn +0,074 (p 0,05); khoảng cách Piper − Piper_no_VITS2_cpn +0,303.

| Tỉ lệ sai từ | từ có phoneme en và en-us khác nhau (1.458) | các từ còn lại (2.788) | tỉ số |
|---|---:|---:|---:|
| Piper external | **34,2%** | 7,5% | 4,56 |
| Piper_no_VITS2_cpn | 5,8% | 5,5% | 1,05 |

- **Difference-in-differences:** **+26,4 điểm** [+23,9; +29,1].
- **Lỗi thêm của Piper nằm ở đâu:** 88% số lỗi từ mà Piper sai thêm so với Piper_no_VITS2_cpn, trong các câu này, rơi vào những từ có phoneme khác nhau giữa hai giọng.
- **Trên các từ còn lại:** hai hệ thống chỉ cách nhau 2 điểm (7,5% so với 5,5%).
- **Ví dụ trong Harvard list 1** (từ: en-us → en; Whisper nghe Piper thành):
  - background: `bˈækɡɹaʊnd` → `bˈakɡɹaʊnd`; "book ground";
  - hogs: `hˈɑːɡz` → `hˈɒɡz`; "herbs";
  - chopped: `tʃˈɑːpt` → `tʃˈɒpt`; "sharp";
  - four: `fˈoːɹ` → `fˈɔː`; "fair".

## Cách đọc

- **Điều được ủng hộ:** dữ liệu khớp với giả thuyết lệch frontend là nguồn chính gây thêm lỗi ASR cho Piper trên Harvard-720. Piper sai ở các từ đặc trưng theo phương ngữ, còn model train bằng en-us thì không.
- **Điều chưa chứng minh:**
  - **Quan hệ nhân quả:** phân tích này chỉ là tương quan. Phép kiểm định nhân quả sẽ synthesis Piper bằng chính phoneme "en" của nó, cộng chiều ngược lại cho model của team, với các dự đoán khai báo trước (`docs/plan-frontend-causal-test.vi.md`).
  - **Giọng lúc train:** được suy ra từ config phát hành, không phải từ dữ liệu train đã tiền xử lý của Piper, vì dữ liệu đó không có.
  - **Giới hạn đo lường:** transcript của Whisper-small, mỗi câu sinh một lần, và phân tích cấp từ chỉ phủ 551 câu ghép được.
- **Hệ quả cho tới khi có phép kiểm định nhân quả:** mọi phép so sánh trên Harvard-720 với checkpoint Piper external trong repository này đều đã đưa vào phoneme bị lệch. Điều đó áp dụng cho các model thế hệ 2 và cho baseline, MRF, SEQ lịch sử. Hãy đọc các phép so sánh đó là đang bất lợi cho Piper.

## Tái lập

Trong môi trường có `banhmi_phonemize`, JiWER và SciPy (môi trường scorer), chạy lệnh ở mục "Reproduction" của [README.md](README.md).

Manifest en cho phép kiểm định nhân quả, `results/manifests/harvard720_espeak_en.json`, được dựng bằng `scripts/evaluation/harvard/build_manifest.py --voice en`. Nó khác ở 691 trên 720 câu, và mọi ký hiệu trong đó đều có trong bảng của Piper.
