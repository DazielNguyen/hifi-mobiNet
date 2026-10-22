# Kế hoạch: kiểm định nhân quả lỗi lệch frontend (Piper external trên Harvard-720)

*English: [plan-frontend-causal-test.md](plan-frontend-causal-test.md)*

Trạng thái: **đã chuẩn bị, chưa chạy.** Viết ngày 2026-10-23, trước khi có bất kỳ kết quả nào của phép kiểm định này.

## 1. Câu hỏi và bằng chứng hiện có

- **Thiết lập hiện tại.** Harvard-720 đưa phoneme espeak **en-us** vào mọi hệ thống. Config phát hành của checkpoint Piper external (`lj-med_1000.ckpt`) ghi giọng espeak **"en"** (Anh–Anh), trong khi mọi model của hifi-mobiNet đều được train bằng en-us.
- **Phép kiểm thăm dò** (`results/frontend-check/`) ủng hộ giả thuyết lệch frontend, nhưng mới chỉ ở mức tương quan:
  - WER của Piper tăng theo khoảng cách phoneme en/en-us của từng câu (ρ = +0,36), còn WER của Piper_no_VITS2_cpn gần như không đổi (ρ = +0,07).
  - Piper sai 34,2% các từ có phiên âm khác nhau giữa hai giọng, so với 7,5% ở các từ còn lại. Piper_no_VITS2_cpn sai 5,8% và 5,5%.
  - Difference-in-differences là +26,4 điểm [23,9; 29,1].
  - Khoảng 88% số lỗi vượt trội của Piper rơi vào các từ có phiên âm khác nhau.

**Phép kiểm định này hỏi:** đưa cho Piper đúng giọng của nó ("en") có xoá được phần lớn khoảng cách WER không? Và đưa en vào model của team, vốn được train bằng en-us, có làm nó kém đi không?

## 2. Thiết kế

Tất cả chạy trên CPU, Harvard-720, protocol giữ nguyên như `docs/evaluation.md` (seed 1234 cho mỗi câu, scales 0.667/1.0/0.8, `harvard/score.py`):

| Bộ | Hệ thống | Phoneme | Vai trò |
|---|---|---|---|
| `piper_enus` | Piper external | en-us (`results/manifests/harvard720.json`) | control cùng phiên; phải tái tạo đúng WAV Piper lịch sử |
| `piper_en` | Piper external | en (`results/manifests/harvard720_espeak_en.json`) | phép thử chính |
| `team_en` | Piper_no_VITS2_cpn (epoch 1403) | en | phép thử ngược (lệch frontend với model train bằng en-us) |
| (đã lưu) | Piper_no_VITS2_cpn | en-us | `results/piper-no-vits2-cpn/` |

- **Manifest en:** dựng bằng `scripts/evaluation/harvard/build_manifest.py`. Script này dựng lại manifest en-us trước và thấy trùng hoàn toàn.
  - Manifest en khác ở 691 trên 720 câu.
  - Mọi ký hiệu của en đều có trong bảng phoneme-ID của Piper, và bảng ID của Piper với của project trùng nhau ở cả 157 ký hiệu chung.
- **Synthesis Piper:** `scripts/evaluation/harvard/synthesize_piper.py`, dùng code VITS gốc của Piper, không sửa (`vendor/piper`), theo đúng quy trình của `synth_piper.py` lịch sử.
- **Chấm điểm:** cả ba bộ mới được chấm trong cùng một phiên, nên phép so UTMOSv2 giữa Piper en và en-us là so cùng phiên. WER tái tạo chính xác giữa các phiên.

## 3. Dự đoán (khai báo trước khi chạy)

| | Dự đoán nếu lệch frontend gây ra phần lớn khoảng cách | Cách hiểu |
|---|---|---|
| **P1 (chính)** | Macro WER của `piper_en` < `piper_enus`; CI bootstrap ghép cặp 95% của hiệu số nằm hoàn toàn dưới 0 | lệch frontend làm Piper kém đi |
| **P2** | `dialect_check` của `piper_en` (đối chứng: Piper_no_VITS2_cpn): difference-in-differences ≤ một phần ba giá trị của lần đầu (≤ +0,088), hoặc CI chứa 0 | phần lỗi vượt trội theo phương ngữ biến mất |
| **P3 (chiều ngược)** | Macro WER của `team_en` > WER en-us đã lưu; CI nằm hoàn toàn trên 0 | cùng kiểu lệch đó làm hại một model train bằng giọng kia |
| **P4 (mô tả)** | Khoảng cách WER giữa Piper_no_VITS2_cpn (en-us) và Piper (en) nhỏ hơn mức 11,35 điểm hiện tại | còn lại bao nhiêu khoảng cách |

- **P1 và P2 đúng:** chứng minh được lệch frontend gây ra các lỗi đặc trưng theo phương ngữ. Phần khoảng cách còn lại (P4) mới là phần mà khác biệt về train, checkpoint và code có thể giải thích.
- **P1 sai:** giả thuyết bị bác bỏ đối với Piper, và khoảng cách có nguyên nhân khác.
- **P3** là bằng chứng bổ trợ, không bắt buộc.
- **UTMOSv2** được báo cáo cho mọi bộ nhưng không thuộc các dự đoán.

## 4. Chạy

Trước khi chạy:
- commit các script đã chuẩn bị và manifest en (script khởi chạy sẽ kiểm tra điều này);
- bảo đảm máy rảnh và có tiến trình giữ WSL chạy (keep-alive).

Trong WSL:

```sh
bash <WSL_PROJECT_HOME>/hifimobinet-private/launch_frontend_test.sh
```

Script mở session tmux `frontend-test`, chạy `scripts/evaluation/harvard/run_frontend_test.sh` với:
- `OUT=training_output/frontend-test`;
- `RESULTS=<repository>/results/frontend-test`.

| Stage | Thời gian dự kiến |
|---|---|
| Synthesis Piper en-us và en, synthesis team en (3 × 720 câu) | khoảng 6 phút |
| Chấm điểm 3 bộ WAV (khoảng 45 phút mỗi bộ) | khoảng 2 giờ 15 phút |
| So sánh và kiểm tra phương ngữ | khoảng 2 phút |
| **Tổng** | **khoảng 2 giờ 25 phút** |

**Theo dõi đợt chạy:**
- `tail -f <clone training>/training_output/frontend-test/run.log` cho biết tiến độ.
- Một popup báo DONE hoặc FAILED.
- Nếu bị gián đoạn, chạy lại `run_frontend_test.sh` với cùng `OUT`: nó tiếp tục sau stage cuối cùng đã xong.

## 5. Sau khi chạy xong

1. Kiểm tra `piper_enus_synth_record.json → reference_inventory.wavs_identical`.
   - **720/720:** môi trường tái tạo đúng synthesis Piper lịch sử.
   - **Không đủ 720:** ghi rõ điều đó. Phép so en với en-us vẫn hợp lệ, vì cả hai bộ đều được tạo trong cùng lần chạy này.
2. Đánh giá P1–P4 từ `piper_en_vs_piper_enus.json`, `dialect_check_piper_en.json`, `team_en_vs_team_enus.json` và `team_enus_vs_piper_en.json`. Báo cáo mọi dự đoán, kể cả dự đoán sai.
3. Viết `results/frontend-test/README.md` và `README.vi.md`, và thêm các mục vào `results/manifest.json`.
4. **Nếu P1 đúng, sửa mọi phép so sánh với Piper:**
   - trong `results/README(.vi).md`: dòng Piper và các ghi chú của nó;
   - trong README của từng model: mọi kết quả "so với Piper";
   - trong `docs/q2-readiness-seq-par(.vi).md`: các nhận định SEQ/MRF/baseline so với Piper, kể cả của thế hệ 1.

   Các dòng Piper lịch sử vẫn được giữ làm bản ghi, kèm nhãn là đã đo với phoneme bị lệch.
5. Commit theo từng phạm vi riêng, không có attribution. Tác giả tự push.

## 6. Prompt cho phiên sẽ chạy phép kiểm định này

> Chạy phép kiểm định nhân quả về frontend theo kế hoạch trong `hifi-mobiNet/docs/plan-frontend-causal-test.md`.
> - Trước tiên kiểm tra mục 4 (script và manifest en đã commit, máy rảnh, WSL keep-alive). Sau đó khởi chạy `<WSL_PROJECT_HOME>/hifimobinet-private/launch_frontend_test.sh` và theo dõi `run.log`.
> - Không thay đổi các dự đoán ở mục 3, protocol hay scorer.
> - Khi log ghi ALL DONE, làm mục 5. Báo cáo mọi dự đoán, kể cả dự đoán sai.
> - Chỉ commit khi tôi yêu cầu, không có attribution. Trao đổi bằng tiếng Việt; tài liệu viết bằng tiếng Anh, kèm README.vi.md tiếng Việt.
