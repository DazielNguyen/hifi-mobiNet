# Hồ sơ quantization

**Code hiện hành cho ba Piper là `piper-common97-v1`, đã tích hợp ngày
05/11/2026.** Các báo cáo ngày 04/11 giữ nguyên nội dung và trạng thái tại
thời điểm viết để truy vết quá trình tìm lỗi. Dòng “chưa tích hợp” trong báo
cáo cũ không mô tả code hiện hành.

| Tài liệu | Vai trò |
|---|---|
| [Benchmark N100 common97, 05/11](../../results/edge-n100-openvino/2026-11-05-common97/README.vi.md) | 12.960 bản ghi timed; RTF giảm 12,73–13,88%; đã kiểm work N100: preparation/IR, cạnh FQ và 54 WAV, gồm 18 WAV INT8 >10 giây; chất lượng độc lập còn pending |
| [Tích hợp common97, 05/11](piper-common97-integration-2026-11-05/README.vi.md) | Phạm vi INT8 đang dùng, kiểm thử code thực và cách cập nhật N100 |
| [Nguyên nhân lỗi đuôi và cách sửa, 04/11](piper-ptq-tail-fix-2026-11-04/README.vi.md) | Báo cáo chính: CumSum FQ, can thiệp đối chứng, phổ WAV, kết quả cũ/mới |
| [Kiểm kê ba cấu hình, 04/11](piper-three-configs-2026-11-04/README.vi.md) | Phạm vi kiến trúc, ứng viên quantization, inventory operator và weights |
| [Đối chiếu WAV ban đầu, 04/11](piper-long-audio-review-2026-11-04/README.vi.md) | Quan sát tín hiệu đầu tiên và giới hạn của giả thuyết lỗi sau 8–10 giây; có trước phân tích nhân quả |
| [Kiểm kê sáu model, 02/11](six-model-audit-2026-11-02/README.vi.md) | Khảo sát trước đó; không phải policy triển khai common97 |

## Nội dung lưu trong Git

Ba báo cáo ngày 04/11 được lưu cùng Markdown, PNG dùng trong báo cáo,
CSV/JSON số liệu, hash nguồn, bản chụp code và script phân tích. Chúng có
các vai trò khác nhau: kiểm kê kiến trúc → khoanh vùng tín hiệu → chứng minh
nguyên nhân và đối chứng cách sửa. Không loại bỏ các bước này như file tạm.

Các bản Word, HTML và SVG của báo cáo lỗi đuôi là bản xuất để đọc/in/trình
bày. Chúng được giữ tại đường dẫn local hiện có, ngoài Git bằng các rule
chỉ áp dụng cho thư mục báo cáo này. WAV tiếp tục nằm ngoài Git theo quy
ước chung của repo. Việc dọn Git không xóa, đổi tên hoặc sửa các file đó.

| Loại file trong báo cáo lỗi đuôi | Trong Git | Bản local |
|---|---|---|
| Markdown, PNG, CSV/JSON, script và manifest | Có | Giữ nguyên bytes |
| `BAO-CAO-PTQ-PIPER-common97.docx` và checksum riêng | Không | Giữ để mở trong Word |
| `BAO-CAO.html`, `NGHE-DOI-CHIEU.html` | Không | Giữ để đọc/nghe offline |
| `figures/*.svg` | Không | Giữ để xuất hình vector |
| `audio/**/*.wav` | Không | Giữ audio gốc/thử nghiệm và hash trong evidence |

Vì thế, một checkout chỉ từ Git vẫn đọc được báo cáo và hình PNG, nhưng
không có các bản xuất và WAV để phát. Các liên kết audio/HTML trong báo cáo
lịch sử dành cho bộ hồ sơ local đầy đủ. Muốn nghe cần copy thêm các file
local, rồi kiểm hash; không thay bằng audio khác hoặc tự báo pass khi thiếu.

## Kiểm tra tính toàn vẹn

`SHA256SUMS.archived-reports` ở thư mục này chỉ liệt kê các file được commit
của ba báo cáo ngày 04/11. Từ thư mục gốc repo trên Linux:

```bash
cd docs/quantization
sha256sum --check SHA256SUMS.archived-reports
```

`SHA256SUMS` và `VERIFICATION.json` **bên trong từng báo cáo** là record gốc
được giữ nguyên. Riêng manifest báo cáo lỗi đuôi bao gồm cả HTML/SVG/WAV
local; file Word có checksum riêng. Không dùng kết quả verification lịch
sử thay cho kiểm tra mới, và không gọi manifest của bộ offline là manifest
đầy đủ của một checkout Git.

Các script dựng lại báo cáo có thể cần nguồn `.local/ptq-ablation-20261104`,
`.local/ptq-listening-20261104` hoặc WAV ở Downloads như hướng dẫn gốc.
Chúng không cần chạy để đọc tài liệu. Bản chụp code trong `evidence/code/`
dùng để đối chiếu lịch sử; code N100 hiện hành nằm trong `src/` và `scripts/`.

Phần dọn Git này không thay đổi policy common97, không chạy lại PTQ,
inference, scoring, benchmark hoặc training; không sửa kết quả nghiên cứu.
