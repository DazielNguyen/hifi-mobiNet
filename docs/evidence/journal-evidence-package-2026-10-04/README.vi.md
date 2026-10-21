# Gói bằng chứng journal ngày 2026-10-04 (nhập ngày 2026-10-22)

*English: [README.md](README.md)*

Thư mục này chứa `journal-evidence-package` của BanhmiTTS ngày 2026-10-04: bằng chứng thế hệ 1
cho baseline-resblock2, Parallel-IR (MRF), Sequential-IR (SEQ) và checkpoint Piper bên ngoài.
Nội dung gồm:
- claim ledger J-E001–J-E033 (`claim-ledger.json`, `evidence/claim_table.md`);
- tài liệu bối cảnh (`JOURNAL_EVIDENCE_CONTEXT.md`, tiếng Việt);
- các bản ghi về Harvard, chia dữ liệu, kiến trúc và ONNX/PTQ (`evidence/B_` … `F_`);
- ghi chú của tác giả (`evidence/author_notes/`), gồm checklist Q2 trước đây và ghi chú MRF/SEQ.

Tổng quan và đánh giá hiện tại: [`../../q2-readiness-seq-par.vi.md`](../../q2-readiness-seq-par.vi.md).

**Cách nhập** (`../historical-import-2026-10-22.json`, mỗi file một bản ghi):
- **Đã có trong repository (32 file):** những file đã được Git theo dõi với cùng nội dung thì không chép lại. Nhóm này gồm các file kết quả Harvard và LJ-500, kết quả ONNX lịch sử, manifest Harvard và split canonical. Bản ghi nêu đường dẫn hiện có, ví dụ `results/historical/harvard/`, `results/manifests/` hoặc `docs/evidence/claim-evidence-2026-10-07/`. Vì vậy các link tương đối trong gói trỏ tới những file đó sẽ không mở được ở đây.
- **Chép nguyên byte:** các file không chứa đường dẫn máy.
- **Bản đã che đường dẫn:** file có đường dẫn máy được chép với các ký hiệu thay thế, không đổi gì khác: `<WORKSPACE>/`, `<WINDOWS_USER_HOME>/`, `<WSL_PROJECT_HOME>/`, `<WSL_USER_HOME>/`, `<CLAUDE_PROJECT_SLUG>`, `<HOST>`, `<EMAIL>`.
  - Bản ghi nhập giữ SHA-256 của cả bản gốc lẫn bản đã che cho từng file.
  - Bản gốc được giữ riêng trong WSL (`<WSL_PROJECT_HOME>/hifimobinet-private/`), ngoài OneDrive và Git, và không dùng để phát hành.
  - Các script đã che chỉ là bản ghi, không chạy được nguyên trạng.
- **Kiểm tra:**
  - quét mẫu thông tin xác thực trên mọi file đã chép;
  - quét đường dẫn máy độc lập sau khi nhập, không tìm thấy chỗ nào;
  - kiểm lại hash của mọi file đích.

**Cách đọc:**
- **Không phải kết luận hiện tại.** Các đánh giá trong gói (`SUPPORTED`, `CONTRADICTED`, …) là của ngày 2026-10-04.
- **Ghi chú của tác giả là bối cảnh, không phải phép đo.** Đặc biệt, `evidence/author_notes/MRF_vs_SEQ_comparison_Q2.md` có những nhận định mà ledger đã đánh dấu là bị bác bỏ (J-E030, J-E031).
- **Bằng chứng mới hơn.** Các run thế hệ 2 tháng 10/2026 thay thế nhiều mục còn mở, ví dụ phần ONNX và PTQ của Q05/Q06 cho các model Config A. Xem bản tổng quan.
