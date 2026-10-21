# Phiên đo thời gian trên máy rảnh, 2026-10-21

*English: [README.md](README.md)*

Đây là các bản ghi của một phiên đo thời gian trên CPU, ngày 2026-10-21, 09:05–09:09 +07.
Phiên chạy sau khi chuỗi train Piper-IR đã hoàn tất và không còn trainer nào.

Ba model được synthesis trên Harvard-720 liền nhau, mỗi model trong một tiến trình
`scripts/evaluation/harvard/synthesize.py` mới:
1. Piper_no_VITS2_cpn, làm control (epoch 1403);
2. Piper-ParIR (epoch 1486);
3. Piper-SeqIR (epoch 1455).

Protocol: PyTorch FP32, 12 thread, độ ưu tiên bình thường, chỉ bấm giờ lời gọi forward,
mỗi câu đo một lần, không warm-up.

| Model | RTF trung bình | So với control | Nhanh hơn control |
|---|---:|---:|---:|
| Piper_no_VITS2_cpn (control) | 0.0510 | — | — |
| Piper-ParIR | 0.0392 | −23,2% | 653 / 720 |
| Piper-SeqIR | 0.0340 | −33,3% | 690 / 720 |

- **Control:** WAV giống từng byte với WAV ngày 2026-10-10 (720/720). RTF của nó bằng 1,21 lần giá trị của chính nó ngày 2026-10-10, nên RTF của phiên này chỉ so sánh được trong nội bộ phiên.
- **ParIR:** WAV trùng với lần đánh giá 2026-10-16 (720/720).

Phân tích, giới hạn và kết quả chất lượng nằm trong
[`../piper-seqir-no-vits2-cpn/README.vi.md`](../piper-seqir-no-vits2-cpn/README.vi.md), mục 4.
Bản ghi synthesis SeqIR của phiên này là `../piper-seqir-no-vits2-cpn/harvard720_synth_record.json`.

| File | Nội dung |
|---|---|
| `piper_no_vits2_cpn_ctrl_synth_record.json` | Synthesis control: hash, protocol, môi trường, thời gian và SHA-256 WAV theo từng câu |
| `piper_parir_timing_synth_record.json` | Synthesis ParIR trong phiên này, cùng các trường |
| `session_control_parir.json`, `session_control_seqir.json` | Tính tái lập của control, độ trôi giữa các phiên, phép thử RTF ghép cặp với control |
| `parir_determinism.json` | Hash và thời lượng WAV của ParIR so với lần đánh giá 2026-10-16 |
| `machine_state_before.txt`, `machine_state_after.txt` | Load average, mức dùng GPU và các tiến trình bận nhất trước và sau khi đo |
