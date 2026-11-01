# Đo sáu model trên Intel N100 / OpenVINO CPU, ngày 01-11-2026

*English: [README.md](README.md)*

Phiên đo đã hoàn tất: **6 model × 2 precision × 3 vòng × 720 câu = 25.920 bản ghi timed**. PC đã kiểm gói, hash nguồn và tính lại thống kê từ từng câu. Đây là một phiên N100 với NNCF INT8 chọn lọc, có backend và protocol riêng so với các phép đo PyTorch/ONNX Runtime trên workstation.

**Timing đã đủ; chất lượng các artifact OpenVINO INT8 này chưa được chứng nhận.** Hai mẫu thời lượng bất thường được giữ nguyên và mô tả bên dưới. Việc nhập evidence không chạy thêm training, inference, export, quantization hoặc scoring.

## Kết quả và cách đọc

RTF chính trong bảng = **tổng thời gian inference / tổng thời lượng audio thực sinh**, gộp cả ba vòng; càng nhỏ càng nhanh. Mean latency dùng cùng 2.160 observations mỗi model/precision. Peak RSS là đỉnh bộ nhớ của process, gồm runtime/input/model; không phải chỉ bộ nhớ tham số.

| Model | RTF FP32 | RTF INT8 | Mean latency FP32 / INT8 (ms) | Peak RSS FP32 / INT8 (GiB) |
|---|---:|---:|---:|---:|
| `baseline-resblock2` | 0.07095 | 0.06880 | 198.36 / 211.17 | 11.08 / 3.83 |
| `parallel-ir` | 0.07053 | 0.06739 | 178.28 / 169.42 | 1.64 / 2.86 |
| `sequential-ir` | 0.06045 | 0.05511 | 154.63 / 146.34 | 1.55 / 3.02 |
| `piper-no-vits2-cpn` | 0.06457 | 0.05870 | 164.43 / 147.30 | 1.80 / 2.13 |
| `piper-parir-no-vits2-cpn` | 0.06322 | 0.05820 | 166.96 / 154.26 | 1.22 / 1.77 |
| `piper-seqir-no-vits2-cpn` | 0.05326 | 0.04788 | 137.25 / 123.52 | 1.27 / 1.80 |

Piper-SeqIR có RTF thấp nhất trong phiên này. RTF INT8 chọn lọc thấp hơn FP32 10,09%. Mọi RTF từng câu được ghi đều nhỏ hơn 1, nhưng kết luận này chỉ áp dụng cho vùng model inference đã đo; chưa chứng minh latency streaming hoặc text-to-WAV đầu cuối.

Baseline INT8 có RTF thấp hơn nhưng **mean latency cao hơn** FP32. Tổng audio INT8 dài hơn (6.629,73 so với 6.039,01 giây). Native RNG không được khóa giống nhau giữa các lần chạy, nên giảm RTF không đồng nghĩa cùng mức tăng tốc xử lý một waveform cố định.

Các thống kê phân phối dùng `latency_s / audio_seconds` của từng câu; khác RTF trọng số thời lượng ở bảng chính. P95 theo nearest-rank, vị trí `ceil(0.95 × n)`.

| Model | Mean RTF FP32 / INT8 | Median RTF FP32 / INT8 | P95 RTF FP32 / INT8 |
|---|---:|---:|---:|
| `baseline-resblock2` | 0.07018 / 0.06932 | 0.06745 / 0.07129 | 0.08476 / 0.08613 |
| `parallel-ir` | 0.07060 / 0.06753 | 0.06853 / 0.06381 | 0.08057 / 0.08380 |
| `sequential-ir` | 0.06056 / 0.05529 | 0.05876 / 0.05093 | 0.07111 / 0.07030 |
| `piper-no-vits2-cpn` | 0.06474 / 0.05887 | 0.06235 / 0.05502 | 0.07833 / 0.07548 |
| `piper-parir-no-vits2-cpn` | 0.06331 / 0.05830 | 0.06231 / 0.05618 | 0.06918 / 0.06762 |
| `piper-seqir-no-vits2-cpn` | 0.05339 / 0.04801 | 0.05228 / 0.04578 | 0.06001 / 0.05945 |

RTF trọng số thời lượng của từng vòng:

| Model | FP32: r0 / r1 / r2 | INT8: r0 / r1 / r2 |
|---|---|---|
| `baseline-resblock2` | 0.06986 / 0.07313 / 0.06977 | 0.06843 / 0.06800 / 0.06998 |
| `parallel-ir` | 0.07070 / 0.07035 / 0.07054 | 0.06711 / 0.06853 / 0.06655 |
| `sequential-ir` | 0.06004 / 0.06005 / 0.06124 | 0.05455 / 0.05463 / 0.05614 |
| `piper-no-vits2-cpn` | 0.06492 / 0.06399 / 0.06481 | 0.05841 / 0.05831 / 0.05939 |
| `piper-parir-no-vits2-cpn` | 0.06274 / 0.06396 / 0.06297 | 0.05747 / 0.05878 / 0.05836 |
| `piper-seqir-no-vits2-cpn` | 0.05289 / 0.05275 / 0.05414 | 0.04810 / 0.04857 / 0.04698 |

Ba vòng lặp lại cùng 720 câu, không phải 2.160 câu test độc lập. Đây là thống kê mô tả một phiên; chưa khẳng định ý nghĩa thống kê qua training seed, thiết bị, phiên đo hay power profile, và không chứng minh tác động riêng của decoder. Không gắn điểm chất lượng cũ vào bản NNCF INT8 mới.

Nguồn: [CSV FP32 gốc](evidence/results-n100-t4-20261101T080423Z/summary_fp32.csv), [CSV INT8 gốc](evidence/results-n100-t4-20261101T080423Z/summary_int8.csv), [hash từng cell và bảng tổng hợp](evidence/results-n100-t4-20261101T080423Z/summary.json), [PC tính lại](verification/received/rtf-recomputed.csv).

## Danh tính model và phạm vi INT8

| Model ID | Cohort | Selected checkpoint |
|---|---|---|
| `baseline-resblock2` | historical Q05 FP32 | `best-epoch=1489-val_loss_mel=19.7882.ckpt` |
| `parallel-ir` | historical Q05 FP32 | `best-epoch=1386-val_loss_mel=18.7681.ckpt` |
| `sequential-ir` | historical Q05 FP32 | `best-epoch=1442-val_loss_mel=19.0136.ckpt` |
| `piper-no-vits2-cpn` | Config A, round-2 FP32 | `best-epoch=1403-val_loss_mel=19.7719.ckpt` |
| `piper-parir-no-vits2-cpn` | Config A, round-2 FP32 | `best-epoch=1486-val_loss_mel=19.0069.ckpt` |
| `piper-seqir-no-vits2-cpn` | Config A, round-2 FP32 | `best-epoch=1455-val_loss_mel=19.0332.ckpt` |

Ba model đầu thuộc nhóm Q05 lịch sử; ba model cuối là Config A, lấy FP32 từ đợt export round 2 đã sửa flow weight norm. Piper external, Low và High không nằm trong sáu model này. Tất cả xuất audio 22.050 Hz. [models.json](evidence/models.json) lưu hash checkpoint nguồn, ONNX, config và export record. Checkpoint không nằm trong gói và không được load khi nhập kết quả.

**INT8 là lượng tử hóa chọn lọc.** NNCF quantize Conv đủ điều kiện trong `/enc_p/`, `/dp/`, `/flow/`, với CPU, preset MIXED, fast bias correction và chung 60 câu calibration chỉ thuộc train. Weighted operations của decoder giữ FP32. Policy đã sửa loại masked duration-spline khỏi activation quantization, vẫn giữ các Conv `pre/convs/proj` của flow đủ điều kiện. Bản ghi runtime cho thấy 121 INT8 weighted kernels mỗi model lịch sử và 97 mỗi model Config A.

Có 13 FP32 parity cases/model giữa OpenVINO và ONNX Runtime với matched noise (78 tổng cộng); đều đạt NRMSE ≤ 0,001 và max absolute error ≤ 0,005. Đây là kiểm kỹ thuật FP32, không phải đánh giá chất lượng INT8. Đã kiểm 36 pilot WAV (3 câu × 2 precision × 6 model); pilot WAV được peak-normalize riêng để nghe, khác waveform trực tiếp dùng đo timing.

## Máy, môi trường và phương pháp đo

| Thiết lập | Giá trị ghi nhận |
|---|---|
| Máy | Intel N100; 4 logical CPU; Ubuntu cài trực tiếp |
| Platform | `Linux-7.0.0-31-generic-x86_64-with-glibc2.39` |
| Python | 3.12.3 |
| OpenVINO / NNCF | 2026.4.1 / 3.4.0 |
| NumPy / ONNX / ONNX Runtime | 2.4.6 / 1.23.2 / 1.31.1 |
| Power trước/sau | `balanced`; `intel_pstate`; governor `powersave`; EPP `balance_performance` |
| Runtime | CPU, batch 1, đồng bộ, 4 threads, 1 stream, `LATENCY`, precision hint `f32`, execution mode `ACCURACY` |
| Input | Phoneme IDs Harvard-720; scales `[0.667, 1.0, 0.8]`; mẫu số RTF là thời lượng output thực tế |
| Warm-up | 10 calls trải theo độ dài câu; không warm-up riêng mọi dynamic shape |
| Lặp | 3 vòng; process worker mới cho mỗi model/precision/vòng |
| Thứ tự | Shuffle các cell bằng seed 1234, đảo ở r1; thứ tự câu dùng seed `1234 + round` |
| Bắt đầu/kết thúc UTC | 08:04:24.224763 / 09:18:46.163322 ngày 01-11-2026 |
| Giờ địa phương UTC+07 | 15:04:24 / 16:18:46; khoảng 74 phút 22 giây |

Timer chỉ bao quanh `InferRequest.infer(share_outputs=True)`, bao gồm overhead gọi Python/OpenVINO. Load/compile, phonemization, chuẩn bị/gán input, kiểm/copy output và ghi file nằm ngoài timer. Chi phí lần đầu gặp shape mới sau warm-up vẫn có thể được tính. Chuỗi governor `powersave` dưới `intel_pstate` không chứng minh CPU bị cố định ở xung thấp; không có trace liên tục về xung hoặc throttling.

Environment lock, trạng thái nguồn, lệnh chạy và môi trường đầu/cuối từng worker được lưu. Sensor cao nhất ghi nhận 76 °C, chỉ lấy ở ranh giới cell; chưa loại trừ throttling hoặc swap trong lúc đo.

## Mẫu bất thường và evidence còn thiếu

1. **Baseline FP32, r1 (vòng thứ hai), Harvard idx 379 tính từ 0:** “That move means the game is over.” Output dài 87,0632 giây; inference 12,5710 giây; RTF 0,14439. Cùng câu ở hai vòng FP32 khác dài 2,3104 và 3,0302 giây. Peak RSS của cell này 11,0809 GiB; hai cell FP32 còn lại 2,5242 và 2,4918 GiB. Sự xuất hiện cùng cell chưa chứng minh quan hệ nhân quả.
2. **Piper-SeqIR FP32, r2, idx 332:** “The idea is to sew both edges straight.” Output dài 7,3491 giây, vẫn nằm trong mọi bảng tổng hợp.

Chi tiết và đối chiếu các vòng: [duration-observations.json](verification/received/duration-observations.json). Không xóa, cắt hoặc tự sửa mẫu. Nếu phân tích độ nhạy sau này, phải tách kết quả đó khỏi bảng gốc này.

Suite không lưu WAV của từng lần timed, nên không có đúng hai waveform dài trên để nghe lại. Listening/WER/UTMOS cho artifact OpenVINO INT8 mới vẫn pending. Nguyên nhân outlier và kết quả ở số thread khác chưa được xác lập. Không chuyển kết luận chất lượng từ artifact cũ sang đây.

## Lỗi calibration và bản sửa đã dùng

| Giai đoạn | Evidence | Diễn giải |
|---|---|---|
| Chẩn đoán ban đầu | [report](diagnostics/numeric-failure/diagnostic-report.json), [log](diagnostics/numeric-failure/diagnostic.log), [numeric-event](diagnostics/numeric-failure/numeric-event.json) | Thống kê calibration bị overflow; không dùng làm kết quả timing đạt. |
| Probe SeqIR | [report](diagnostics/spline-probe/probe-report.json), [empty-branch audit](diagnostics/spline-probe/empty-branch-audit.json), [diff code](diagnostics/spline-probe/candidate-code/suite-change.diff) | Nhánh spline quan sát rỗng ở 60/60 input; probe loại masked spline khỏi quantization hoàn tất, không bị numeric guard chặn. |
| Chuẩn bị sáu model cuối | `evidence/work-n100-clean-20261101T075458Z/<model>/preparation.json`, [prepare log](evidence/work-n100-clean-20261101T075458Z.prepare.log) | Thư mục work mới; đủ sáu model theo revision `n100-spline-fp32-v1-20261101`; numeric guard đạt. |
| Đo cuối | [schedule](evidence/results-n100-t4-20261101T080423Z/schedule.json), [run log](evidence/results-n100-t4-20261101T080423Z.run.log) | Đủ 36 cell; không thấy RuntimeWarning/overflow/invalid/divide-by-zero/Traceback trong log phiên cuối. |

Probe và preparation mới tách khỏi các round ORT PTQ trước đây. Bản sửa thay đổi ranh giới quantization NNCF, không sửa checkpoint hay trọng số FP32 nguồn. SHA-256 suite cuối: `fb20086c7907a7730fd3e3446544c6c816809c88b2164e640e1b10882af902cf`. Suite ban đầu vẫn có trong `evidence/code-history/`.

## Cấu trúc thư mục và nguồn gốc

| Đường dẫn | Nội dung |
|---|---|
| `evidence/results-n100-t4-20261101T080423Z/` | 36 cell: `result.json`, 720 dòng `utterances.jsonl`, runtime nodes; 36 worker logs; schedule và bảng tổng hợp |
| `evidence/work-n100-clean-20261101T075458Z/` | Preparation, FP32 parity và execution nodes FP32/INT8 của sáu model |
| `evidence/*.log`, `*.environment-lock.txt`, `*.power-*.txt` | Log prepare/run cuối, lock thư viện và power profile |
| `evidence/metadata/`, `evidence/reference/`, `evidence/*.py` | Input/split/config cố định, provenance nguồn/export và snapshot code thực thi |
| `diagnostics/` | Lần lỗi số học và probe SeqIR thành công riêng; không gộp vào bảng timing cuối |
| `provenance/` | Danh sách member/hash của archive nguồn và file lệnh đánh số 05–10 |
| `verification/received/` | PC audit archive, tính lại metric, ghi nhận outlier và bản sao bảng gốc |
| [verification/import-check.json](verification/import-check.json) | Kết quả kiểm mới khi nhập repo, đối chiếu phần việc có sẵn và thử verifier với file thiếu/bị sửa cùng bản chỉ có nội dung Git |
| `manifest.json` | Member nguồn, size, SHA-256 gốc, đường dẫn repo, hash bản dẫn xuất và từng phép biến đổi/loại trừ |
| `.local/originals/` | Bản gốc local bị Git bỏ qua, gồm 72 ONNX/XML/BIN/WAV; clone Git sẽ không có các artifact này |
| `SHA256SUMS`, `verify.py` | Checksum cho phần Git và script kiểm bằng Python standard library |

Có 240 file text được nhập vào Git: **232 bản giống nguyên bytes và 8 bản dẫn xuất** chỉ che prefix thư mục người dùng. Bản gốc chưa che vẫn giữ local. Checksum preflight/PC ban đầu giữ nguyên nội dung và tham chiếu bytes gốc; hãy dùng verifier của package này cho bản dẫn xuất trong repo. Một `.pyc` sinh tự động được loại khỏi import, vẫn có danh tính trong archive nguồn và manifest. Không đưa dataset audio cache, venv, checkpoint hay archive chuyển máy vào Git.

Archive nguồn vẫn ngoài repo: `N100-RESULTS-20261101T092554Z/results-n100-t4-20261101T080423Z.tar.gz`, **1.178.279.698 bytes**, SHA-256 **`0655d3b9d4f146a0a4423f63c58e07b9c21234675f6f8f1e02e1f4f75a4b69b1`**. Gói đã qua kiểm CRC gzip, danh sách member, giải nén và đối chiếu hash file sau giải nén (283 file, 345 entry). Lần import này kiểm lại bytes đã copy theo các danh tính đó. Danh sách 120 hash preflight gốc có một bytecode bị loại khỏi import; mục này được ghi rõ, không coi là file đã copy.

## Kiểm lại, không chạy model

Từ thư mục gốc repo, dùng Python 3.10+ đã có:

```sh
python results/edge-n100-openvino/2026-11-01/verify.py
python results/edge-n100-openvino/2026-11-01/verify.py --with-local-artifacts
```

Lệnh đầu dùng được khi chỉ clone Git: kiểm hash, liên kết provenance, đủ 25.920 dòng, từng cell và tính lại CSV/JSON tổng hợp. Lệnh hai kiểm thêm toàn bộ bản gốc local; thiếu artifact sẽ báo lỗi. Không lệnh nào import OpenVINO hay chạy model. Kết quả kiểm đạt không chứng nhận chất lượng hoặc giải thích outlier.

Để đối chiếu lịch sử, N100 đã chạy `openvino_suite.py prepare --work work-n100-clean-20261101T075458Z` rồi `openvino_suite.py run --work work-n100-clean-20261101T075458Z --threads 4 --rounds 3 --output results-n100-t4-20261101T080423Z`. Các shell guard, version pin, preflight và đóng gói nằm trong [LENH-05-DEN-10-N100.txt](provenance/commands/LENH-05-DEN-10-N100.txt). Đây là lệnh đã dùng, không phải yêu cầu chạy lại trên PC. Thí nghiệm N100 tiếp theo cần thư mục work/output và kết quả riêng.
