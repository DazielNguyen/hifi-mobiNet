# Benchmark common97 trên Intel N100 — 05/11/2026

*English: [README.md](README.md)*

**Đã kiểm lại đủ 3 model × 2 precision × 3 vòng × 720 câu = 12.960 bản ghi timed.**
Trong cùng phiên, INT8 common97 giảm RTF trọng số thời lượng **12,73–13,88%**
so với FP32. Piper-SeqIR đạt RTF thấp nhất: **0,04579** ở INT8.
Đây là kết quả tốc độ inference trên N100; chất lượng WER/CER/UTMOS/human MOS
của các artifact này chưa được đánh giá trong gói nhận về.

Nguồn: `20261105T075322Z-benchmark-common97.4SOO5I` do người dùng copy từ N100
về Downloads. Giữ nguyên **77 file / 14.530.595 bytes**, không loại câu hoặc sửa
log. PC chỉ đọc, kiểm hash, tính lại thống kê và nhập evidence; không chạy
inference, PTQ, scoring hay benchmark mới.

**Bổ sung log chuẩn bị:** đã nhận riêng `20261105T073706Z-common97.6nBwhE`,
chỉ gồm [prepare-common97.log](preparation-evidence/prepare-common97.log)
(367.051 bytes), giữ nguyên bytes và kiểm SHA-256. Log có ba lượt calibration
100% 60/60, kết thúc bằng `COMMON97 READY: three Piper models prepared and local
catalog activated.` Không thấy Traceback/RuntimeWarning hoặc thông báo lỗi số học
trong log nhận được. Log báo thành công; liên kết artifact với benchmark đã được
kiểm riêng qua thư mục work bổ sung dưới đây, không suy ra từ dòng READY.

Đường dẫn N100 ghi ở cuối log, nay đã nhận đúng thư mục work:

```text
/home/capstone/benchmarks/hifi-mobiNet-n100-openvino-2026-10-31/hifi-mobiNet-n100-openvino-2026-10-31/work-common97-20261105T073706Z
```

Catalog được log trỏ tới, **đã nhận và kiểm hash ngày 06/11/2026**:

```text
/home/capstone/benchmarks/hifi-mobiNet-n100-common97-2026-11-05/.local/n100-common97-models.json
```

**Bổ sung kiểm tra và khởi động demo:** đã nhận thêm **4 file / 3.552 bytes**,
giữ nguyên trong [demo-evidence/](demo-evidence/), đối chiếu hash với Downloads:

| Thư mục nhận | File | Nội dung xác nhận được từ record |
|---|---|---|
| `20261105T074110Z-check.8oqXmf` | `check.log`, `doctor.json` | Hai file giống bytes; báo `ready`, frontend en-us khớp 720/720 |
| `20261105T074122Z-server.CkuEHv` | `server.log` | Báo demo sẵn sàng cổng 8501, trỏ đúng session bên dưới |
| `20261105T074123Z-a3c2e898` | `server.json` | Khởi tạo server trên Intel N100; cùng môi trường và catalog hash với doctor |

Doctor chạy lúc **14:41:11**, server lúc **14:41:23 ngày 05/11/2026 (UTC+07)**,
trước benchmark. CPU, kernel, Python và năm package được ghi nhận khớp phiên
benchmark. Hash source trong server khớp nguyên file
[n100_demo.py](reference/n100_demo.py) lấy từ repo; code preflight kiểm hash
preparation/IR và frontend trước khi ghi trạng thái sẵn sàng.

Hai record cùng ghi catalog SHA-256
`d2f9895b90882430984ce9cb9751324a382bffbc4a60bf0fb0448db02c5e9f42`,
khớp chính xác file [n100-common97-models.json](catalog-evidence/n100-common97-models.json)
(10.760 bytes) vừa nhận từ Downloads. Catalog có sáu mục FP32 và ba mục INT8
common97 cho Piper ResBlock2, ParIR và SeqIR. Cả hai record đều ghi
**`inference_performed: false`**; thư mục server nhận về chỉ có `server.json`,
không có WAV/JSON của lần sinh audio. Vì vậy đây là evidence của bước kiểm tra
và khởi động. Evidence graph/WAV chuẩn bị đã nhận riêng trong thư mục work.

**Đã nhận đúng work N100:** `C:\Users\duyng\Downloads\work-common97-20261105T073706Z`
có **84 file / 478.136.232 bytes**, đủ ba model, mỗi model 28 file. Đã đọc và tính
SHA-256 mới cho tất cả file, đối chiếu các hash gốc có trong preparation/benchmark.
15 JSON được copy nguyên bytes vào [work-evidence/](work-evidence/). 69 file
XML/BIN/ONNX/WAV giữ local tại Downloads, được liệt kê đầy đủ trong
[work-manifest.json](work-manifest.json), không đưa weights/audio vào Git.

| Kiểm tra từ artifact nhận về | ResBlock2 | ParIR | SeqIR |
|---|---:|---:|---:|
| Preparation khớp hash trong các cell benchmark | 6/6 | 6/6 | 6/6 |
| XML/BIN FP32 và INT8 khớp hash preparation | 4/4 | 4/4 | 4/4 |
| FQ trực tiếp trong XML INT8, đúng cạnh Conv | 96 | 96 | 96 |
| Kernel INT8 trong runtime record | 97 | 97 | 97 |
| WAV smoke FP32 + INT8, đúng format/số mẫu | 6/6 | 6/6 | 6/6 |
| WAV alignment INT8 khớp hash/số mẫu | 12/12 | 12/12 | 12/12 |
| WAV alignment INT8 >10 giây | 6 | 6 | 6 |
| WAV alignment dài nhất, đo từ file | 23,452 s | 21,513 s | 22,291 s |

Mỗi graph có 384 hằng range FQ được đọc trực tiếp từ BIN: hữu hạn, low < high,
256 mức. Mọi cạnh đầu ra FQ chỉ đi tới input activation (port 0) của Conv trong
allowlist common97; không thấy FQ rò sang duration/alignment hoặc decoder.
Kiểm lại 60 mẫu calibration theo split và phoneme IDs: đều thuộc train, không
có val/test. Policy, implementation, numeric guard và regression config khớp
hash được lưu trong cả ba preparation. Chi tiết: [work-verification.json](verification/work-verification.json).

Các record FP32 parity báo 13/13/model đạt; công thức và ngưỡng khớp. Các record
alignment báo 12/12/model đạt, coverage 1 và 0 frame mất. Đây là **kiểm tính nhất
quán record**, không chạy lại inference để tái tạo tensor nội bộ. Đã kiểm header,
đọc đủ PCM của cả 54 WAV mono 22.050 Hz/16-bit và đối chiếu duration. 36 WAV
alignment có hash gốc; 18 WAV smoke chỉ có hash import mới và số mẫu gốc để đối chiếu.
Chưa nghe/chấm chất lượng trong lần kiểm này; không kết luận hết rè chỉ từ hash.

**Đã nối đủ bằng hash phần INT8 giữa demo và benchmark:** catalog kích hoạt
lúc `2026-11-05T07:40:35.325686+00:00`, sau khi chuẩn bị đủ ba model và trước
doctor/server. Policy và implementation khớp common97; cả ba `int8_preparation`
khớp hash benchmark, cả sáu XML/BIN INT8 khớp file work thực tế. Kết quả kiểm
catalog nằm ở mục `catalog` trong [verification.json](verification/verification.json).

Catalog giữ đường FP32 từ `work-n100-clean-20261101T075458Z`, còn INT8 dùng
`work-common97-20261105T073706Z`. Với ba Piper, hash BIN FP32 trong catalog giống
BIN FP32 benchmark mới, nhưng hash XML khác; không coi hai graph FP32 giống
bytes hoặc gán timing benchmark cho đường FP32 demo chỉ từ catalog này.
Đây là cấu hình catalog đã được code chuẩn bị giữ lại, không phải lỗi copy file.

Không cần copy lại catalog hoặc chạy lại PTQ/benchmark. Nếu kiểm một lần sinh
trên demo, cần WAV và JSON tương ứng; WAV chuẩn bị không phải audio của session
server. Đánh giá WER/CER/UTMOS/MOS vẫn là công việc riêng. Hash catalog xác nhận
phiên khởi động đã ghi, không khẳng định trạng thái server đang chạy hiện tại.

## Kết quả chính

RTF = tổng thời gian inference / tổng thời lượng audio thực sinh, gộp ba vòng.
Mean latency tính trên 2.160 quan sát/model/precision. Peak RSS là giá trị lớn
nhất của ba process worker, gồm runtime/input/model, không phải riêng weights.

| Model | RTF FP32 | RTF INT8 common97 | RTF giảm | Mean latency FP32 / INT8 (ms) | Peak RSS FP32 / INT8 (GiB) |
|---|---:|---:|---:|---:|---:|
| Piper ResBlock2 | 0,06473 | 0,05649 | 12,73% | 164,81 / 144,13 | 1,750 / 1,692 |
| Piper ParIR | 0,06549 | 0,05663 | 13,53% | 172,83 / 147,83 | 1,232 / 1,129 |
| Piper SeqIR | 0,05317 | 0,04579 | 13,88% | 136,63 / 118,08 | 1,166 / 1,056 |

Mean latency giảm lần lượt 12,55%, 14,47%, 13,58%. Native RNG không được khóa
để FP32 và INT8 sinh cùng waveform; tổng audio INT8 khác FP32 lần lượt khoảng
+0,20%, −1,09%, +0,35%. Vì thế phần trăm giảm RTF không đồng nhất với phần
trăm giảm latency, và không phải phép tăng tốc trên một waveform cố định.

| Model | Mean RTF FP32 / INT8 | Median RTF FP32 / INT8 | P95 RTF FP32 / INT8 | P95 latency FP32 / INT8 (ms) |
|---|---:|---:|---:|---:|
| Piper ResBlock2 | 0,06490 / 0,05664 | 0,06227 / 0,05394 | 0,07907 / 0,07068 | 220,48 / 198,26 |
| Piper ParIR | 0,06561 / 0,05671 | 0,06447 / 0,05523 | 0,07203 / 0,06383 | 219,41 / 189,47 |
| Piper SeqIR | 0,05330 / 0,04589 | 0,05219 / 0,04480 | 0,06040 / 0,05264 | 172,48 / 148,27 |

Các RTF phân phối là `latency_s / audio_seconds` từng câu. P95 dùng nearest-rank,
vị trí `ceil(0,95 × n)`. Ba vòng lặp cùng 720 câu, không phải 2.160 câu độc lập.

| Model | RTF FP32 r0 / r1 / r2 | RTF INT8 r0 / r1 / r2 |
|---|---|---|
| Piper ResBlock2 | 0,06384 / 0,06489 / 0,06545 | 0,05585 / 0,05719 / 0,05644 |
| Piper ParIR | 0,06487 / 0,06657 / 0,06504 | 0,05733 / 0,05670 / 0,05587 |
| Piper SeqIR | 0,05299 / 0,05283 / 0,05369 | 0,04641 / 0,04543 / 0,04552 |

Nguồn tính lại: [CSV](verification/recomputed.csv), [JSON từng cell và tổng hợp](verification/recomputed.json).
Bảng nhận từ N100: [FP32](evidence/results/summary_fp32.csv),
[INT8](evidence/results/summary_int8.csv), [summary và hash](evidence/results/summary.json).
Tất cả số liệu tổng hợp đều đã đối chiếu từ `utterances.jsonl`, không chỉ đọc báo cáo cũ.

## Model, recipe và những gì xác minh được

| Model ID | Selected checkpoint |
|---|---|
| `piper-no-vits2-cpn` | `best-epoch=1403-val_loss_mel=19.7719.ckpt` |
| `piper-parir-no-vits2-cpn` | `best-epoch=1486-val_loss_mel=19.0069.ckpt` |
| `piper-seqir-no-vits2-cpn` | `best-epoch=1455-val_loss_mel=19.0332.ckpt` |

Danh tính nguồn được tham chiếu qua [models.json](reference/models.json), có
SHA-256 khớp `manifest_sha256` ở mọi cell. Đây là ba Config A, FP32 từ đợt 2 đã
sửa flow weight norm; không bao gồm Q05 lịch sử, Piper external, Low hoặc High.
Không load checkpoint/ONNX. Ba ONNX nhận về là `fp32_matched_validation.onnx`
phục vụ parity, không phải ONNX deployment gốc; chỉ kiểm hash inventory của chúng.

Suite thực thi có SHA-256
`7c064dcd88bd4727e781af34aa4c1a83365c878108dc3717540699c72a592323`,
khớp [snapshot code](reference/openvino_suite.py) lấy từ repo. Code kiểm policy,
IR hash và graph trước khi đo. Cả 9 cell INT8 có runtime metadata phủ đúng
**97 Convolution: 37 `/enc_p/` + 20 `/dp/` + 40 `/flow/`** theo
[policy common97](reference/piper-common97.json). Không có weighted kernel INT8
ngoài phạm vi. Decoder weighted kernels đều FP32: 23 ở ResBlock2, 32 ở ParIR,
23 ở SeqIR. Cả 9 cell FP32 có toàn bộ weighted runtime nodes FP32.

**Đã bổ sung audit tĩnh trực tiếp XML/BIN nhận từ N100.** SHA-256 của mỗi
`preparation.json` khớp sáu cell/model; preparation ràng buộc bốn file IR và
12 WAV alignment/model. Graph và range FQ được đọc lại từ file thực tế bằng
[verify_work.py](verify_work.py), không chỉ đếm kernel trong report cũ.
[verification.json](verification/verification.json) lưu chuỗi liên kết này.
Không thay thế preparation N100 bằng bản PC/WSL hoặc phiên N100 cũ.

File `quality_status` gốc được kế thừa từ bước prepare nên vẫn chứa cụm
“N100 RTF not certified by preparation”. Đây là giới hạn của bước prepare,
không phủ nhận các timing mới đã hoàn tất. File gốc được giữ nguyên.

## Thiết bị và protocol

| Thiết lập | Giá trị ghi nhận |
|---|---|
| CPU | Intel N100, 4 logical CPUs |
| Platform | `Linux-7.0.0-42-generic-x86_64-with-glibc2.39` |
| Python | 3.12.3 |
| OpenVINO / NNCF | 2026.4.1 / 3.4.0 |
| NumPy / ONNX / ONNX Runtime | 2.4.6 / 1.23.2 / 1.31.1 |
| Runtime | CPU, batch 1, đồng bộ, 4 threads, 1 stream, `LATENCY`, `f32` hint, `ACCURACY` |
| Input | Harvard-720 phoneme IDs; scales `[0.667, 1.0, 0.8]`; output 22.050 Hz |
| Warm-up | 10 lời gọi theo độ dài input, ngoài vùng timed |
| Lặp/thứ tự | 3 vòng; worker riêng/cell; shuffle seed 1234, đảo cell ở r1; câu dùng seed `1234 + round` |
| Governor | `powersave` ở cả 4 CPU tại đầu/cuối mỗi worker |
| Sensor lớn nhất | 72 °C, chỉ tại ranh giới worker |
| Khoảng thời gian UTC | 07:53:23–08:26:50, 05/11/2026 |
| Giờ UTC+07 | 14:53:23–15:26:50; khoảng 33 phút 27 giây |

Timer chỉ bao quanh `InferRequest.infer(share_outputs=True)`. Load/compile,
phonemization, tạo/gán input, copy/kiểm output và ghi file nằm ngoài timer.
Chi phí gặp shape mới sau warm-up vẫn có thể nằm trong timing. Không dùng số
đo này làm latency streaming hoặc text-to-WAV đầu cuối.

Không có bản ghi power profile/EPP hoặc trace xung/thermal liên tục của phiên
này. Không suy ra profile `balanced` từ phiên cũ hoặc khẳng định máy không bị
throttling chỉ dựa trên các mẫu sensor. Kernel cũng đã đổi từ `7.0.0-31` của
01/11 sang `7.0.0-42`; so sánh nhân quả code cũ/mới cần control cùng phiên.

## Phạm vi kết luận và evidence còn thiếu

- Các số đo hữu hạn, duration/RTF đúng công thức, không thiếu/trùng ID. RTF lớn
  nhất 0,12903; cả 12.960 quan sát đều có RTF < 1 trong vùng inference đã đo.
- Audio ở các lượt timed chỉ dài **1,463–4,203 giây**. Không có câu >10 giây trong
  timing này, nên không dùng nó để chứng nhận đã hết lỗi đuôi câu dài.
- `peak_abs` được ghi trong khoảng 0,2357–0,8495; không có peak >1. Đây không
  chứng minh không méo tiếng, không rè, hoặc nội dung được đọc đầy đủ.
- `benchmark.log` có đủ 18 cell theo lịch. 18 worker log đều **0 byte** và được
  giữ nguyên; việc không có warning trong log không thay thế kiểm các record.
- Đã nhận và kiểm work N100 gồm preparation, IR, parity, graph/alignment audit
  và WAV câu dài; catalog demo cũng đã khớp hash. Full pip lock, frontend binary và dữ liệu tensor
  parity/coverage gốc vẫn chưa nhận; không dùng riêng dòng READY làm chứng nhận.
- Suite không lưu waveform của mỗi lần timed. WER/CER/UTMOS/human MOS mới còn
  pending; không ghép điểm chất lượng của ONNX Runtime hoặc PyTorch vào đây.
- Chưa có phép kiểm ý nghĩa thống kê qua nhiều phiên, thiết bị hay training seed.

Lịch sử [N100 01/11](../2026-11-01/README.vi.md) vẫn giữ nguyên. Các graph INT8
cũ có lỗi activation quantization ở duration/alignment, không được gộp timing
hoặc chất lượng với common97. [Báo cáo tích hợp](../../../docs/quantization/piper-common97-integration-2026-11-05/README.vi.md)
giải thích phạm vi mới và các phép kiểm PC/WSL riêng.

## File và cách kiểm lại

| Đường dẫn | Vai trò |
|---|---|
| `evidence/` | 77 file nhận từ N100, nguyên bytes; gồm log, schedule, summary, 18 × result/runtime/utterances |
| `preparation-evidence/` | Một log chuẩn bị nhận bổ sung, nguyên bytes |
| `demo-evidence/` | 4 file check/server từ 3 thư mục; giữ nguyên cả hai file check giống bytes |
| `work-evidence/` | 15 JSON từ work N100 |
| `catalog-evidence/` | Catalog common97 nguyên bytes; tổng cộng 98 file nhận được lưu trong repo |
| `work-manifest.json` | Inventory 84 file work, gồm 69 file nhị phân/audio giữ tại Downloads; không bỏ qua file nào |
| `reference/` | 11 file từ repo: suite/demo/common97/numeric guard/config/manifest/metadata; hash được các record tương ứng ràng buộc |
| `manifest.json` | Nguồn, size/SHA-256 các bản sao trong repo và danh sách evidence thiếu |
| `verification/` | Thống kê tính lại, scope runtime, thời gian, quan sát duration và các giới hạn |
| `SHA256SUMS` | Checksum toàn bộ package trừ chính file checksum |
| `verify.py` | Python standard library; không import OpenVINO hoặc chạy model |
| `verify_work.py` | Đọc toàn bộ work external, kiểm hash, graph XML/BIN, WAV và đối chiếu record; không inference |

Từ gốc repo:

```sh
python results/edge-n100-openvino/2026-11-05-common97/verify.py
```

Đối chiếu thêm bản Downloads trên PC:

```powershell
.venv/Scripts/python.exe results/edge-n100-openvino/2026-11-05-common97/verify.py --source "C:\Users\duyng\Downloads\20261105T075322Z-benchmark-common97.4SOO5I" --preparation-source "C:\Users\duyng\Downloads\20261105T073706Z-common97.6nBwhE" --demo-source-root "C:\Users\duyng\Downloads" --catalog-source "C:\Users\duyng\Downloads\n100-common97-models.json"
.venv/Scripts/python.exe -B results/edge-n100-openvino/2026-11-05-common97/verify_work.py --source "C:\Users\duyng\Downloads\work-common97-20261105T073706Z"
```

Verifier kiểm toàn bộ inventory/hash, 36 hash nội bộ trong summary, schedule,
warm-up, ID/thứ tự câu, môi trường và thời gian đầu/cuối, công thức từng dòng,
18 runtime records, JSON/CSV từng cell và tổng hợp. Verifier cũng kiểm inventory
bốn file demo mới, doctor/check giống bytes, hash catalog giữa hai record,
frontend, môi trường, thứ tự thời gian, hash source demo và preparation với benchmark.
Catalog thực tế được đối chiếu hash với doctor/server, metadata/recipe và ba
preparation INT8; phân biệt rõ đường FP32 lịch sử với FP32 benchmark mới.
`verify.py` kiểm các file trong repo; `verify_work.py` mới đọc lại 84 file work
external. Thiếu hoặc sửa file sẽ báo lỗi. Không có manifest transfer toàn bộ từ
N100; hash benchmark ràng buộc preparation, IR và WAV alignment; các file khác
có hash tại thời điểm import. Giữ thư mục Downloads để chạy lại audit external.
