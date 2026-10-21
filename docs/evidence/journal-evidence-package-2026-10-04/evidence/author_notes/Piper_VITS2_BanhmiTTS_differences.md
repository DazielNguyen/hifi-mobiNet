# Khác biệt giữa Piper, VITS2 và BanhmiTTS — các vấn đề và hệ quả cho bài Q2

Báo cáo này đối chiếu BanhmiTTS với hai hệ thống mà nó kế thừa hoặc so sánh:
**Piper** (bản cài đặt VITS1 mà BanhmiTTS được fork từ đó, và là checkpoint SOTA
bên ngoài dùng để so sánh) và **VITS2** (bài báo mà BanhmiTTS lấy các thành
phần backbone). Mục tiêu: liệt kê chính xác chỗ nào khác, chỗ nào là lỗi,
và mỗi vấn đề ảnh hưởng thế nào tới các kết luận trong bài.

**Nguồn kiểm chứng** (đều đã đọc trực tiếp, không suy đoán):
- Bài báo VITS2 — Kong et al., 2023, arXiv:2307.16430.
- Mã nguồn tham chiếu VITS2 không chính thức phổ biến nhất — `p0p4k/vits2_pytorch`
  (`models.py`, `train.py`, `configs/vits2_ljs_base.json`), tải nguyên văn.
  Tác giả bài báo (SK Telecom) chưa từng công bố code chính thức.
- Mã nguồn Piper gốc — `piper/src/python/piper_train` và hparams lưu trong
  checkpoint `lj-med_1000.ckpt` (HuggingFace `rhasspy/piper-checkpoints`).
- Mã nguồn BanhmiTTS — `banhmi_train/`, commit `0a9bec2`.

---

## Tóm tắt

1. **BanhmiTTS không phải VITS2 đúng nghĩa.** Nó cài 3 kỹ thuật VITS2
   (transformer trong flow, duration discriminator, MAS có nhiễu) ở mức tương
   đương bản cộng đồng p0p4k, nhưng có **1 lỗi số học**, **1 điểm lệch lớn** về
   cách train duration predictor, và **thiếu 2 thành phần** của bài báo.
2. **"Baseline" của BanhmiTTS không phải Piper.** Hai hệ thống dùng chung
   decoder (HiFi-GAN `resblock 2`, LeakyReLU, không MRD) nhưng khác backbone,
   độ chính xác số học, số epoch và cách chia dữ liệu.
3. **Baseline của BanhmiTTS kém Piper gốc về UTMOSv2 (3.132 vs 3.534)** dù cùng
   decoder, trong khi lại tốt hơn về WER. Nguyên nhân **chưa xác định** — đây là
   điểm phản biện dễ hỏi nhất.
4. **So sánh SEQ với Piper là so sánh cấp hệ thống**, không phải ablation: hai
   bên khác nhau ở backbone, decoder, SnakeBeta, MRD, số epoch, độ chính xác và
   tập dữ liệu. Phép so sánh còn **nghiêng về phía Piper** (Piper train trên
   ~95% LJSpeech, nhiều khả năng đã thấy phần lớn test-set), nên việc SEQ vẫn
   thắng là kết quả thận trọng.
5. **So sánh giữa baseline, MRF và SEQ vẫn hợp lệ**: các vấn đề ở mục 3 và 5
   ảnh hưởng đều lên cả ba model.

---

## 1. Ba hệ thống

| | Piper (`lj-med_1000.ckpt`) | VITS2 (bài báo) | BanhmiTTS |
|---|---|---|---|
| Bản chất | Bản cài đặt VITS1, dùng cho triển khai thực tế | Đề xuất học thuật, không có code chính thức | Fork từ Piper, thêm thành phần kiểu VITS2 và decoder mới |
| Dữ liệu | LJSpeech, ~95% làm tập train | LJSpeech 12.500 / 100 / 500 | LJSpeech 12.500 / 100 / 500 (`canonical_split.json`) |
| Phần cứng | 1 GPU, batch 22 | 4× V100, batch 256 | 2× RTX 4070 Ti, batch 16 × 2 |
| Độ dài train | 1000 epoch (~557k batch) | 800k step (mạng chính) + 30k step (duration predictor) | 1500 epoch (~588k batch) |
| Độ chính xác | fp32 | mixed precision | bf16 |

---

## 2. Bảng so sánh thành phần

| Thành phần | Piper | VITS2 (bài báo) | BanhmiTTS baseline | BanhmiTTS MRF / SEQ |
|---|---|---|---|---|
| Text encoder | Transformer 6 lớp | + điều kiện speaker (đa speaker) | Như Piper | Như Piper |
| Blank token | Có (xen token 0) | **Không dùng** | Có | Có |
| Posterior encoder | Linear spec 513 bin | **Mel 80 bin** | Linear spec 513 bin | Linear spec 513 bin |
| Flow | Coupling WN thuần (VITS1) | Coupling + transformer block có residual | Coupling + transformer | Coupling + transformer |
| Duration predictor | SDP dạng flow, loss NLL | Generator G(z_d, h_text), loss `L_adv + L_mse`, train riêng ở cuối | SDP dạng flow, loss NLL + `L_adv` | Như baseline |
| Duration discriminator | Không | Có | Có | Có |
| MAS | Không nhiễu | Nhiễu 0.01, giảm 2e-6 mỗi step | Có nhiễu (giảm nhanh gấp đôi, xem 3.1) | Như baseline |
| Decoder | HiFi-GAN `resblock 2`, LeakyReLU | HiFi-GAN (`resblock 1` trong config p0p4k) | HiFi-GAN `resblock 2`, LeakyReLU | `ResBlockInverted`, SnakeBeta |
| Discriminator âm thanh | MPD | MPD | MPD | MPD + MRD |
| Optimizer | AdamW (0.8, 0.99), wd 0.01, lr 2e-4, decay 0.999875/epoch | Giống | Giống | Giống |

---

## 3. Vấn đề khi đối chiếu BanhmiTTS với bài báo VITS2

### 3.1. MAS noise giảm về 0 nhanh gấp đôi — lỗi số học

- **Bài báo (§2.2):** ε = nhiễu chuẩn × độ lệch chuẩn của P × noise scale;
  noise scale bắt đầu 0.01 và giảm 2×10⁻⁶ **mỗi step**.
- **BanhmiTTS:** công thức đúng (`synthesizer.py`), lịch giảm dùng
  `self.global_step`. Nhưng PyTorch Lightning 1.7 tính `global_step` cho **cả hai
  optimizer** (generator và discriminator), nên mỗi batch tăng 2 đơn vị.
  Checkpoint xác nhận: ~784 `global_step` mỗi epoch = 2 × 392 batch.
- **Hệ quả:** nhiễu về 0 sau 2.500 batch (~6.4 epoch) thay vì 5.000. Tính theo
  lượng dữ liệu (batch 32 so với 256 của bài báo), giai đoạn khám phá alignment
  ngắn hơn khoảng 16 lần. Ablation trong bài báo cho thấy bỏ hẳn nhiễu làm MOS
  giảm 0.15, nên ảnh hưởng nhiều khả năng nhỏ nhưng không bằng 0.
- **Phạm vi:** cả baseline, MRF, SEQ đều chịu như nhau.
- **Sửa:** dùng `global_step // 2` (hoặc đếm số batch). Chỉ có hiệu lực với lần
  train sau.

### 3.2. Duration predictor train khác bài báo — điểm lệch lớn

- **Bài báo (§2.1, Eq. 1–3):** duration predictor là generator nhẹ nhận
  h_text và nhiễu z_d, train bằng `L_adv(G) + L_mse`, và được **train riêng ở
  bước cuối** (30k step sau 800k step của mạng chính). Mục đích: giảm chi phí so
  với SDP dạng flow của VITS1.
- **BanhmiTTS:** giữ nguyên SDP dạng flow của VITS1 (loss NLL), rồi cộng thêm
  `L_adv(G)` trên mẫu sinh ngược của nó. **Không có `L_mse`**, train chung từ
  đầu. Đây chính là cách làm `use_sdp=True` của p0p4k, không phải của bài báo.
- **Hệ quả:**
  - Không có lợi ích hiệu năng mà bài báo nêu (duration predictor không nhẹ hơn).
  - Module flow này nhạy với bf16 — chính là nơi đã gây sập NaN nhiều lần
    (xem `nan_collapse_root_cause_and_fix.md`).
- **Sửa:** cần thay kiến trúc duration predictor và train lại.

### 3.3. Posterior encoder dùng linear spectrogram — thiếu thành phần

- **Bài báo (§3):** "we used the same spectrograms [mel 80 bin] as the input of
  the posterior encoder". Config LJSpeech của p0p4k cũng đặt
  `use_mel_posterior_encoder: true`.
- **BanhmiTTS:** linear spectrogram 513 bin, giống VITS1/Piper.
- **Sửa:** cần train lại.

### 3.4. Vẫn dùng blank token — thiếu thành phần

- **Bài báo (§3):** "we did not use the blank token".
- **BanhmiTTS:** giữ cách mã hoá phoneme của Piper (xen token 0 giữa các
  phoneme). Đây cũng là điều kiện để dùng chung bảng phoneme với checkpoint
  Piper khi so sánh.
- **Sửa:** cần đổi preprocessing và train lại.

### 3.5. Transformer block trong flow — đúng vị trí, nhưng comment trong code sai

- **Bài báo (§2.3, Hình 1b):** transformer block đặt trước khối convolution, có
  residual. **Không công bố siêu tham số.**
- **BanhmiTTS:** `pre → attention (+ residual) → WN`, khớp hình trong bài báo.
  FFN của attention dùng kernel 1, filter 384 — hợp lệ vì bài báo không quy định.
- **Vấn đề:** comment trong `flow_block.py` ghi "khớp
  `ResidualCouplingTransformersLayer2` của p0p4k, đã kiểm chứng". Thực tế p0p4k
  Layer2 dùng kernel 5, filter 192, và config LJSpeech của p0p4k còn dùng biến
  thể `pre_conv` (attention 2 lớp đặt trước phép chiếu). Comment cần sửa; code
  không cần sửa.

### 3.6. Output duration discriminator không được mask ở padding

- Output ở các vị trí padding là `sigmoid(bias)` chứ không phải 0, và loss LSGAN
  lấy trung bình cả những vị trí này. Kế thừa nguyên từ p0p4k/EdgeTTS. Ảnh hưởng
  nhỏ (chỉ kéo bias đầu ra về 0.5), không phải lỗi riêng của BanhmiTTS.

### 3.7. Các điểm không áp dụng hoặc đã khớp

- **Text encoder có điều kiện speaker:** chỉ cho đa speaker; LJSpeech có 1
  speaker nên bỏ qua là đúng.
- **Duration discriminator:** đầu vào (h_text đã detach + log-duration) và loss
  LSGAN khớp Eq. 1–2. Kiến trúc giống p0p4k V2 cộng thêm dropout; bài báo không
  mô tả kiến trúc cụ thể.
- **Optimizer:** khớp hoàn toàn (0.999^(1/8) = 0.999875).
- **Batch và độ dài train:** khác do giới hạn phần cứng (batch 32 so với 256).

---

## 4. Vấn đề khi so sánh với checkpoint Piper gốc

### 4.1. Kết quả (PyTorch FP32, n=500, cùng test-set canonical)

| | Piper gốc | Baseline | MRF(1,1,1) | SEQ(1,1,1) |
|---|---:|---:|---:|---:|
| RTF ↓ | 0.0386 | 0.0434 | 0.0370 | **0.0324** |
| UTMOSv2 ↑ | 3.534 | 3.132 | 3.537 | **3.581** |
| WER ↓ | 0.1632 | 0.1230 | 0.1043 | **0.1033** |

SEQ thắng Piper ở cả 3 trục, có ý nghĩa thống kê (Mann-Whitney U: RTF
p=1.7×10⁻⁴⁸, UTMOSv2 p=4.3×10⁻³, WER p=1.2×10⁻¹⁶).

Checkpoint Piper được chạy bằng chính code `piper_train` gốc, không chỉnh sửa,
trong môi trường Python riêng. Code BanhmiTTS không load được checkpoint này
một cách đúng đắn, vì flow của BanhmiTTS luôn có attention còn Piper thì không.

### 4.2. Baseline kém Piper về UTMOSv2 dù cùng decoder — chưa rõ nguyên nhân

Baseline của BanhmiTTS và Piper dùng **cùng decoder** (HiFi-GAN `resblock 2`,
LeakyReLU, không MRD), nhưng UTMOSv2 của baseline thấp hơn 0.40. WER thì ngược
lại: baseline tốt hơn (0.1230 so với 0.1632).

Các khả năng (chưa kiểm chứng, không loại trừ nhau):
- Piper train fp32, BanhmiTTS train bf16 (kèm sai số mel loss ở mục 5.1).
- Piper nhiều khả năng đã thấy phần lớn test-set khi train (mục 4.4).
- Các điểm lệch VITS2 ở mục 3 (đặc biệt 3.2) có thể làm giảm chất lượng thay
  vì tăng.

Chiều ngược nhau giữa UTMOSv2 và WER cho thấy đây không đơn giản là baseline
train thiếu. Phản biện có thể hỏi baseline có bị yếu đi không — nên nêu thẳng
trong bài cùng các khả năng trên.

### 4.3. Hai hệ thống khác nhau ở nhiều yếu tố cùng lúc

SEQ và Piper khác nhau ở: backbone (VITS2-style vs VITS1), decoder
(`ResBlockInverted` vs HiFi-GAN), activation (SnakeBeta vs LeakyReLU),
discriminator (MPD + MRD vs MPD), số epoch (1500 vs 1000), độ chính xác (bf16 vs
fp32), batch (32 vs 22) và tập train. Kết quả chỉ nói được **hệ thống SEQ tốt
hơn hệ thống Piper**, không nói được yếu tố nào tạo ra khác biệt.

### 4.4. Rò rỉ test-set — nghiêng về phía Piper

Hparams trong checkpoint Piper: `num_test_examples: 0`, `validation_split: 0.05`.
Tức Piper train trên ~95% LJSpeech (sau khi lọc câu dài hơn 400 phoneme), nên
phần lớn 500 câu test canonical nhiều khả năng nằm trong tập train của Piper.
Không đo được chính xác mức trùng, vì Piper chia dữ liệu bằng `random_split`
theo trạng thái RNG riêng của nó. Rò rỉ này có lợi cho Piper, nên việc SEQ vẫn
thắng là kết quả thận trọng — nhưng bắt buộc phải nêu trong bài.

### 4.5. Tương thích bảng phoneme

Cả hai dùng chung cách mã hoá phoneme kiểu Piper (`banhmi_phonemize` mô phỏng
`piper_phonemize`), nên test-set đã mã hoá dùng trực tiếp được cho checkpoint
Piper. Kiểm tra thực tế: audio sinh ra có độ dài và biên độ hợp lý, WER 0.163.

---

## 5. Vấn đề nội bộ BanhmiTTS (ảnh hưởng đều lên baseline, MRF, SEQ)

### 5.1. Mel loss tính một phần ở bf16

STFT đã được ép chạy fp32, nhưng phép nhân với mel filterbank trong
`spec_to_mel_torch` nằm ngoài vùng fp32 nên chạy ở bf16 dưới autocast. Đo thực
tế: sai số log-mel trung bình 0.0017, lớn nhất 0.0077 so với fp32 thuần. Nhỏ, và
như nhau cho cả ba model, nhưng làm `val_loss_mel` không so trực tiếp được với
hệ thống train fp32 như Piper.

### 5.2. Mỗi cấu hình chỉ train một lần

Mỗi model chỉ có 1 lần train (1 seed). Kiểm định thống kê được tính trên 500
câu, không phải trên nhiều lần train, nên chưa đo được độ dao động giữa các
lần train.

### 5.3. Baseline khác MRF/SEQ ở cả SnakeBeta và MRD

Baseline dùng LeakyReLU và không MRD; MRF/SEQ dùng SnakeBeta và MRD (gắn cố
định theo loại resblock từ commit `0a9bec2`). Hiệu năng (params, FLOPs, RTF) và
so sánh SEQ với MRF không bị ảnh hưởng, nhưng mức tăng **chất lượng** so với
baseline là của cả gói (khối mới + SnakeBeta + MRD).

---

## 6. Hệ quả cho bài Q2

### Claim an toàn
- SEQ nhẹ hơn, ít FLOPs hơn và nhanh hơn baseline và MRF, ở cả FP32 và INT8,
  trên runtime triển khai thực tế (ONNX Runtime).
- SEQ tốt hơn MRF về RTF và UTMOSv2; WER ngang nhau. Đây là so sánh có kiểm soát
  hoàn toàn.
- Hệ thống SEQ tốt hơn checkpoint Piper công bố ở cả 3 trục, kèm điều kiện ở
  mục 4.3–4.4.

### Cần sửa câu chữ
- **Không ghi "dựa trên VITS2".** Đề xuất: *"a VITS backbone with VITS2-style
  components (transformer-augmented flow, duration discriminator, noise-scaled
  MAS), following the open-source implementation of p0p4k"*, rồi liệt kê các
  điểm lệch ở mục 3 trong phần implementation details.
- **Không gọi so sánh với Piper là ablation.** Trình bày là so sánh cấp hệ thống.
- **Mức tăng chất lượng so với baseline** nên quy cho cả hệ thống SEQ, không
  riêng khối `ResBlockInverted`.

### Nên đưa vào Limitations
1. Các điểm lệch VITS2 (mục 3.1–3.4).
2. Baseline kém Piper về UTMOSv2, nguyên nhân chưa xác định (mục 4.2).
3. Rò rỉ test-set nghiêng về phía Piper (mục 4.4).
4. Chỉ 1 dataset, 1 speaker; mỗi cấu hình 1 lần train.
5. UTMOSv2 là chỉ số tự động, chưa có MOS người nghe.

---

## 7. Những gì sửa được

| Việc | Cần train lại? | Ghi chú |
|---|---|---|
| Sửa comment sai trong `flow_block.py` (mục 3.5) | Không | Chỉ là tài liệu |
| Sửa lịch giảm MAS noise (mục 3.1) | Có hiệu lực từ lần train sau | Không đổi model đã train |
| Đưa phép nhân mel về fp32 (mục 5.1) | Có hiệu lực từ lần train sau | Không đổi model đã train |
| Duration predictor theo bài báo (mục 3.2) | Có | Đổi kiến trúc và loss |
| Mel posterior encoder (mục 3.3) | Có | |
| Bỏ blank token (mục 3.4) | Có | Mất tương thích phoneme với Piper |
| MOS người nghe | Không | Việc đáng làm nhất nếu còn thời gian |
