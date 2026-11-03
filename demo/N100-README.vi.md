# Demo HiFi-MobiNet qua LAN trên Intel N100

Nhập văn bản và nghe audio trong trình duyệt **PC**; frontend và inference chạy
trên **N100 / Ubuntu / OpenVINO CPU**. PC không cần Python, model hay sound card
gắn vào N100. Loa/tai nghe đang dùng với PC sẽ phát file WAV nhận từ N100.

```text
PC: trình duyệt -- văn bản qua LAN --> N100: Banhmi frontend → OpenVINO CPU
PC: loa/tai nghe <-- WAV qua LAN ----- N100: đóng gói audio PCM16
```

Đây là server riêng `hifimobinet.n100_server`, giao diện ở `demo/n100_web/`.
App cũ `demo/streamlit_app.py`, pipeline
training và bộ evidence đã lưu không bị thay đổi. Gói demo chỉ chứa code,
frontend nguồn và danh mục SHA-256; dùng lại IR có sẵn trên N100, không export
hoặc quantize lại và không mang checkpoint vào tiến trình demo.

## 1. Chuyển gói sang N100

Copy ZIP, `.zip.sha256` và `LENH-N100-LAN-DEMO.txt` qua ổ cứng cá nhân.
Trong Files trên N100, đặt ZIP và `.sha256` cùng thư mục, mở Terminal tại đó:

```bash
sha256sum -c hifi-mobiNet-n100-lan-demo-2026-11-02.zip.sha256
```

Chỉ giải nén sau khi báo `OK`. Giải nén thư mục bên trong ZIP vào
`~/benchmarks/` để có đường dẫn:

```text
~/benchmarks/hifi-mobiNet-n100-lan-demo-2026-11-02/demo/n100_web/index.html
```

Các lệnh đầy đủ có trong [LENH-N100-LAN-DEMO.txt](LENH-N100-LAN-DEMO.txt).
Copy nguyên từng khối, chạy ở Terminal **N100**, chờ thành công rồi sang khối kế
tiếp. Không `source` các script; chạy bằng `bash` để lỗi không đóng Terminal.

## 2. Cài phần giao diện một lần

```bash
cd "$HOME/benchmarks/hifi-mobiNet-n100-lan-demo-2026-11-02" && bash scripts/n100_demo/01-setup.sh
```

Cần Internet nếu thiếu frontend để build
`vendor/banhmi-phonemize` với eSpeak từ commit được ghim trong CMakeLists.
Script dùng `~/venvs/openvino-n100/bin/python`, không tạo venv mới. Các phiên
bản runtime được giữ theo session đã đo: OpenVINO 2026.4.1, NNCF 3.4.0,
NumPy 2.4.6, ONNX 1.23.2, ONNX Runtime 1.31.1. Server web dùng thư viện chuẩn
Python, không cài framework hoặc tạo môi trường giao diện mới. Streamlit
1.51.0 xung đột với `protobuf==7.36.2` và `packaging==26.3` đang có trong
environment lock N100, vì thế không đưa Streamlit vào môi trường này.

Pip còn bị ràng buộc bởi `pip freeze` trước cài; nếu có xung đột thì bước cài
dừng và giữ log, không tự bỏ ràng buộc hay nâng runtime. `sudo` chỉ dùng để cài
công cụ build frontend trên Ubuntu. Mật khẩu nhập tại N100, không đặt trong file.
Chờ dòng `BUOC 01 OK`.

Nếu đã dùng gói đầu tiên và gặp `could not find git for clone of
sonic-git-populate`, máy đang thiếu Git phục vụ build frontend. Script bước 01
đã bổ sung dependency này. Để sửa ngay trên bản đã copy sang N100, chạy
`sudo apt-get update && sudo apt-get install --yes git`, rồi chạy lại bước 01.
Không cần thay file hoặc sửa `SHA256SUMS` trong gói đang dùng. Khối lệnh đầy đủ
nằm trong [SUA-LOI-01-THIEU-GIT.txt](SUA-LOI-01-THIEU-GIT.txt).

## 3. Kiểm tra trước buổi demo

```bash
cd "$HOME/benchmarks/hifi-mobiNet-n100-lan-demo-2026-11-02" && bash scripts/n100_demo/02-check.sh
```

Chờ `BUOC 02 OK`. Bước này kiểm tra **trên máy đang chạy**, không dựa vào dòng
PASS cũ:

- Linux, CPU Intel N100, phiên bản môi trường.
- Manifest, sáu preparation records và 24 file XML/BIN đúng SHA-256.
- Frontend đang cài tạo đúng phoneme IDs cho cả 720 câu Harvard đã lưu.

Đây là kiểm tra chuẩn bị, chưa chạy inference. App kiểm tra lại khi khởi tạo.
Model chỉ được compile khi bấm tạo audio. Sau inference đầu tiên, app kiểm tra
CPU execution và precision của kernel trước khi phát WAV.

Đường dẫn dữ liệu mặc định:

```text
~/benchmarks/hifi-mobiNet-n100-openvino-2026-10-31/
  hifi-mobiNet-n100-openvino-2026-10-31/
    models.json
    metadata/harvard720.json
    work-n100-clean-20261101T075458Z/<model-id>/
      preparation.json
      fp32.xml + fp32.bin
      int8.xml + int8.bin
```

Nếu đã chuyển cả bộ dữ liệu sang vị trí khác, đặt đường dẫn **thư mục chứa
models.json và work-n100-clean...** trước khi chạy các bước:

```bash
export HIFIMOBINET_N100_BUNDLE="/duong/dan/thuc/den/thu-muc-bo-model"
```

Không trỏ vào `work-n100` cũ hoặc các lần probe. Catalog chỉ chấp nhận kết quả
final đã sửa masked duration spline, lưu trong evidence ngày 2026-11-01.

## 4. Mở server và dùng trên PC

```bash
cd "$HOME/benchmarks/hifi-mobiNet-n100-lan-demo-2026-11-02" && bash scripts/n100_demo/03-start.sh
```

Giữ Terminal này mở và chờ `N100 DEMO READY`. Script in địa chỉ của N100. PC và N100 phải ở LAN có thể
liên lạc với nhau; PC có thể dùng Wi-Fi của cùng router. Trên trình duyệt PC mở:

```text
http://<IPv4-hiện-tại-của-N100>:8501
```

Ví dụ `http://192.168.1.3:8501` **chỉ khi N100 hiện vẫn có IP đó**. Không dùng
`localhost` trên PC. Không chạy Python server trên PC.

Chọn model, để FP32 cho lần demo đầu, nhập một câu tiếng Anh ngắn rồi bấm
**Tạo giọng nói trên N100**. Chờ audio xuất hiện, bấm nút **▶** trong player.
Trình duyệt phát qua loa PC; app không cần thiết bị âm thanh tại N100.
Có nút tải WAV và JSON ghi thông tin lần tạo. Có thể chọn INT8 để nghe thử,
nhưng INT8 là chọn lọc, decoder vẫn FP32 và đánh giá chất lượng còn pending.

Nếu PC chưa mở được trang, thử `http://127.0.0.1:8501` ngay trên N100. Nếu
trang local hoạt động, kiểm tra IP, cùng mạng, client isolation của router và
firewall. Không cần mở cổng router ra Internet. API chỉ chấp nhận yêu cầu JSON
cùng origin với trang demo, không bật CORS và không gửi telemetry. App dành
cho LAN demo tin cậy, không có đăng
nhập và không được cấu hình như dịch vụ Internet công cộng.

## Kịch bản trình bày khoảng 3 phút

1. Chỉ rõ dây LAN nối N100 với router, PC chỉ mở trình duyệt. Hiện phần
   **Thông tin máy chủ**: hostname, Intel N100, OpenVINO và frontend kiểm tra.
2. Mời hội đồng đưa một câu tiếng Anh mới; nhập trực tiếp và bấm tạo.
   Không có cache WAV hoặc chọn bản ghi âm sẵn trong app này.
3. Trên màn hình N100 mở System Monitor để quan sát tiến trình Python sử dụng
   CPU khi tạo. N100 có thể không có màn hình trong demo; JSON lần tạo vẫn lưu
   hostname, PID, model, SHA-256 và các phép đo ở phía server.
4. Bấm phát trên PC. Giải thích: “N100 tính toán giọng nói; PC nhận WAV qua LAN
   và phát bằng loa. LAN là kênh điều khiển và chuyển kết quả.”
5. Đổi sang model khác với cùng câu, hoặc INT8 chọn lọc, rồi tạo mới. Cho thấy
   thời gian inference và độ dài audio. Chỉ gọi đây là số đo demo từng câu.

Có thể ngắt **đường Internet của router** mà vẫn giữ LAN sau khi cài xong;
app không gọi cloud để tạo audio. Rút dây LAN giữa PC và N100 sẽ làm PC không
gửi được yêu cầu mới; điều đó không phải phép thử inference offline.

## Model và ý nghĩa các số đo

| Model | Epoch checkpoint chọn | Nhánh |
|---|---:|---|
| HiFi-GAN ResBlock2 | 1489 | historical Q05 |
| Parallel-IR | 1386 | historical Q05 |
| Sequential-IR | 1442 | historical Q05 |
| Piper_no_VITS2_cpn | 1403 | Config A, round 2 |
| Piper_ParIR_no_VITS2_cpn | 1486 | Config A, round 2 |
| Piper_SeqIR_no_VITS2_cpn | 1455 | Config A, round 2 |

Cả sáu có lựa chọn FP32 và selective INT8 v2 đã chuẩn bị trên N100. Piper Low
và High không thuộc gói này. Exact identities nằm trong `n100_models.json`.

- CPU, batch 1, 4 inference threads, một stream, LATENCY, ACCURACY, FP32 hint;
  selective INT8 vẫn được kiểm tra bằng runtime kernel metadata.
- `inference_s` chỉ bao quanh `InferRequest.infer`; không gồm frontend, nạp/
  compile model, sao chép output, tạo WAV, mạng hay phát audio. RTF từng câu =
  `inference_s / audio_seconds`. Không warm-up tự động; JSON đánh dấu lần đầu
  sau load. So sánh benchmark chính thức xem bộ kết quả N100 đã lưu.
- `load_compile_s` tách riêng; bằng 0 nếu tái sử dụng model hiện tại.
- Mỗi yêu cầu chạy inference mới. RNG nằm trong graph nên thời lượng/WAV có
  thể thay đổi khi cùng text và model; không hứa tương đương RNG giữa FP32/INT8.
- Chỉ giữ một compiled model và xử lý một yêu cầu tại một thời điểm. Yêu cầu
  đồng thời bị báo bận. Đổi model sẽ giải phóng model cũ trước khi compile.
- Tối đa 200 ký tự, frontend tối đa 800 IDs. Giới hạn phát audio 60 giây được
  kiểm tra **sau inference**, không bảo đảm giới hạn RAM/thời gian inference.
  Dùng một câu ngắn cho buổi demo; chưa thiết lập timeout cưỡng bức tiến trình.
- WAV mono 22,050 Hz, PCM16 có peak scaling; waveform thô được kiểm tra finite
  và không im lặng. Audio demo không thay thế đánh giá chất lượng có kiểm soát.

## Log và trạng thái xác minh

Mọi log mới nằm trong `~/benchmarks/n100-demo-runs/`, mỗi bước/session có thư
mục riêng. Setup lưu freeze trước/sau và log build; doctor lưu JSON; server
lưu `server.json`, runtime nodes, từng WAV + JSON và log lỗi. Không ghi đè
kết quả benchmark hoặc input model. Text do người dùng nhập được lưu cùng
JSON trên N100 để đối chiếu với WAV.

Kiểm thử trên PC dùng backend giả lập và audio tổng hợp để kiểm tra đường đi
text → inference → WAV, metrics, lỗi, UI và đóng gói. Chúng **không chứng minh**
frontend Linux build thành công hay inference/audio qua LAN đã chạy thực trên
N100. Phải hoàn tất bước 01–03 và thử một câu mới tại N100/PC để xác nhận các
phần đó. Không có inference, export, PTQ hoặc benchmark mới được chạy trên PC
để chuẩn bị gói demo.

Tham khảo API: [Python HTTP server](https://docs.python.org/3/library/http.server.html),
[OpenVINO InferRequest](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.InferRequest.html).
