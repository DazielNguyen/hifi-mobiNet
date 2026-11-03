"use strict";
const el = id => document.getElementById(id);
const modelNames = new Map();
let objectUrls = [];

function clearAudio() {
  el("audio").pause();
  el("audio").removeAttribute("src");
  el("audio").load();
  objectUrls.forEach(url => URL.revokeObjectURL(url));
  objectUrls = [];
  el("result").hidden = true;
}
function blobUrl(blob) {
  const url = URL.createObjectURL(blob);
  objectUrls.push(url);
  return url;
}
el("precision").addEventListener("change", () => {
  el("int8-note").hidden = el("precision").value !== "int8";
});
el("generate").addEventListener("submit", async event => {
  event.preventDefault();
  clearAudio();
  el("submit").disabled = true;
  el("status").className = "";
  el("status").textContent = "N100 đang tạo âm thanh…";
  const body = {text: el("text").value, model_id: el("model").value, precision: el("precision").value};
  try {
    const response = await fetch("/api/synthesize", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Yêu cầu không thành công");
    const record = data.record;
    const wav = Uint8Array.from(atob(data.wav_base64), char => char.charCodeAt(0));
    const wavUrl = blobUrl(new Blob([wav], {type: "audio/wav"}));
    const recordText = JSON.stringify(record, null, 2);
    el("audio").src = wavUrl;
    el("download-wav").href = wavUrl;
    el("download-wav").download = record.wav_file;
    el("download-record").href = blobUrl(new Blob([recordText], {type: "application/json"}));
    el("download-record").download = record.request_id + ".json";
    el("identity").textContent = modelNames.get(record.model_id) + " · " + record.precision.toUpperCase();
    el("generated-text").textContent = record.text;
    el("inference").textContent = record.inference_s.toFixed(3) + " s";
    el("duration").textContent = record.audio_seconds.toFixed(3) + " s";
    el("rtf").textContent = record.rtf.toFixed(3);
    el("load").textContent = record.load_compile_s.toFixed(3) + " s";
    el("record").textContent = recordText;
    el("result").hidden = false;
    el("status").textContent = "Đã tạo xong. Bấm ▶ bên dưới để nghe bằng loa máy tính.";
  } catch (error) {
    el("status").className = "error";
    el("status").textContent = "Chưa tạo được audio: " + error.message;
  } finally {
    el("submit").disabled = false;
  }
});

async function connect() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Máy chủ chưa sẵn sàng");
    const data = await response.json();
    for (const model of data.models) {
      modelNames.set(model.id, model.label);
      el("model").add(new Option(model.label, model.id));
    }
    el("model").value = "piper-seqir-no-vits2-cpn";
    const host = data.server.host;
    el("server").textContent = host.hostname + " · " + data.server.openvino_cpu + " · OpenVINO " + host.packages.openvino + " · CPU / 4 luồng";
    el("server-record").textContent = JSON.stringify(data.server, null, 2);
    ["model", "precision", "submit"].forEach(id => { el(id).disabled = false; });
  } catch (error) {
    el("server").textContent = "Không kết nối được N100: " + error.message + ". Kiểm tra server và tải lại trang.";
  }
}
connect();
