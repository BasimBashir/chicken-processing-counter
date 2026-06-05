const rtspUrl = document.getElementById("rtspUrl");
const btnConnect = document.getElementById("btnConnect");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnCountStart = document.getElementById("btnCountStart");
const btnCountStop = document.getElementById("btnCountStop");
const feedImg = document.getElementById("feedImg");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const totalCount = document.getElementById("totalCount");
const emptyCount = document.getElementById("emptyCount");
const singleCount = document.getElementById("singleCount");
const slaughteredCount = document.getElementById("slaughteredCount");
const fpsVal = document.getElementById("fpsVal");
// Live-retune the running "default" stream (best-effort: ignored if no stream).
function patchLiveStream(body) {
    fetch("/api/streams/default", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    }).catch(() => {});
}

// Update the config DEFAULT (applies to future streams) and the live stream now.
function applyParam(key, value) {
    fetch("/api/config", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [key]: value }),
    }).catch(() => {});
    patchLiveStream({ [key]: value });
}

const fmt2 = v => Number(v).toFixed(2);
const fmtInt = v => String(Math.round(Number(v)));

// All live-tunable range controls: slider id -> {config key, value formatter, parser}.
const RANGE_CONTROLS = [
    { id: "confSlider",      key: "confidence",          fmt: fmt2,   parse: parseFloat },
    { id: "confEmptySlider", key: "conf_empty_shackles", fmt: fmt2,   parse: parseFloat },
    { id: "nmsSlider",       key: "nms_iou",             fmt: fmt2,   parse: parseFloat },
    { id: "roiSlider",       key: "roi_position",        fmt: fmt2,   parse: parseFloat },
    { id: "zoneSlider",      key: "zone_half",           fmt: fmtInt, parse: v => parseInt(v, 10) },
    { id: "speedSlider",     key: "conveyor_speed_px",   fmt: fmtInt, parse: parseFloat },
    { id: "maxDistSlider",   key: "max_distance",        fmt: fmtInt, parse: v => parseInt(v, 10) },
    { id: "maxDisapSlider",  key: "max_disappeared",     fmt: fmtInt, parse: v => parseInt(v, 10) },
];

const imgszSelect = document.getElementById("imgszSelect");
let pollInterval = null;

RANGE_CONTROLS.forEach(c => {
    const el = document.getElementById(c.id);
    if (!el) return;
    const valEl = document.getElementById(c.id.replace("Slider", "Value"));
    el.addEventListener("input", () => { if (valEl) valEl.textContent = c.fmt(el.value); });
    el.addEventListener("change", () => applyParam(c.key, c.parse(el.value)));
});
if (imgszSelect) {
    imgszSelect.addEventListener("change", () => applyParam("imgsz", parseInt(imgszSelect.value, 10)));
}

fetch("/api/config").then(r => r.json()).then(cfg => {
    if (cfg.rtsp_url) rtspUrl.value = cfg.rtsp_url;
    RANGE_CONTROLS.forEach(c => {
        if (cfg[c.key] == null) return;
        const el = document.getElementById(c.id);
        const valEl = document.getElementById(c.id.replace("Slider", "Value"));
        if (el) el.value = cfg[c.key];
        if (valEl) valEl.textContent = c.fmt(cfg[c.key]);
    });
    if (imgszSelect && cfg.imgsz != null) imgszSelect.value = String(cfg.imgsz);
});

btnConnect.addEventListener("click", async () => {
    const url = rtspUrl.value.trim();
    const body = url ? { url } : {};
    try {
        const resp = await fetch("/api/stream/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert(err.detail || "Failed to connect");
            return;
        }
        feedImg.src = `/api/stream/feed?t=${Date.now()}`;
        feedImg.style.display = "block";
        feedPlaceholder.style.display = "none";
        btnConnect.disabled = true;
        btnDisconnect.disabled = false;
        btnCountStart.disabled = false;
        setStatus(true, "Connected");
        startPolling();
    } catch (err) {
        console.error(err);
    }
});

btnDisconnect.addEventListener("click", async () => {
    await fetch("/api/stream/stop", { method: "POST" });
    feedImg.style.display = "none";
    feedPlaceholder.style.display = "block";
    btnConnect.disabled = false;
    btnDisconnect.disabled = true;
    btnCountStart.disabled = true;
    btnCountStop.disabled = true;
    setStatus(false, "Disconnected");
    stopPolling();
});

btnCountStart.addEventListener("click", async () => {
    await fetch("/api/stream/counting/start", { method: "POST" });
    btnCountStart.disabled = true;
    btnCountStop.disabled = false;
});

btnCountStop.addEventListener("click", async () => {
    await fetch("/api/stream/counting/stop", { method: "POST" });
    btnCountStart.disabled = false;
    btnCountStop.disabled = true;
});

function setStatus(active, text) {
    statusDot.className = `status-dot ${active ? "active" : "inactive"}`;
    statusLabel.textContent = text;
}

function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const resp = await fetch("/api/stream/status");
            const s = await resp.json();
            totalCount.textContent = s.total_count;
            emptyCount.textContent = s.counts?.empty_shackles ?? 0;
            singleCount.textContent = s.counts?.single_legged ?? 0;
            slaughteredCount.textContent = s.counts?.slaughtered_chicken ?? 0;
            fpsVal.textContent = s.fps;
            if (!s.is_connected) {
                setStatus(false, s.error || "Disconnected");
                btnConnect.disabled = false;
                btnDisconnect.disabled = true;
                stopPolling();
            }
        } catch (_) {}
    }, 500);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}
