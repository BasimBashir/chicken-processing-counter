const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const results = document.getElementById("results");
const totalCount = document.getElementById("totalCount");
const emptyCount = document.getElementById("emptyCount");
const singleCount = document.getElementById("singleCount");
const slaughteredCount = document.getElementById("slaughteredCount");
const originalImg = document.getElementById("originalImg");
const annotatedImg = document.getElementById("annotatedImg");
const downloadBtn = document.getElementById("downloadBtn");

uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
    originalImg.src = URL.createObjectURL(file);
    uploadZone.querySelector(".label").textContent = "Processing...";
    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/image/detect", { method: "POST", body: formData });
        totalCount.textContent = resp.headers.get("X-Total-Count") || "0";
        emptyCount.textContent = resp.headers.get("X-Count-Empty-Shackles") || "0";
        singleCount.textContent = resp.headers.get("X-Count-Single-Legged") || "0";
        slaughteredCount.textContent = resp.headers.get("X-Count-Slaughtered-Chicken") || "0";

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        annotatedImg.src = url;
        downloadBtn.href = url;
        results.style.display = "block";
        uploadZone.querySelector(".label").textContent = "Drop another image or click to browse";
    } catch (err) {
        uploadZone.querySelector(".label").textContent = "Error - try again";
        console.error(err);
    }
}
