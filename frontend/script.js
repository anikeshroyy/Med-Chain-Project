// ===== CONFIGURATION =====
// Change this URL when you deploy your ML backend to HuggingFace Spaces
const API_BASE_URL = "http://127.0.0.1:8000";

// ===== DOM ELEMENTS =====
const dropzone = document.getElementById("upload-dropzone");
const fileInput = document.getElementById("file-input");
const dropzoneContent = document.getElementById("dropzone-content");
const previewContainer = document.getElementById("preview-container");
const previewImage = document.getElementById("preview-image");
const btnRemove = document.getElementById("btn-remove");
const btnAnalyze = document.getElementById("btn-analyze");

const resultsEmpty = document.getElementById("results-empty");
const resultsLoading = document.getElementById("results-loading");
const resultsContent = document.getElementById("results-content");
const loaderText = document.getElementById("loader-text");

const diagnosisBadge = document.getElementById("diagnosis-badge");
const confidenceValue = document.getElementById("confidence-value");
const confidenceFill = document.getElementById("confidence-fill");
const heatmapImage = document.getElementById("heatmap-image");
const hashValue = document.getElementById("hash-value");

const apiStatusDot = document.getElementById("api-status-dot");
const apiStatusText = document.getElementById("api-status-text");

// ===== STATE =====
let selectedFile = null;

// ===== API HEALTH CHECK =====
async function checkAPIHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            apiStatusDot.className = "status-dot connected";
            apiStatusText.textContent = "API Connected";
        } else {
            throw new Error();
        }
    } catch {
        apiStatusDot.className = "status-dot disconnected";
        apiStatusText.textContent = "API Offline";
    }
}

// Check on load, then every 30 seconds
checkAPIHealth();
setInterval(checkAPIHealth, 30000);

// ===== FILE UPLOAD HANDLING =====

// Click to open file dialog
dropzone.addEventListener("click", (e) => {
    if (e.target.closest(".btn-remove")) return;
    fileInput.click();
});

// File selected via dialog
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop events
dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Process the file
function handleFile(file) {
    const validTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!validTypes.includes(file.type)) {
        alert("Please upload a JPG or PNG image.");
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropzoneContent.style.display = "none";
        previewContainer.style.display = "block";
        btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);

    // Reset results
    showState("empty");
}

// Remove image
btnRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    previewImage.src = "";
    previewContainer.style.display = "none";
    dropzoneContent.style.display = "flex";
    btnAnalyze.disabled = true;
    showState("empty");
}

// ===== RESULTS STATE MANAGEMENT =====
function showState(state) {
    resultsEmpty.style.display = state === "empty" ? "flex" : "none";
    resultsLoading.style.display = state === "loading" ? "flex" : "none";
    resultsContent.style.display = state === "results" ? "flex" : "none";
}

// ===== SHA-256 HASH IN BROWSER =====
async function computeSHA256(file) {
    const buffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ===== ANALYZE BUTTON =====
btnAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;

    btnAnalyze.disabled = true;
    showState("loading");

    // Animated loading messages
    const loadingMessages = [
        "Preprocessing image...",
        "Running AI model (MobileNetV2)...",
        "Generating Grad-CAM heatmap...",
        "Computing SHA-256 hash...",
        "Finalizing results...",
    ];

    let msgIndex = 0;
    const msgInterval = setInterval(() => {
        msgIndex++;
        if (msgIndex < loadingMessages.length) {
            loaderText.textContent = loadingMessages[msgIndex];
        }
    }, 1200);

    try {
        // Compute SHA-256 hash locally in the browser
        const fileHash = await computeSHA256(selectedFile);

        // Send image to API
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`API returned status ${response.status}`);
        }

        const data = await response.json();

        clearInterval(msgInterval);
        displayResults(data, fileHash);
    } catch (err) {
        clearInterval(msgInterval);
        console.error("Analysis failed:", err);

        // If API is offline, run in demo mode with a warning
        const useDemoMode = confirm(
            "Could not connect to the AI backend.\n\nWould you like to see a demo with sample results?\n\n(To use real AI predictions, start the backend server.)"
        );

        if (useDemoMode) {
            const fileHash = await computeSHA256(selectedFile);
            displayDemoResults(fileHash);
        } else {
            showState("empty");
            btnAnalyze.disabled = false;
        }
    }
});

// ===== DISPLAY REAL RESULTS =====
function displayResults(data, fileHash) {
    const label = data.diagnosis || data.label || "UNKNOWN";
    const confidence = data.confidence || 0;
    const heatmapBase64 = data.heatmap_image || data.heatmap || null;

    // Diagnosis badge
    diagnosisBadge.textContent = label;
    diagnosisBadge.className = "diagnosis-badge";
    if (label === "NORMAL") {
        diagnosisBadge.classList.add("normal");
    } else if (label === "PNEUMONIA") {
        diagnosisBadge.classList.add("pneumonia");
    }

    // Confidence
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    confidenceFill.style.width = `${confidence}%`;
    confidenceFill.className = "confidence-fill";
    if (label === "PNEUMONIA" && confidence > 70) {
        confidenceFill.classList.add("high-danger");
    }

    // Heatmap
    if (heatmapBase64) {
        heatmapImage.src = `data:image/png;base64,${heatmapBase64}`;
        document.getElementById("heatmap-card").style.display = "block";
    } else {
        document.getElementById("heatmap-card").style.display = "none";
    }

    // Hash
    hashValue.textContent = fileHash;

    showState("results");
    btnAnalyze.disabled = false;
}

// ===== DEMO MODE (when API is offline) =====
function displayDemoResults(fileHash) {
    const isNormal = Math.random() > 0.5;
    const label = isNormal ? "NORMAL" : "PNEUMONIA";
    const confidence = 75 + Math.random() * 20;

    // Diagnosis badge
    diagnosisBadge.textContent = label;
    diagnosisBadge.className = "diagnosis-badge";
    diagnosisBadge.classList.add(isNormal ? "normal" : "pneumonia");

    // Confidence
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    confidenceFill.style.width = `${confidence}%`;
    confidenceFill.className = "confidence-fill";
    if (!isNormal) confidenceFill.classList.add("high-danger");

    // Hide heatmap in demo mode
    document.getElementById("heatmap-card").style.display = "none";

    // Hash (this is real, computed in the browser)
    hashValue.textContent = fileHash;

    showState("results");
    btnAnalyze.disabled = false;
}

// ===== SMOOTH SCROLL FOR NAV LINKS =====
document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
        const targetId = link.getAttribute("href");
        if (targetId === "#") return;
        const target = document.querySelector(targetId);
        if (target) {
            e.preventDefault();
            const headerOffset = 64;
            const elementPos = target.getBoundingClientRect().top + window.scrollY;
            window.scrollTo({ top: elementPos - headerOffset, behavior: "smooth" });
        }
    });
});

// ===== ACTIVE NAV LINK ON SCROLL =====
const sections = document.querySelectorAll("section[id]");
const navLinks = document.querySelectorAll(".nav-link");

function updateActiveNav() {
    const scrollY = window.scrollY + 100;
    sections.forEach((section) => {
        const top = section.offsetTop;
        const height = section.offsetHeight;
        const id = section.getAttribute("id");

        if (scrollY >= top && scrollY < top + height) {
            navLinks.forEach((link) => {
                link.classList.remove("active");
                if (link.getAttribute("href") === `#${id}`) {
                    link.classList.add("active");
                }
            });
        }
    });
}

window.addEventListener("scroll", updateActiveNav);
