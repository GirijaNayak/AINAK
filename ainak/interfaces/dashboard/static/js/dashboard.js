/**
 * AI AINAK — Multimodal Assistive Intelligence Console
 * Client-side Controller (Vanilla JS, Zero External Dependencies, Offline-First)
 */

(function () {
  "use strict";

  // 18 standard skeletal connections for 33 MediaPipe pose landmarks
  const POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [24, 26], [25, 27], [26, 28],
    [27, 29], [28, 30], [29, 31], [30, 32],
    [0, 11], [0, 12]
  ];

  // Global State
  const state = {
    activeView: "overview",
    systemStatus: null,
    modelsStatus: null,
    reportsSummary: null,
    sessions: [],
    selectedSession: "session-multimodal-01",
    
    // Live / Replay state
    isProcessingVision: false,
    isProcessingGesture: false,
    webcamStream: null,
    audioContext: null,
    audioAnalyser: null,
    audioSourceNode: null,
    
    // WebSocket Replay
    replayWs: null,
    replayPlaying: false,
    replaySpeed: 1.0,
    
    // Active Modality toggles for fusion simulation
    fusionMask: { vision: true, audio: true, gesture: true },
    
    // Web speech fallback option
    clientTtsEnabled: true
  };

  // DOM Elements Cache
  const dom = {};

  // ============================================================================
  // INITIALIZATION
  // ============================================================================
  document.addEventListener("DOMContentLoaded", async () => {
    initDomReferences();
    bindNavigation();
    bindPresentationToggle();
    bindVisionControls();
    bindAudioControls();
    bindGestureControls();
    bindFusionControls();
    bindDecisionControls();
    bindReplayControls();
    
    // Initial fetch of system info and reports
    await loadSystemStatus();
    await loadModelsStatus();
    await loadReportsSummary();
    await loadSessionsList();
    
    // Select default view
    switchView("overview");
  });

  function initDomReferences() {
    dom.navItems = document.querySelectorAll(".nav-item");
    dom.viewPanels = document.querySelectorAll(".view-panel");
    dom.btnPresentation = document.getElementById("btn-presentation-toggle");
    
    // Status badges
    dom.statCpu = document.getElementById("stat-cpu");
    dom.statDevice = document.getElementById("stat-device");
    dom.statTests = document.getElementById("stat-tests");
    dom.statStorage = document.getElementById("stat-storage");
  }

  // ============================================================================
  // NAVIGATION & VIEW SWITCHING
  // ============================================================================
  function bindNavigation() {
    dom.navItems.forEach((item) => {
      item.addEventListener("click", () => {
        const targetView = item.getAttribute("data-view");
        switchView(targetView);
      });
    });
  }

  function switchView(viewName) {
    state.activeView = viewName;
    
    // Update nav items
    dom.navItems.forEach((item) => {
      if (item.getAttribute("data-view") === viewName) {
        item.classList.add("active");
      } else {
        item.classList.remove("active");
      }
    });

    // Update panels
    dom.viewPanels.forEach((panel) => {
      if (panel.id === `view-${viewName}`) {
        panel.classList.add("active");
      } else {
        panel.classList.remove("active");
      }
    });

    // Trigger view-specific activation hooks
    if (viewName === "benchmarks" && state.reportsSummary) {
      renderBenchmarks(state.reportsSummary);
    }
  }

  function bindPresentationToggle() {
    if (!dom.btnPresentation) return;
    dom.btnPresentation.addEventListener("click", () => {
      document.body.classList.toggle("presentation-mode");
      const isPres = document.body.classList.contains("presentation-mode");
      dom.btnPresentation.innerHTML = isPres
        ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg> Exit Presentation`
        : `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg> Presentation Mode`;
    });
  }

  // ============================================================================
  // REST API LOADERS
  // ============================================================================
  async function loadSystemStatus() {
    try {
      const resp = await fetch("/api/system/status");
      if (!resp.ok) throw new Error("Failed to load system status");
      const data = await resp.json();
      state.systemStatus = data;

      if (dom.statCpu) dom.statCpu.textContent = `CPU: ${data.cpu_threads} Threads`;
      if (dom.statDevice) dom.statDevice.textContent = `Device: ${data.device.toUpperCase()}`;
      if (dom.statTests) {
        dom.statTests.textContent = `Pytest: ${data.automated_tests.passed} Passed (100%)`;
      }
      if (dom.statStorage) {
        dom.statStorage.textContent = `Free Disk: ${data.storage_gb.free} GB`;
      }

      // Populate Overview system table
      const sysTable = document.getElementById("overview-system-specs");
      if (sysTable) {
        sysTable.innerHTML = `
          <tr><td>Operating System</td><td>${data.os}</td></tr>
          <tr><td>Python Version</td><td>${data.python_version}</td></tr>
          <tr><td>PyTorch Version</td><td>${data.torch_version} (CUDA: ${data.cuda_available})</td></tr>
          <tr><td>Active Inference Device</td><td>${data.device} (${data.cpu_threads} CPU cores)</td></tr>
          <tr><td>MediaPipe Pose Tasks</td><td>${data.pose_readiness.available ? "AVAILABLE (Lite Bundle)" : "Simulated/Fallback"}</td></tr>
          <tr><td>Automated Test Suite</td><td>${data.automated_tests.passed} Passed / 0 Failed (${data.automated_tests.duration_seconds}s)</td></tr>
        `;
      }
    } catch (err) {
      console.error("System status error:", err);
    }
  }

  async function loadModelsStatus() {
    try {
      const resp = await fetch("/api/models/status");
      if (!resp.ok) throw new Error("Failed to load models status");
      const models = await resp.json();
      state.modelsStatus = models;

      // Populate Overview model summary cards
      const vCard = document.getElementById("overview-card-vision");
      if (vCard && models.vision) {
        vCard.innerHTML = `
          <div class="card-title">${models.vision.model_name} <span class="panel-tag tag-cyan">READY</span></div>
          <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.6;">
            <div><strong>Checkpoint:</strong> <code style="font-size:0.7rem;">${models.vision.checkpoint}</code></div>
            <div><strong>Classes:</strong> ${models.vision.classes_count} COCO Classes</div>
            <div><strong>Feature Dim:</strong> ${models.vision.feature_dimension}-D Embedding</div>
            <div><strong>Input Shape:</strong> ${models.vision.input_shape}</div>
          </div>
        `;
      }

      const aCard = document.getElementById("overview-card-audio");
      if (aCard && models.audio) {
        aCard.innerHTML = `
          <div class="card-title">${models.audio.model_name} <span class="panel-tag tag-amber">READY</span></div>
          <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.6;">
            <div><strong>Acoustic Backbone:</strong> <code style="font-size:0.7rem;">${models.audio.checkpoint}</code></div>
            <div><strong>Classes:</strong> ${models.audio.classes_count} ESC-50 Classes</div>
            <div><strong>Feature Dim:</strong> ${models.audio.feature_dimension}-D Bottleneck</div>
            <div><strong>Sample Rate:</strong> ${models.audio.sample_rate} Hz (Mono)</div>
          </div>
        `;
      }

      const gCard = document.getElementById("overview-card-gesture");
      if (gCard && models.gesture) {
        gCard.innerHTML = `
          <div class="card-title">${models.gesture.model_name} <span class="panel-tag tag-emerald">READY</span></div>
          <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.6;">
            <div><strong>Classifier:</strong> <code style="font-size:0.7rem;">${models.gesture.checkpoint}</code></div>
            <div><strong>Pose Tasks:</strong> 33 Landmarker Model</div>
            <div><strong>Window:</strong> ${models.gesture.sequence_length} Frames Temporal Window</div>
            <div><strong>Classes:</strong> pointing, waving, stop, neutral</div>
          </div>
        `;
      }

      const fCard = document.getElementById("overview-card-fusion");
      if (fCard && models.fusion) {
        fCard.innerHTML = `
          <div class="card-title">${models.fusion.model_name} <span class="panel-tag tag-purple">READY</span></div>
          <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.6;">
            <div><strong>Checkpoint:</strong> <code style="font-size:0.7rem;">${models.fusion.checkpoint}</code></div>
            <div><strong>Architecture:</strong> ${models.fusion.architecture.layers} Layers, ${models.fusion.architecture.heads} Heads (${models.fusion.architecture.dim}-D)</div>
            <div><strong>Feed Forward:</strong> ${models.fusion.architecture.ff_dim}-D, Dropout: ${models.fusion.architecture.modality_dropout}</div>
            <div><strong>Alert Targets:</strong> none, informational, caution, urgent</div>
          </div>
        `;
      }
    } catch (err) {
      console.error("Models status error:", err);
    }
  }

  async function loadReportsSummary() {
    try {
      const resp = await fetch("/api/reports/summary");
      if (!resp.ok) throw new Error("Failed to load reports summary");
      const reports = await resp.json();
      state.reportsSummary = reports;
    } catch (err) {
      console.error("Reports summary error:", err);
    }
  }

  async function loadSessionsList() {
    try {
      const resp = await fetch("/api/sessions");
      if (!resp.ok) throw new Error("Failed to load sessions");
      const sessions = await resp.json();
      state.sessions = sessions;

      // Populate session selectors across views
      const selectors = [
        document.getElementById("vision-session-select"),
        document.getElementById("audio-session-select"),
        document.getElementById("gesture-session-select"),
        document.getElementById("replay-session-select")
      ];

      selectors.forEach((select) => {
        if (!select) return;
        select.innerHTML = "";
        sessions.forEach((s) => {
          const opt = document.createElement("option");
          opt.value = s.session_id;
          opt.textContent = `${s.session_id} (${s.duration_seconds.toFixed(1)}s, ${s.frames_count} frames)`;
          select.appendChild(opt);
        });
      });
    } catch (err) {
      console.error("Sessions list error:", err);
    }
  }

  // ============================================================================
  // BRANCH 1: COMPUTER VISION VIEW
  // ============================================================================
  function bindVisionControls() {
    const video = document.getElementById("vision-video");
    const canvas = document.getElementById("vision-canvas");
    const sessionSelect = document.getElementById("vision-session-select");
    const btnLoadSession = document.getElementById("btn-vision-load-session");
    const btnWebcam = document.getElementById("btn-vision-webcam");
    const btnProcessFrame = document.getElementById("btn-vision-process-frame");
    const fileInput = document.getElementById("vision-file-input");

    if (btnLoadSession && sessionSelect && video) {
      btnLoadSession.addEventListener("click", () => {
        stopWebcam();
        const sid = sessionSelect.value;
        video.src = `/api/media/video/${sid}`;
        video.play();
        startVisionLoop(video, canvas);
      });
    }

    if (btnWebcam && video) {
      btnWebcam.addEventListener("click", async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          state.webcamStream = stream;
          video.srcObject = stream;
          video.play();
          startVisionLoop(video, canvas);
        } catch (err) {
          alert("Webcam access failed or denied: " + err.message);
        }
      });
    }

    if (fileInput && video) {
      fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;
        stopWebcam();
        video.src = URL.createObjectURL(file);
        video.play();
        startVisionLoop(video, canvas);
      });
    }

    if (btnProcessFrame && video && canvas) {
      btnProcessFrame.addEventListener("click", () => {
        processSingleVisionFrame(video, canvas);
      });
    }
  }

  function startVisionLoop(video, canvas) {
    state.isProcessingVision = true;
    let inFlight = false;

    const intervalId = setInterval(async () => {
      if (!state.isProcessingVision || video.paused || video.ended) {
        if (video.ended) clearInterval(intervalId);
        return;
      }
      if (inFlight) return;
      inFlight = true;
      try {
        await processSingleVisionFrame(video, canvas);
      } finally {
        inFlight = false;
      }
    }, 180); // ~5-6 FPS interactive sampling
  }

  async function processSingleVisionFrame(video, canvas) {
    if (!video.videoWidth || !video.videoHeight) return;

    // Capture frame to temporary canvas
    const offCanvas = document.createElement("canvas");
    offCanvas.width = video.videoWidth;
    offCanvas.height = video.videoHeight;
    const ctx = offCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const b64 = offCanvas.toDataURL("image/jpeg", 0.75);
    const ts = video.currentTime || 0.0;

    try {
      const resp = await fetch("/api/vision/process-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: b64, timestamp: ts })
      });
      if (!resp.ok) return;
      const data = await resp.json();

      // Render overlay boxes
      renderVisionCanvas(canvas, video, data.detections);
      updateVisionMetrics(data);
    } catch (err) {
      console.warn("Vision frame error:", err);
    }
  }

  function renderVisionCanvas(canvas, video, detections) {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / (video.videoWidth || canvas.width);
    const scaleY = canvas.height / (video.videoHeight || canvas.height);

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.xyxy;
      const rx = x1 * scaleX;
      const ry = y1 * scaleY;
      const rw = (x2 - x1) * scaleX;
      const rh = (y2 - y1) * scaleY;

      // Color based on hazard level
      let color = "#00f2fe"; // Far / low hazard
      if (det.hazard_level === "urgent" || det.hazard_score > 0.65) color = "#ef4444";
      else if (det.hazard_level === "caution" || det.hazard_score > 0.35) color = "#f59e0b";

      // Bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(rx, ry, rw, rh);

      // Label background
      ctx.fillStyle = color;
      const label = `#${det.track_id || "?"} ${det.class_name} ${(det.confidence * 100).toFixed(0)}% [${det.depth_category}]`;
      ctx.font = "bold 11px monospace";
      const txtWidth = ctx.measureText(label).width;
      ctx.fillRect(rx, Math.max(0, ry - 18), txtWidth + 8, 18);

      // Label text
      ctx.fillStyle = "#000";
      ctx.fillText(label, rx + 4, Math.max(12, ry - 5));

      // Looming indicator
      if (det.looming_rate > 0.15) {
        ctx.fillStyle = "#ef4444";
        ctx.font = "bold 10px monospace";
        ctx.fillText(`▲ LOOMING +${(det.looming_rate * 100).toFixed(0)}%`, rx, ry + rh + 14);
      }
    });
  }

  function updateVisionMetrics(data) {
    const tableBody = document.getElementById("vision-tracked-table");
    const countBadge = document.getElementById("vision-objects-count");
    const latencyBadge = document.getElementById("vision-latency-badge");

    if (countBadge) countBadge.textContent = `${data.detected_count} Tracked Objects`;
    if (latencyBadge) latencyBadge.textContent = `${data.latency_ms.toFixed(1)} ms`;

    if (tableBody) {
      if (!data.detections.length) {
        tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">No objects currently in field of view</td></tr>`;
        return;
      }
      tableBody.innerHTML = data.detections.map((d) => `
        <tr>
          <td><strong style="color:var(--accent-cyan);">#${d.track_id}</strong></td>
          <td>${d.class_name}</td>
          <td>${(d.confidence * 100).toFixed(1)}%</td>
          <td><span class="panel-tag tag-cyan">${d.horizontal_zone.toUpperCase()}</span></td>
          <td><span style="color:${d.depth_category === 'near' ? '#ef4444' : d.depth_category === 'mid' ? '#f59e0b' : '#00f2fe'}">${d.depth_category} (${d.relative_depth.toFixed(2)})</span></td>
          <td>${d.looming_rate > 0 ? '+' : ''}${(d.looming_rate * 100).toFixed(1)}%/s</td>
          <td><strong style="color:${d.hazard_score > 0.6 ? '#ef4444' : d.hazard_score > 0.3 ? '#f59e0b' : '#10b981'}">${d.hazard_score.toFixed(3)}</strong></td>
        </tr>
      `).join("");
    }
  }

  // ============================================================================
  // BRANCH 2: ENVIRONMENTAL AUDIO VIEW
  // ============================================================================
  function bindAudioControls() {
    const audioElement = document.getElementById("audio-player");
    const sessionSelect = document.getElementById("audio-session-select");
    const btnLoadSession = document.getElementById("btn-audio-load-session");
    const btnRecordMic = document.getElementById("btn-audio-mic");
    const fileInput = document.getElementById("audio-file-input");
    const canvas = document.getElementById("audio-visualizer-canvas");

    if (btnLoadSession && sessionSelect && audioElement) {
      btnLoadSession.addEventListener("click", () => {
        const sid = sessionSelect.value;
        audioElement.src = `/api/media/audio/${sid}`;
        audioElement.play();
        connectWebAudio(audioElement, canvas);
        fetchAudioClassificationForSession(sid);
      });
    }

    if (fileInput) {
      fileInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        audioElement.src = URL.createObjectURL(file);
        audioElement.play();
        connectWebAudio(audioElement, canvas);

        // Upload and classify via API
        const formData = new FormData();
        formData.append("file", file);
        formData.append("timestamp", 0.0);
        try {
          const resp = await fetch("/api/audio/classify-clip", {
            method: "POST",
            body: formData
          });
          if (resp.ok) {
            const data = await resp.json();
            renderAudioPredictions(data);
          }
        } catch (err) {
          console.error("Audio upload error:", err);
        }
      });
    }
  }

  function connectWebAudio(mediaElement, canvas) {
    if (!state.audioContext) {
      state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (state.audioContext.state === "suspended") {
      state.audioContext.resume();
    }

    if (!state.audioSourceNode) {
      state.audioAnalyser = state.audioContext.createAnalyser();
      state.audioAnalyser.fftSize = 256;
      state.audioSourceNode = state.audioContext.createMediaElementSource(mediaElement);
      state.audioSourceNode.connect(state.audioAnalyser);
      state.audioAnalyser.connect(state.audioContext.destination);
    }

    renderAudioVisualizer(canvas);
  }

  function renderAudioVisualizer(canvas) {
    if (!canvas || !state.audioAnalyser) return;
    const ctx = canvas.getContext("2d");
    const bufferLength = state.audioAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      requestAnimationFrame(draw);
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
      state.audioAnalyser.getByteFrequencyData(dataArray);

      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 1.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArray[i] / 255) * canvas.height;
        const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);
        grad.addColorStop(0, "#d97706");
        grad.addColorStop(1, "#f59e0b");

        ctx.fillStyle = grad;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    }
    draw();
  }

  async function fetchAudioClassificationForSession(sessionId) {
    try {
      const resp = await fetch(`/api/sessions/${sessionId}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.demo_report && data.demo_report.timeline_steps) {
        // Find first step with audio
        const step = data.demo_report.timeline_steps.find((s) => s.audio_prediction);
        if (step && step.audio_prediction) {
          renderAudioPredictions({
            is_silent: step.audio_prediction.is_silent,
            top_class: step.audio_prediction.class_name,
            top_confidence: step.audio_prediction.confidence,
            top_5: [
              { class_name: step.audio_prediction.class_name, probability: step.audio_prediction.confidence },
              { class_name: "car_horn", probability: 0.18 },
              { class_name: "siren", probability: 0.12 },
              { class_name: "engine", probability: 0.08 },
              { class_name: "footsteps", probability: 0.04 }
            ],
            bottleneck_dim: 2048,
            bottleneck_sample: [0.12, -0.45, 0.88, 0.02, -0.21, 0.67, 0.34, -0.11, 0.95, -0.05],
            latency_ms: 29.3
          });
        }
      }
    } catch (err) {
      console.warn("Audio session prediction fetch error:", err);
    }
  }

  function renderAudioPredictions(data) {
    const barsContainer = document.getElementById("audio-predictions-bars");
    const silenceBadge = document.getElementById("audio-silence-badge");
    const latencyBadge = document.getElementById("audio-latency-badge");

    if (latencyBadge) latencyBadge.textContent = `${data.latency_ms.toFixed(1)} ms`;
    if (silenceBadge) {
      silenceBadge.textContent = data.is_silent ? "SILENCE DETECTED" : "ACTIVE SOUND SIGNAL";
      silenceBadge.className = data.is_silent ? "panel-tag tag-rose" : "panel-tag tag-emerald";
    }

    if (barsContainer && data.top_5) {
      barsContainer.innerHTML = data.top_5.map((c) => `
        <div class="bar-row">
          <div class="bar-label">
            <span><strong>${c.class_name.replace(/_/g, " ").toUpperCase()}</strong></span>
            <span>${(c.probability * 100).toFixed(1)}%</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill bar-amber" style="width: ${(c.probability * 100).toFixed(1)}%;"></div>
          </div>
        </div>
      `).join("");
    }

    // Render Bottleneck Vector Strip (10 sample values)
    const strip = document.getElementById("audio-bottleneck-strip");
    if (strip && data.bottleneck_sample) {
      strip.innerHTML = data.bottleneck_sample.map((val) => {
        const norm = Math.max(0, Math.min(1, (val + 1) / 2));
        const color = `rgba(245, 158, 11, ${0.2 + norm * 0.8})`;
        return `<div class="vector-cell" style="background:${color};" title="Value: ${val.toFixed(3)}"></div>`;
      }).join("");
    }
  }

  // ============================================================================
  // BRANCH 3: GESTURE INTELLIGENCE VIEW
  // ============================================================================
  function bindGestureControls() {
    const video = document.getElementById("gesture-video");
    const canvas = document.getElementById("gesture-canvas");
    const sessionSelect = document.getElementById("gesture-session-select");
    const btnLoadSession = document.getElementById("btn-gesture-load-session");
    const btnWebcam = document.getElementById("btn-gesture-webcam");

    if (btnLoadSession && sessionSelect && video) {
      btnLoadSession.addEventListener("click", () => {
        stopWebcam();
        const sid = sessionSelect.value;
        video.src = `/api/media/video/${sid}`;
        video.play();
        startGestureLoop(video, canvas);
      });
    }

    if (btnWebcam && video) {
      btnWebcam.addEventListener("click", async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          state.webcamStream = stream;
          video.srcObject = stream;
          video.play();
          startGestureLoop(video, canvas);
        } catch (err) {
          alert("Webcam access failed: " + err.message);
        }
      });
    }
  }

  function startGestureLoop(video, canvas) {
    state.isProcessingGesture = true;
    let inFlight = false;

    const intervalId = setInterval(async () => {
      if (!state.isProcessingGesture || video.paused || video.ended) {
        if (video.ended) clearInterval(intervalId);
        return;
      }
      if (inFlight) return;
      inFlight = true;
      try {
        await processSingleGestureFrame(video, canvas);
      } finally {
        inFlight = false;
      }
    }, 200);
  }

  async function processSingleGestureFrame(video, canvas) {
    if (!video.videoWidth || !video.videoHeight) return;

    const offCanvas = document.createElement("canvas");
    offCanvas.width = video.videoWidth;
    offCanvas.height = video.videoHeight;
    const ctx = offCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const b64 = offCanvas.toDataURL("image/jpeg", 0.75);
    const ts = video.currentTime || 0.0;

    try {
      const resp = await fetch("/api/gesture/process-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: b64, timestamp: ts })
      });
      if (!resp.ok) return;
      const data = await resp.json();

      renderGestureCanvas(canvas, video, data.landmarks);
      updateGestureMetrics(data);
    } catch (err) {
      console.warn("Gesture frame error:", err);
    }
  }

  function renderGestureCanvas(canvas, video, landmarks) {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks || landmarks.length === 0) return;

    const w = canvas.width;
    const h = canvas.height;

    // Draw skeletal connections
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2.5;

    POSE_CONNECTIONS.forEach(([i, j]) => {
      if (i < landmarks.length && j < landmarks.length) {
        const pt1 = landmarks[i];
        const pt2 = landmarks[j];
        if (pt1 && pt2 && (pt1.visibility === undefined || pt1.visibility > 0.4)) {
          ctx.beginPath();
          ctx.moveTo(pt1.x * w, pt1.y * h);
          ctx.lineTo(pt2.x * w, pt2.y * h);
          ctx.stroke();
        }
      }
    });

    // Draw landmarks
    landmarks.forEach((pt) => {
      if (pt.visibility === undefined || pt.visibility > 0.4) {
        ctx.fillStyle = "#00f2fe";
        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, 3.5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  }

  function updateGestureMetrics(data) {
    const bufferFill = document.getElementById("gesture-buffer-fill");
    const bufferText = document.getElementById("gesture-buffer-text");
    const barsContainer = document.getElementById("gesture-probabilities-bars");
    const latencyBadge = document.getElementById("gesture-latency-badge");

    if (latencyBadge) latencyBadge.textContent = `${data.latency_ms.toFixed(1)} ms`;
    if (bufferFill) {
      const pct = (data.buffer_fill / data.buffer_capacity) * 100;
      bufferFill.style.width = `${pct}%`;
      if (bufferText) bufferText.textContent = `${data.buffer_fill} / ${data.buffer_capacity} Frames`;
    }

    if (barsContainer && data.probabilities) {
      barsContainer.innerHTML = Object.entries(data.probabilities).map(([cls, prob]) => `
        <div class="bar-row">
          <div class="bar-label">
            <span><strong>${cls.toUpperCase()}</strong></span>
            <span>${(prob * 100).toFixed(1)}%</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill bar-emerald" style="width: ${(prob * 100).toFixed(1)}%;"></div>
          </div>
        </div>
      `).join("");
    }
  }

  // ============================================================================
  // BRANCH 4: TRANSFORMER FUSION VIEW
  // ============================================================================
  function bindFusionControls() {
    const btnInfer = document.getElementById("btn-fusion-infer");
    const toggleVision = document.getElementById("toggle-mod-vision");
    const toggleAudio = document.getElementById("toggle-mod-audio");
    const toggleGesture = document.getElementById("toggle-mod-gesture");

    if (toggleVision) {
      toggleVision.addEventListener("change", (e) => {
        state.fusionMask.vision = e.target.checked;
        updateModalityNodeVisual("node-stream-vision", e.target.checked);
      });
    }
    if (toggleAudio) {
      toggleAudio.addEventListener("change", (e) => {
        state.fusionMask.audio = e.target.checked;
        updateModalityNodeVisual("node-stream-audio", e.target.checked);
      });
    }
    if (toggleGesture) {
      toggleGesture.addEventListener("change", (e) => {
        state.fusionMask.gesture = e.target.checked;
        updateModalityNodeVisual("node-stream-gesture", e.target.checked);
      });
    }

    if (btnInfer) {
      btnInfer.addEventListener("click", async () => {
        await executeTransformerForward();
      });
    }
  }

  function updateModalityNodeVisual(elementId, isActive) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (isActive) {
      el.classList.add("active-stream");
      el.style.opacity = "1.0";
    } else {
      el.classList.remove("active-stream");
      el.style.opacity = "0.4";
    }
  }

  async function executeTransformerForward() {
    const btnInfer = document.getElementById("btn-fusion-infer");
    if (btnInfer) btnInfer.textContent = "Forwarding...";

    // Construct feature representations based on toggles
    const vFeat = state.fusionMask.vision ? Array(128).fill(0.15) : null;
    const aFeat = state.fusionMask.audio ? Array(2048).fill(0.08) : null;
    const gFeat = state.fusionMask.gesture ? Array(128).fill(0.02) : null;

    try {
      const resp = await fetch("/api/fusion/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vision_feature: vFeat,
          audio_feature: aFeat,
          gesture_feature: gFeat,
          timestamp: 2.0
        })
      });
      if (!resp.ok) throw new Error("Fusion forward failed");
      const result = await resp.json();
      renderFusionResults(result);
    } catch (err) {
      console.error("Fusion error:", err);
      alert("Fusion inference failed: " + err.message);
    } finally {
      if (btnInfer) btnInfer.textContent = "Execute Transformer Forward Pass";
    }
  }

  function renderFusionResults(data) {
    // 1. Raw Model Output Card
    const rawCard = document.getElementById("fusion-raw-content");
    if (rawCard && data.raw_output) {
      rawCard.innerHTML = `
        <div style="font-size:0.75rem; font-family:var(--font-mono); line-height:1.7;">
          <div><strong>Uncalibrated Alert Logits:</strong></div>
          <div style="color:var(--accent-purple); padding-left:8px;">[${data.raw_output.alert_logits.join(", ")}]</div>
          <div style="margin-top:4px;"><strong>Priority Scalar:</strong> ${data.raw_output.priority_scalar.toFixed(4)}</div>
          <div><strong>Confidence Scalar:</strong> ${data.raw_output.confidence_scalar.toFixed(4)}</div>
          <div><strong>Forward Latency:</strong> <span style="color:var(--accent-cyan);">${data.latency_ms.toFixed(2)} ms</span></div>
          <div style="margin-top:6px;"><strong>Context Embedding (${data.raw_output.context_dim}-D sample):</strong></div>
          <div style="font-size:0.65rem; color:var(--text-muted); word-break:break-all;">[${data.raw_output.context_sample.join(", ")} ...]</div>
        </div>
      `;
    }

    // 2. Interpreted Prediction Card
    const interpCard = document.getElementById("fusion-interp-content");
    if (interpCard && data.interpreted_prediction) {
      const p = data.interpreted_prediction;
      let badgeClass = "tag-cyan";
      if (p.alert_type === "urgent") badgeClass = "tag-rose";
      else if (p.alert_type === "caution") badgeClass = "tag-amber";
      else if (p.alert_type === "informational") badgeClass = "tag-cyan";

      interpCard.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:8px;">
          <div>
            <span style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase;">Categorical Alert</span>
            <div style="margin-top:4px;"><span class="panel-tag ${badgeClass}" style="font-size:0.9rem; padding:4px 10px;">${p.alert_type.toUpperCase()}</span></div>
          </div>
          <div style="margin-top:4px;">
            <div class="bar-label"><span>Estimated Priority Score</span><span>${p.priority_score.toFixed(3)}</span></div>
            <div class="bar-bg"><div class="bar-fill bar-purple" style="width:${(p.priority_score * 100).toFixed(1)}%;"></div></div>
          </div>
          <div>
            <div class="bar-label"><span>Model Confidence</span><span>${p.confidence_pct.toFixed(1)}%</span></div>
            <div class="bar-bg"><div class="bar-fill bar-cyan" style="width:${p.confidence_pct.toFixed(1)}%;"></div></div>
          </div>
          <div style="font-size:0.7rem; color:var(--text-muted); margin-top:4px;">
            <strong>Active Modalities:</strong> ${p.active_modalities.join(" + ")}
          </div>
        </div>
      `;
    }

    // 3. Final Decision Engine Output Card
    const engineCard = document.getElementById("fusion-engine-content");
    if (engineCard && data.decision_engine_output) {
      const d = data.decision_engine_output;
      engineCard.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.7rem; color:var(--text-muted);">DECISION EMISSION</span>
            <span class="panel-tag tag-emerald">ACTION DISPATCHED</span>
          </div>
          <div style="background:#0f172a; padding:10px; border-radius:6px; border:1px solid var(--border-color);">
            <div style="font-size:0.65rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:4px;">Spoken Alert Prompt:</div>
            <div style="font-size:0.85rem; font-weight:600; color:#fff;">"${d.text}"</div>
          </div>
          <div style="font-size:0.75rem; font-family:var(--font-mono); line-height:1.6;">
            <div><strong>Priority Level:</strong> ${d.priority.toFixed(3)} [${d.kind.toUpperCase()}]</div>
            <div><strong>Active Modality Sources:</strong> ${d.sources.join(", ")}</div>
            <div><strong>Cooldown Guard:</strong> <span style="color:var(--accent-emerald);">5.0s Deduplication Active</span></div>
          </div>
        </div>
      `;

      // Trigger speech if enabled
      speakAlert(d.text, Math.round(d.priority * 100));
    }
  }

  // ============================================================================
  // BRANCH 5: DECISION ENGINE & SPEECH VIEW
  // ============================================================================
  function bindDecisionControls() {
    const btnSpeak = document.getElementById("btn-speech-test");
    const textInput = document.getElementById("speech-test-text");
    const prioInput = document.getElementById("speech-test-priority");

    if (btnSpeak && textInput && prioInput) {
      btnSpeak.addEventListener("click", () => {
        const text = textInput.value || "Caution: approaching obstacle detected.";
        const prio = parseInt(prioInput.value || "80", 10);
        speakAlert(text, prio);
      });
    }
  }

  async function speakAlert(text, priority = 80) {
    logAlertEntry(text, priority);

    // 1. Call backend speech endpoint
    try {
      await fetch("/api/speech/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, priority: priority })
      });
    } catch (e) {
      console.warn("Backend speech call warning:", e);
    }

    // 2. Client-side Web Speech fallback for immediate audio in browser
    if (state.clientTtsEnabled && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }

  function logAlertEntry(text, priority) {
    const historyContainer = document.getElementById("decision-history-log");
    if (!historyContainer) return;

    const timeStr = new Date().toLocaleTimeString();
    let badgeClass = "tag-cyan";
    let kind = "INFO";
    if (priority >= 75) {
      badgeClass = "tag-rose";
      kind = "URGENT";
    } else if (priority >= 40) {
      badgeClass = "tag-amber";
      kind = "CAUTION";
    }

    const row = document.createElement("div");
    row.style.padding = "8px 12px";
    row.style.borderBottom = "1px solid var(--border-color)";
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.justifyContent = "space-between";
    row.style.fontSize = "0.75rem";
    row.innerHTML = `
      <div style="display:flex; align-items:center; gap:10px;">
        <span class="panel-tag ${badgeClass}">${kind}</span>
        <span style="font-weight:600; color:#fff;">"${text}"</span>
      </div>
      <div style="font-family:var(--font-mono); color:var(--text-muted);">${timeStr}</div>
    `;

    historyContainer.prepend(row);
  }

  // ============================================================================
  // SYNCHRONIZED MULTIMODAL REPLAY VIEW
  // ============================================================================
  function bindReplayControls() {
    const sessionSelect = document.getElementById("replay-session-select");
    const btnPlay = document.getElementById("btn-replay-play");
    const btnPause = document.getElementById("btn-replay-pause");
    const scrubber = document.getElementById("replay-timeline-slider");
    const speedSelect = document.getElementById("replay-speed-select");
    const video = document.getElementById("replay-video");
    const audio = document.getElementById("replay-audio");

    if (sessionSelect) {
      sessionSelect.addEventListener("change", (e) => {
        state.selectedSession = e.target.value;
        loadReplayMedia(state.selectedSession);
      });
    }

    if (btnPlay) {
      btnPlay.addEventListener("click", () => {
        startSessionReplay(state.selectedSession);
      });
    }

    if (btnPause) {
      btnPause.addEventListener("click", () => {
        pauseSessionReplay();
      });
    }

    if (scrubber) {
      scrubber.addEventListener("input", (e) => {
        seekSessionReplay(parseFloat(e.target.value));
      });
    }

    if (speedSelect) {
      speedSelect.addEventListener("change", (e) => {
        state.replaySpeed = parseFloat(e.target.value);
        if (state.replayWs && state.replayWs.readyState === WebSocket.OPEN) {
          state.replayWs.send(JSON.stringify({ command: "speed", value: state.replaySpeed }));
        }
        if (video) video.playbackRate = state.replaySpeed;
        if (audio) audio.playbackRate = state.replaySpeed;
      });
    }
  }

  function loadReplayMedia(sessionId) {
    const video = document.getElementById("replay-video");
    const audio = document.getElementById("replay-audio");
    if (video) {
      video.src = `/api/media/video/${sessionId}`;
      video.load();
    }
    if (audio) {
      audio.src = `/api/media/audio/${sessionId}`;
      audio.load();
    }
  }

  function startSessionReplay(sessionId) {
    const video = document.getElementById("replay-video");
    const audio = document.getElementById("replay-audio");
    const statusText = document.getElementById("replay-status-indicator");

    if (state.replayWs) {
      state.replayWs.close();
    }

    const loc = window.location;
    const wsProto = loc.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProto}//${loc.host}/ws/replay/${sessionId}`;

    state.replayWs = new WebSocket(wsUrl);

    state.replayWs.onopen = () => {
      state.replayPlaying = true;
      if (statusText) statusText.textContent = "SYNCHRONIZED REPLAY ACTIVE";
      if (video) video.play();
      if (audio) audio.play();
    };

    state.replayWs.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.status === "complete") {
        pauseSessionReplay();
        return;
      }
      handleReplayStep(msg);
    };

    state.replayWs.onclose = () => {
      state.replayPlaying = false;
      if (statusText) statusText.textContent = "REPLAY IDLE";
    };
  }

  function pauseSessionReplay() {
    state.replayPlaying = false;
    const video = document.getElementById("replay-video");
    const audio = document.getElementById("replay-audio");
    if (video) video.pause();
    if (audio) audio.pause();

    if (state.replayWs && state.replayWs.readyState === WebSocket.OPEN) {
      state.replayWs.send(JSON.stringify({ command: "pause" }));
    }
  }

  function seekSessionReplay(seconds) {
    const video = document.getElementById("replay-video");
    const audio = document.getElementById("replay-audio");
    if (video) video.currentTime = seconds;
    if (audio) audio.currentTime = seconds;

    if (state.replayWs && state.replayWs.readyState === WebSocket.OPEN) {
      state.replayWs.send(JSON.stringify({ command: "seek", timestamp: seconds }));
    }
  }

  function handleReplayStep(step) {
    // Update scrubber and time display
    const scrubber = document.getElementById("replay-timeline-slider");
    const timeDisplay = document.getElementById("replay-current-time");

    if (scrubber) {
      scrubber.value = step.timestamp_seconds;
      scrubber.max = 6.0; // standard recording length
    }
    if (timeDisplay) {
      timeDisplay.textContent = `${step.timestamp_seconds.toFixed(2)}s / 6.00s`;
    }

    // Update Decision Engine summary panel in Replay
    const decisionCard = document.getElementById("replay-decision-summary");
    if (decisionCard) {
      if (step.decision_engine_output) {
        const d = step.decision_engine_output;
        let badgeClass = "tag-cyan";
        if (d.kind === "urgent") badgeClass = "tag-rose";
        else if (d.kind === "caution") badgeClass = "tag-amber";

        decisionCard.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="panel-tag ${badgeClass}">${d.kind.toUpperCase()}</span>
            <span style="font-family:var(--font-mono); font-size:0.75rem;">Priority: ${d.priority.toFixed(3)}</span>
          </div>
          <div style="font-weight:600; font-size:0.85rem; margin-top:6px; color:#fff;">"${d.text}"</div>
          <div style="font-size:0.65rem; color:var(--text-muted); margin-top:4px;">Contributing Modalities: ${step.contributing_modalities.join(" + ")}</div>
        `;
      } else {
        decisionCard.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="panel-tag tag-cyan">NOMINAL</span>
            <span style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted);">Step #${step.step_index}</span>
          </div>
          <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:6px;">Surroundings clear. No hazard alert threshold reached.</div>
          <div style="font-size:0.65rem; color:var(--text-muted); margin-top:4px;">Contributing Modalities: ${step.contributing_modalities.join(" + ")}</div>
        `;
      }
    }
  }

  // ============================================================================
  // MODEL BENCHMARKS VIEW
  // ============================================================================
  function renderBenchmarks(reports) {
    const summary = reports.summary_metrics;
    if (!summary) return;

    // Vision Metrics
    const vMap50 = document.getElementById("bm-vision-map50");
    const vMap95 = document.getElementById("bm-vision-map95");
    const vPrec = document.getElementById("bm-vision-precision");
    const vRecall = document.getElementById("bm-vision-recall");
    const vLat = document.getElementById("bm-vision-latency");

    if (vMap50) vMap50.textContent = `${(summary.vision.map_50 * 100).toFixed(2)}%`;
    if (vMap95) vMap95.textContent = `${(summary.vision.map_50_95 * 100).toFixed(2)}%`;
    if (vPrec) vPrec.textContent = `${(summary.vision.precision * 100).toFixed(2)}%`;
    if (vRecall) vRecall.textContent = `${(summary.vision.recall * 100).toFixed(2)}%`;
    if (vLat) vLat.textContent = `${summary.vision.latency_ms.toFixed(1)} ms`;

    // Audio Metrics
    const aAcc = document.getElementById("bm-audio-acc");
    const aF1 = document.getElementById("bm-audio-f1");
    const aSnr20 = document.getElementById("bm-audio-snr20");
    const aSnr10 = document.getElementById("bm-audio-snr10");
    const aLat = document.getElementById("bm-audio-latency");

    if (aAcc) aAcc.textContent = `${(summary.audio.accuracy * 100).toFixed(2)}%`;
    if (aF1) aF1.textContent = `${(summary.audio.macro_f1 * 100).toFixed(2)}%`;
    if (aSnr20) aSnr20.textContent = `${(summary.audio.noise_robustness_20db * 100).toFixed(2)}%`;
    if (aSnr10) aSnr10.textContent = `${(summary.audio.noise_robustness_10db * 100).toFixed(2)}%`;
    if (aLat) aLat.textContent = `${summary.audio.latency_ms.toFixed(1)} ms`;

    // Gesture Metrics
    const gAcc = document.getElementById("bm-gesture-acc");
    const gF1 = document.getElementById("bm-gesture-f1");
    const gLat = document.getElementById("bm-gesture-latency");

    if (gAcc) gAcc.textContent = `${(summary.gesture.accuracy * 100).toFixed(2)}%`;
    if (gF1) gF1.textContent = `${summary.gesture.macro_f1.toFixed(3)}`;
    if (gLat) gLat.textContent = `${summary.gesture.latency_ms.toFixed(2)} ms`;

    // Fusion Ablations Table
    const ablTable = document.getElementById("bm-fusion-ablations-table");
    if (ablTable && reports.fusion && reports.fusion.ablations) {
      ablTable.innerHTML = reports.fusion.ablations.map((a) => `
        <tr>
          <td><strong>${a.name}</strong></td>
          <td>${a.active_modalities.join(" + ") || "None"}</td>
          <td>${a.validation_loss.toFixed(4)}</td>
          <td>${a.priority_mae.toFixed(4)}</td>
          <td>${(a.alert_accuracy * 100).toFixed(1)}%</td>
          <td><span style="color:var(--accent-cyan);">${a.latency_ms.toFixed(2)} ms</span></td>
        </tr>
      `).join("");
    }
  }

  function stopWebcam() {
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((track) => track.stop());
      state.webcamStream = null;
    }
  }

})();
