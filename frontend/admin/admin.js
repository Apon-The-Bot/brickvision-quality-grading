const CLASSES = [
  "G1 - First Class Brick",
  "G2 - Second Class Brick",
  "G3 - Third Class Brick",
  "G4 - Jhama / Rejected"
];

const uploadForm = document.getElementById("uploadForm");
const uploadMessage = document.getElementById("uploadMessage");
const modelStats = document.getElementById("modelStats");
const trainBtn = document.getElementById("trainBtn");
const trainLog = document.getElementById("trainLog");
const clearLogBtn = document.getElementById("clearLogBtn");
const viewReportBtn = document.getElementById("viewReportBtn");
const reportContainer = document.getElementById("reportContainer");
const reportText = document.getElementById("reportText");
const reportGraph = document.getElementById("reportGraph");
const addAnotherBtn = document.getElementById("addAnotherBtn");
const uploadRowsContainer = document.getElementById("uploadRowsContainer");
const splitSelect = document.getElementById("split");

let polling = null;

function initializeUploadForm() {
  uploadRowsContainer.innerHTML = "";
  addUploadRow();
}

function addUploadRow() {
  const rowCount = uploadRowsContainer.querySelectorAll(".upload-row").length;
  if (rowCount >= CLASSES.length) return;

  const row = document.createElement("div");
  row.className = "upload-row";
  row.style.background = "#fdfdfd";
  row.style.border = "1px solid #e2ebd9";
  row.style.borderRadius = "16px";
  row.style.padding = "15px";
  row.style.marginBottom = "12px";
  row.style.position = "relative";

  row.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <span style="font-weight: bold; color: #168246; font-size: 14px;">Class Selection</span>
      <button type="button" class="remove-row-btn" style="background: #fef2f2; color: #b01a1a; padding: 4px 8px; border-radius: 8px; font-size: 12px; font-weight: bold; cursor: pointer; border: 1px solid #fecaca; display: none;">Remove</button>
    </div>
    
    <label style="font-size: 12px; margin-top: 4px; display: block; font-weight: 800;">Class Name</label>
    <select class="class-select" required style="margin-bottom: 8px; width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #cad8ce; font-size: 14px; background: white;"></select>
    
    <label style="font-size: 12px; margin-top: 4px; display: block; font-weight: 800;">Select Images</label>
    <input class="class-files-input" type="file" accept="image/*" multiple required style="padding: 8px; width: 100%; border: 1px solid #cad8ce; border-radius: 8px; font-size: 14px; background: white;" />
  `;

  uploadRowsContainer.appendChild(row);

  const select = row.querySelector(".class-select");
  CLASSES.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    select.appendChild(opt);
  });

  const selectedValues = Array.from(uploadRowsContainer.querySelectorAll(".class-select"))
    .map(s => s.value)
    .filter(val => val !== "");
  const availableClass = CLASSES.find(c => !selectedValues.includes(c)) || CLASSES[0];
  select.value = availableClass;

  const removeBtn = row.querySelector(".remove-row-btn");
  removeBtn.addEventListener("click", () => {
    row.remove();
    updateRemoveButtonsVisibility();
  });

  updateRemoveButtonsVisibility();
}

function updateRemoveButtonsVisibility() {
  const rows = uploadRowsContainer.querySelectorAll(".upload-row");
  rows.forEach(row => {
    const btn = row.querySelector(".remove-row-btn");
    btn.style.display = rows.length > 1 ? "block" : "none";
  });
}

async function loadModelStats() {
  try {
    const res = await fetch("/admin/model-stats");
    const data = await res.json();
    let html = `
      <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead>
          <tr style="border-bottom: 2px solid #e2ebd9; text-align: left;">
            <th style="padding: 8px;">Class Name</th>
            <th style="padding: 8px; text-align: center;">train</th>
            <th style="padding: 8px; text-align: center;">validation</th>
            <th style="padding: 8px; text-align: center;">test</th>
            <th style="padding: 8px; text-align: right;">Total</th>
          </tr>
        </thead>
        <tbody>
    `;
    let grandTrain = 0, grandVal = 0, grandTest = 0, grandTotal = 0;
    for (const [cls, counts] of Object.entries(data.stats)) {
      grandTrain += counts.train;
      grandVal += counts.validation;
      grandTest += counts.test;
      grandTotal += counts.total;
      html += `
        <tr style="border-bottom: 1px solid #f0f4ec;">
          <td style="padding: 10px 8px; font-weight: 600;">${cls}</td>
          <td style="padding: 10px 8px; text-align: center;">${counts.train}</td>
          <td style="padding: 10px 8px; text-align: center;">${counts.validation}</td>
          <td style="padding: 10px 8px; text-align: center;">${counts.test}</td>
          <td style="padding: 10px 8px; text-align: right; font-weight: bold;">${counts.total}</td>
        </tr>
      `;
    }
    html += `
        <tr style="border-top: 2px solid #cad8ce; font-weight: 800; background: #fbfdfa;">
          <td style="padding: 12px 8px;">Grand Total</td>
          <td style="padding: 12px 8px; text-align: center;">${grandTrain}</td>
          <td style="padding: 12px 8px; text-align: center;">${grandVal}</td>
          <td style="padding: 12px 8px; text-align: center;">${grandTest}</td>
          <td style="padding: 12px 8px; text-align: right; color: #168246;">${grandTotal}</td>
        </tr>
      </tbody>
    </table>
    `;
    modelStats.innerHTML = html;
  } catch (err) {
    modelStats.innerHTML = "<p>Error loading model stats.</p>";
  }
}

async function loadTrainLog() {
  try {
    const res = await fetch("/admin/train-status");
    const data = await res.json();
    trainLog.textContent = data.log || "No log available.";
    trainLog.scrollTop = trainLog.scrollHeight;
  } catch (err) {
    trainLog.textContent = "Error loading log.";
  }
}

viewReportBtn.addEventListener("click", async () => {
  if (reportContainer.style.display === "none") {
    try {
      const res = await fetch("/admin/evaluation-report");
      const data = await res.json();
      reportText.innerHTML = `<pre style="font-family: inherit; font-size: 13px; margin: 0; line-height: 1.6; white-space: pre-wrap;">${data.report}</pre>`;
      reportGraph.src = "/admin/training-history?t=" + Date.now();
      reportContainer.style.display = "block";
      viewReportBtn.textContent = "Hide Report";
    } catch (e) {
      alert("Failed to load report.");
    }
  } else {
    reportContainer.style.display = "none";
    viewReportBtn.textContent = "View Report";
  }
});

clearLogBtn.addEventListener("click", async () => {
  if (confirm("Clear current training log?")) {
    await fetch("/admin/clear-log", { method: "POST" });
    loadTrainLog();
  }
});

trainBtn.addEventListener("click", async () => {
  trainBtn.disabled = true;
  trainBtn.textContent = "Training...";
  try {
    await fetch("/admin/train", { method: "POST" });
    setTimeout(() => {
      loadTrainLog();
      trainBtn.disabled = false;
      trainBtn.textContent = "Start Training";
    }, 1500);
  } catch (e) {
    trainBtn.disabled = false;
    trainBtn.textContent = "Start Training";
  }
});

addAnotherBtn.addEventListener("click", addUploadRow);

initializeUploadForm();
loadModelStats();
loadTrainLog();
