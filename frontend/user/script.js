const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const previewWrap = document.getElementById("previewWrap");
const previewImage = document.getElementById("previewImage");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const buttonLabel = predictBtn.querySelector(".button-label");
const message = document.getElementById("message");
const resultCard = document.getElementById("resultCard");

let selectedFile = null;
let previewUrl = null;

function setButtonLoading(isLoading) {
  predictBtn.disabled = isLoading || !selectedFile;
  buttonLabel.textContent = isLoading ? "বিশ্লেষণ চলছে..." : "মান পরীক্ষা করুন";
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    message.textContent = "দয়া করে শুধুমাত্র ইটের ছবি নির্বাচন করুন।";
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewWrap.classList.remove("hidden");
  dropZone.classList.add("hidden");
  predictBtn.disabled = false;
  message.textContent = "";
}

function clearSelection() {
  selectedFile = null;
  imageInput.value = "";
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  previewWrap.classList.add("hidden");
  dropZone.classList.remove("hidden");
  predictBtn.disabled = true;
  message.textContent = "";
}

imageInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    setFile(e.target.files[0]);
  }
});

clearBtn.addEventListener("click", clearSelection);

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    setFile(e.dataTransfer.files[0]);
  }
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setButtonLoading(true);
  message.textContent = "";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "সার্ভারে সমস্যা হয়েছে।");
    }

    resultCard.className = "result-card status-" + (data.status_key || "g1");
    resultCard.innerHTML = `
      <div style="padding: 20px; background: white; border-radius: 20px; border: 1px solid #dbe7de;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
          <div>
            <span style="font-size: 11px; font-weight: bold; text-transform: uppercase; color: #168246; background: #eaf6ee; padding: 4px 10px; border-radius: 20px;">মূল্যায়িত গ্রেড</span>
            <h3 style="margin: 8px 0 0; font-size: 24px; color: #1b2c20;">${data.predicted_class}</h3>
          </div>
          <div style="text-align: right;">
            <span style="font-size: 26px; font-weight: 800; color: #168246;">${data.confidence}%</span>
            <small style="display: block; font-size: 11px; color: #66746a;">কনফিডেন্স স্কোর</small>
          </div>
        </div>

        <div style="height: 8px; width: 100%; background: #eef4ee; border-radius: 4px; overflow: hidden; margin-bottom: 16px;">
          <div style="height: 100%; width: ${data.confidence}%; background: #168246; border-radius: 4px;"></div>
        </div>

        <div style="margin-bottom: 14px;">
          <strong style="font-size: 13px; color: #1b2c20; display: block; margin-bottom: 4px;">বিশ্লেষণ সারসংক্ষেপ:</strong>
          <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #435447;">${data.explanation}</p>
        </div>

        <div style="background: #f7faf7; border-left: 4px solid #168246; padding: 12px 14px; border-radius: 0 12px 12px 0;">
          <strong style="font-size: 13px; color: #168246; display: block; margin-bottom: 4px;">প্রকৌশল সুপারিশ ও ব্যবহার:</strong>
          <p style="margin: 0; font-size: 13px; line-height: 1.6; color: #2e4033;">${data.recommendation}</p>
        </div>
      </div>
    `;
  } catch (err) {
    message.textContent = err.message || "ছবিটি পরীক্ষা করতে সমস্যা হয়েছে।";
  } finally {
    setButtonLoading(false);
  }
});
