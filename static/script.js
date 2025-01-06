// let chooseFileButton = document.getElementById("choose-file-button");
// let captureImageButton = document.getElementById("capture-image-button");
// let analyzeButton = document.getElementById("analyze-button");
// let fileInput = document.getElementById("file-input");
// let videoElement = document.getElementById("video-element");
// let cameraContainer = document.getElementById("camera-container");
// let previewImage = document.getElementById("preview-image");
// let captureButton = document.getElementById("capture-button");
// let resultSection = document.getElementById("result-section");
// let recommendationForm = document.getElementById("recommendation-form");
// let recommendationResult = document.getElementById("recommendation-result");

// chooseFileButton.addEventListener("click", () => {
//   fileInput.click();
// });

// fileInput.addEventListener("change", (event) => {
//   let file = event.target.files[0];
//   let reader = new FileReader();
//   reader.onload = (e) => {
//     previewImage.src = e.target.result;
//     previewImage.style.display = "block";
//     analyzeButton.disabled = false;
//   };
//   reader.readAsDataURL(file);
// });

// captureImageButton.addEventListener("click", () => {
//   cameraContainer.style.display = "flex";
//   navigator.mediaDevices
//     .getUserMedia({ video: true })
//     .then((stream) => {
//       videoElement.srcObject = stream;
//       captureButton.style.display = "block";
//     })
//     .catch((err) => {
//       console.error("Error accessing camera: ", err);
//     });
// });

// captureButton.addEventListener("click", () => {
//   let canvas = document.createElement("canvas");
//   canvas.width = videoElement.videoWidth;
//   canvas.height = videoElement.videoHeight;
//   let context = canvas.getContext("2d");
//   context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
//   let dataUrl = canvas.toDataURL("image/png");
//   previewImage.src = dataUrl;
//   previewImage.style.display = "block";
//   analyzeButton.disabled = false;
//   videoElement.srcObject.getTracks().forEach((track) => track.stop());
// });

// analyzeButton.addEventListener("click", () => {
//   let formData = new FormData();
//   if (fileInput.files[0]) {
//     formData.append("file", fileInput.files[0]);
//   } else {
//     let dataUrl = previewImage.src;
//     let base64Image = dataUrl.split(",")[1];
//     formData.append("image_data", base64Image);
//   }

//   fetch("/predict", {
//     method: "POST",
//     body: formData,
//   })
//     .then((response) => response.json())
//     .then((data) => {
//       resultSection.innerHTML = `<span>${data.prediction}</span>`;
//     })
//     .catch((error) => {
//       console.error("Error:", error);
//       resultSection.innerHTML = `<span style="color: red;">An error occurred</span>`;
//     });
// });
// document.getElementById("recommend-button").addEventListener("click", () => {
//   const formData = new FormData(recommendationForm);

//   fetch("/recommend", {
//     method: "POST",
//     body: formData,
//   })
//     .then((response) => response.json())
//     .then((data) => {
//       const recommendations = data.recommendations
//         .map((product) => `<li>${product}</li>`)
//         .join("");
//       recommendationResult.innerHTML = `<h3>Recommended Products:</h3><ul>${recommendations}</ul>`;
//     })
//     .catch((error) => {
//       console.error("Error during recommendation:", error);
//       recommendationResult.innerHTML = `<span style="color: red;">An error occurred</span>`;
//     });
// });
// // Consultation form handling
// document.getElementById("consult-button").addEventListener("click", () => {
//   const formData = new FormData(document.getElementById("consultation-form"));

//   fetch("/consult", {
//     method: "POST",
//     body: formData,
//   })
//     .then((response) => response.json())
//     .then((data) => {
//       document.getElementById(
//         "consultation-result"
//       ).innerHTML = `<span style="color: green;">${data.message}</span>`;
//     })
//     .catch((error) => {
//       console.error("Error during consultation request:", error);
//       document.getElementById(
//         "consultation-result"
//       ).innerHTML = `<span style="color: red;">An error occurred</span>`;
//     });
// });

document.addEventListener("DOMContentLoaded", function () {
  const uploadForm = document.getElementById("upload-form");
  if (uploadForm) {
    uploadForm.addEventListener("submit", async function (event) {
      event.preventDefault();
      const formData = new FormData(uploadForm);
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      document.getElementById("result").innerText = JSON.stringify(result);
    });
  }

  const recommendForm = document.getElementById("recommend-form");
  if (recommendForm) {
    recommendForm.addEventListener("submit", async function (event) {
      event.preventDefault();
      const formData = new FormData(recommendForm);
      const response = await fetch("/recommend", {
        method: "POST",
        body: new URLSearchParams(new FormData(recommendForm)),
      });
      const result = await response.json();
      // document.getElementById("recommendations").innerText = JSON.stringify(
      //   result.recommendations
      // );
      document.getElementById("recommendations").innerHTML =
        result.recommendations
          .map(
            (recommendation) => `
        <div class="recommendation-card">
          <h3>${recommendation}</h3>
        </div>
      `
          )
          .join("");
    });
  }

  const consultForm = document.getElementById("consult-form");
  if (consultForm) {
    consultForm.addEventListener("submit", async function (event) {
      event.preventDefault();
      const formData = new FormData(consultForm);
      const response = await fetch("/consult", {
        method: "POST",
        body: new URLSearchParams(new FormData(consultForm)),
      });
      const result = await response.json();
      document.getElementById("consultation-result").innerText = result.message;
    });
  }
});
