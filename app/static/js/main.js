/**
 * main.js — FoodWise frontend utilities
 */

// Show loading state on form submit
document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector(".predict-form");
  const btn  = document.querySelector(".btn-predict");

  if (form && btn) {
    form.addEventListener("submit", (e) => {
      // Basic HTML5 validation first
      if (!form.checkValidity()) return;
      btn.textContent = "⏳ Predicting…";
      btn.disabled = true;
    });
  }

  // Animate waste meter bar on result page
  const resultValue = document.querySelector(".result-value");
  if (resultValue) {
    resultValue.style.opacity = "0";
    resultValue.style.transform = "scale(0.8)";
    requestAnimationFrame(() => {
      resultValue.style.transition = "opacity .5s ease, transform .5s ease";
      resultValue.style.opacity = "1";
      resultValue.style.transform = "scale(1)";
    });
  }
});
