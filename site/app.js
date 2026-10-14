// Copyright 2026 Compounding Intelligence. SPDX-License-Identifier: Apache-2.0
// OpenRecipes: copy buttons, opening the recipe a link points at, and the index and GPU filters. No dependencies.
(function () {
  "use strict";
  function flash(button, text) {
    var label = button.getAttribute("data-label") || "Copy";
    button.textContent = text;
    setTimeout(function () { button.textContent = label; }, 1600);
  }
  function selectAndCopy(node) {
    var range = document.createRange();
    range.selectNodeContents(node);
    var sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    try { return document.execCommand("copy"); } catch (e) { return false; }
  }
  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-copy]");
    if (!button) return;
    var node = document.getElementById(button.getAttribute("data-copy"));
    if (!node) return;
    var text = node.textContent;
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(
        function () { flash(button, "Copied"); },
        function () { flash(button, selectAndCopy(node) ? "Copied" : "Select and copy"); });
    } else {
      flash(button, selectAndCopy(node) ? "Copied" : "Select and copy");
    }
  });
  function openTarget() {
    var id = decodeURIComponent(window.location.hash.slice(1));
    if (!id) return;
    var node = document.getElementById(id);
    if (node && node.tagName === "DETAILS") node.open = true;
  }
  var toggle = document.querySelector(".idx .tg");
  if (toggle) {
    toggle.hidden = false;
    toggle.addEventListener("click", function () {
      var on = toggle.getAttribute("aria-pressed") !== "true";
      toggle.setAttribute("aria-pressed", on ? "true" : "false");
      toggle.closest(".idx").classList.toggle("only", on);
    });
  }
  var seg = document.querySelector(".seg");
  if (seg) seg.addEventListener("click", function (event) {
    var button = event.target.closest("button[data-gpu]");
    if (!button) return;
    var gpu = button.getAttribute("data-gpu");
    seg.querySelectorAll("button").forEach(function (b) {
      b.classList.toggle("on", b === button);
      b.setAttribute("aria-pressed", b === button ? "true" : "false");
    });
    document.querySelectorAll(".mb").forEach(function (block) {
      var shown = 0;
      block.querySelectorAll("tr[data-gpu]").forEach(function (row) {
        row.hidden = !!gpu && row.getAttribute("data-gpu") !== gpu;
        if (!row.hidden) shown++;
      });
      block.hidden = !shown;
    });
  });
  window.addEventListener("hashchange", openTarget);
  openTarget();
})();
