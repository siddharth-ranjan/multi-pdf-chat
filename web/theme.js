// Runs before first paint: apply the saved theme and skip the upload screen when a chat is being restored
(function () {
  var root = document.documentElement;
  try {
    var theme = localStorage.getItem("theme");
    if (theme === "light" || theme === "dark") root.dataset.theme = theme;
  } catch (e) {}
  try {
    if (sessionStorage.getItem("chatpdf-session")) root.classList.add("has-session");
  } catch (e) {}
})();
