"use strict";

(() => {
  const STORE_KEY = "chatpdf-session";
  const HEARTBEAT_MS = 30000;
  const HISTORY_MESSAGES = 12;

  const $ = (selector) => document.querySelector(selector);
  const els = {
    root: document.documentElement,
    app: $("#app"),
    uploadView: $("#upload-view"),
    chatView: $("#chat-view"),
    dropzone: $("#dropzone"),
    fileInput: $("#file-input"),
    dzIdle: $("#dz-idle"),
    dzBusy: $("#dz-busy"),
    fileList: $("#file-list"),
    bar: $("#progress-bar"),
    uploadStatus: $("#upload-status"),
    uploadNotice: $("#upload-notice"),
    limits: $("#limits"),
    docs: $("#topbar-docs"),
    newChat: $("#new-chat"),
    themeToggle: $("#theme-toggle"),
    banner: $("#banner"),
    bannerText: $("#banner-text"),
    bannerAction: $("#banner-action"),
    thread: $("#thread"),
    composer: $("#composer"),
    question: $("#question"),
    send: $("#send"),
  };

  const svg = (body, extra = "") =>
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ${extra}>${body}</svg>`;
  const ICON = {
    doc: svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>'),
    send: svg('<path d="M12 19V5M5 12l7-7 7 7"/>', 'stroke-width="2.4"'),
    stop: svg('<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none"/>'),
    copy: svg('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/>'),
    check: svg('<path d="M20 6 9 17l-5-5"/>', 'stroke-width="2.5"'),
    sparkle: svg('<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9zM19 16l.7 1.8 1.8.7-1.8.7L19 21l-.7-1.8-1.8-.7 1.8-.7z"/>'),
    arrow: svg('<path d="M5 12h14M13 6l6 6-6 6"/>'),
    retry: svg('<path d="M3 12a9 9 0 1 0 2.6-6.4L3 8"/><path d="M3 3v5h5"/>'),
  };

  const state = {
    sessionId: null,
    docs: [],
    messages: [],
    suggestions: null,
    config: { max_files: 3, max_total_mb: 10, max_pages: 100, idle_minutes: 15 },
    controller: null,
    heartbeat: null,
    pollToken: 0,
    expired: false,
  };

  // ---------- helpers ----------

  const escapeHtml = (text) =>
    text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  const formatSize = (bytes) =>
    bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

  const errorDetail = (body) => {
    if (!body) return "";
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail[0]) return body.detail[0].msg || "";
    return "";
  };

  const isTouch = () => window.matchMedia("(hover: none)").matches;

  function save() {
    try {
      if (!state.sessionId) return sessionStorage.removeItem(STORE_KEY);
      const messages = state.messages.map(({ role, content, sources, status, error }) => ({ role, content, sources, status, error }));
      sessionStorage.setItem(STORE_KEY, JSON.stringify({ sessionId: state.sessionId, docs: state.docs, messages }));
    } catch (e) { /* storage unavailable: chat just won't survive a reload */ }
  }

  // ---------- markdown (escaped first, so model output can't inject HTML) ----------

  function renderInline(text, sourceIds) {
    return text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/\[(\d{1,2})\]/g, (match, n) =>
        sourceIds.has(Number(n))
          ? `<button type="button" class="cite" data-id="${n}" aria-label="Show source ${n}">${n}</button>`
          : match);
  }

  function renderMarkdown(source, sources) {
    const ids = new Set((sources || []).map((s) => s.id));
    const lines = escapeHtml(source).split("\n");
    let html = "";
    let paragraph = [];
    let list = null;
    let code = null;

    const flushParagraph = () => {
      if (paragraph.length) html += `<p>${renderInline(paragraph.join("<br>"), ids)}</p>`;
      paragraph = [];
    };
    const closeList = () => {
      if (list) html += `</${list}>`;
      list = null;
    };
    const openList = (tag) => {
      if (list !== tag) {
        closeList();
        html += `<${tag}>`;
        list = tag;
      }
    };

    for (const line of lines) {
      if (/^\s*```/.test(line)) {
        if (code) {
          html += `<pre><code>${code.join("\n")}</code></pre>`;
          code = null;
        } else {
          flushParagraph();
          closeList();
          code = [];
        }
        continue;
      }
      if (code) { code.push(line); continue; }

      let m;
      if (!line.trim()) {
        flushParagraph();
        closeList();
      } else if ((m = line.match(/^\s*(#{1,6})\s+(.*)$/))) {
        flushParagraph();
        closeList();
        const level = Math.min(m[1].length + 2, 5);
        html += `<h${level}>${renderInline(m[2], ids)}</h${level}>`;
      } else if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
        flushParagraph();
        openList("ul");
        html += `<li>${renderInline(m[1], ids)}</li>`;
      } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
        flushParagraph();
        openList("ol");
        html += `<li>${renderInline(m[1], ids)}</li>`;
      } else {
        closeList();
        paragraph.push(line.trim());
      }
    }
    if (code) html += `<pre><code>${code.join("\n")}</code></pre>`;
    flushParagraph();
    closeList();
    return html;
  }

  // Sources the answer actually cites, in order of first mention
  function citedSources(message) {
    const byId = new Map((message.sources || []).map((s) => [s.id, s]));
    const seen = [];
    for (const match of message.content.matchAll(/\[(\d{1,2})\]/g)) {
      const source = byId.get(Number(match[1]));
      if (source && !seen.includes(source)) seen.push(source);
    }
    return seen;
  }

  // ---------- theme ----------

  function currentTheme() {
    return els.root.dataset.theme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }

  els.themeToggle.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    els.root.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* not persisted */ }
  });

  // ---------- upload ----------

  function showNotice(text, kind = "error") {
    els.uploadNotice.textContent = text;
    els.uploadNotice.className = `notice ${kind === "error" ? "error" : ""}`;
    els.uploadNotice.hidden = !text;
  }

  function setBusy(files) {
    const busy = Boolean(files);
    els.dropzone.classList.toggle("busy", busy);
    els.dropzone.setAttribute("aria-disabled", String(busy));
    els.dzIdle.hidden = busy;
    els.dzBusy.hidden = !busy;
    els.fileList.replaceChildren();
    setProgress(0);
    if (!busy) return;
    for (const file of files) {
      const item = document.createElement("li");
      item.className = "file-item";
      item.innerHTML = '<span class="file-badge">PDF</span><div class="file-meta"><div class="file-name"></div><div class="file-size"></div></div>';
      item.querySelector(".file-name").textContent = file.name;
      item.querySelector(".file-size").textContent = formatSize(file.size);
      els.fileList.append(item);
    }
  }

  let creepTimer = null;
  function setProgress(fraction) {
    els.bar.style.width = `${Math.round(fraction * 100)}%`;
  }
  // Indexing has no real progress signal, so ease towards 92% while waiting
  function creepProgress(from) {
    clearInterval(creepTimer);
    let value = from;
    creepTimer = setInterval(() => {
      value += (0.92 - value) * 0.035;
      setProgress(value);
    }, 200);
  }

  function pickFiles(fileList) {
    if (els.dropzone.classList.contains("busy")) return;
    const all = [...fileList];
    const files = all.filter((f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
    const { max_files: maxFiles, max_total_mb: maxMb } = state.config;
    if (!files.length) return showNotice(all.length ? "Only PDF files are supported." : "");
    if (files.length > maxFiles) return showNotice(`You can upload up to ${maxFiles} PDFs at a time.`);
    const total = files.reduce((sum, f) => sum + f.size, 0);
    if (total > maxMb * 1048576) return showNotice(`These files add up to ${formatSize(total)}; the limit is ${maxMb} MB.`);
    upload(files);
  }

  function upload(files) {
    showNotice("");
    setBusy(files);
    els.uploadStatus.textContent = files.length > 1 ? `Uploading ${files.length} files…` : "Uploading…";

    const form = new FormData();
    files.forEach((file) => form.append("files", file, file.name));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/sessions");
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) setProgress((e.loaded / e.total) * 0.3); };
    xhr.upload.onload = () => {
      els.uploadStatus.textContent = "Reading pages and building your index…";
      creepProgress(0.3);
    };
    xhr.onload = () => {
      clearInterval(creepTimer);
      if (xhr.status === 201 && xhr.response) {
        setProgress(1);
        setTimeout(() => startChat(xhr.response), 250);
      } else {
        setBusy(null);
        showNotice(errorDetail(xhr.response) || "Upload failed. Please try again.");
      }
    };
    xhr.onerror = () => {
      clearInterval(creepTimer);
      setBusy(null);
      showNotice("Couldn't reach ChatPDF. Check your connection and try again.");
    };
    xhr.send(form);
  }

  els.dropzone.addEventListener("click", () => {
    if (!els.dropzone.classList.contains("busy")) els.fileInput.click();
  });
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.dropzone.click();
    }
  });
  els.fileInput.addEventListener("change", () => {
    pickFiles(els.fileInput.files);
    els.fileInput.value = "";
  });

  // Drops anywhere on the upload screen count; elsewhere they are ignored instead of opening the PDF
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => {
    if (els.uploadView.hidden) return;
    e.preventDefault();
    dragDepth += 1;
    els.dropzone.classList.add("dragging");
  });
  window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) els.dropzone.classList.remove("dragging");
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    els.dropzone.classList.remove("dragging");
    if (!els.uploadView.hidden && e.dataTransfer?.files?.length) pickFiles(e.dataTransfer.files);
  });

  // ---------- views ----------

  function showUpload(notice) {
    els.root.classList.remove("has-session");
    els.app.dataset.view = "upload";
    els.chatView.hidden = true;
    els.uploadView.hidden = false;
    els.docs.hidden = true;
    els.newChat.hidden = true;
    setBusy(null);
    showNotice(notice || "", "info");
    window.scrollTo(0, 0);
  }

  function showChat() {
    els.app.dataset.view = "chat";
    els.uploadView.hidden = true;
    els.chatView.hidden = false;
    els.newChat.hidden = false;
    els.banner.hidden = true;
    renderDocs();
    renderThread();
    setComposerEnabled(true);
    startHeartbeat();
    if (!isTouch()) els.question.focus();
  }

  function renderDocs() {
    els.docs.replaceChildren();
    for (const doc of state.docs) {
      const chip = document.createElement("span");
      chip.className = "doc-chip";
      chip.title = `${doc.name} · ${doc.pages} page${doc.pages === 1 ? "" : "s"}`;
      chip.innerHTML = `${ICON.doc}<span class="doc-name"></span><span class="doc-pages">${doc.pages}p</span>`;
      chip.querySelector(".doc-name").textContent = doc.name;
      els.docs.append(chip);
    }
    els.docs.hidden = !state.docs.length;
  }

  function startChat({ session_id: sessionId, docs }) {
    Object.assign(state, { sessionId, docs, messages: [], suggestions: null, expired: false });
    save();
    showChat();
    pollSuggestions();
  }

  async function resetChat(notice) {
    const oldId = state.sessionId;
    state.controller?.abort();
    stopHeartbeat();
    state.pollToken += 1;
    Object.assign(state, { sessionId: null, docs: [], messages: [], suggestions: null, expired: false });
    save();
    showUpload(notice);
    if (oldId) fetch(`/api/sessions/${oldId}`, { method: "DELETE", keepalive: true }).catch(() => {});
  }

  function markExpired() {
    if (state.expired) return;
    state.expired = true;
    stopHeartbeat();
    state.pollToken += 1;
    try { sessionStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ }
    els.bannerText.textContent = "This chat expired, so its documents were deleted. Upload them again to keep asking.";
    els.banner.hidden = false;
    setComposerEnabled(false);
    renderWelcomeIfEmpty();
  }

  els.newChat.addEventListener("click", () => resetChat());
  els.bannerAction.addEventListener("click", () => resetChat());

  // ---------- heartbeat & suggestions ----------

  function startHeartbeat() {
    stopHeartbeat();
    state.heartbeat = setInterval(checkSession, HEARTBEAT_MS);
  }
  function stopHeartbeat() {
    clearInterval(state.heartbeat);
    state.heartbeat = null;
  }

  async function checkSession() {
    if (!state.sessionId || state.expired) return null;
    try {
      const res = await fetch(`/api/sessions/${state.sessionId}`, { cache: "no-store" });
      if (res.status === 404) {
        markExpired();
        return null;
      }
      return res.ok ? await res.json() : null;
    } catch (e) {
      return null;
    }
  }

  async function pollSuggestions() {
    const token = ++state.pollToken;
    for (let attempt = 0; attempt < 60 && token === state.pollToken; attempt++) {
      const body = await checkSession();
      if (token !== state.pollToken) return;
      if (body?.suggestions) {
        state.suggestions = body.suggestions;
        renderWelcomeIfEmpty();
        return;
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
    if (token === state.pollToken && !state.suggestions) {
      state.suggestions = [];
      renderWelcomeIfEmpty();
    }
  }

  // ---------- thread ----------

  function renderThread() {
    els.thread.replaceChildren();
    if (!state.messages.length) return renderWelcome();
    for (const message of state.messages) {
      const view = appendMessage(message);
      if (message.role === "assistant") updateAssistant(view, message, true);
    }
    window.scrollTo(0, document.body.scrollHeight);
  }

  function renderWelcomeIfEmpty() {
    if (!state.messages.length && !els.chatView.hidden) renderWelcome();
  }

  function renderWelcome() {
    const pages = state.docs.reduce((sum, d) => sum + d.pages, 0);
    const count = state.docs.length;
    const section = document.createElement("section");
    section.className = "welcome";
    section.innerHTML = `
      <span class="welcome-badge">${ICON.check}Ready</span>
      <h2>What would you like to know?</h2>
      <p>${count} document${count === 1 ? "" : "s"} · ${pages} page${pages === 1 ? "" : "s"} indexed. Ask anything, or start with a suggestion.</p>
      <div class="suggest-area"></div>`;
    const area = section.querySelector(".suggest-area");

    if (state.expired) {
      section.querySelector("p").textContent = "Upload your PDFs again to start a new chat.";
      section.querySelector(".welcome-badge").hidden = true;
    } else if (state.suggestions === null) {
      area.innerHTML = `<p class="suggest-status">${ICON.sparkle}<span>Reading your documents to suggest questions…</span></p>
        <div class="suggestion-list" aria-hidden="true"><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div>`;
    } else if (state.suggestions.length) {
      const list = document.createElement("div");
      list.className = "suggestion-list";
      for (const text of state.suggestions) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "suggestion";
        button.dataset.question = text;
        button.innerHTML = `<span class="s-icon">${ICON.sparkle}</span><span class="s-text"></span><span class="s-arrow">${ICON.arrow}</span>`;
        button.querySelector(".s-text").textContent = text;
        list.append(button);
      }
      area.append(list);
    }
    els.thread.replaceChildren(section);
  }

  function appendMessage(message) {
    const article = document.createElement("article");
    if (message.role === "user") {
      article.className = "msg user";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = message.content;
      article.append(bubble);
    } else {
      article.className = "msg bot";
      article.innerHTML = `<div class="avatar">${ICON.doc}</div><div class="msg-body"><div class="md"></div><div class="msg-extra"></div></div>`;
    }
    els.thread.append(article);
    return article;
  }

  function updateAssistant(view, message, final) {
    const md = view.querySelector(".md");
    const extra = view.querySelector(".msg-extra");
    const streaming = message.status === "streaming";
    view.classList.toggle("streaming", streaming && Boolean(message.content));

    if (message.content) {
      md.innerHTML = renderMarkdown(message.content, message.sources);
    } else if (streaming) {
      md.innerHTML = '<span class="typing"><span class="dots"><i></i><i></i><i></i></span>Searching your documents…</span>';
    } else {
      md.innerHTML = "";
    }
    if (!final) return;

    extra.replaceChildren();
    const cited = citedSources(message);
    if (cited.length) {
      const row = document.createElement("div");
      row.className = "sources";
      for (const source of cited) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "source";
        chip.dataset.id = source.id;
        chip.innerHTML = `<span class="src-num">${source.id}</span><span class="src-name"></span><span class="src-page">p. ${source.page}</span>`;
        chip.querySelector(".src-name").textContent = source.doc;
        chip.title = `${source.doc}, page ${source.page}`;
        row.append(chip);
      }
      extra.append(row);
    }

    if (message.status === "error") {
      const box = document.createElement("div");
      box.className = "msg-error";
      box.innerHTML = `<span></span><button type="button" class="ghost retry">${ICON.retry}Try again</button>`;
      box.firstElementChild.textContent = message.error || "Something went wrong.";
      extra.append(box);
    } else if (message.content) {
      const actions = document.createElement("div");
      actions.className = "msg-actions";
      actions.innerHTML = `<button type="button" class="ghost copy">${ICON.copy}<span>Copy</span></button>`;
      if (message.status === "stopped") actions.insertAdjacentHTML("beforeend", '<span class="msg-note">Stopped</span>');
      extra.append(actions);
    }
  }

  function toggleSource(view, message, id) {
    const extra = view.querySelector(".msg-extra");
    const open = extra.querySelector(".source-detail");
    const wasOpen = open && open.dataset.id === String(id);
    open?.remove();
    view.querySelectorAll(".cite.active, .source.active").forEach((el) => el.classList.remove("active"));
    if (wasOpen) return;

    const source = message.sources.find((s) => s.id === id);
    if (!source) return;
    view.querySelectorAll(`[data-id="${id}"]`).forEach((el) => el.classList.add("active"));
    const detail = document.createElement("div");
    detail.className = "source-detail";
    detail.dataset.id = id;
    detail.innerHTML = `<div class="sd-head">${ICON.doc}<span></span></div><p class="sd-quote"></p>`;
    detail.querySelector(".sd-head span").textContent = `${source.doc} · Page ${source.page}`;
    detail.querySelector(".sd-quote").textContent = `“${source.snippet}”`;
    const row = extra.querySelector(".sources");
    row ? row.after(detail) : extra.prepend(detail);
  }

  els.thread.addEventListener("click", async (e) => {
    const suggestion = e.target.closest(".suggestion");
    if (suggestion) return ask(suggestion.dataset.question);

    const view = e.target.closest(".msg.bot");
    if (!view) return;
    const index = [...els.thread.querySelectorAll(".msg")].indexOf(view);
    const message = state.messages[index];
    if (!message) return;

    const sourceButton = e.target.closest(".cite, .source");
    if (sourceButton) return toggleSource(view, message, Number(sourceButton.dataset.id));

    const copy = e.target.closest(".copy");
    if (copy) {
      const text = message.content.replace(/\s?\[\d{1,2}\]/g, "");
      try {
        await navigator.clipboard.writeText(text);
        copy.innerHTML = `${ICON.check}<span>Copied</span>`;
        setTimeout(() => { copy.innerHTML = `${ICON.copy}<span>Copy</span>`; }, 1600);
      } catch (err) { /* clipboard blocked */ }
      return;
    }

    if (e.target.closest(".retry") && index === state.messages.length - 1 && !state.controller) {
      const question = state.messages[index - 1]?.content;
      state.messages.splice(index - 1, 2);
      view.previousElementSibling?.remove();
      view.remove();
      if (question) ask(question);
    }
  });

  // ---------- asking ----------

  const nearBottom = () => window.innerHeight + window.scrollY >= document.body.scrollHeight - 160;

  async function ask(rawQuestion) {
    const question = rawQuestion.trim();
    if (!question || state.controller || !state.sessionId || state.expired) return;

    const history = state.messages
      .filter((m) => m.content && m.status !== "error")
      .slice(-HISTORY_MESSAGES)
      .map(({ role, content }) => ({ role, content: content.slice(0, 8000) }));

    if (!state.messages.length) els.thread.replaceChildren();
    const userMessage = { role: "user", content: question };
    const bot = { role: "assistant", content: "", sources: [], status: "streaming" };
    state.messages.push(userMessage, bot);
    appendMessage(userMessage);
    const view = appendMessage(bot);
    updateAssistant(view, bot, false);
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });

    els.question.value = "";
    autosize();
    const controller = new AbortController();
    state.controller = controller;
    setStreaming(true);

    let frame = 0;
    const scheduleRender = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        const stick = nearBottom();
        updateAssistant(view, bot, false);
        if (stick) window.scrollTo(0, document.body.scrollHeight);
      });
    };
    const handle = (event) => {
      if (event.type === "sources") bot.sources = event.sources;
      else if (event.type === "delta") { bot.content += event.text; scheduleRender(); }
      else if (event.type === "error") { bot.status = "error"; bot.error = event.message; }
      else if (event.type === "done") bot.status = "done";
    };

    try {
      const res = await fetch(`/api/sessions/${state.sessionId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
        signal: controller.signal,
      });
      if (res.status === 404) {
        bot.status = "error";
        bot.error = "This chat expired. Upload your PDFs again to continue.";
        markExpired();
      } else if (!res.ok) {
        const body = await res.json().catch(() => null);
        bot.status = "error";
        bot.error = errorDetail(body) || "Something went wrong. Please try again.";
      } else {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let newline;
          while ((newline = buffer.indexOf("\n")) >= 0) {
            const line = buffer.slice(0, newline).trim();
            buffer = buffer.slice(newline + 1);
            if (line) handle(JSON.parse(line));
          }
        }
        if (bot.status === "streaming") {
          bot.status = "error";
          bot.error = "The answer was cut off. Please try again.";
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        bot.status = "stopped";
      } else {
        bot.status = "error";
        bot.error = "Couldn't reach ChatPDF. Check your connection and try again.";
      }
    } finally {
      cancelAnimationFrame(frame);
      state.controller = null;
      setStreaming(false);
      if (bot.status === "stopped" && !bot.content) {
        state.messages.splice(state.messages.indexOf(bot) - 1, 2);
        view.previousElementSibling?.remove();
        view.remove();
        if (!state.messages.length) renderWelcome();
        els.question.value = question;
        autosize();
      } else {
        const stick = nearBottom();
        updateAssistant(view, bot, true);
        if (stick) window.scrollTo(0, document.body.scrollHeight);
      }
      save();
    }
  }

  // ---------- composer ----------

  function autosize() {
    els.question.style.height = "auto";
    els.question.style.height = `${Math.min(els.question.scrollHeight, 200)}px`;
    updateSend();
  }

  function updateSend() {
    if (state.controller) {
      els.send.disabled = false;
      return;
    }
    els.send.disabled = !els.question.value.trim() || els.question.disabled;
  }

  function setStreaming(on) {
    els.send.classList.toggle("stop", on);
    els.send.innerHTML = on ? ICON.stop : ICON.send;
    els.send.setAttribute("aria-label", on ? "Stop generating" : "Send");
    updateSend();
  }

  function setComposerEnabled(enabled) {
    els.question.disabled = !enabled;
    els.question.placeholder = enabled ? "Ask about your documents…" : "Upload PDFs to start a new chat";
    updateSend();
  }

  els.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.controller) state.controller.abort();
    else ask(els.question.value);
  });
  els.question.addEventListener("input", autosize);
  els.question.addEventListener("keydown", (e) => {
    // On phones Enter adds a new line; the send button submits
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !isTouch()) {
      e.preventDefault();
      els.composer.requestSubmit();
    }
  });

  // ---------- start ----------

  async function loadConfig() {
    try {
      const res = await fetch("/api/config");
      if (!res.ok) return;
      state.config = await res.json();
      const { max_files: files, max_total_mb: mb, max_pages: pages } = state.config;
      els.limits.textContent = `Up to ${files} files · ${mb} MB total · ${pages} pages`;
    } catch (e) { /* keep defaults */ }
  }

  async function restore() {
    let saved = null;
    try { saved = JSON.parse(sessionStorage.getItem(STORE_KEY) || "null"); } catch (e) { /* ignore */ }
    if (!saved?.sessionId) return showUpload();

    Object.assign(state, {
      sessionId: saved.sessionId,
      docs: saved.docs || [],
      messages: (saved.messages || []).filter((m) => m.status !== "streaming"),
    });
    const body = await checkSession();
    if (!body) {
      Object.assign(state, { sessionId: null, docs: [], messages: [], expired: false });
      save();
      return showUpload(state.expired === false ? "Your previous chat expired, so its documents were deleted. Upload them again to continue." : "");
    }
    state.suggestions = body.suggestions;
    showChat();
    if (!body.suggestions) pollSuggestions();
  }

  setStreaming(false);
  loadConfig();
  restore();
})();
