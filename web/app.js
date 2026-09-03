/* ==========================================================================
   Sentinel RAG — client

   Talks to exactly four endpoints and invents nothing:
     POST /session  -> { session_id }
     POST /query    -> { answer, abstained, blocked, sources[], top_score,
                         answered_by, thought_process[], conversational, ... }
     GET  /scope    -> { subject, points, topics[] }
     GET  /health   -> { status, guardrails, corpus{}, models{} }

   Conversation history is kept in localStorage and labelled "local" in the UI,
   because the server deliberately has no chat persistence - sessions are
   in-process and expire. Showing a server-backed history would be a lie.
   ========================================================================== */
(() => {
  'use strict';

  const REQUEST_TIMEOUT_MS = 120_000;
  const HEALTH_INTERVAL_MS = 30_000;
  const HISTORY_KEY = 'sentinel.history.v1';
  const THEME_KEY = 'sentinel.theme';
  const HISTORY_LIMIT = 40;

  const $ = (id) => document.getElementById(id);
  const el = {
    app: $('app'), rail: $('rail'), scrim: $('scrim'),
    railOpen: $('railOpen'), railClose: $('railClose'),
    newChat: $('newChat'), historySearch: $('historySearch'),
    historyList: $('historyList'), historyEmpty: $('historyEmpty'),
    topics: $('topics'), scopeSummary: $('scopeSummary'),
    statusBar: $('statusBar'), statusDot: $('statusDot'), statusText: $('statusText'),
    thread: $('thread'), composer: $('composer'), input: $('input'),
    send: $('send'), charCount: $('charCount'), suggestions: $('suggestions'),
    turnPill: $('turnPill'), topbarHint: $('topbarHint'),
    inspector: $('inspector'), inspectorToggle: $('inspectorToggle'),
    sourcesList: $('sourcesList'), sourcesEmpty: $('sourcesEmpty'),
    contextList: $('contextList'), contextEmpty: $('contextEmpty'),
    traceList: $('traceList'), traceEmpty: $('traceEmpty'),
    tabCountSources: $('tabCountSources'),
    queryInfo: $('queryInfo'),
    qiOutcome: $('qiOutcome'), qiRelevance: $('qiRelevance'),
    qiPassages: $('qiPassages'), qiModel: $('qiModel'), qiTime: $('qiTime'),
    welcomeTemplate: $('welcomeTemplate'),
  };

  // Conversation ids are issued by the server and are unguessable. The client
  // never chooses one: a client-chosen id let any caller read another user's
  // conversation by guessing it.
  let sessionId = null;
  let listEl = null;
  let inFlight = null;
  let turns = 0;
  let activeChatId = null;
  let history = [];

  /* ------------------------------------------------------------ utilities */

  function escapeHtml(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // Only http(s) and mailto survive. Anything else - javascript:, data: - is
  // dropped rather than rendered as a link.
  function safeHref(url) {
    const trimmed = String(url ?? '').trim();
    return /^(https?:\/\/|mailto:)/i.test(trimmed) ? escapeHtml(trimmed) : null;
  }

  function inline(text) {
    let out = escapeHtml(text);
    out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
    out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, '$1<em>$2</em>');
    out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, href) => {
      const safe = safeHref(href);
      return safe ? `<a href="${safe}" target="_blank" rel="noopener noreferrer">${label}</a>` : label;
    });
    // The generator is instructed to cite passages as [2]; render those as chips
    // so a claim visibly points at a source.
    out = out.replace(/\[(\d{1,2})\](?!\()/g, '<span class="cite">$1</span>');
    return out;
  }

  function renderBlocks(src) {
    const lines = String(src ?? '').split('\n');
    const html = [];
    let list = null;

    const closeList = () => { if (list) { html.push(`</${list}>`); list = null; } };

    for (const raw of lines) {
      const line = raw.trimEnd();
      if (!line.trim()) { closeList(); continue; }

      const h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) { closeList(); html.push(`<h3>${inline(h[2])}</h3>`); continue; }

      const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ol) {
        if (list !== 'ol') { closeList(); html.push('<ol>'); list = 'ol'; }
        html.push(`<li>${inline(ol[1])}</li>`); continue;
      }

      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      if (ul) {
        if (list !== 'ul') { closeList(); html.push('<ul>'); list = 'ul'; }
        html.push(`<li>${inline(ul[1])}</li>`); continue;
      }

      closeList();
      html.push(`<p>${inline(line)}</p>`);
    }
    closeList();
    return html.join('');
  }

  function renderMarkdown(src) {
    const parts = String(src ?? '').split(/```/);
    let out = '';
    parts.forEach((part, i) => {
      if (i % 2 === 0) { out += renderBlocks(part); return; }
      const nl = part.indexOf('\n');
      const lang = nl === -1 ? '' : part.slice(0, nl).trim();
      const code = nl === -1 ? part : part.slice(nl + 1);
      out +=
        '<div class="codeblock"><div class="codeblock__bar">' +
        `<span class="codeblock__lang">${escapeHtml(lang || 'text')}</span>` +
        '<button class="codeblock__copy" type="button">Copy</button></div>' +
        `<pre><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre></div>`;
    });
    return out;
  }

  const ICON = {
    shield: '<svg viewBox="0 0 32 32" fill="none"><path d="M16 3l11 5.5v9C27 24 22 28.5 16 30 10 28.5 5 24 5 17.5v-9z" stroke="currentColor" stroke-width="1.6"/><path d="M11 16.4l3.4 3.4L21 13" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    warn: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M8 2.6 14.4 13H1.6z" stroke-linejoin="round"/><path d="M8 6.6v3M8 11.4h.01" stroke-linecap="round"/></svg>',
    err: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="8" cy="8" r="6.2"/><path d="M8 4.8v3.6M8 10.9h.01" stroke-linecap="round"/></svg>',
    copy: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="5.4" y="5.4" width="8" height="8" rx="1.6"/><path d="M10.6 5.4V4a1.6 1.6 0 0 0-1.6-1.6H4A1.6 1.6 0 0 0 2.4 4v5a1.6 1.6 0 0 0 1.6 1.6h1.4"/></svg>',
    trash: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 4.4h10M6.4 4.4V3.2A1 1 0 0 1 7.4 2.2h1.2a1 1 0 0 1 1 1v1.2M4.4 4.4l.5 8a1 1 0 0 0 1 .95h4.2a1 1 0 0 0 1-.95l.5-8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  };

  /* -------------------------------------------------------------- theme */

  function applyTheme(mode) {
    document.documentElement.setAttribute('data-theme', mode);
    document.querySelectorAll('[data-theme-set]').forEach((b) =>
      b.classList.toggle('is-active', b.dataset.themeSet === mode));
    try { localStorage.setItem(THEME_KEY, mode); } catch { /* storage unavailable */ }
  }

  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch { /* ignore */ }
    if (saved !== 'light' && saved !== 'dark') {
      saved = window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }
    applyTheme(saved);
    document.querySelectorAll('[data-theme-set]').forEach((b) =>
      b.addEventListener('click', () => applyTheme(b.dataset.themeSet)));
  }

  /* ------------------------------------------------------ local history */

  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      history = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(history)) history = [];
    } catch { history = []; }
  }

  function saveHistory() {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, HISTORY_LIMIT)));
    } catch { /* quota or private mode - history is a convenience, not state */ }
  }

  function relativeTime(ts) {
    const mins = Math.floor((Date.now() - ts) / 60000);
    if (mins < 1) return 'now';
    if (mins < 60) return `${mins}m`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h`;
    const days = Math.floor(hrs / 24);
    return days < 7 ? `${days}d` : new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }

  function renderHistory() {
    const filter = (el.historySearch.value || '').toLowerCase().trim();
    const shown = history.filter((c) => !filter || c.title.toLowerCase().includes(filter));

    el.historyList.innerHTML = '';
    el.historyEmpty.hidden = shown.length > 0;
    el.historyEmpty.textContent = history.length
      ? 'No conversations match that search.'
      : 'No conversations yet.';

    for (const chat of shown) {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'history__item' + (chat.id === activeChatId ? ' is-active' : '');
      btn.innerHTML =
        `<span class="history__label">${escapeHtml(chat.title)}</span>` +
        `<span class="history__time">${escapeHtml(relativeTime(chat.at))}</span>`;
      btn.addEventListener('click', () => openChat(chat.id));

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'history__del';
      del.title = 'Delete conversation';
      del.setAttribute('aria-label', `Delete "${chat.title}"`);
      del.innerHTML = ICON.trash;
      del.addEventListener('click', (e) => {
        e.stopPropagation();
        history = history.filter((c) => c.id !== chat.id);
        saveHistory();
        if (chat.id === activeChatId) newConversation();
        else renderHistory();
      });

      btn.appendChild(del);
      li.appendChild(btn);
      el.historyList.appendChild(li);
    }
  }

  function recordTurn(question, record) {
    let chat = history.find((c) => c.id === activeChatId);
    if (!chat) {
      chat = { id: activeChatId, title: question.slice(0, 68), at: Date.now(), turns: [] };
      history.unshift(chat);
    }
    chat.at = Date.now();
    chat.turns.push({ q: question, r: record });
    // Move to the top so the most recent conversation is first.
    history = [chat, ...history.filter((c) => c.id !== chat.id)].slice(0, HISTORY_LIMIT);
    saveHistory();
    renderHistory();
  }

  function openChat(id) {
    const chat = history.find((c) => c.id === id);
    if (!chat) return;
    activeChatId = id;
    if (listEl) { listEl.remove(); listEl = null; }
    el.thread.innerHTML = '';
    turns = 0;
    for (const t of chat.turns) {
      addUser(t.q);
      const node = addPending();
      fillAnswer(node, t.r, { replay: true });
      turns += 1;
    }
    setTurnPill();
    // A replayed conversation has no live server session behind it, so the next
    // question starts a fresh one rather than pretending the model remembers.
    sessionId = null;
    renderHistory();
    if (window.innerWidth <= 860) closeRail();
  }

  /* ------------------------------------------------------------- thread */

  function ensureList() {
    if (!listEl) {
      listEl = document.createElement('div');
      listEl.className = 'thread__inner';
      el.thread.appendChild(listEl);
    }
    return listEl;
  }

  function hideWelcome() {
    const w = $('welcome');
    if (w) w.remove();
  }

  function scrollToEnd() {
    requestAnimationFrame(() => { el.thread.scrollTop = el.thread.scrollHeight; });
  }

  function showWelcome() {
    const tpl = el.welcomeTemplate.content.cloneNode(true);
    el.thread.appendChild(tpl);
    applyScopeFacts();
  }

  function addUser(text) {
    hideWelcome();
    const node = document.createElement('div');
    node.className = 'msg msg--user';
    node.innerHTML = `<div class="msg__bubble">${escapeHtml(text)}</div>`;
    ensureList().appendChild(node);
    scrollToEnd();
  }

  function addPending() {
    const node = document.createElement('div');
    node.className = 'msg msg--bot';
    node.innerHTML =
      `<div class="msg__role"><span class="msg__avatar">${ICON.shield}</span>Sentinel</div>` +
      '<div class="msg__body"><div class="typing"><span></span><span></span><span></span></div></div>';
    ensureList().appendChild(node);
    scrollToEnd();
    return node;
  }

  function outcomeOf(data) {
    if (data.blocked) return 'blocked';
    if (data.abstained) return 'abstained';
    if (data.conversational) return 'conversational';
    return 'answered';
  }

  function fillAnswer(node, data, opts = {}) {
    const answer = data.answer ?? '';
    const outcome = outcomeOf(data);
    const head = `<div class="msg__role"><span class="msg__avatar">${ICON.shield}</span>Sentinel</div>`;

    if (outcome === 'blocked') {
      node.innerHTML = head +
        `<div class="notice notice--blocked">${ICON.warn}<div>` +
        '<p class="notice__title">Outside scope</p>' +
        `<div class="notice__text">${renderMarkdown(answer)}</div></div></div>`;
    } else if (outcome === 'abstained') {
      // Abstention is the product working, not an error. It reads as a
      // deliberate state, distinct from both an answer and a failure.
      node.innerHTML = head +
        `<div class="notice notice--abstain">${ICON.warn}<div>` +
        '<p class="notice__title">Not covered by the documentation</p>' +
        `<div class="notice__text">${renderMarkdown(answer)}</div></div></div>`;
    } else {
      node.innerHTML = head +
        `<div class="msg__body">${renderMarkdown(answer)}</div>` +
        '<div class="msg__foot"></div>';
      const foot = node.querySelector('.msg__foot');
      const copy = document.createElement('button');
      copy.type = 'button';
      copy.className = 'icon-btn';
      copy.title = 'Copy answer';
      copy.setAttribute('aria-label', 'Copy answer');
      copy.innerHTML = ICON.copy;
      copy.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(answer);
          copy.title = 'Copied';
          setTimeout(() => { copy.title = 'Copy answer'; }, 1500);
        } catch { copy.title = 'Press Ctrl+C'; }
      });
      foot.appendChild(copy);
    }

    if (!opts.replay) updateInspector(data);
    scrollToEnd();
  }

  function fillError(node, title, detail) {
    node.innerHTML =
      `<div class="msg__role"><span class="msg__avatar">${ICON.shield}</span>Sentinel</div>` +
      `<div class="notice notice--error">${ICON.err}<div>` +
      `<p class="notice__title">${escapeHtml(title)}</p>` +
      `<p class="notice__text">${escapeHtml(detail)}</p></div></div>`;
    scrollToEnd();
  }

  /* ---------------------------------------------------------- inspector */

  function scoreClass(score) {
    if (score >= 0.7) return '';
    if (score >= 0.35) return ' is-mid';
    return ' is-low';
  }

  function updateInspector(data) {
    const sources = Array.isArray(data.sources) ? data.sources : [];
    const steps = Array.isArray(data.thought_process) ? data.thought_process : [];

    el.tabCountSources.textContent = String(sources.length);

    // Sources
    el.sourcesList.innerHTML = '';
    el.sourcesEmpty.hidden = sources.length > 0;
    if (!sources.length) {
      el.sourcesEmpty.textContent = data.abstained
        ? 'No passage cleared the relevance threshold, so nothing was cited.'
        : data.blocked
          ? 'The request was refused at the gate, before retrieval.'
          : 'This turn was answered from the conversation, without retrieval.';
    }
    sources.forEach((src, i) => {
      const score = typeof src.score === 'number' ? src.score : 0;
      const href = safeHref(src.url);
      const d = document.createElement('details');
      d.className = 'source';
      d.innerHTML =
        '<summary class="source__head">' +
        `<span class="source__idx">${i + 1}</span>` +
        '<span class="source__main">' +
        `<span class="source__title">${escapeHtml(src.title || src.source || 'Passage')}</span>` +
        `<span class="source__section">${escapeHtml(src.section || '')}</span>` +
        '</span>' +
        `<span class="source__score${scoreClass(score)}">${score.toFixed(2)}</span>` +
        '</summary>' +
        '<div class="source__body">' +
        `<p class="source__excerpt">${escapeHtml(src.excerpt || '')}</p>` +
        '<div class="source__meta">' +
        `<span>${escapeHtml(src.topic || '')}</span>` +
        (href ? `<a class="source__link" href="${href}" target="_blank" rel="noopener noreferrer">Open source ↗</a>` : '') +
        '</div></div>';
      el.sourcesList.appendChild(d);
    });

    // Context — the exact passages handed to the model
    el.contextList.innerHTML = '';
    el.contextEmpty.hidden = sources.length > 0;
    if (sources.length) {
      sources.forEach((src, i) => {
        const wrap = document.createElement('div');
        wrap.className = 'source';
        wrap.innerHTML =
          '<div class="source__head">' +
          `<span class="source__idx">${i + 1}</span>` +
          `<span class="source__main"><span class="source__title">${escapeHtml(src.section || src.title || '')}</span></span>` +
          '</div>' +
          `<div class="source__body"><p class="source__excerpt">${escapeHtml(src.excerpt || '')}</p></div>`;
        el.contextList.appendChild(wrap);
      });
    }

    // Trace — the pipeline's own decisions
    el.traceList.innerHTML = '';
    el.traceEmpty.hidden = steps.length > 0;
    steps.forEach((step, i) => {
      const row = document.createElement('div');
      row.className = 'trace__step';
      row.innerHTML = `<span class="trace__idx">${i + 1}</span><span>${escapeHtml(step)}</span>`;
      el.traceList.appendChild(row);
    });

    // Query info
    const outcome = outcomeOf(data);
    const label = {
      answered: ['Grounded answer', 'is-ok'],
      abstained: ['Declined — not covered', 'is-warn'],
      blocked: ['Blocked at the gate', 'is-danger'],
      conversational: ['Conversational', ''],
    }[outcome];
    el.qiOutcome.textContent = label[0];
    el.qiOutcome.className = label[1];
    const top = typeof data.top_score === 'number' ? data.top_score : null;
    el.qiRelevance.textContent = top === null ? '—' : top.toFixed(3);
    el.qiPassages.textContent = String(sources.length);
    el.qiModel.textContent = data.answered_by || '—';
    el.qiTime.textContent = data.elapsed ? `${data.elapsed.toFixed(2)}s` : '—';
    el.queryInfo.hidden = false;
  }

  function switchTab(name) {
    document.querySelectorAll('.tab').forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', String(on));
    });
    document.querySelectorAll('.panel').forEach((p) =>
      p.classList.toggle('is-active', p.id === `panel-${name}`));
  }

  /* ---------------------------------------------------------------- net */

  async function startSession() {
    // The server owns conversation identity. If it declines we run without
    // memory rather than inventing an id the server would reject.
    try {
      const res = await fetch('/session', { method: 'POST' });
      sessionId = res.ok ? ((await res.json()).session_id ?? null) : null;
    } catch { sessionId = null; }
  }

  async function loadScope() {
    try {
      const res = await fetch('/scope', { cache: 'no-store' });
      if (!res.ok) return;
      const scope = await res.json();
      const topics = Array.isArray(scope.topics) ? scope.topics : [];
      window.__scope = scope;
      applyScopeFacts();
      if (!topics.length) return;

      el.scopeSummary.textContent =
        `${scope.points} passages · ${topics.length} areas`;
      el.topics.innerHTML = topics.map((t) =>
        '<li><span class="topics__dot"></span>' +
        `<span class="topics__label" title="${escapeHtml(t.label)}">${escapeHtml(t.label.split(' - ')[0])}</span>` +
        `<span class="topics__count">${t.passages}</span></li>`).join('');
    } catch { /* the static fallback copy stands */ }
  }

  function applyScopeFacts() {
    const scope = window.__scope;
    if (!scope) return;
    const p = $('factPassages'), t = $('factTopics');
    if (p) p.textContent = String(scope.points ?? '—');
    if (t) t.textContent = String((scope.topics || []).length || '—');
  }

  function setStatus(kind, text, title) {
    el.statusDot.className = `dot ${kind}`;
    el.statusText.textContent = text;
    el.statusBar.title = title || text;
  }

  async function checkHealth() {
    try {
      const res = await fetch('/health', { cache: 'no-store' });
      if (!res.ok) { setStatus('is-down', 'Backend error'); return; }
      const h = await res.json();
      const bits = [];
      if (!h.guardrails) bits.push('guardrails down');
      if (!h.corpus?.available) bits.push('corpus unreachable');
      if (h.status === 'ok') {
        setStatus('is-ok', 'Connected',
          `Corpus ${h.corpus?.points ?? '?'} passages · generation ${h.models?.generation ?? '?'}`);
      } else {
        setStatus('is-degraded', 'Degraded', bits.join(', ') || 'Some subsystems are unavailable');
      }
    } catch {
      setStatus('is-down', 'Offline', 'The backend is not reachable from this page');
    }
  }

  async function ask(text) {
    const node = addPending();
    const controller = new AbortController();
    inFlight = controller;
    const timer = setTimeout(() => controller.abort('timeout'), REQUEST_TIMEOUT_MS);
    const started = performance.now();

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: text, session_id: sessionId }),
        signal: controller.signal,
      });

      // Check status before parsing — a 500 with an HTML error page would
      // otherwise surface as a confusing JSON parse failure.
      if (!res.ok) {
        let detail = `The backend responded with HTTP ${res.status}.`;
        if (res.status === 429) detail = 'Rate limit reached. Wait a moment and try again.';
        if (res.status === 404) { sessionId = null; detail = 'That conversation expired. Your next message will start a new one.'; }
        try {
          const body = await res.json();
          if (body?.detail) detail = String(body.detail);
          if (body?.correlation_id) detail += ` (ref ${body.correlation_id})`;
        } catch { /* status alone is enough */ }
        fillError(node, 'Request failed', detail);
        return;
      }

      let data;
      try { data = await res.json(); }
      catch {
        fillError(node, 'Unreadable response', 'The backend returned a response that was not valid JSON.');
        return;
      }

      data.elapsed = (performance.now() - started) / 1000;
      fillAnswer(node, data);
      turns += 1;
      setTurnPill();
      recordTurn(text, data);
    } catch (err) {
      if (err?.name === 'AbortError') {
        fillError(node, 'Timed out', 'The request took too long. The pipeline runs several model calls per question.');
      } else {
        fillError(node, 'Network error', 'Could not reach the backend. Check that it is running.');
      }
    } finally {
      clearTimeout(timer);
      inFlight = null;
      setBusy(false);
    }
  }

  /* ------------------------------------------------------------- shell */

  function setTurnPill() {
    el.turnPill.textContent = `${turns} turn${turns === 1 ? '' : 's'}`;
  }

  function setBusy(busy) {
    el.send.disabled = busy || !el.input.value.trim();
    el.input.disabled = busy;
    el.topbarHint.textContent = busy ? 'Thinking…' : 'Grounded in the official documentation';
    if (!busy) el.input.focus();
  }

  function autoGrow() {
    el.input.style.height = 'auto';
    el.input.style.height = `${Math.min(el.input.scrollHeight, 200)}px`;
    el.charCount.textContent = `${el.input.value.length} / 4000`;
  }

  function submit() {
    const value = el.input.value.trim();
    if (!value || inFlight) return;
    if (!sessionId) startSession();
    addUser(value);
    el.input.value = '';
    autoGrow();
    setBusy(true);
    ask(value);
  }

  function newConversation() {
    if (inFlight) { inFlight.abort('cancelled'); inFlight = null; }
    activeChatId = `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
    turns = 0;
    setTurnPill();
    if (listEl) { listEl.remove(); listEl = null; }
    el.thread.innerHTML = '';
    showWelcome();
    el.queryInfo.hidden = true;
    el.sourcesList.innerHTML = '';
    el.contextList.innerHTML = '';
    el.traceList.innerHTML = '';
    el.sourcesEmpty.hidden = false;
    el.sourcesEmpty.textContent = 'Ask a question to see the passages the answer was built from.';
    el.contextEmpty.hidden = false;
    el.traceEmpty.hidden = false;
    el.tabCountSources.textContent = '0';
    startSession();
    renderHistory();
    setBusy(false);
  }

  // Panes auto-follow the breakpoint until the user expresses a preference;
  // after that their choice wins. Without this the visibility set at load never
  // updated, so widening the window left both side panes hidden.
  let railChoice = null;       // null = follow breakpoint, true = shown
  let inspectorChoice = null;

  function setRail(show) {
    el.app.classList.toggle('rail-hidden', !show);
    el.scrim.hidden = !(show && window.innerWidth <= 860);
  }

  function setInspector(show) {
    el.app.classList.toggle('inspector-hidden', !show);
    el.inspectorToggle.setAttribute('aria-pressed', String(show));
  }

  function openRail() { railChoice = true; setRail(true); }
  function closeRail() { railChoice = false; setRail(false); }

  function toggleInspector() {
    const show = el.app.classList.contains('inspector-hidden');
    inspectorChoice = show;
    setInspector(show);
  }

  function applyBreakpoint() {
    const w = window.innerWidth;
    setRail(railChoice === null ? w > 860 : railChoice);
    setInspector(inspectorChoice === null ? w > 1180 : inspectorChoice);
  }

  /* --------------------------------------------------------------- init */

  initTheme();
  loadHistory();
  loadScope();
  checkHealth();
  setInterval(checkHealth, HEALTH_INTERVAL_MS);
  newConversation();

  applyBreakpoint();

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(applyBreakpoint, 120);
  });

  el.composer.addEventListener('submit', (e) => { e.preventDefault(); submit(); });
  el.input.addEventListener('input', () => { autoGrow(); el.send.disabled = !el.input.value.trim() || !!inFlight; });
  el.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  });

  el.newChat.addEventListener('click', newConversation);
  el.historySearch.addEventListener('input', renderHistory);
  el.statusBar.addEventListener('click', checkHealth);
  el.inspectorToggle.addEventListener('click', toggleInspector);
  el.railOpen.addEventListener('click', openRail);
  el.railClose.addEventListener('click', closeRail);
  el.scrim.addEventListener('click', closeRail);

  el.suggestions.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip || inFlight) return;
    el.input.value = chip.dataset.q || chip.textContent.trim();
    autoGrow();
    submit();
  });

  document.querySelectorAll('.tab').forEach((t) =>
    t.addEventListener('click', () => switchTab(t.dataset.tab)));

  // Copy buttons are created inside rendered markdown, so delegate.
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('.codeblock__copy');
    if (!btn) return;
    const code = btn.closest('.codeblock')?.querySelector('code')?.textContent ?? '';
    try {
      await navigator.clipboard.writeText(code);
      btn.textContent = 'Copied';
    } catch { btn.textContent = 'Press Ctrl+C'; }
    setTimeout(() => { btn.textContent = 'Copy'; }, 1600);
  });

  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      openRail();
      el.historySearch.focus();
    }
  });
})();
