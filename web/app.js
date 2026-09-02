/* ==========================================================================
   Sentinel-RAG — client
   No framework, no build step, no external requests: the page is served from
   the same FastAPI process that answers /query.
   ========================================================================== */
(() => {
  'use strict';

  // Guardrails + LangGraph + two LLM calls can legitimately exceed 60s.
  // The previous client timed out at 60s and reported a healthy backend as
  // offline, so this is deliberately generous.
  const REQUEST_TIMEOUT_MS = 120_000;
  const HEALTH_INTERVAL_MS = 30_000;

  const $ = (id) => document.getElementById(id);
  const el = {
    app: document.querySelector('.app'),
    thread: $('thread'),
    welcome: $('welcome'),
    form: $('composer'),
    input: $('input'),
    send: $('send'),
    status: $('status'),
    statusLabel: $('statusLabel'),
    threadLabel: $('threadLabel'),
    turnCount: $('turnCount'),
    newChat: $('newChat'),
    themeToggle: $('themeToggle'),
    menuToggle: $('menuToggle'),
    scrim: $('scrim'),
    suggestions: $('suggestions'),
    topbarHint: $('topbarHint'),
  };

  // Conversation ids are issued by the server and are unguessable. The client
  // never chooses one: a client-chosen id let any caller read another user's
  // conversation by guessing it.
  let sessionId = null;
  let turns = 0;
  let inFlight = null;      // AbortController for the active request
  let listEl = null;        // lazily-created message container

  /* ------------------------------------------------------------ utils --- */

  const uuid = () =>
    (crypto.randomUUID?.() ??
      `${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 10)}`);

  /** Escape first, always. Everything downstream may only add tags we authored. */
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /** Only http(s) and mailto survive — blocks javascript: and data: URLs. */
  function safeHref(url) {
    const trimmed = String(url).trim();
    return /^(https?:\/\/|mailto:)/i.test(trimmed) ? escapeHtml(trimmed) : null;
  }

  /* --------------------------------------------------------- markdown --- */
  /* A deliberately small subset — headings, emphasis, code, lists, links —
     covering what the model actually emits. Input is escaped before any of
     this runs, so no model output can inject markup. */

  function inline(text) {
    let out = escapeHtml(text);
    out = out.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
    out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?]|$)/g, '$1<em>$2</em>');
    out = out.replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,!?]|$)/g, '$1<em>$2</em>');
    out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, url) => {
      const href = safeHref(url);
      return href
        ? `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`
        : m;
    });
    return out;
  }

  function renderBlocks(src) {
    const lines = src.split('\n');
    let html = '';
    let list = null;          // 'ul' | 'ol' | null
    let para = [];

    const flushPara = () => {
      if (para.length) { html += `<p>${inline(para.join(' '))}</p>`; para = []; }
    };
    const closeList = () => { if (list) { html += `</${list}>`; list = null; } };

    for (const raw of lines) {
      const line = raw.trimEnd();

      if (!line.trim()) { flushPara(); closeList(); continue; }

      const heading = line.match(/^(#{1,3})\s+(.*)$/);
      if (heading) {
        flushPara(); closeList();
        const lvl = heading[1].length;
        html += `<h${lvl}>${inline(heading[2])}</h${lvl}>`;
        continue;
      }

      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ul || ol) {
        flushPara();
        const want = ul ? 'ul' : 'ol';
        if (list !== want) { closeList(); html += `<${want}>`; list = want; }
        html += `<li>${inline((ul || ol)[1])}</li>`;
        continue;
      }

      closeList();
      para.push(line.trim());
    }
    flushPara(); closeList();
    return html;
  }

  function renderMarkdown(src) {
    // Split on fenced blocks so their contents are never treated as markdown.
    const parts = String(src ?? '').split(/```/);
    let html = '';
    parts.forEach((part, i) => {
      if (i % 2 === 1) {
        const nl = part.indexOf('\n');
        const lang = (nl === -1 ? '' : part.slice(0, nl)).trim();
        const code = nl === -1 ? part : part.slice(nl + 1);
        html +=
          `<div class="codeblock">` +
            `<div class="codeblock__bar">` +
              `<span class="codeblock__lang">${escapeHtml(lang || 'text')}</span>` +
              `<button class="copy" type="button" data-copy>Copy</button>` +
            `</div>` +
            `<pre><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>` +
          `</div>`;
      } else {
        html += renderBlocks(part);
      }
    });
    return html;
  }

  /* ------------------------------------------------------------ icons --- */
  const shieldSvg =
    `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 3 5 8v8c0 6.6 4.5 12.7 11 14 6.5-1.3 11-7.4 11-14V8z"/></svg>`;
  const warnSvg =
    `<svg class="notice__icon" viewBox="0 0 16 16" fill="none" stroke-width="1.6" aria-hidden="true">` +
    `<path d="M8 2.2 1.6 13.4h12.8z" stroke-linejoin="round"/><path d="M8 6.6v3.1" stroke-linecap="round"/>` +
    `<circle cx="8" cy="11.4" r=".75" fill="currentColor" stroke="none"/></svg>`;
  const errSvg =
    `<svg class="notice__icon" viewBox="0 0 16 16" fill="none" stroke-width="1.6" aria-hidden="true">` +
    `<circle cx="8" cy="8" r="6.2"/><path d="M8 4.9v3.6" stroke-linecap="round"/>` +
    `<circle cx="8" cy="11.1" r=".75" fill="currentColor" stroke="none"/></svg>`;

  /* --------------------------------------------------------- thread ----- */

  function ensureList() {
    if (!listEl) {
      listEl = document.createElement('div');
      listEl.className = 'thread__inner';
      el.thread.appendChild(listEl);
    }
    return listEl;
  }

  function hideWelcome() {
    if (el.welcome && el.welcome.isConnected) el.welcome.remove();
  }

  function scrollToEnd() {
    // rAF so layout has settled before we measure.
    requestAnimationFrame(() => {
      el.thread.scrollTop = el.thread.scrollHeight;
    });
  }

  function addUser(text) {
    hideWelcome();
    const node = document.createElement('div');
    node.className = 'msg msg--user msg--enter';
    node.innerHTML = `<div class="msg__bubble">${escapeHtml(text)}</div>`;
    ensureList().appendChild(node);
    scrollToEnd();
  }

  function addPending() {
    const node = document.createElement('div');
    node.className = 'msg msg--enter';
    node.innerHTML =
      `<div class="msg__role"><span class="msg__avatar">${shieldSvg}</span>Sentinel</div>` +
      `<div class="thinking"><span class="thinking__dots"><i></i><i></i><i></i></span>` +
      `<span>Searching your documentation…</span></div>`;
    ensureList().appendChild(node);
    scrollToEnd();
    return node;
  }

  /** Retrieval emits "SOURCE: <file>\nCONTENT: <text>" — split it for display. */
  function parseSource(raw) {
    const s = String(raw ?? '');
    const m = s.match(/^SOURCE:\s*(.*?)\r?\nCONTENT:\s*([\s\S]*)$/);
    if (m) return { file: m[1].trim() || 'Unknown', text: m[2].trim() };
    return { file: null, text: s.replace(/^CONTENT:\s*/, '').trim() };
  }

  function stepsHtml(steps) {
    if (!steps?.length) return '';
    const rows = steps
      .map((s, i) => `<div class="step"><span class="step__idx">${i + 1}</span><span>${escapeHtml(s)}</span></div>`)
      .join('');
    return (
      `<details class="disclosure"><summary class="disclosure__summary">` +
        `<svg class="disclosure__chevron" viewBox="0 0 16 16" fill="none" stroke-width="1.7" aria-hidden="true">` +
        `<path d="M6 3.5 10.5 8 6 12.5" stroke-linecap="round" stroke-linejoin="round"/></svg>` +
        `Reasoning<span class="disclosure__count">${steps.length}</span></summary>` +
        `<div class="disclosure__content">${rows}</div></details>`
    );
  }

  function sourcesHtml(sources) {
    if (!sources?.length) return '';
    // Citations are objects carrying document, section, relevance and a link
    // back to the upstream page, so a reader can verify a claim rather than
    // being shown an anonymous slab of retrieved text.
    const items = sources
      .map((src, i) => {
        const title = src.title || src.source || 'Retrieved passage';
        const section = src.section ? ` &middot; ${escapeHtml(src.section)}` : '';
        const score = typeof src.score === 'number' ? src.score.toFixed(2) : '';
        const link = src.url
          ? ` <a class="source__link" href="${escapeHtml(src.url)}" target="_blank" rel="noopener noreferrer">source</a>`
          : '';
        return (
          `<details class="source"><summary class="source__head">` +
            `<span class="source__badge">${i + 1}</span>` +
            `<span class="source__file">${escapeHtml(title)}${section}</span>` +
            (score ? `<span class="source__score" title="cross-encoder relevance">${score}</span>` : '') +
          `</summary><p class="source__text">${escapeHtml(src.excerpt || '')}</p>` +
          `<p class="source__meta">${escapeHtml(src.topic || '')}${link}</p></details>`
        );
      })
      .join('');
    return (
      `<details class="disclosure"><summary class="disclosure__summary">` +
        `<svg class="disclosure__chevron" viewBox="0 0 16 16" fill="none" stroke-width="1.7" aria-hidden="true">` +
        `<path d="M6 3.5 10.5 8 6 12.5" stroke-linecap="round" stroke-linejoin="round"/></svg>` +
        `Sources<span class="disclosure__count">${sources.length}</span></summary>` +
        `<div class="disclosure__content">${items}</div></details>`
    );
  }

  function fillAnswer(node, data) {
    const steps = Array.isArray(data.thought_process) ? data.thought_process : [];
    const sources = Array.isArray(data.sources) ? data.sources : [];
    const answer = data.answer ?? '';
    const status = String(data.status ?? '');

    // Three outcomes, and the user needs to tell them apart: the gate refused
    // the request, the corpus does not cover it, or it was answered.
    const blocked = data.blocked === true || /guardrail/i.test(status);
    const abstained = data.abstained === true;

    const head = `<div class="msg__role"><span class="msg__avatar">${shieldSvg}</span>Sentinel</div>`;

    if (blocked) {
      // A refusal is the product working, not an error — so it reads as a
      // deliberate notice rather than a failure state.
      node.innerHTML =
        head +
        `<div class="notice notice--blocked">${warnSvg}` +
          `<div><p class="notice__title">Outside scope</p>` +
          `<p class="notice__text">${escapeHtml(answer)}</p></div></div>` +
        stepsHtml(steps);
    } else if (abstained) {
      // Abstention is the headline behaviour: no documentation covered this, so
      // nothing was invented. Shown as a distinct state rather than as prose,
      // because "I don't know" being visibly deliberate is the whole point.
      node.innerHTML =
        head +
        `<div class="notice notice--abstain">${warnSvg}` +
          `<div><p class="notice__title">Not in the documentation</p>` +
          `<div class="notice__text">${renderMarkdown(answer)}</div></div></div>` +
        stepsHtml(steps);
    } else {
      node.innerHTML =
        head +
        `<div class="msg__body">${renderMarkdown(answer)}</div>` +
        stepsHtml(steps) +
        sourcesHtml(sources);
    }
    scrollToEnd();
  }

  function fillError(node, title, detail) {
    node.innerHTML =
      `<div class="msg__role"><span class="msg__avatar">${shieldSvg}</span>Sentinel</div>` +
      `<div class="notice notice--error">${errSvg}` +
        `<div><p class="notice__title">${escapeHtml(title)}</p>` +
        `<p class="notice__text">${escapeHtml(detail)}</p></div></div>`;
    scrollToEnd();
  }

  async function loadScope() {
    // The sidebar lists what is actually indexed. A hardcoded list is how the
    // previous version ended up advertising topics the corpus did not contain.
    const list = $('topics');
    if (!list) return;
    try {
      const res = await fetch('/scope', { cache: 'no-store' });
      if (!res.ok) return;
      const scope = await res.json();
      const topics = Array.isArray(scope.topics) ? scope.topics : [];
      if (!topics.length) return;
      list.innerHTML = topics
        .map(
          (t) =>
            `<li><span class="topics__dot"></span>${escapeHtml(t.label)}` +
            `<span class="topics__count">${t.passages}</span></li>`
        )
        .join('');
      const sub = $('scopeSummary');
      if (sub) {
        sub.textContent = `${scope.points} passages across ${topics.length} areas`;
      }
    } catch { /* leave the static fallback in place */ }
  }

  /* ------------------------------------------------------------- net ---- */

  async function ask(text) {
    const node = addPending();
    const controller = new AbortController();
    inFlight = controller;
    const timer = setTimeout(() => controller.abort('timeout'), REQUEST_TIMEOUT_MS);

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: text, session_id: sessionId }),
        signal: controller.signal,
      });

      // Check the status before parsing — a 500 with an HTML error page would
      // otherwise surface as a confusing JSON parse failure.
      if (!res.ok) {
        let detail = `The backend responded with HTTP ${res.status}.`;
        try {
          const body = await res.text();
          if (body) detail += ` ${body.slice(0, 240)}`;
        } catch { /* body unreadable — the status alone is enough */ }
        fillError(node, 'Request failed', detail);
        return;
      }

      let data;
      try {
        data = await res.json();
      } catch {
        fillError(node, 'Unreadable response', 'The backend returned a response that was not valid JSON.');
        return;
      }

      fillAnswer(node, data);
      turns += 1;
      el.turnCount.textContent = String(turns);
      setStatus('online', 'Connected');
    } catch (err) {
      if (controller.signal.aborted && controller.signal.reason === 'cancelled') {
        node.remove();
        return;
      }
      if (controller.signal.aborted) {
        fillError(node, 'Timed out',
          `No response within ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s. The backend may still be starting up — it imports the guardrails and agent graph before accepting traffic.`);
      } else {
        fillError(node, 'Cannot reach the backend',
          'The request failed. Check that the API is running and reachable, then try again.');
        setStatus('offline', 'Unreachable');
      }
    } finally {
      clearTimeout(timer);
      inFlight = null;
      setBusy(false);
    }
  }

  function setStatus(state, label) {
    el.status.dataset.state = state;
    el.statusLabel.textContent = label;
  }

  async function checkHealth() {
    try {
      const res = await fetch('/health', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json().catch(() => ({}));
        setStatus('online', data.guardrails === false ? 'Degraded' : 'Connected');
      } else {
        setStatus('offline', `HTTP ${res.status}`);
      }
    } catch {
      setStatus('offline', 'Unreachable');
    }
  }

  /* ------------------------------------------------------------- ui ----- */

  function setBusy(busy) {
    el.send.disabled = busy || !el.input.value.trim();
    el.input.disabled = busy;
    el.topbarHint.textContent = busy ? 'Thinking…' : 'Grounded in your documentation';
    if (!busy) el.input.focus();
  }

  function autoGrow() {
    el.input.style.height = 'auto';
    el.input.style.height = `${Math.min(el.input.scrollHeight, 190)}px`;
  }

  function submit(text) {
    const value = String(text ?? el.input.value).trim();
    if (!value || inFlight) return;
    addUser(value);
    el.input.value = '';
    autoGrow();
    setBusy(true);
    ask(value);
  }

  async function startSession() {
    // The server owns conversation identity. If it declines, we simply run
    // without memory rather than inventing an id the server would reject.
    try {
      const res = await fetch('/session', { method: 'POST' });
      if (res.ok) {
        sessionId = (await res.json()).session_id ?? null;
      } else {
        sessionId = null;
      }
    } catch {
      sessionId = null;
    }
    el.threadLabel.textContent = sessionId ? sessionId.slice(0, 8) : 'no memory';
  }

  function newConversation() {
    if (inFlight) { inFlight.abort('cancelled'); inFlight = null; }
    startSession();
    turns = 0;
    el.turnCount.textContent = '0';
    if (listEl) { listEl.remove(); listEl = null; }
    // Rebuild the empty state so a fresh session looks like a fresh session.
    location.hash = '';
    el.thread.innerHTML = '';
    el.thread.appendChild(buildWelcome());
    el.welcome = $('welcome');
    wireSuggestions();
    setBusy(false);
  }

  function buildWelcome() {
    const tpl = document.createElement('div');
    tpl.innerHTML = WELCOME_HTML;
    return tpl.firstElementChild;
  }

  // Captured once at load so "New conversation" can restore it verbatim.
  const WELCOME_HTML = el.welcome ? el.welcome.outerHTML : '';

  function wireSuggestions() {
    document.querySelectorAll('.suggestion').forEach((b) => {
      b.addEventListener('click', () => submit(b.dataset.q));
    });
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('sentinel-theme', theme); } catch { /* private mode */ }
  }

  function setSidebar(open) {
    el.app.dataset.sidebar = open ? 'open' : 'closed';
    el.scrim.hidden = !open;
  }

  /* ----------------------------------------------------------- init ---- */

  // Theme: stored choice wins, otherwise follow the OS.
  try {
    const saved = localStorage.getItem('sentinel-theme');
    if (saved === 'light' || saved === 'dark') applyTheme(saved);
    else if (window.matchMedia?.('(prefers-color-scheme: light)').matches) applyTheme('light');
  } catch { /* storage unavailable — dark default stands */ }

  startSession();
  loadScope();

  el.form.addEventListener('submit', (e) => { e.preventDefault(); submit(); });

  el.input.addEventListener('input', () => {
    autoGrow();
    el.send.disabled = !el.input.value.trim() || !!inFlight;
  });

  el.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submit();
    }
  });

  el.newChat.addEventListener('click', newConversation);

  el.themeToggle.addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
  });

  el.menuToggle.addEventListener('click', () =>
    setSidebar(el.app.dataset.sidebar !== 'open'));
  el.scrim.addEventListener('click', () => setSidebar(false));

  // Copy buttons are delegated: code blocks are created after this runs.
  el.thread.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-copy]');
    if (!btn) return;
    const code = btn.closest('.codeblock')?.querySelector('code')?.textContent ?? '';
    try {
      await navigator.clipboard.writeText(code);
      btn.textContent = 'Copied';
      btn.dataset.copied = 'true';
    } catch {
      btn.textContent = 'Press ⌘C';
    }
    setTimeout(() => { btn.textContent = 'Copy'; delete btn.dataset.copied; }, 1600);
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && el.app.dataset.sidebar === 'open') setSidebar(false);
  });

  wireSuggestions();
  setBusy(false);
  checkHealth();
  setInterval(checkHealth, HEALTH_INTERVAL_MS);
  el.input.focus();
})();
