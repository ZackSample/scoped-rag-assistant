/*
  Browser app for the technician-facing UI.
  It uses the FastAPI backend directly and keeps only lightweight UI state in
  the browser. Durable history lives in the database behind the API.
*/

const state = {
  // All UI state stays here so screen rerenders are predictable.
  machines: [],
  activeMachine: null,
  currentView: "machines",
  sessionByMachine: {},
  messagesByMachine: {},
  historySessions: [],
  activeCitation: null,
  // The source viewer: every source of the answer it was opened from (one tab
  // each), the page being shown, and that document's page count once known.
  panelSources: [],
  viewerPage: 1,
  viewerPageCount: null,
  machineFilter: "",
  activeContextByMachine: {},
  activityByMachine: {},
  streamingByMachine: {},
  // Full-search turns reroute streamed answer text into the activity panel;
  // these track the reroute per machine for the in-flight turn only.
  panelStreamByMachine: {},
  fullSearchModeByMachine: {},
  // The live activity panel is collapsed by default; remembering the reader's
  // choice here is what survives the full innerHTML rebuild per stream token.
  activityOpenByMachine: {},
  citationTextPreview: null,
  currentUser: null,
  zoom: 1,
  scrollToNewestAnswer: false,
};

const app = document.querySelector("#app");

async function api(path, options = {}) {
  // Small wrapper keeps API errors readable in the chat instead of silent.
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const text = await response.text();
    const error = new Error(text || `Request failed: ${response.status}`);
    // Carried so callers can tell "not signed in" from "the server broke".
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function machineMessages() {
  if (!state.activeMachine) return [];
  const key = state.activeMachine.machine_number;
  state.messagesByMachine[key] ||= [];
  return state.messagesByMachine[key];
}

function setSessionId(sessionId) {
  if (!state.activeMachine || !sessionId) return;
  state.sessionByMachine[state.activeMachine.machine_number] = sessionId;
}

function getSessionId() {
  if (!state.activeMachine) return null;
  return state.sessionByMachine[state.activeMachine.machine_number] || null;
}

function getActiveContext() {
  if (!state.activeMachine) return null;
  return state.activeContextByMachine[state.activeMachine.machine_number] || null;
}

function setActiveContext(citations) {
  if (!state.activeMachine) return;
  const key = state.activeMachine.machine_number;
  // Active context tells follow-up questions which cited pages are still relevant.
  state.activeContextByMachine[key] = citations?.length ? { citations, createdAt: new Date().toISOString() } : null;
}

function pushActivity(kind, text) {
  // Append rather than replace. A single overwritten line lost everything the
  // pipeline had already reported - which search it ran, what it found, what
  // the evidence check concluded - and left the technician with one vague
  // sentence for the whole wait.
  if (!state.activeMachine || !text) return;
  const key = state.activeMachine.machine_number;
  const items = state.activityByMachine[key]?.items || [];
  const last = items[items.length - 1];
  // Progress stages repeat verbatim across passes; collapse exact duplicates.
  if (last && last.kind === kind && last.text === text) return;
  state.activityByMachine[key] = { items: [...items, { kind, text }] };
}

function clearActivity() {
  if (!state.activeMachine) return;
  delete state.activityByMachine[state.activeMachine.machine_number];
}

function getActivity() {
  if (!state.activeMachine) return null;
  return state.activityByMachine[state.activeMachine.machine_number] || null;
}

function appendStreamingAnswer(text) {
  if (!state.activeMachine) return;
  const key = state.activeMachine.machine_number;
  // On a full-search turn the bubble shows only the final summary; everything
  // the model writes along the way belongs in the activity panel.
  if (state.fullSearchModeByMachine[key]) {
    state.panelStreamByMachine[key] = (state.panelStreamByMachine[key] || "") + text;
    return;
  }
  state.streamingByMachine[key] = (state.streamingByMachine[key] || "") + text;
}

function getStreamingAnswer() {
  if (!state.activeMachine) return null;
  return state.streamingByMachine[state.activeMachine.machine_number] || null;
}

function clearStreamingAnswer() {
  if (!state.activeMachine) return;
  delete state.streamingByMachine[state.activeMachine.machine_number];
}

function enterFullSearchMode() {
  if (!state.activeMachine) return;
  const key = state.activeMachine.machine_number;
  state.fullSearchModeByMachine[key] = true;
  // Whatever already streamed into the bubble moves into the panel so the
  // bubble stays clean for the summary.
  const streamed = state.streamingByMachine[key];
  if (streamed) {
    state.panelStreamByMachine[key] = (state.panelStreamByMachine[key] || "") + streamed;
    delete state.streamingByMachine[key];
  }
}

function getPanelStream() {
  if (!state.activeMachine) return null;
  return state.panelStreamByMachine[state.activeMachine.machine_number] || null;
}

function clearPanelStream() {
  if (!state.activeMachine) return;
  delete state.panelStreamByMachine[state.activeMachine.machine_number];
}

function resetStreamingState() {
  if (!state.activeMachine) return;
  const key = state.activeMachine.machine_number;
  delete state.streamingByMachine[key];
  delete state.panelStreamByMachine[key];
  delete state.fullSearchModeByMachine[key];
}

// Small inline icons, so the UI needs no icon font or build step.
const ICONS = {
  back: '<path d="M15 18l-6-6 6-6"/>',
  prev: '<path d="M15 18l-6-6 6-6"/>',
  next: '<path d="M9 18l6-6-6-6"/>',
  close: '<path d="M18 6L6 18M6 6l12 12"/>',
  open: '<path d="M14 3h7v7M10 14L21 3M21 14v7H3V3h7"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
  history: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>',
  up: '<path d="M7 10v11M2 10h5l4-8a3 3 0 0 1 3 3v3h6a2 2 0 0 1 2 2.3l-1.4 8A2 2 0 0 1 18.6 21H7"/>',
  down: '<path d="M17 14V3M22 14h-5l-4 8a3 3 0 0 1-3-3v-3H4a2 2 0 0 1-2-2.3l1.4-8A2 2 0 0 1 5.4 3H17"/>',
};

function icon(name, size = 18) {
  return `<svg class="icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
}

function initials(user) {
  const parts = String(user?.display_name || user?.username || "?").split(/[\s._-]+/).filter(Boolean);
  return ((parts[0]?.[0] || "?") + (parts[1]?.[0] || "")).toUpperCase();
}

function topBar({ eyebrow = "", title, back = false, actions = "" }) {
  // One bar across the top of every screen: back (or the app mark), what you
  // are looking at, then actions and who is signed in.
  const user = state.currentUser;
  return `
    <header class="top">
      ${back ? `<button class="iconbtn" id="top-back" aria-label="Back">${icon("back", 20)}</button>` : `<div class="mark">RAG</div>`}
      <div class="title">
        ${eyebrow ? `<small>${escapeHtml(eyebrow)}</small>` : ""}
        <b>${escapeHtml(title)}</b>
      </div>
      <div class="grow"></div>
      ${actions}
      ${themeToggleHtml()}
      ${user ? `<button class="iconbtn" id="sign-out" title="Sign out">${icon("logout")}</button>` : ""}
      ${user ? `<div class="who" title="${escapeAttr(user.display_name || user.username)}">${escapeHtml(initials(user))}</div>` : ""}
    </header>
  `;
}

function machineSubtitle(machine) {
  return [machine.model, machine.serial ? `Serial ${machine.serial}` : ""].filter(Boolean).join(" | ");
}

const STARTERS = [
  "How much hydraulic oil does it hold?",
  "A heater zone won't come up to temperature",
  "How do I adjust the mold height?",
  "What is the clamp tonnage and shot size?",
];

function emptyStateHtml(machine) {
  return `
    <div class="hello">
      <b>What's going on with machine ${escapeHtml(machine.machine_number)}?</b>
      Answers come from this machine's documents and the shared troubleshooting library.
      <div class="starters">
        ${STARTERS.map((text) => `<button type="button" data-starter="${escapeAttr(text)}">${escapeHtml(text)}</button>`).join("")}
      </div>
    </div>
  `;
}

function startFreshMachineSession(machine) {
  const key = machine.machine_number;
  // Changing machines starts clean so answers never bleed across machine scopes.
  state.activeMachine = machine;
  state.sessionByMachine[key] = null;
  state.messagesByMachine[key] = [];
  state.activeContextByMachine[key] = null;
  delete state.activityByMachine[key];
  delete state.streamingByMachine[key];
  delete state.panelStreamByMachine[key];
  delete state.fullSearchModeByMachine[key];
  delete state.activityOpenByMachine[key];
  closeViewer();
}

function wireSignOut() {
  document.querySelector("#sign-out")?.addEventListener("click", async () => {
    await api("/auth/logout", { method: "POST" });
    // Full reload rather than a re-render: it drops every scrap of in-memory
    // state with it. On a shared device the next technician must not inherit
    // the previous one's chat, citations or machine selection.
    window.location.href = "/login";
  });
}

async function loadMachines() {
  try {
    state.currentUser = (await api("/auth/me")).user || null;
  } catch {
    // Auth may be disabled entirely in local development.
    state.currentUser = null;
  }
  let data;
  try {
    data = await api("/machine-registry");
  } catch (error) {
    // /app/ is served without a gate (see app/auth/middleware.py), so an
    // unauthenticated load reaches this code instead of being redirected. The
    // first real API call is what discovers it. Deliberately keyed on the 401
    // rather than on /auth/me saying "not authenticated", because with
    // AUTH_REQUIRED=false that is the normal local-development answer.
    if (error.status === 401) {
      window.location.href = "/login";
      return;
    }
    throw error;
  }
  state.machines = data.machines;
  renderMachineSelector();
}

function renderMachineSelector() {
  state.currentView = "machines";
  // The landing page enforces machine choice before chat is available.
  app.innerHTML = `
    <div class="shell">
      ${topBar({ title: "Scoped RAG Assistant" })}
      <main class="start">
        <div class="start-inner">
          <h1 class="start-title">Which machine are we working with?</h1>
          <label class="search">
            ${icon("search", 20)}
            <input id="machine-filter" value="${escapeAttr(state.machineFilter)}" placeholder="Machine number or model" inputmode="search" autocomplete="off" />
          </label>
          <div class="grid" id="machine-grid">${machineTilesHtml()}</div>
        </div>
      </main>
    </div>
  `;

  const filter = document.querySelector("#machine-filter");
  filter.addEventListener("input", () => {
    // Only the grid is redrawn, so the search box keeps focus while typing.
    state.machineFilter = filter.value;
    document.querySelector("#machine-grid").innerHTML = machineTilesHtml();
  });
  filter.addEventListener("keydown", (event) => {
    if (event.key === "Enter") document.querySelector("#machine-grid [data-machine]")?.click();
  });
  document.querySelector("#machine-grid").addEventListener("click", (event) => {
    const tile = event.target.closest("[data-machine]");
    if (!tile) return;
    startFreshMachineSession(state.machines.find((machine) => machine.machine_number === tile.dataset.machine));
    renderChat();
  });
  wireSignOut();
  wireThemeToggle();
}

function machineTilesHtml() {
  const needle = state.machineFilter.trim().toLowerCase();
  const shown = state.machines.filter(
    (machine) => !needle || [machine.machine_number, machine.model, machine.serial].some((value) => value?.toLowerCase().includes(needle))
  );
  if (!shown.length) {
    return `<p class="empty">${state.machines.length ? `No machine matches "${escapeHtml(state.machineFilter)}".` : "No machine folders found under DOCS_ROOT yet."}</p>`;
  }
  return shown.map((machine) => `
    <button class="tile" data-machine="${escapeAttr(machine.machine_number)}" title="${escapeAttr(machine.model || machine.display_name)}">
      <b>${escapeHtml(machine.machine_number)}</b>
      <small>${escapeHtml(machine.model || machine.display_name)}</small>
    </button>
  `).join("");
}

function renderChat() {
  state.currentView = "chat";

  // Captured before the rebuild wipes the scroll position. 120px of slack, so
  // "near the bottom" still counts as following along - an exact comparison
  // would drop out of follow mode the moment a streaming token nudged the
  // height mid-measurement.
  const previousMessages = document.querySelector("#messages");
  const previousScrollTop = previousMessages ? previousMessages.scrollTop : 0;
  const wasNearBottom = previousMessages
    ? previousMessages.scrollHeight - previousMessages.scrollTop - previousMessages.clientHeight < 120
    : true;

  const machine = state.activeMachine;
  const messages = machineMessages();
  // The source viewer only takes space when a cited page is open.
  const layoutClass = state.activeCitation ? "chat with-panel" : "chat";

  app.innerHTML = `
    <div class="shell chat-screen">
      ${topBar({
        eyebrow: `Machine ${machine.machine_number}`,
        title: machineSubtitle(machine) || machine.display_name,
        back: true,
        actions: `
          <button class="iconbtn" id="open-history" title="History">${icon("history")}<span class="label">History</span></button>
          <button class="iconbtn" id="new-question" title="New chat">${icon("plus")}<span class="label">New chat</span></button>
        `,
      })}
      <div class="${layoutClass}">
        <div class="thread-col">
          <div class="thread" id="messages">
            <div class="thread-inner">
              ${messages.map((message, index) => renderMessage(message, index === messages.length - 1, index)).join("") || emptyStateHtml(machine)}
              ${renderLiveActivity()}
              ${renderStreamingAnswer()}
            </div>
          </div>
          <div class="composer-wrap">
            <form class="composer" id="composer">
              <textarea id="question" rows="1" placeholder="${messages.length ? "Ask a follow-up" : "Describe the symptom, alarm, part or procedure"}" required></textarea>
              <button class="send" type="submit" aria-label="Ask">${icon("send", 22)}</button>
            </form>
          </div>
        </div>
        ${state.activeCitation ? renderSourcesPanel() : ""}
      </div>
    </div>
  `;

  document.querySelector("#top-back").addEventListener("click", () => {
    clearActivity();
    state.activeMachine = null;
    state.activeCitation = null;
    renderMachineSelector();
  });
  document.querySelector("#open-history").addEventListener("click", loadHistory);
  wireSignOut();
  document.querySelectorAll("[data-starter]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelector("#question").value = button.dataset.starter;
      document.querySelector("#composer").requestSubmit();
    });
  });
  const newQuestion = document.querySelector("#new-question");
  if (newQuestion) {
    newQuestion.addEventListener("click", () => {
      // New question clears follow-up context and starts a fresh session thread.
      setActiveContext(null);
      state.sessionByMachine[state.activeMachine.machine_number] = null;
      state.messagesByMachine[state.activeMachine.machine_number] = [];
      state.activityByMachine[state.activeMachine.machine_number] = null;
      state.activeCitation = null;
      renderChat();
    });
  }
  document.querySelector("#composer").addEventListener("submit", submitQuestion);
  const questionBox = document.querySelector("#question");
  if (questionBox) {
    // Auto-grow. The box starts at one line and only takes the space a long
    // question actually needs; max-height in CSS stops it eating the chat.
    const fit = () => {
      questionBox.style.height = "auto";
      questionBox.style.height = `${questionBox.scrollHeight}px`;
    };
    questionBox.addEventListener("input", fit);
    // Enter sends, Shift+Enter is a newline - a shop-floor tool should not need
    // the reader to aim for a button after every question.
    questionBox.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        document.querySelector("#composer").requestSubmit();
      }
    });
  }
  document.querySelectorAll("[data-citation]").forEach((button) => {
    button.addEventListener("click", () => {
      // Open the viewer with every source of that answer as a tab.
      const sources = machineMessages()[Number(button.dataset.msg)]?.citations;
      openCitation(JSON.parse(button.dataset.citation), sources);
    });
  });
  document.querySelectorAll("[data-more-detail]").forEach((button) => {
    button.addEventListener("click", () => askForMoreDetail());
  });
  document.querySelectorAll("[data-toggle-detail]").forEach((button) => {
    button.addEventListener("click", () => {
      const message = machineMessages().find(
        (candidate) => String(candidate.answer_id) === button.dataset.toggleDetail
      );
      if (!message) return;
      message.detail_open = !message.detail_open;
      renderChat();
    });
  });
  wireLiveActivity();
  // Scroll anchoring. renderChat() rebuilds the whole transcript, and it used
  // to jam the view to the bottom every time - so opening a citation on an
  // answer you were part-way through reading threw you to the end of the chat.
  //
  // The rule: follow the bottom only if the reader was already there (or close
  // to it), which is what someone watching an answer stream in is doing.
  // Otherwise hold their position. `wasNearBottom` is measured BEFORE the
  // rebuild, at the top of this function.
  const messagesEl = document.querySelector("#messages");
  if (state.scrollToNewestAnswer) {
    // A finished answer lands at ITS OWN TOP, not the transcript bottom. The
    // reader was pinned to the bottom watching it stream, so following the
    // bottom scrolled them past the whole answer to the feedback and citation
    // sections underneath it - they had to scroll up to read what they asked
    // for, and the more-detail button made that worse.
    state.scrollToNewestAnswer = false;
    const answers = messagesEl.querySelectorAll(".a");
    const newest = answers[answers.length - 1];
    if (newest) {
      messagesEl.scrollTop = newest.offsetTop - messagesEl.offsetTop;
    } else {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  } else if (wasNearBottom) {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } else {
    messagesEl.scrollTop = previousScrollTop;
  }
  wireArtifactButtons();
  wireFeedback();
  wireThemeToggle();
}

async function loadHistory() {
  state.historySessions = (await api(`/chat/sessions/${state.activeMachine.machine_number}`)).sessions;
  renderHistory();
}

function renderHistory() {
  const machine = state.activeMachine;
  app.innerHTML = `
    <div class="shell">
      ${topBar({ eyebrow: `Machine ${machine.machine_number}`, title: "Earlier conversations", back: true })}
      <main class="start">
        <div class="start-inner">
          <div class="recent">
            ${state.historySessions.map((session) => `
              <button data-session-id="${session.id}">
                ${escapeHtml(session.title || `Session ${session.id}`)}
                <small>${session.message_count} messages &middot; ${escapeHtml(session.last_message_at)}</small>
              </button>
            `).join("") || `<p class="empty">No saved conversations for this machine yet.</p>`}
          </div>
        </div>
      </main>
    </div>
  `;

  document.querySelector("#top-back").addEventListener("click", renderChat);
  wireSignOut();
  wireThemeToggle();
  document.querySelectorAll("[data-session-id]").forEach((button) => {
    button.addEventListener("click", () => loadSession(button.dataset.sessionId));
  });
}

async function loadSession(sessionId) {
  const payload = await api(`/chat/sessions/${state.activeMachine.machine_number}/${sessionId}`);
  const key = state.activeMachine.machine_number;
  state.sessionByMachine[key] = payload.session.id;
  state.messagesByMachine[key] = messagesFromLoadedSession(payload.session);
  const lastAssistant = [...state.messagesByMachine[key]].reverse().find((message) => message.role === "assistant");
  state.activeContextByMachine[key] = lastAssistant?.citations?.length
    ? { citations: lastAssistant.citations, createdAt: new Date().toISOString() }
    : null;
  closeViewer();
  clearActivity();
  renderChat();
}

function messagesFromLoadedSession(session) {
  // Messages are stored separately from answers. Reattach answer citations to
  // assistant messages in order so loaded history keeps clickable citations.
  const answerQueue = [...(session.answers || [])];
  return (session.messages || []).map((message) => {
    if (message.role !== "assistant") return { role: message.role, content: message.content };
    const answer = answerQueue.shift();
    const fullAnswer = answer?.full_answer || null;
    return {
      role: "assistant",
      content: message.content,
      // Reopened history keeps the instant More-detail reveal: the gate is
      // recomputed from the stored lengths, same rule as the live turn.
      full_content: fullAnswer,
      has_full_detail: Boolean(fullAnswer && fullAnswer.length - message.content.length > 200),
      detail_open: false,
      citations: answer?.citations || [],
      source_badges: answer?.source_badges || [],
      // Rebuild the trail from what was stored, so reopening an old answer
      // shows how it was reached and not just what it concluded.
      activity: activityFromStored(answer),
      answer_id: answer?.id || null,
      // Only this user's own rating comes back, so a reopened session shows
      // the thumb they pressed rather than someone else's on a shared device.
      feedback: answer?.feedback || null,
    };
  });
}

function activityFromStored(answer) {
  if (!answer) return [];
  // Stored progress events keep their stage as `kind`; reasoning is a plain
  // list of strings, so it gets the kind applied here.
  const progress = (answer.activity || []).map((event) => ({
    kind: event.stage || "progress",
    text: event.message || "",
  }));
  const reasoning = (answer.reasoning || []).map((text) => ({ kind: "reasoning", text }));
  return [...progress, ...reasoning].filter((item) => item.text);
}

// Spelled out on purpose for anything involving stored energy: a technician
// reading a quick answer at the machine should see this without scrolling.
const SAFETY_PATTERN = /\b(lock ?out|lockout|tag ?out|hot surface|high voltage|under pressure|hydraulic pressure)\b/i;

function renderMessage(message, isLatest = false, index = 0) {
  if (message.role === "assistant") {
    // Full-search answers carry both bodies; the toggle swaps them locally
    // with no network call - the full text was stored with the answer.
    const bodyText = message.detail_open && message.full_content ? message.full_content : message.content;
    const citations = message.citations || [];
    return `
      <article class="a">
        ${renderActivityLog(message.activity || [], { collapsed: true, sourceCount: citations.length })}
        ${renderSourceBadges(message.source_badges || [])}
        <div class="answer-body">${renderAnswerText(bodyText, citations, index)}</div>
        ${SAFETY_PATTERN.test(bodyText) ? `<div class="notice">Lock out the machine before working on hydraulics, heaters or electrical.</div>` : ""}
        ${renderDetailToggle(message)}
        ${renderMoreDetail(message, isLatest)}
        ${renderSourceCards(citations, index)}
        ${renderRelatedAnswer(message.related_previous_answer)}
        ${renderFeedback(message)}
      </article>
    `;
  }
  if (message.role === "user") {
    return `<div class="q">${escapeHtml(message.content)}</div>`;
  }
  return `<div class="system-note">${escapeHtml(message.content)}</div>`;
}

const MORE_DETAIL_QUESTION = "Yes - give me the full detail on that.";

function renderDetailToggle(message) {
  // The instant reveal for full-search answers. Unlike the legacy expand
  // button below, this works on every answer that carries a full body -
  // including reopened history - because it is a local swap, not a request.
  if (!message.has_full_detail || !message.full_content || !message.answer_id) return "";
  const label = message.detail_open ? "Show the short version" : "More detail";
  return `
    <div class="more-detail">
      <button type="button" class="ghost" data-toggle-detail="${message.answer_id}">${label}</button>
    </div>
  `;
}

function renderMoreDetail(message, isLatest) {
  // Only on the newest answer: the follow-up lands at the end of the thread, so
  // a button on an older answer would silently expand something else.
  if (!message.offers_detail || !isLatest) return "";
  // There is always more that could be said, so the model keeps offering. One
  // expansion per question is the useful part; offering to expand the expansion
  // is just a button that never goes away.
  const messages = machineMessages();
  const asked = messages[messages.length - 2];
  if (asked?.role === "user" && asked.content === MORE_DETAIL_QUESTION) return "";
  return `
    <div class="more-detail">
      <button type="button" class="ghost" data-more-detail>Would you like more detail?</button>
    </div>
  `;
}

async function askForMoreDetail() {
  machineMessages().push({ role: "user", content: MORE_DETAIL_QUESTION });
  renderChat();
  // intent:"expand" tells the backend to reuse the evidence behind the answer
  // being expanded instead of retrieving again.
  await askBackend({ question: MORE_DETAIL_QUESTION, intent: "expand" });
}

function renderStreamingAnswer() {
  const partial = getStreamingAnswer();
  if (!partial) return "";
  // Markdown is rendered here too. It is still provisional text that the final
  // result event replaces, but leaving it raw meant the technician watched
  // literal ** stream past and then snap into bold at the end.
  return `
    <article class="a" aria-live="polite">
      <div class="answer-body cursor">${renderMarkdown(escapeHtml(partial))}</div>
    </article>
  `;
}

const ACTIVITY_LABELS = {
  reasoning: "Thinking",
  tool_call: "Looking",
  retrieval: "Searching",
  retrieval_result: "Found",
  judge: "Checking",
  judge_result: "Verdict",
  database: "Database",
  final_answer: "Writing",
};

function renderActivityLog(items, { collapsed = false, sourceCount = 0 } = {}) {
  if (!items?.length) return "";

  const rows = items.map((item, index) => {
    const isLast = index === items.length - 1;
    // Live: the newest step is in progress. Finished: every step is done.
    const state = collapsed || !isLast ? "done" : "current";
    const marker = state === "done" ? `<span class="activity-tick">&check;</span>` : `<span class="activity-dot"></span>`;
    return `
      <div class="activity-item ${item.kind} ${state}">
        ${marker}
        <span class="activity-label">${escapeHtml(ACTIVITY_LABELS[item.kind] || "Working")}</span>
        <span class="activity-text">${escapeHtml(item.text)}</span>
      </div>
    `;
  }).join("");

  if (!collapsed) {
    // Collapsed-by-default heartbeat. The summary row IS the progress display:
    // a spinner plus the current stage, updating with every event; the full
    // step list (and, on full-search turns, the model's in-progress writing)
    // is one tap away. Open state lives in `state` because this whole panel is
    // rebuilt on every streamed token.
    const current = items[items.length - 1];
    const stage = ACTIVITY_LABELS[current.kind] || "Working";
    const snippet = current.text.length > 90 ? `${current.text.slice(0, 90)}...` : current.text;
    const open = isActivityOpen() ? " open" : "";
    const draft = getPanelStream();
    const draftBlock = draft
      ? `
        <div class="activity-draft">
          <span class="activity-draft-label">The model is writing (full answer streams here, summary lands in the chat):</span>
          <div class="answer-body">${renderMarkdown(escapeHtml(draft))}</div>
        </div>
      `
      : "";
    return `
      <details class="activity-log live" id="live-activity"${open} aria-live="polite">
        <summary>
          <span class="activity-spinner" aria-hidden="true" style="animation-delay: -${((Date.now() % 900) / 1000).toFixed(3)}s"></span>
          <span class="activity-heartbeat"><strong>${escapeHtml(stage)}</strong> &mdash; ${escapeHtml(snippet)}</span>
        </summary>
        ${rows}
        ${draftBlock}
      </details>
    `;
  }

  // One quiet line above the answer; the full trail is a tap away.
  const searches = items.filter((item) => item.kind === "retrieval" || (item.kind === "tool_call" && /^Searching/.test(item.text))).length;
  const summary = `${searches} search${searches === 1 ? "" : "es"}, ${sourceCount} source${sourceCount === 1 ? "" : "s"} \u00b7 how this was worked out`;
  return `
    <details class="activity-log collapsed">
      <summary>${escapeHtml(summary)}</summary>
      ${rows}
    </details>
  `;
}

function isActivityOpen() {
  if (!state.activeMachine) return false;
  return Boolean(state.activityOpenByMachine[state.activeMachine.machine_number]);
}

function wireLiveActivity() {
  const panel = document.querySelector("#live-activity");
  if (!panel) return;
  panel.addEventListener("toggle", () => {
    if (!state.activeMachine) return;
    // Guard: rebuilds re-emit the stored state, so only a change that differs
    // from it is a real reader action worth persisting.
    const key = state.activeMachine.machine_number;
    if (Boolean(state.activityOpenByMachine[key]) !== panel.open) {
      state.activityOpenByMachine[key] = panel.open;
    }
  });
}

function renderLiveActivity() {
  return renderActivityLog(getActivity()?.items || [], { collapsed: false });
}

function renderSourceBadges(badges) {
  if (!badges.length) return "";
  return `<div class="source-badges">${badges.map((badge) => `<span>${escapeHtml(badge)}</span>`).join("")}</div>`;
}

function renderAnswerText(text, citations, msgIndex = 0) {
  // First link loose source text, then apply markdown, then replace exact
  // [cite_n] markers. Order matters: linkInlineSourceCitations escapes the
  // text, markdown works on that escaped string, and citation buttons are
  // injected last so their HTML is not re-escaped or mangled by markdown.
  let html = linkInlineSourceCitations(text, citations, msgIndex);
  html = renderMarkdown(html);
  citations.forEach((citation) => {
    const button = citationButtonHtml(citation, citationNumber(citation), msgIndex, "cite");
    html = html.replaceAll(`[${citation.id}]`, button);
  });
  return html;
}

function renderMarkdown(escaped) {
  // Deliberately small: the model is asked for structured answers, and without
  // this the technician reads literal "**Clamping force:**" and "| col |".
  // Input is ALREADY HTML-escaped, so this only ever adds tags - it can never
  // introduce markup from the model's text.
  const lines = escaped.split("\n");
  const out = [];
  let listType = null;

  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };

  const inline = (value) => value
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    // The inner alternation requires the emphasis to hug non-whitespace at both
    // ends, the way real markdown does. Without it "5 * 3 = 15 and a lone *"
    // became one long italic run - and multiplication shows up in these
    // documents constantly (tonnage, shot size, cycle maths).
    // Written without lookbehind on purpose: Safari only gained it in 16.4 and
    // shop-floor tablets are of unknown vintage.
    .replace(/(^|[\s(])\*([^\s*][^*\n]*[^\s*]|[^\s*])\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");

  // Plain lines are collected and flushed as one <p> rather than emitted raw.
  // Raw emission relied on the container's `white-space: pre-wrap` to produce
  // line breaks, which then ALSO applied to the newlines between generated
  // block tags - so every list and heading gained a blank line it never asked
  // for, and answers rendered two to three times taller than they are.
  let para = [];
  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${para.join("<br>")}</p>`);
      para = [];
    }
  };

  const isTableRow = (value) => /^\s*\|.*\|\s*$/.test(value);
  const isTableDivider = (value) => /^\s*\|[\s:|-]+\|\s*$/.test(value);
  const cells = (value) =>
    value.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];

    // Tables: a header row, a |---|---| divider, then body rows. The model
    // emits these constantly for specs and comparisons, and without this the
    // technician read raw pipe characters.
    if (isTableRow(line) && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      flushPara();
      closeList();
      const header = cells(line);
      const body = [];
      let cursor = index + 2;
      while (cursor < lines.length && isTableRow(lines[cursor])) {
        body.push(cells(lines[cursor]));
        cursor += 1;
      }
      index = cursor - 1;
      out.push(
        `<div class="table-scroll"><table>` +
          `<thead><tr>${header.map((cell) => `<th>${inline(cell)}</th>`).join("")}</tr></thead>` +
          `<tbody>${body
            .map((row) => `<tr>${row.map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`)
            .join("")}</tbody>` +
          `</table></div>`
      );
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushPara();
      closeList();
      // Answers sit inside the chat bubble, so headings start at h4 rather
      // than competing with the page's own heading levels.
      const level = Math.min(6, heading[1].length + 3);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const numbered = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushPara();
      const wanted = bullet ? "ul" : "ol";
      if (listType !== wanted) {
        closeList();
        // `start` carries the model's own number through. Procedures interleave
        // numbered steps with bullets under each one, which closes the <ol> and
        // opens a fresh one - so without this, every step rendered as "1.".
        out.push(wanted === "ol" ? `<ol start="${Number(numbered[1])}">` : "<ul>");
        listType = wanted;
      }
      out.push(`<li>${inline(bullet ? bullet[1] : numbered[2])}</li>`);
      continue;
    }

    closeList();
    if (line.trim() === "") {
      // A blank line ends the paragraph; it does not become one.
      flushPara();
    } else {
      para.push(inline(line));
    }
  }
  flushPara();
  closeList();
  return out.join("");
}

function linkInlineSourceCitations(text, citations, msgIndex = 0) {
  // Turn model-written citations like "(Source: manual.pdf, p. 326)" into
  // tappable buttons without changing the model answer stored in history.
  const sourcePattern = /\((Source:[^)]+|from [^)]+|same as above)\)/gi;
  let output = "";
  let cursor = 0;
  let lastCitation = null;
  for (const match of text.matchAll(sourcePattern)) {
    const matchedText = match[0];
    const innerText = match[1];
    const normalizedInner = innerText.toLowerCase().replace(/^source:\s*/, "").trim();
    const citation = normalizedInner === "same as above"
      ? lastCitation
      : matchCitationFromText(innerText, citations);
    output += escapeHtml(text.slice(cursor, match.index));
    output += citation ? citationButtonHtml(citation, innerText, msgIndex, "cite cite-text") : escapeHtml(matchedText);
    if (citation) lastCitation = citation;
    cursor = match.index + matchedText.length;
  }
  output += escapeHtml(text.slice(cursor));
  return output;
}

function matchCitationFromText(text, citations) {
  const normalizedText = normalizeCitationMatchText(text);
  const pages = [...text.matchAll(/\b(?:p|pp|page|pages)\.?\s*(\d+)/gi)].map((match) => Number(match[1]));
  return citations.find((citation) => {
    const file = normalizeCitationMatchText(citation.source_file || "");
    const label = normalizeCitationMatchText(citation.label || "");
    const fileMatches = (file && normalizedText.includes(file)) || (label && normalizedText.includes(label));
    const pageMatches = !pages.length || pages.includes(Number(citation.page_start));
    return fileMatches && pageMatches;
  }) || citations.find((citation) => {
    const fileStem = normalizeCitationMatchText((citation.source_file || "").replace(/\.pdf$/i, ""));
    return fileStem && normalizedText.includes(fileStem);
  }) || null;
}

function normalizeCitationMatchText(value) {
  const plusAsSpace = String(value || "").replace(/\+/g, " ");
  try {
    return decodeURIComponent(plusAsSpace).toLowerCase().replace(/[^a-z0-9]+/g, "");
  } catch {
    return plusAsSpace.toLowerCase().replace(/[^a-z0-9]+/g, "");
  }
}

function citationNumber(citation) {
  return String(citation.id || "").replace(/^cite_/, "");
}

function fileKind(citation) {
  const suffix = (citation.source_file || citation.source_path || "").split(".").pop()?.toLowerCase();
  if (suffix === "pdf") return "pdf";
  if (["png", "jpg", "jpeg", "webp", "tif", "tiff"].includes(suffix)) return "image";
  if (suffix === "xlsx") return "sheet";
  if (suffix === "pptx") return "slide";
  return "file";
}

function hasPageImage(citation) {
  return ["pdf", "image"].includes(fileKind(citation));
}

function unitLabel(citation) {
  return { sheet: "sheet", slide: "slide" }[fileKind(citation)] || "page";
}

function pageImageUrl(citation, page, width) {
  return `/documents/page.png?source_path=${encodeURIComponent(citation.source_path)}&page=${page}&w=${width}`;
}

function isActiveSource(citation) {
  const active = state.activeCitation;
  return Boolean(active && active.source_path === citation.source_path && active.page_start === citation.page_start);
}

function renderSourceCards(citations, msgIndex) {
  // Each source as a card with a thumbnail of the cited page, so a technician
  // can see at a glance whether it is a table, a drawing or prose.
  if (!citations.length) return "";
  return `
    <div class="sources">
      ${citations.map((citation) => `
        <button class="source ${isActiveSource(citation) ? "on" : ""}" data-msg="${msgIndex}" data-citation="${escapeAttr(JSON.stringify(citation))}">
          ${hasPageImage(citation)
            ? `<img src="${pageImageUrl(citation, citation.page_start || 1, 120)}" alt="" loading="lazy" />`
            : `<span class="thumb"></span>`}
          <span>
            <b>[${escapeHtml(citationNumber(citation))}] ${escapeHtml(citation.source_file || "Source")}</b>
            <small>${unitLabel(citation)} ${escapeHtml(citation.page_start ?? "")}${citation.source_scope === "general" ? " &middot; general library" : ""}</small>
          </span>
        </button>
      `).join("")}
    </div>
  `;
}

function citationButtonHtml(citation, label, msgIndex = 0, className = "cite") {
  return `<button class="${className}" data-msg="${msgIndex}" data-citation="${escapeAttr(JSON.stringify(citation))}" title="${escapeAttr(citation.label || "")}">${escapeHtml(label)}</button>`;
}

function renderRelatedAnswer(related) {
  // Shown BESIDE a real answer, never instead of one. Collapsed by default so
  // it is available without competing with the answer the technician asked for.
  if (!related) return "";
  return `
    <details class="related-answer">
      <summary>A similar question was answered before on this machine</summary>
      <p class="muted">Previous question: ${escapeHtml(related.question)}</p>
      <div class="answer-body">${renderMarkdown(escapeHtml(related.answer_preview))}${related.answer_preview.length >= 700 ? "..." : ""}</div>
      <p class="muted">Match score: ${related.match_score}</p>
    </details>
  `;
}

// Mirrors FEEDBACK_REASONS in app/db/feedback.py. Duplicated rather than
// fetched: it is five fixed strings, and an extra round trip before the panel
// can open would make a one-second interaction feel like a page load. The
// server is still the authority - it drops any code it does not recognise.
const FEEDBACK_REASONS = {
  wrong_info: "Information is wrong",
  not_in_answer: "It is in the manual but the answer missed it",
  too_vague: "Too vague to act on",
  wrong_machine: "Not about this machine",
  other: "Something else",
};

function renderFeedback(message) {
  // Answers restored from history before this feature shipped have no id, and
  // there is nothing to attach a rating to.
  if (!message.answer_id) return "";

  const feedback = message.feedback || null;
  const rating = feedback?.rating || null;
  const reasons = new Set(feedback?.reasons || []);

  const thumb = (value, glyph, label) => `
    <button class="feedback-thumb ${value} ${rating === value ? "active" : ""}"
            data-feedback-rating="${value}"
            aria-pressed="${rating === value}"
            title="${label}">${glyph}</button>
  `;

  // The detail panel only exists for a downvote, and only until it is sent. An
  // upvote needs no follow-up question - asking one would tax the one action we
  // most want technicians to keep doing.
  const detail = rating === "down" && message.feedbackDetailOpen ? `
    <div class="feedback-detail">
      <p class="muted">What was wrong? Optional - the thumbs down is already saved.</p>
      <div class="feedback-reasons">
        ${Object.entries(FEEDBACK_REASONS).map(([code, label]) => `
          <button class="feedback-reason ${reasons.has(code) ? "active" : ""}"
                  data-feedback-reason="${code}">${escapeHtml(label)}</button>
        `).join("")}
      </div>
      <textarea class="feedback-comment" data-feedback-comment
                placeholder="Anything else worth knowing? (optional)"
                rows="2">${escapeHtml(message.feedbackDraft ?? feedback?.comment ?? "")}</textarea>
      <button class="ghost feedback-send" data-feedback-send>Send</button>
    </div>
  ` : "";

  return `
    <div class="answer-feedback" data-answer-id="${message.answer_id}">
      <div class="feedback-row">
        <span class="muted">${rating ? "Thanks - recorded." : "Was this helpful?"}</span>
        ${thumb("up", icon("up"), "Helpful")}
        ${thumb("down", icon("down"), "Not helpful")}
      </div>
      ${detail}
    </div>
  `;
}

function messageByAnswerId(answerId) {
  return machineMessages().find((message) => String(message.answer_id) === String(answerId));
}

function repaintFeedback(answerId) {
  // A targeted repaint rather than renderChat(). A full rerender rebuilds the
  // composer, which would throw away a half-typed question, and scrolls the
  // transcript to the bottom - both unacceptable side effects of clicking a
  // thumb on an answer further up the page.
  const container = document.querySelector(`.answer-feedback[data-answer-id="${answerId}"]`);
  const message = messageByAnswerId(answerId);
  if (container && message) container.outerHTML = renderFeedback(message);
}

async function sendFeedback(message) {
  // Failures are deliberately quiet. A rating is a courtesy the technician is
  // doing us; interrupting their actual job with an error about it would be a
  // bad trade, and the answer itself is unaffected either way.
  //
  // What is NOT acceptable is showing "Thanks - recorded" when nothing was
  // recorded. The reply's `stored` flag is checked as well as the status
  // code, and the thumb is put back if the rating did not land.
  try {
    const result = await api("/chat/feedback", {
      method: "POST",
      body: JSON.stringify({
        answer_id: message.answer_id,
        rating: message.feedback.rating,
        reasons: message.feedback.reasons || [],
        comment: message.feedback.comment || null,
      }),
    });
    if (result?.stored === false) throw new Error(result.reason || "not stored");
  } catch (error) {
    console.warn("Feedback was not recorded:", error.message);
    message.feedback = null;
    message.feedbackDetailOpen = false;
    repaintFeedback(message.answer_id);
  }
}

function wireFeedback() {
  const container = document.querySelector("#messages");
  if (!container) return;

  // Delegated, so the controls survive repaintFeedback() replacing their own
  // markup out from under them.
  container.addEventListener("click", async (event) => {
    const block = event.target.closest(".answer-feedback");
    if (!block) return;
    const answerId = block.dataset.answerId;
    const message = messageByAnswerId(answerId);
    if (!message) return;

    const thumb = event.target.closest("[data-feedback-rating]");
    if (thumb) {
      const rating = thumb.dataset.feedbackRating;
      // Changing your mind replaces the rating and drops the reasons with it -
      // they described the old verdict and would otherwise be carried onto a
      // rating that contradicts them.
      const changed = message.feedback?.rating !== rating;
      message.feedback = changed
        ? { rating, reasons: [], comment: null }
        : message.feedback;
      if (changed) message.feedbackDraft = "";
      message.feedbackDetailOpen = rating === "down";
      repaintFeedback(answerId);
      await sendFeedback(message);
      return;
    }

    const reason = event.target.closest("[data-feedback-reason]");
    if (reason) {
      const code = reason.dataset.feedbackReason;
      const current = new Set(message.feedback?.reasons || []);
      current.has(code) ? current.delete(code) : current.add(code);
      message.feedback = { ...message.feedback, reasons: [...current] };
      repaintFeedback(answerId);
      // Saved on each toggle, not on Send: most people never press Send, and
      // the reasons are the part worth counting.
      await sendFeedback(message);
      return;
    }

    if (event.target.closest("[data-feedback-send]")) {
      message.feedback = { ...message.feedback, comment: message.feedbackDraft || null };
      message.feedbackDetailOpen = false;
      repaintFeedback(answerId);
      await sendFeedback(message);
    }
  });

  container.addEventListener("input", (event) => {
    const box = event.target.closest("[data-feedback-comment]");
    if (!box) return;
    const message = messageByAnswerId(event.target.closest(".answer-feedback").dataset.answerId);
    // Held on the message so a repaint mid-typing does not lose the text.
    if (message) message.feedbackDraft = box.value;
  });
}

async function submitQuestion(event) {
  event.preventDefault();
  const textarea = document.querySelector("#question");
  const question = textarea.value.trim();
  if (!question) return;
  machineMessages().push({ role: "user", content: question });
  textarea.value = "";
  renderChat();
  await askBackend({ question });
}

async function askBackend({ question, intent = "question" }) {
  const activeContext = getActiveContext();
  clearActivity();
  pushActivity("progress", "Starting...");
  renderChat();

  try {
    // Send enough context for follow-ups without resending the whole chat.
    const requestBody = JSON.stringify({
      machine_number: state.activeMachine.machine_number,
      question,
      session_id: getSessionId(),
      context_mode: activeContext ? "follow_up" : "new_question",
      intent,
      active_citations: activeContext?.citations || [],
      conversation_tail: machineMessages()
        .filter((message) => message.role === "user" || message.role === "assistant")
        .slice(-6)
        // Follow-up context must not degrade to summaries: where an answer
        // has a stored full body, that is what the model reads back.
        .map((message) => ({ role: message.role, content: message.full_content || message.content })),
    });

    const payload = await streamAsk(requestBody);
    setSessionId(payload.session_id);
    // Read the trail BEFORE clearing it - it is being moved onto the message,
    // not thrown away, and clearing first left every answer with an empty one.
    const trail = getActivity()?.items || [];
    clearActivity();
    resetStreamingState();

    machineMessages().push({
      role: "assistant",
      content: payload.answer,
      full_content: payload.full_answer || null,
      has_full_detail: Boolean(payload.has_full_detail),
      detail_open: false,
      citations: payload.citations || [],
      source_badges: payload.source_badges || [],
      offers_detail: Boolean(payload.offers_detail),
      activity: trail,
      related_previous_answer: payload.related_previous_answer || null,
      // What a thumbs up/down gets attached to. Without it the answer cannot
      // be rated at all - see renderFeedback().
      answer_id: payload.answer_id || null,
      feedback: null,
    });
    state.scrollToNewestAnswer = true;
    setActiveContext(payload.citations || []);
  } catch (error) {
    clearActivity();
    resetStreamingState();
    if (error.status === 401) {
      // The session expired mid-shift. A "something went wrong" message would
      // leave the technician retyping the question at a dead session - send
      // them to login instead, same as the boot path.
      window.location.href = "/login";
      return;
    }
    machineMessages().push({ role: "system", content: `Something went wrong: ${error.message}` });
  }
  renderChat();
}

async function streamAsk(requestBody) {
  // Reads the backend's live progress events (retrieval pass, judge check,
  // writing the answer, etc.) off the wire as they actually happen, instead
  // of faking a status message on a timer. Ends with a "result" event that
  // has the same shape /chat/ask used to return directly.
  const response = await fetch("/chat/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: requestBody,
  });
  if (!response.ok || !response.body) {
    const text = await response.text();
    const error = new Error(text || `Request failed: ${response.status}`);
    // Carried so the ask handler can tell "your session is gone" (bounce to
    // login) apart from a pipeline failure (render as a system message).
    error.status = response.status;
    throw error;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; a frame may arrive split
    // across multiple chunks, so only consume complete frames from the buffer.
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const dataLine = frame.split("\n").find((line) => line.startsWith("data: "));
      if (!dataLine) continue;

      const event = JSON.parse(dataLine.slice("data: ".length));
      if (event.type === "progress") {
        pushActivity(event.stage || "progress", event.message);
        renderChat();
      } else if (event.type === "reasoning") {
        pushActivity("reasoning", event.text);
        renderChat();
      } else if (event.type === "answer_mode") {
        // run_full_search executed: the bubble now waits for the summary and
        // everything streamed (so far and from here on) shows in the panel.
        enterFullSearchMode();
        renderChat();
      } else if (event.type === "answer_discard") {
        // That text was superseded - pre-tool commentary, or a draft the
        // verification pass told the model to rewrite from scratch.
        clearStreamingAnswer();
        clearPanelStream();
        renderChat();
      } else if (event.type === "answer_delta") {
        appendStreamingAnswer(event.text);
        renderChat();
      } else if (event.type === "error") {
        throw new Error(event.message);
      } else if (event.type === "result") {
        return event;
      }
    }
  }
  throw new Error("Stream ended without a result.");
}

// ---- Source viewer -----------------------------------------------------------
// Opened from a citation chip or source card: the cited page beside the chat on
// a wide screen, or as a full sheet on a tablet. Every source of that answer is
// a tab, and Prev/Next pages through the document from the cited page.

function openCitation(citation, sources) {
  state.activeCitation = citation;
  state.panelSources = sources?.length ? sources : [citation];
  state.viewerPage = Number(citation.page_start) || 1;
  state.viewerPageCount = null;
  state.citationTextPreview = null;
  renderChat();
  loadViewerInfo(citation);
  if (!hasPageImage(citation)) loadViewerText();
}

function closeViewer() {
  state.activeCitation = null;
  state.panelSources = [];
  state.viewerPageCount = null;
  state.citationTextPreview = null;
}

async function loadViewerInfo(citation) {
  try {
    const info = await api(`/documents/info?source_path=${encodeURIComponent(citation.source_path)}`);
    // Ignore a late reply for a document the reader has already left.
    if (state.activeCitation?.source_path !== citation.source_path) return;
    state.viewerPageCount = info.page_count || null;
    renderChat();
  } catch {
    /* Without a count, Next simply stays enabled. */
  }
}

async function loadViewerText() {
  // Sheets and slides: a rendered image when the server can make one,
  // otherwise the extracted text, so the panel never just breaks.
  const citation = state.activeCitation;
  const page = state.viewerPage;
  try {
    const preview = await api("/page-preview", {
      method: "POST",
      body: JSON.stringify({ source_path: citation.source_path, page_number: page }),
    });
    if (state.activeCitation !== citation || state.viewerPage !== page) return;
    state.citationTextPreview = preview.kind === "image"
      ? { image: `data:${preview.media_type || "image/png"};base64,${preview.content}` }
      : { text: preview.content || "(Nothing on this page.)" };
  } catch (error) {
    state.citationTextPreview = { text: `Preview not available: ${error.message}` };
  }
  renderChat();
}

function goToPage(page) {
  const last = state.viewerPageCount || Infinity;
  const next = Math.min(Math.max(1, page), last);
  if (!state.activeCitation || next === state.viewerPage) return;
  state.viewerPage = next;
  state.citationTextPreview = null;
  renderChat();
  if (!hasPageImage(state.activeCitation)) loadViewerText();
}

function renderSourcesPanel() {
  const citation = state.activeCitation;
  const page = state.viewerPage;
  const count = state.viewerPageCount;
  const unit = unitLabel(citation);
  const tabs = state.panelSources.map((source) => `
    <button class="panel-tab ${isActiveSource(source) ? "on" : ""}" data-panel-tab="${escapeAttr(JSON.stringify(source))}">
      ${escapeHtml(citationNumber(source))} &middot; ${escapeHtml(truncate(source.source_file || "Source", 26))}
    </button>
  `).join("");

  let body;
  if (hasPageImage(citation)) {
    body = `<img src="${pageImageUrl(citation, page, 1100)}" alt="${escapeAttr(`${citation.source_file}, ${unit} ${page}`)}" />`;
  } else if (state.citationTextPreview?.image) {
    body = `<img src="${state.citationTextPreview.image}" alt="${escapeAttr(`${citation.source_file}, ${unit} ${page}`)}" />`;
  } else if (state.citationTextPreview?.text) {
    body = `<pre class="page-text">${escapeHtml(state.citationTextPreview.text)}</pre>`;
  } else {
    body = `<p class="empty">Loading ${unit}...</p>`;
  }

  return `
    <aside class="panel" aria-label="Source viewer">
      <div class="panel-head">
        ${tabs}
        <span class="grow"></span>
        <button class="iconbtn" id="close-viewer" aria-label="Close">${icon("close", 20)}</button>
      </div>
      <div class="panel-body">${body}</div>
      <div class="panel-foot">
        <button class="iconbtn" id="viewer-prev" ${page <= 1 ? "disabled" : ""}>${icon("prev")} Prev</button>
        <span id="viewer-position">${unit[0].toUpperCase() + unit.slice(1)} ${page}${count ? ` of ${count}` : ""}${page === Number(citation.page_start) ? " &middot; cited" : ""}</span>
        <button class="iconbtn" id="viewer-open">${icon("open", 16)} Open</button>
        <button class="iconbtn" id="viewer-next" ${count && page >= count ? "disabled" : ""}>Next ${icon("next")}</button>
      </div>
    </aside>
  `;
}

function truncate(text, length) {
  return text.length > length ? `${text.slice(0, length - 1)}\u2026` : text;
}

function wireArtifactButtons() {
  document.querySelector("#close-viewer")?.addEventListener("click", () => {
    closeViewer();
    renderChat();
  });
  document.querySelector("#viewer-prev")?.addEventListener("click", () => goToPage(state.viewerPage - 1));
  document.querySelector("#viewer-next")?.addEventListener("click", () => goToPage(state.viewerPage + 1));
  document.querySelector("#viewer-open")?.addEventListener("click", () => openPdf(state.activeCitation, state.viewerPage));
  document.querySelectorAll("[data-panel-tab]").forEach((tab) => {
    tab.addEventListener("click", () => openCitation(JSON.parse(tab.dataset.panelTab), state.panelSources));
  });
}

// Keyboard: Escape closes the viewer, arrow keys turn pages - unless the reader
// is typing, where arrows belong to the text box.
document.addEventListener("keydown", (event) => {
  if (!state.activeCitation || state.currentView !== "chat") return;
  if (event.target.closest?.("textarea, input")) return;
  if (event.key === "Escape") {
    closeViewer();
    renderChat();
  } else if (event.key === "ArrowLeft") {
    goToPage(state.viewerPage - 1);
  } else if (event.key === "ArrowRight") {
    goToPage(state.viewerPage + 1);
  }
});

async function openPdf(citation, page = null) {
  // Create the tab during the tap/click event so mobile browsers do not treat
  // the eventual navigation as a blocked popup. Do not use "noopener" here:
  // some browsers return null for noopener windows, which would make the app
  // tab navigate to the document instead of the new tab.
  const pdfWindow = window.open("", "_blank");
  if (pdfWindow) {
    pdfWindow.document.write("<p>Opening document...</p>");
  }
  try {
    const payload = await api("/pdf-link", {
      method: "POST",
      body: JSON.stringify({ source_path: citation.source_path }),
    });
    if (!pdfWindow) throw new Error("The browser blocked the new tab. Allow popups for this app and try again.");
    // #page= is honoured by browser PDF viewers and harmless elsewhere.
    pdfWindow.location.href = payload.url + (page && fileKind(citation) === "pdf" ? `#page=${page}` : "");
  } catch (error) {
    if (pdfWindow) pdfWindow.close();
    machineMessages().push({ role: "system", content: `Could not open the document: ${error.message}` });
    renderChat();
  }
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("`", "&#96;");
}

// Theming. Light is the default; the choice persists per browser and applies
// before first paint (the stylesheet keys everything off html[data-theme]).
const THEME_KEY = "imm_theme";

function storedTheme() {
  try {
    return localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light";
  } catch (err) {
    return "light";
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch (err) {
    /* Embedded webviews may block storage; the attribute still applies. */
  }
}

function themeToggleHtml() {
  const label = storedTheme() === "dark" ? "Light mode" : "Dark mode";
  return `<button class="iconbtn theme-toggle" type="button" title="${label}" aria-label="${label}">${icon("moon")}</button>`;
}

function wireThemeToggle() {
  document.querySelectorAll(".theme-toggle").forEach((button) => {
    button.addEventListener("click", () => {
      const next = storedTheme() === "dark" ? "light" : "dark";
      applyTheme(next);
      const label = next === "dark" ? "Light mode" : "Dark mode";
      button.title = label;
      button.setAttribute("aria-label", label);
    });
  });
}

applyTheme(storedTheme());

loadMachines().catch((error) => {
  app.innerHTML = `<main class="screen"><h1>Could not load machines</h1><p>${escapeHtml(error.message)}</p></main>`;
});
