const API_BASE = "/api";

const state = {
  data: null,
  progressTimer: null,
};

const axisLabels = {
  issue_clarification: "論点整理",
  decision_progress: "意思決定",
  risk_detection: "リスク",
  actionability: "行動化",
  groundedness: "根拠性",
  novelty: "新規性",
  summarization: "要約",
};

const penaltyLabels = {
  duplication: "重複",
  verbosity: "冗長",
  off_topic: "脱線",
  unsupported_assertion: "根拠不足",
};

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function scoreBadge(score) {
  let cls = "low";
  if (score >= 6) cls = "high";
  else if (score >= 3) cls = "mid";
  else if (score < 0) cls = "negative";
  return `<span class="score ${cls}">${escapeHtml(score)}</span>`;
}

function renderEmpty() {
  return byId("emptyTemplate").innerHTML;
}

function renderUtterance(u, options = {}) {
  const scoreItems = Object.entries(u.scores || {})
    .filter(([, value]) => value > 0)
    .map(([key, value]) => `<span class="chip">${axisLabels[key] || key}: ${value}</span>`)
    .join("");

  const penaltyItems = Object.entries(u.penalties || {})
    .filter(([, value]) => value < 0)
    .map(([key, value]) => `<span class="chip penalty">${penaltyLabels[key] || key}: ${value}</span>`)
    .join("");

  return `
    <article class="utterance ${options.highlight ? "highlight" : ""}">
      <div class="utterance-head">
        <span class="speaker">${escapeHtml(u.speaker)}</span>
        <span class="meta">${escapeHtml(u.timestamp)}</span>
        <span class="type-label">${escapeHtml(u.speech_type)}</span>
        ${scoreBadge(u.total_score)}
      </div>
      <p class="utterance-text">${escapeHtml(u.text)}</p>
      <p class="reason">${escapeHtml(u.reason)}</p>
      <div class="scores-grid">${scoreItems}${penaltyItems}</div>
    </article>
  `;
}

function renderTopList(container, items) {
  container.innerHTML = items.length ? items.map((item) => renderUtterance(item)).join("") : renderEmpty();
}

async function loadSamples() {
  try {
    const response = await fetch(`${API_BASE}/samples`);
    const samples = await response.json();
    byId("sampleCount").textContent = `${samples.length}件`;
    byId("sampleList").innerHTML = samples.length
      ? samples
          .map(
            (sample) =>
              `<button class="button button-ghost" type="button" data-sample="${escapeHtml(sample.filename)}">${escapeHtml(
                sample.filename.replace(".json", "")
              )}</button>`
          )
          .join("")
      : '<p class="empty">サンプルがありません。</p>';
  } catch (error) {
    byId("sampleCount").textContent = "失敗";
    byId("sampleList").innerHTML = '<p class="empty">サンプルを読み込めませんでした。</p>';
  }
}

function setupEvents() {
  byId("sampleList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-sample]");
    if (button) analyzeSample(button.dataset.sample);
  });

  byId("fileInput").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      await analyzeData(data);
    } catch (error) {
      alert(`JSONの読み込みに失敗しました: ${error.message}`);
      showInput();
    }
  });

  byId("resetBtn").addEventListener("click", () => {
    state.data = null;
    showInput();
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      showResultView(tab.dataset.tab);
    });
  });
}

async function analyzeSample(filename) {
  showLoading();
  try {
    const response = await fetch(`${API_BASE}/analyze/sample/${encodeURIComponent(filename)}`, { method: "POST" });
    await handleAnalysisResponse(response);
  } catch (error) {
    alert(`分析に失敗しました: ${error.message}`);
    showInput();
  }
}

async function analyzeData(data) {
  showLoading();
  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    await handleAnalysisResponse(response);
  } catch (error) {
    alert(`分析に失敗しました: ${error.message}`);
    showInput();
  }
}

async function handleAnalysisResponse(response) {
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Keep the HTTP fallback.
    }
    throw new Error(message);
  }

  state.data = await response.json();
  renderAll(state.data);
  showResults();
}

function showLoading() {
  byId("inputView").classList.add("hidden");
  byId("summaryView").classList.add("hidden");
  byId("timelineView").classList.add("hidden");
  byId("speakersView").classList.add("hidden");
  byId("tabBar").classList.add("hidden");
  byId("resetBtn").classList.add("hidden");
  byId("loadingView").classList.remove("hidden");
  animateProgress();
}

function showInput() {
  clearInterval(state.progressTimer);
  byId("inputView").classList.remove("hidden");
  byId("loadingView").classList.add("hidden");
  byId("summaryView").classList.add("hidden");
  byId("timelineView").classList.add("hidden");
  byId("speakersView").classList.add("hidden");
  byId("tabBar").classList.add("hidden");
  byId("resetBtn").classList.add("hidden");
}

function showResults() {
  clearInterval(state.progressTimer);
  byId("loadingView").classList.add("hidden");
  byId("tabBar").classList.remove("hidden");
  byId("resetBtn").classList.remove("hidden");
  showResultView("summary");
}

function showResultView(name) {
  ["summary", "timeline", "speakers"].forEach((view) => {
    byId(`${view}View`).classList.toggle("hidden", view !== name);
  });
}

function animateProgress() {
  const messages = [
    "会議データを読み込んでいます...",
    "前後の文脈を組み立てています...",
    "発言ごとの評価を行っています...",
    "スコアを集計しています...",
    "表示用のサマリーを整えています...",
  ];
  let index = 0;
  byId("progressText").textContent = messages[index];
  clearInterval(state.progressTimer);
  state.progressTimer = setInterval(() => {
    index = (index + 1) % messages.length;
    byId("progressText").textContent = messages[index];
  }, 2600);
}

function renderAll(data) {
  renderSummary(data);
  renderTimeline(data);
  renderSpeakers(data);
}

function renderSummary(data) {
  byId("meetingTitle").textContent = data.title;
  byId("meetingGoal").textContent = `目的: ${data.goal}`;
  byId("overallComment").textContent = data.overall_comment;

  renderTopList(byId("topUtterances"), data.top_utterances || []);
  renderTopList(byId("topIssue"), data.top_issue_clarification || []);
  renderTopList(byId("topDecision"), data.top_decision_progress || []);
  renderTopList(byId("topRisk"), data.top_risk_detection || []);
  renderTopList(byId("topAction"), data.top_actionability || []);

  const comments = data.improvement_comments || [];
  byId("improvementSection").classList.toggle("hidden", comments.length === 0);
  byId("improvementComments").innerHTML = comments
    .map((comment) => `<div class="improvement-item">${escapeHtml(comment)}</div>`)
    .join("");
}

function renderTimeline(data) {
  const utterances = data.evaluated_utterances || [];
  const maxScore = Math.max(...utterances.map((u) => u.total_score), 1);
  byId("timelineList").innerHTML = utterances.length
    ? utterances
        .map((u) => renderUtterance(u, { highlight: u.total_score >= maxScore * 0.7 && u.total_score > 3 }))
        .join("")
    : renderEmpty();
}

function renderSpeakers(data) {
  const utteranceMap = Object.fromEntries((data.evaluated_utterances || []).map((u) => [u.utterance_id, u]));
  const speakers = data.speaker_summaries || [];

  byId("speakersList").innerHTML = speakers.length
    ? speakers
        .map((speaker) => {
          const topUtterances = (speaker.top_utterances || [])
            .map((id) => utteranceMap[id])
            .filter(Boolean)
            .map((utterance) => renderUtterance(utterance))
            .join("");

          return `
            <section class="speaker-panel">
              <div class="speaker-head">
                <h3>${escapeHtml(speaker.speaker)}</h3>
                <span class="style-label">${escapeHtml(speaker.style_label)}</span>
              </div>
              <div class="speaker-stats">
                <span class="chip">発言 ${speaker.utterance_count}件</span>
                <span class="chip">合計 ${speaker.total_contribution_score}</span>
                <span class="chip">平均 ${speaker.average_total_score}</span>
              </div>
              <div class="bars">${renderBars(speaker.average_scores || {})}</div>
              <div class="top-list">${topUtterances || renderEmpty()}</div>
            </section>
          `;
        })
        .join("")
    : renderEmpty();
}

function renderBars(scores) {
  return Object.entries(axisLabels)
    .map(([key, label]) => {
      const value = Number(scores[key] || 0);
      const width = Math.max(0, Math.min(100, (value / 3) * 100));
      return `
        <div class="bar-row">
          <div class="bar-label">${label}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <div class="bar-value">${value.toFixed(1)}</div>
        </div>
      `;
    })
    .join("");
}

loadSamples();
setupEvents();
