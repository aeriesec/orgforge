const state = {
  config: null,
  overview: null,
  collections: [],
  activeView: "overview",
  activeCollection: "artifacts",
  activeDocument: null,
  activeFile: null,
  activeTicket: null,
  activeAccount: null,
  caches: {},
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function formatMoney(value) {
  return Number(value || 0).toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function truncate(value, limit = 160) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return text.slice(0, limit - 1).trim() + "...";
}

function showError(message) {
  const banner = $("#errorBanner");
  banner.textContent = message;
  banner.classList.toggle("hidden", !message);
}

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  }
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function metric(label, value) {
  return `
    <div class="metric">
      <span>${escapeHtml(label)}</span>
      <strong>${formatCount(value)}</strong>
    </div>
  `;
}

function miniMetric(label, value) {
  return `
    <div class="mini-metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `;
}

function chips(items) {
  return (items || [])
    .filter((item) => item !== undefined && item !== null && item !== "")
    .map((item) => `<span>${escapeHtml(item)}</span>`)
    .join("");
}

function recordButton(record, index, type, extraClass = "") {
  const meta = chips(record.meta || []);
  return `
    <button class="record ${extraClass}" data-index="${index}" data-type="${type}">
      <p class="record-title">${escapeHtml(record.label || record.name || record.path)}</p>
      <p class="record-summary">${escapeHtml(record.summary || record.snippet || record.path || "")}</p>
      <div class="record-meta">${meta}</div>
    </button>
  `;
}

function switchView(view) {
  state.activeView = view;
  $$(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  $$(".view").forEach((section) => section.classList.remove("active-view"));
  $(`#${view}View`).classList.add("active-view");
  $("#viewTitle").textContent =
    {
      overview: "Overview",
      inbox: "Inbox",
      slack: "Slack",
      jira: "Jira",
      docs: "Docs & Meetings",
      crm: "CRM",
      org: "Org State",
      timeline: "Timeline",
      raw: "Raw Data",
      search: "Search",
    }[view] || "Overview";
  loadActiveView().catch((error) => showError(error.message));
}

function renderOverview() {
  const overview = state.overview;
  if (!overview) return;
  const metrics = overview.metrics || {};
  $("#metricGrid").innerHTML = [
    metric("Artifacts", metrics.artifacts),
    metric("Events", metrics.events),
    metric("Jira Tickets", metrics.jira_tickets),
    metric("Slack Messages", metrics.slack_messages),
    metric("Emails", metrics.emails),
    metric("Export Files", metrics.files),
  ].join("");

  $("#appCards").innerHTML = [
    ["inbox", "Inbox", "Threaded email conversations from customers, vendors, and internal replies."],
    ["slack", "Slack", "Channels, DMs, threads, bot messages, and replies in conversation order."],
    ["jira", "Jira", "Ticket board grouped by workflow state, with department and artifact context."],
    ["docs", "Docs & Meetings", "Confluence pages, Zoom transcripts, and knowledge-domain coverage."],
    ["crm", "CRM", "Salesforce accounts, opportunities, touchpoints, and related customer email."],
    ["timeline", "Timeline", "Ground-truth event log for causal checks and agent evaluation."],
  ]
    .map(
      ([target, title, summary]) => `
        <button class="app-card" data-target="${target}">
          <strong>${escapeHtml(title)}</strong>
          <span>${escapeHtml(summary)}</span>
        </button>
      `
    )
    .join("");
  $$("#appCards .app-card").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.target));
  });

  const collections = overview.collections || [];
  $("#collectionCount").textContent = `${collections.length} collections`;
  $("#collectionInventory").innerHTML =
    collections
      .map(
        (collection) => `
          <div class="inventory-row">
            <div>
              <strong>${escapeHtml(collection.name)}</strong>
              <div class="muted">${escapeHtml((collection.fields || []).join(", "))}</div>
            </div>
            <span class="count-pill">${formatCount(collection.count)}</span>
          </div>
        `
      )
      .join("") || `<div class="empty-state">No collections found.</div>`;

  const labels = {
    event_types: "Event Types",
    artifact_types: "Artifact Types",
    jira_status: "Jira Status",
    slack_channels: "Slack Channels",
    email_direction: "Email Direction",
    file_kinds: "File Kinds",
  };
  $("#breakdownGrid").innerHTML = Object.entries(labels)
    .map(([key, title]) => breakdown(title, overview.breakdowns?.[key] || []))
    .join("");
}

function breakdown(title, rows) {
  const max = Math.max(...rows.map((row) => row.count), 1);
  const body =
    rows
      .slice(0, 8)
      .map((row) => {
        const width = Math.max(4, Math.round((row.count / max) * 100));
        return `
          <div class="bar-row">
            <span class="bar-label" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</span>
            <span class="bar-track"><span class="bar-fill" style="width:${width}%"></span></span>
            <strong>${formatCount(row.count)}</strong>
          </div>
        `;
      })
      .join("") || `<div class="muted">No data</div>`;
  return `
    <div class="breakdown">
      <h4>${escapeHtml(title)}</h4>
      ${body}
    </div>
  `;
}

async function loadInbox() {
  const data = await api("/api/app/inbox", { q: $("#inboxQuery").value.trim() });
  state.caches.inbox = data;
  $("#inboxMeta").textContent = `${formatCount(data.total_threads)} threads · ${formatCount(data.total_messages)} messages`;
  $("#inboxThreadList").innerHTML =
    data.threads
      .map((thread, index) =>
        recordButton(
          {
            label: thread.subject,
            summary: thread.preview,
            meta: [
              `${thread.count} messages`,
              thread.directions.join(" + "),
              formatDateTime(thread.last),
            ],
          },
          index,
          "email-thread"
        )
      )
      .join("") || `<div class="empty-state">No matching email threads.</div>`;
  $$("#inboxThreadList .record").forEach((button) => {
    button.addEventListener("click", () =>
      selectInboxThread(Number(button.dataset.index)).catch((error) => showError(error.message))
    );
  });
  if (data.threads[0]) selectInboxThread(0).catch((error) => showError(error.message));
}

async function selectInboxThread(index) {
  const thread = state.caches.inbox.threads[index];
  $$("#inboxThreadList .record").forEach((item) => item.classList.remove("active"));
  $$("#inboxThreadList .record")[index]?.classList.add("active");
  $("#inboxThreadTitle").textContent = thread.subject;
  $("#inboxThreadMeta").textContent = thread.participants.join(", ");
  $("#inboxMessages").innerHTML = `<div class="empty-state">Loading thread...</div>`;
  const detail = await api("/api/app/inbox-thread", { id: thread.id });
  $("#inboxMessages").innerHTML = detail.messages
    .map((message) => {
      const direction = message.direction === "outbound" ? "outbound" : message.direction === "internal" ? "bot" : "inbound";
      return `
        <article class="message ${direction}">
          <div class="message-header">
            <strong>${escapeHtml(message.from_name || message.from_addr)} → ${escapeHtml(message.to_name || message.to_addr)}</strong>
            <span>${escapeHtml(formatDateTime(message.timestamp))}</span>
          </div>
          <p class="message-subject">${escapeHtml(message.subject)}</p>
          <p class="message-body">${escapeHtml(message.body || "")}</p>
        </article>
      `;
    })
    .join("") || `<div class="empty-state">No messages in this thread.</div>`;
}

async function loadSlack() {
  const data = await api("/api/app/slack", { q: $("#slackQuery").value.trim() });
  state.caches.slack = data;
  $("#slackChannelMeta").textContent = `${formatCount(data.channels.length)} channels`;
  $("#slackChannelList").innerHTML =
    data.channels
      .map((channel, index) =>
        recordButton(
          {
            label: channel.kind === "dm" ? channel.name.replace(/^dm_/, "DM ") : `#${channel.name}`,
            summary: `${channel.thread_count} threads · ${channel.message_count} messages`,
            meta: [channel.kind, formatDateTime(channel.last)],
          },
          index,
          "slack-channel",
          "compact-record"
        )
      )
      .join("") || `<div class="empty-state">No Slack channels found.</div>`;
  $$("#slackChannelList .record").forEach((button) => {
    button.addEventListener("click", () =>
      selectSlackChannel(Number(button.dataset.index)).catch((error) => showError(error.message))
    );
  });
  if (data.channels[0]) selectSlackChannel(0).catch((error) => showError(error.message));
}

async function selectSlackChannel(index) {
  const channel = state.caches.slack.channels[index];
  state.caches.activeSlackChannel = channel;
  $$("#slackChannelList .record").forEach((item) => item.classList.remove("active"));
  $$("#slackChannelList .record")[index]?.classList.add("active");
  $("#slackThreadPanelTitle").textContent =
    channel.kind === "dm" ? channel.name.replace(/^dm_/, "DM ") : `#${channel.name}`;
  $("#slackThreadMeta").textContent = `${formatCount(channel.thread_count)} threads`;
  $("#slackThreadList").innerHTML = `<div class="empty-state">Loading threads...</div>`;
  $("#slackMessages").innerHTML = "";
  const detail = await api("/api/app/slack-channel", {
    channel: channel.name,
    q: $("#slackQuery").value.trim(),
  });
  channel.threads = detail.threads;
  $("#slackThreadList").innerHTML =
    channel.threads
      .map((thread, threadIndex) =>
        recordButton(
          {
            label: thread.preview || thread.id,
            summary: thread.participants.join(", "),
            meta: [`${thread.count} messages`, formatDateTime(thread.last)],
          },
          threadIndex,
          "slack-thread"
        )
      )
      .join("") || `<div class="empty-state">No threads in this channel.</div>`;
  $$("#slackThreadList .record").forEach((button) => {
    button.addEventListener("click", () =>
      selectSlackThread(Number(button.dataset.index)).catch((error) => showError(error.message))
    );
  });
  if (channel.threads[0]) selectSlackThread(0).catch((error) => showError(error.message));
}

async function selectSlackThread(index) {
  const thread = state.caches.activeSlackChannel.threads[index];
  $$("#slackThreadList .record").forEach((item) => item.classList.remove("active"));
  $$("#slackThreadList .record")[index]?.classList.add("active");
  $("#slackMessageTitle").textContent = truncate(thread.preview || thread.id, 72);
  $("#slackMessageMeta").textContent = `${thread.participants.join(", ")} · ${formatDateTime(thread.first)}`;
  $("#slackMessages").innerHTML = `<div class="empty-state">Loading messages...</div>`;
  const detail = await api("/api/app/slack-thread", {
    channel: thread.channel,
    id: thread.id,
  });
  $("#slackMessages").innerHTML = detail.messages
    .map((message) => `
      <article class="message ${message.is_bot ? "bot" : ""}">
        <div class="message-header">
          <strong>${escapeHtml(message.user || "unknown")}</strong>
          <span>${escapeHtml(formatDateTime(message.ts))}</span>
        </div>
        <p class="message-body">${escapeHtml(message.text || "")}</p>
      </article>
    `)
    .join("") || `<div class="empty-state">No messages in this thread.</div>`;
}

async function loadJira() {
  const data = await api("/api/app/jira", { q: $("#jiraQuery").value.trim() });
  state.caches.jira = data;
  $("#jiraSummary").innerHTML = [
    miniMetric("Tickets", formatCount(data.tickets.length)),
    miniMetric("Departments", formatCount(data.by_dept.length)),
    miniMetric("Workflow States", formatCount(data.columns.length)),
  ].join("");
  $("#jiraBoard").innerHTML = data.columns
    .map(
      (column, columnIndex) => `
        <section class="kanban-column">
          <div class="kanban-header">
            <h3>${escapeHtml(column.status)}</h3>
            <span class="count-pill">${formatCount(column.count)}</span>
          </div>
          <div class="kanban-cards">
            ${column.tickets
              .map((ticket, ticketIndex) => ticketCard(ticket, columnIndex, ticketIndex))
              .join("")}
          </div>
        </section>
      `
    )
    .join("");
  $$(".ticket-card").forEach((button) => {
    button.addEventListener("click", () =>
      selectTicket(Number(button.dataset.column), Number(button.dataset.ticket))
    );
  });
  if (data.columns[0]?.tickets[0]) selectTicket(0, 0);
}

function ticketCard(ticket, columnIndex, ticketIndex) {
  return `
    <button class="ticket-card" data-column="${columnIndex}" data-ticket="${ticketIndex}">
      <span class="ticket-id">${escapeHtml(ticket.id || ticket._id)}</span>
      <div class="ticket-title">${escapeHtml(ticket.title)}</div>
      <div class="ticket-footer">
        ${chips([
          ticket.dept || "vendor",
          ticket.assignee || "unassigned",
          ticket.story_points ? `${ticket.story_points} pts` : ticket.priority,
        ])}
      </div>
    </button>
  `;
}

function selectTicket(columnIndex, ticketIndex) {
  const ticket = state.caches.jira.columns[columnIndex].tickets[ticketIndex];
  state.activeTicket = ticket;
  $$(".ticket-card").forEach((item) => item.classList.remove("active"));
  $(`.ticket-card[data-column="${columnIndex}"][data-ticket="${ticketIndex}"]`)?.classList.add("active");
  $("#jiraDetailTitle").textContent = `${ticket.id || ticket._id}: ${ticket.title}`;
  $("#jiraDetail").textContent = JSON.stringify(ticket, null, 2);
}

async function loadDocs() {
  const data = await api("/api/app/docs", { q: $("#docsQuery").value.trim() });
  state.caches.docs = data;
  $("#docsSummary").innerHTML = [
    miniMetric("Confluence", formatCount(data.counts.confluence)),
    miniMetric("Meetings", formatCount(data.counts.meetings)),
    miniMetric("Domains", formatCount(data.counts.domains)),
  ].join("");
  $("#docsMeta").textContent = `${formatCount(data.documents.length)} documents`;
  $("#docsList").innerHTML =
    data.documents
      .map((doc, index) =>
        recordButton(
          {
            label: doc.title,
            summary: doc.type === "zoom_transcript" ? doc.participants.join(", ") : truncate(doc.content),
            meta: [doc.type === "zoom_transcript" ? "Zoom" : "Confluence", doc.author, formatDateTime(doc.timestamp)],
          },
          index,
          "doc"
        )
      )
      .join("") || `<div class="empty-state">No matching docs.</div>`;
  $$("#docsList .record").forEach((button) => {
    button.addEventListener("click", () => selectDoc(Number(button.dataset.index)));
  });
  renderDomains(data.domains || []);
  if (data.documents[0]) selectDoc(0);
}

function selectDoc(index) {
  const doc = state.caches.docs.documents[index];
  $$("#docsList .record").forEach((item) => item.classList.remove("active"));
  $$("#docsList .record")[index]?.classList.add("active");
  $("#docReaderTitle").textContent = doc.title;
  $("#docReaderMeta").textContent =
    doc.type === "zoom_transcript"
      ? `${doc.participants.join(", ")} · ${formatDateTime(doc.timestamp)}`
      : `${doc.author || "Unknown author"} · ${formatDateTime(doc.timestamp)}`;
  if (doc.type === "zoom_transcript") {
    $("#docReader").innerHTML = renderTranscript(doc.content);
  } else {
    $("#docReader").innerHTML = `<div class="markdown-preview">${renderMarkdown(doc.content || "")}</div>`;
  }
}

function renderTranscript(content) {
  const turns = String(content || "")
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const [speaker, ...rest] = line.split(":");
      return { speaker: speaker || "Speaker", text: rest.join(":").trim() || line };
    });
  return `
    <div class="message-stack">
      ${turns
        .map(
          (turn) => `
            <article class="message">
              <div class="message-header"><strong>${escapeHtml(turn.speaker)}</strong></div>
              <p class="message-body">${escapeHtml(turn.text)}</p>
            </article>
          `
        )
        .join("")}
    </div>
  `;
}

function renderDomains(domains) {
  $("#domainsMeta").textContent = `${formatCount(domains.length)} domains`;
  $("#domainGrid").innerHTML =
    domains
      .map(
        (domain) => `
          <article class="domain-card">
            <h4>${escapeHtml(domain.domain)}</h4>
            <p><strong>Owner:</strong> ${escapeHtml(domain.primary_owner || "unknown")}</p>
            <p><strong>Former owner:</strong> ${escapeHtml(domain.former_owner || "none")}</p>
            <p><strong>Coverage:</strong> ${escapeHtml(Math.round((domain.documentation_coverage || 0) * 100))}%</p>
            <div class="record-meta">${chips(domain.system_tags || [])}</div>
          </article>
        `
      )
      .join("") || `<div class="empty-state">No domains found.</div>`;
}

async function loadCrm() {
  const data = await api("/api/app/crm", { q: $("#crmQuery").value.trim() });
  state.caches.crm = data;
  $("#crmSummary").innerHTML = [
    miniMetric("Accounts", formatCount(data.totals.accounts)),
    miniMetric("ARR", formatMoney(data.totals.arr)),
    miniMetric("Opportunities", formatCount(data.totals.opportunities)),
    miniMetric("Pipeline", formatMoney(data.totals.pipeline)),
  ].join("");
  $("#crmMeta").textContent = `${formatCount(data.accounts.length)} accounts`;
  $("#accountList").innerHTML =
    data.accounts
      .map((account, index) =>
        recordButton(
          {
            label: account.name,
            summary: `${account.industry} · ${account.billing_region}`,
            meta: [account.owner, account.tier, formatMoney(account.arr)],
          },
          index,
          "account"
        )
      )
      .join("") || `<div class="empty-state">No matching accounts.</div>`;
  $$("#accountList .record").forEach((button) => {
    button.addEventListener("click", () => selectAccount(Number(button.dataset.index)));
  });
  if (data.accounts[0]) selectAccount(0);
}

function selectAccount(index) {
  const account = state.caches.crm.accounts[index];
  state.activeAccount = account;
  $$("#accountList .record").forEach((item) => item.classList.remove("active"));
  $$("#accountList .record")[index]?.classList.add("active");
  $("#accountDetailTitle").textContent = account.name;
  $("#accountDetailMeta").textContent = `${account.owner} · ${account.status}`;
  $("#accountDetail").innerHTML = `
    <section class="detail-section">
      <h4>Account</h4>
      <div class="field-grid">
        ${fieldRow("Owner", account.owner)}
        ${fieldRow("ARR", formatMoney(account.arr))}
        ${fieldRow("Tier", account.tier)}
        ${fieldRow("Region", account.billing_region)}
        ${fieldRow("Primary Contact", account.primary_contact_name)}
        ${fieldRow("Contact Email", account.primary_contact_email)}
      </div>
    </section>
    <section class="detail-section">
      <h4>Opportunities</h4>
      ${
        (account.opportunities || [])
          .map(
            (opp) => `
              <article class="domain-card">
                <h4>${escapeHtml(opp.opportunity_id)} · ${escapeHtml(opp.stage)}</h4>
                <p>${escapeHtml(opp.type)} · ${formatMoney(opp.amount)} · ${escapeHtml(opp.probability)}%</p>
                <p>${escapeHtml(opp.next_step || "")}</p>
              </article>
            `
          )
          .join("") || `<p class="muted">No opportunities.</p>`
      }
    </section>
    <section class="detail-section">
      <h4>Related Email</h4>
      ${
        (account.related_emails || [])
          .map(
            (email) => `
              <article class="domain-card">
                <h4>${escapeHtml(email.subject)}</h4>
                <p>${escapeHtml(email.from_name)} → ${escapeHtml(email.to_name)} · ${escapeHtml(formatDateTime(email.timestamp))}</p>
                <p>${escapeHtml(truncate(email.body, 240))}</p>
              </article>
            `
          )
          .join("") || `<p class="muted">No direct email touchpoints.</p>`
      }
    </section>
  `;
}

function fieldRow(label, value) {
  return `
    <div class="field-row">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value || "None")}</strong>
    </div>
  `;
}

async function loadOrg() {
  const data = await api("/api/app/org", { q: $("#orgQuery").value.trim() });
  state.caches.org = data;
  $("#orgSummary").innerHTML = [
    miniMetric("System Health", data.system_health ?? "n/a"),
    miniMetric("Departments", formatCount(data.dept_plans.length)),
    miniMetric("Knowledge Domains", formatCount(data.domains.length)),
    miniMetric("Departures", formatCount(data.departed_employees.length)),
  ].join("");
  $("#deptPlanMeta").textContent = `${formatCount(data.dept_plans.length)} plans`;
  $("#deptPlanList").innerHTML =
    data.dept_plans
      .map(
        (plan) => `
          <article class="record">
            <p class="record-title">${escapeHtml(plan.dept)} · ${escapeHtml(plan.theme)}</p>
            <p class="record-summary">Lead: ${escapeHtml(plan.lead)} · ${escapeHtml(formatDateTime(plan.timestamp))}</p>
            <div class="record-meta">${chips((plan.engineer_plans || []).map((person) => `${person.name}: ${person.focus_note}`))}</div>
          </article>
        `
      )
      .join("") || `<div class="empty-state">No department plans.</div>`;
  renderOrgInsights(data);
}

function renderOrgInsights(data) {
  $("#orgInsightGrid").innerHTML = [
    insightCard(
      "Highest Stress",
      (data.top_stress || [])
        .map((item) => `${escapeHtml(item.name)}: ${escapeHtml(item.stress)}`)
        .join("<br>")
    ),
    insightCard(
      "Strongest Relationships",
      (data.top_relationships || [])
        .map((row) => `${escapeHtml(row[0])} &harr; ${escapeHtml(row[1])}: ${escapeHtml(row[2])}`)
        .join("<br>")
    ),
    insightCard(
      "Estranged Pairs",
      (data.estranged_pairs || [])
        .map((row) => `${escapeHtml(row[0])} &harr; ${escapeHtml(row[1])}: ${escapeHtml(row[2])}`)
        .join("<br>")
    ),
    insightCard(
      "Departed Employees",
      (data.departed_employees || [])
        .map((person) => `${escapeHtml(person.name)}: ${escapeHtml((person.knowledge_domains || []).join(", "))}`)
        .join("<br>")
    ),
    insightCard(
      "Knowledge Domains",
      (data.domains || [])
        .map((domain) => `${escapeHtml(domain.domain)}: ${Math.round((domain.documentation_coverage || 0) * 100)}%`)
        .join("<br>")
    ),
    insightCard(
      "Morale",
      (data.morale_history || [])
        .map((value, index) => `Day ${index + 1}: ${escapeHtml(value)}`)
        .join("<br>")
    ),
  ].join("");
}

function insightCard(title, html) {
  return `
    <article class="insight-card">
      <h4>${escapeHtml(title)}</h4>
      <p>${html || "No data"}</p>
    </article>
  `;
}

async function loadTimeline() {
  const currentType = $("#timelineTypeSelect").value;
  const data = await api("/api/app/timeline", {
    q: $("#timelineQuery").value.trim(),
    type: currentType,
  });
  state.caches.timeline = data;
  const select = $("#timelineTypeSelect");
  select.innerHTML =
    `<option value="">All event types</option>` +
    data.types
      .map((type) => `<option value="${escapeHtml(type.label)}">${escapeHtml(type.label)} (${formatCount(type.count)})</option>`)
      .join("");
  select.value = currentType;
  $("#timelineList").innerHTML =
    data.days
      .map(
        (day) => `
          <section class="timeline-day">
            <h3>${escapeHtml(day.date)} · ${formatCount(day.events.length)} events</h3>
            ${day.events.map(timelineEvent).join("")}
          </section>
        `
      )
      .join("") || `<div class="empty-state">No matching events.</div>`;
}

function timelineEvent(event) {
  const factText = truncate(JSON.stringify(event.facts || {}), 360);
  return `
    <article class="timeline-event">
      <div class="timeline-time">
        <strong>${escapeHtml(formatDateTime(event.timestamp))}</strong>
        <div>${escapeHtml(event.type)}</div>
        <div>${escapeHtml((event.actors || []).join(", "))}</div>
      </div>
      <div>
        <strong>${escapeHtml(event.summary || event.type)}</strong>
        <p>${escapeHtml(factText)}</p>
      </div>
    </article>
  `;
}

function renderCollectionSelect() {
  const options = state.collections
    .map(
      (collection) =>
        `<option value="${escapeHtml(collection.name)}">${escapeHtml(collection.name)} (${formatCount(collection.count)})</option>`
    )
    .join("");
  $("#collectionSelect").innerHTML = options;
  if (state.collections.some((item) => item.name === state.activeCollection)) {
    $("#collectionSelect").value = state.activeCollection;
  } else if (state.collections[0]) {
    state.activeCollection = state.collections[0].name;
    $("#collectionSelect").value = state.activeCollection;
  }
}

async function loadDocuments() {
  const data = await api("/api/documents", {
    collection: state.activeCollection,
    q: $("#collectionQuery").value.trim(),
    limit: 80,
  });
  $("#documentsTitle").textContent = data.collection;
  $("#documentsMeta").textContent = `${formatCount(data.total)} matching`;
  $("#documentList").innerHTML =
    data.documents.map((doc, index) => recordButton(doc, index, "document")).join("") ||
    `<div class="empty-state">No matching documents.</div>`;
  $$("#documentList .record").forEach((button) => {
    button.addEventListener("click", () => {
      $$("#documentList .record").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const doc = data.documents[Number(button.dataset.index)];
      state.activeDocument = doc;
      $("#documentDetailTitle").textContent = doc.label;
      $("#documentDetail").textContent = JSON.stringify(doc.doc, null, 2);
    });
  });
  if (data.documents[0]) $("#documentList .record")?.click();
}

async function loadFiles() {
  const data = await api("/api/files", {
    q: $("#fileQuery").value.trim(),
    kind: $("#fileKindSelect").value,
    limit: 400,
  });
  const select = $("#fileKindSelect");
  const currentKind = select.value;
  select.innerHTML =
    `<option value="">All files</option>` +
    data.kinds
      .map((kind) => `<option value="${escapeHtml(kind)}">${escapeHtml(kind)}</option>`)
      .join("");
  select.value = currentKind;
  $("#filesMeta").textContent = `${formatCount(data.total)} matching`;
  $("#fileList").innerHTML =
    data.files
      .map((file, index) =>
        recordButton(
          {
            label: file.path,
            summary: `${file.kind} · ${formatBytes(file.size)}`,
            meta: [file.extension, file.modified],
          },
          index,
          "file"
        )
      )
      .join("") || `<div class="empty-state">No matching files.</div>`;
  $$("#fileList .record").forEach((button) => {
    button.addEventListener("click", async () => {
      $$("#fileList .record").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const file = data.files[Number(button.dataset.index)];
      await loadFile(file.path);
    });
  });
  if (data.files[0]) $("#fileList .record")?.click();
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function loadFile(path) {
  const file = await api("/api/file", { path });
  state.activeFile = file;
  $("#fileDetailTitle").textContent = file.path;
  if (file.binary) {
    $("#fileDetail").innerHTML = `<div class="empty-state">Binary file preview is not available.</div>`;
    return;
  }
  if (file.extension === ".json" && file.json) {
    $("#fileDetail").innerHTML = `<pre>${escapeHtml(JSON.stringify(file.json, null, 2))}</pre>`;
    return;
  }
  if (file.extension === ".md") {
    $("#fileDetail").innerHTML = `<div class="markdown-preview">${renderMarkdown(file.text)}</div>`;
    return;
  }
  $("#fileDetail").innerHTML = `<pre>${escapeHtml(file.text || "")}</pre>`;
}

function renderMarkdown(text) {
  const lines = String(text || "").split("\n");
  let html = "";
  let inList = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      continue;
    }
    if (trimmed.startsWith("### ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h3>${inlineMarkdown(trimmed.slice(4))}</h3>`;
    } else if (trimmed.startsWith("## ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h2>${inlineMarkdown(trimmed.slice(3))}</h2>`;
    } else if (trimmed.startsWith("# ")) {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<h1>${inlineMarkdown(trimmed.slice(2))}</h1>`;
    } else if (trimmed.startsWith("- ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${inlineMarkdown(trimmed.slice(2))}</li>`;
    } else {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += `<p>${inlineMarkdown(trimmed)}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

async function runGlobalSearch() {
  const query = $("#globalSearchInput").value.trim();
  if (!query) {
    $("#searchResults").innerHTML = `<div class="empty-state">Enter a term to search.</div>`;
    return;
  }
  const data = await api("/api/search", { q: query, limit: 10 });
  const mongo = data.mongo || [];
  const files = data.files || [];
  $("#searchResults").innerHTML = `
    <section class="panel result-group">
      <div class="panel-header">
        <h3>MongoDB</h3>
        <span class="muted">${formatCount(mongo.length)} hits</span>
      </div>
      <div class="record-list">
        ${
          mongo
            .map((hit) =>
              recordButton({ ...hit, meta: [`collection: ${hit.collection}`, ...(hit.meta || [])] }, 0, "search")
            )
            .join("") || `<div class="empty-state">No MongoDB matches.</div>`
        }
      </div>
    </section>
    <section class="panel result-group">
      <div class="panel-header">
        <h3>Files</h3>
        <span class="muted">${formatCount(files.length)} hits</span>
      </div>
      <div class="record-list">
        ${
          files
            .map((file) =>
              recordButton(
                {
                  label: file.path,
                  summary: file.snippet || file.kind,
                  meta: [file.kind, file.extension, formatBytes(file.size)],
                },
                0,
                "search"
              )
            )
            .join("") || `<div class="empty-state">No file matches.</div>`
        }
      </div>
    </section>
  `;
}

async function loadActiveView() {
  showError("");
  if (state.activeView === "inbox") return loadInbox();
  if (state.activeView === "slack") return loadSlack();
  if (state.activeView === "jira") return loadJira();
  if (state.activeView === "docs") return loadDocs();
  if (state.activeView === "crm") return loadCrm();
  if (state.activeView === "org") return loadOrg();
  if (state.activeView === "timeline") return loadTimeline();
  if (state.activeView === "raw") {
    await loadDocuments();
    return loadFiles();
  }
}

async function refreshAll() {
  try {
    showError("");
    state.config = await api("/api/config");
    $("#sourceDb").textContent = state.config.db_name;
    $("#sourceExport").textContent = state.config.export_dir;
    $("#dbSelect").innerHTML = (state.config.databases || [state.config.db_name])
      .map(
        (name) =>
          `<option value="${escapeHtml(name)}" ${name === state.config.db_name ? "selected" : ""}>${escapeHtml(name)}</option>`
      )
      .join("");
    state.overview = await api("/api/overview");
    renderOverview();
    const collections = await api("/api/collections");
    state.collections = collections.collections || [];
    renderCollectionSelect();
    await loadActiveView();
  } catch (error) {
    showError(error.message);
  }
}

function bindEvents() {
  $$(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  $("#refreshButton").addEventListener("click", refreshAll);

  $("#inboxSearchButton").addEventListener("click", () => loadInbox().catch((error) => showError(error.message)));
  $("#inboxQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadInbox().catch((error) => showError(error.message));
  });
  $("#slackSearchButton").addEventListener("click", () => loadSlack().catch((error) => showError(error.message)));
  $("#slackQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadSlack().catch((error) => showError(error.message));
  });
  $("#jiraSearchButton").addEventListener("click", () => loadJira().catch((error) => showError(error.message)));
  $("#jiraQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadJira().catch((error) => showError(error.message));
  });
  $("#docsSearchButton").addEventListener("click", () => loadDocs().catch((error) => showError(error.message)));
  $("#docsQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadDocs().catch((error) => showError(error.message));
  });
  $("#crmSearchButton").addEventListener("click", () => loadCrm().catch((error) => showError(error.message)));
  $("#crmQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadCrm().catch((error) => showError(error.message));
  });
  $("#orgSearchButton").addEventListener("click", () => loadOrg().catch((error) => showError(error.message)));
  $("#orgQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadOrg().catch((error) => showError(error.message));
  });
  $("#timelineSearchButton").addEventListener("click", () => loadTimeline().catch((error) => showError(error.message)));
  $("#timelineTypeSelect").addEventListener("change", () => loadTimeline().catch((error) => showError(error.message)));
  $("#timelineQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadTimeline().catch((error) => showError(error.message));
  });

  $("#collectionSelect").addEventListener("change", async (event) => {
    state.activeCollection = event.target.value;
    await loadDocuments().catch((error) => showError(error.message));
  });
  $("#collectionSearchButton").addEventListener("click", () =>
    loadDocuments().catch((error) => showError(error.message))
  );
  $("#collectionQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadDocuments().catch((error) => showError(error.message));
  });
  $("#fileSearchButton").addEventListener("click", () =>
    loadFiles().catch((error) => showError(error.message))
  );
  $("#fileKindSelect").addEventListener("change", () =>
    loadFiles().catch((error) => showError(error.message))
  );
  $("#fileQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadFiles().catch((error) => showError(error.message));
  });
  $$(".raw-tab").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".raw-tab").forEach((tab) => tab.classList.remove("active"));
      $$(".raw-pane").forEach((pane) => pane.classList.remove("active-raw-pane"));
      button.classList.add("active");
      $(`#raw${button.dataset.rawTab[0].toUpperCase()}${button.dataset.rawTab.slice(1)}`).classList.add("active-raw-pane");
    });
  });

  $("#globalSearchButton").addEventListener("click", () =>
    runGlobalSearch().catch((error) => showError(error.message))
  );
  $("#globalSearchInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runGlobalSearch().catch((error) => showError(error.message));
  });
  $("#copyDocumentButton").addEventListener("click", async () => {
    if (state.activeDocument) {
      await navigator.clipboard.writeText(JSON.stringify(state.activeDocument.doc, null, 2));
    }
  });
  $("#copyTicketButton").addEventListener("click", async () => {
    if (state.activeTicket) {
      await navigator.clipboard.writeText(JSON.stringify(state.activeTicket, null, 2));
    }
  });
  $("#copyFileButton").addEventListener("click", async () => {
    if (state.activeFile?.text) await navigator.clipboard.writeText(state.activeFile.text);
  });
  $("#dbSelect").addEventListener("change", () => {
    state.caches = {};
    api("/api/select-db", { db: $("#dbSelect").value })
      .then(refreshAll)
      .catch((error) => showError(error.message));
  });
}

bindEvents();
refreshAll();
