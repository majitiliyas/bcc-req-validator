/* DAIEK web UI — чистый JS, без сборщиков. Все числа приходят от ядра; здесь только отображение. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const nf = (v, d = 0) => Number(v).toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const BOB = (f) => `/static/bob/${f}`;

  // ---------------- Боб: портреты и реплики ----------------
  const bob = { el: $("#bob"), a: $("#bob-a"), b: $("#bob-b"), who: $("#bob-who"), say: $("#bob-say"), chip: $("#bob-chip"), cur: "a" };
  function bobSet(state, img, who, text, chip) {
    bob.el.dataset.state = state;
    const next = bob.cur === "a" ? bob.b : bob.a;
    const now = bob.cur === "a" ? bob.a : bob.b;
    if (!now.src.endsWith(img)) {
      next.src = BOB(img);
      next.onload = () => { next.classList.add("on"); now.classList.remove("on"); };
      bob.cur = bob.cur === "a" ? "b" : "a";
    }
    bob.who.textContent = who;
    bob.say.textContent = text;
    if (chip !== undefined) bob.chip.textContent = chip;
  }
  const IDLE = () => bobSet("idle", "02_slide1_bob_intro.jpg", "Боб",
    "Скинули новое ТЗ? Закидывай в DAIEK — поймаем баги до старта спринта.",
    "Ожидание: «ТЗ готово». Реальность: проверим.");

  const STAGE_LINES = [
    ["04_slide3_bob_overwhelmed_tz.jpg", "Боб, ты думал, это ТЗ прочитается за одну кружку кофе? DAIEK уже разбирает его на блоки и прячет номера карт до того, как текст увидит модель."],
    ["04_slide3_bob_overwhelmed_tz.jpg", "Ожидание: «там пара пунктов». Реальность: каждое требование получает свой REQ-ID и дословную цитату."],
    ["06_slide5_bob_selected_daiek.jpg", "Две модели читают ТЗ независимо — как два ревьюера, которые не списывают друг у друга."],
    ["06_slide5_bob_selected_daiek.jpg", "Grounding Gate: цитата не нашлась в документе — значит, это не находка, а галлюцинация. Выбрасываем."],
    ["06_slide5_bob_selected_daiek.jpg", "Критик в чистом контексте и FATAL-гейт: никакие баллы не «отмоют» критический дефект."],
  ];

  // ---------------- Пайплайн ----------------
  const steps = $$("#pipeline li");
  let anim = null;
  function pipeReset() { steps.forEach((s) => s.classList.remove("run", "done", "skip")); }
  function pipeRun(stepMs) {
    pipeReset();
    let i = 0;
    const tick = () => {
      if (i > 0) steps[i - 1].classList.replace("run", "done");
      if (i >= steps.length) { anim = null; return; }
      steps[i].classList.add("run");
      const [img, line] = STAGE_LINES[i];
      bobSet("processing", img, "Голос за кадром", line, `Этап ${i + 1} из 5: ${steps[i].querySelector("b").textContent}`);
      i += 1;
      anim = setTimeout(tick, i === steps.length ? stepMs * 1.4 : stepMs);
    };
    tick();
    return { stop: () => { clearTimeout(anim); anim = null; } };
  }
  function pipeFinish(view) {
    const demo = !view.llm, cannot = view.verdict.status === "CANNOT_PROCEED";
    steps.forEach((s, i) => {
      s.classList.remove("run", "done", "skip");
      if (cannot && i > 0) return;  // документ не прочитан: дальше первого этапа не идём
      s.classList.add("done");
      if (demo && (i === 2 || i === 4)) s.classList.add("skip");
    });
  }

  // ---------------- Загрузка и запуск ----------------
  const busyEls = [$("#btn-case"), $("#btn-flawed")];
  const setBusy = (b) => { busyEls.forEach((x) => (x.disabled = b)); $("#drop").style.pointerEvents = b ? "none" : ""; };
  const mode = () => $('input[name="mode"]:checked').value;

  async function run(kind, file) {
    setBusy(true);
    $("#results").hidden = true;
    const llm = kind === "upload" && mode() === "llm";
    const p = pipeRun(kind === "case" ? 520 : llm ? 9000 : 380);
    const started = Date.now();
    let res;
    try {
      let r;
      if (kind === "case") r = await fetch("/api/case");
      else if (kind === "flawed") r = await fetch("/api/sample/flawed", { method: "POST" });
      else {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("mode", mode());
        r = await fetch("/api/analyze", { method: "POST", body: fd });
      }
      res = await r.json();
    } catch (e) {
      res = { ok: false, error: "Сервер не ответил.", hint: "Проверьте, что `req-validator web` запущен." };
    }
    const minShow = kind === "case" ? 2900 : 2200; // дать пайплайну «доиграть», чтобы жюри увидело этапы
    const wait = minShow - (Date.now() - started);
    if (wait > 0) await sleep(wait);
    p.stop();
    setBusy(false);
    if (!res.ok) return showError(res);
    pipeFinish(res.view);
    render(res.view);
  }

  function showError(res) {
    pipeReset();
    bobSet("cannot", "04_slide3_bob_overwhelmed_tz.jpg", "Голос за кадром",
      `Боб, спокойно: ${res.error} ${res.hint || ""}`.trim(), "Ничего не сломалось — просто нужен другой режим или файл.");
    toast(`${res.error} ${res.hint || ""}`);
  }
  function toast(text) {
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = text;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 6000);
  }

  $("#btn-case").addEventListener("click", () => run("case"));
  $("#btn-flawed").addEventListener("click", () => run("flawed"));
  $("#file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) run("upload", f); e.target.value = ""; });
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) run("upload", f); });

  // ---------------- Рендер результата ----------------
  let current = null;
  function render(v) {
    current = v;
    $("#results").hidden = false;
    renderVerdict(v);
    renderKpis(v);
    renderDefects(v);
    renderFunnel(v);
    renderRoi(v);
    renderPassport(v);
    renderDownloads(v);
    bobForVerdict(v);
    activate("defects");
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function bobForVerdict(v) {
    const s = v.verdict.status;
    const m = v.metrics;
    if (s === "REJECTED") {
      const ids = v.defects.map((d) => d.id);
      const twin = v.source === "snapshot" || (ids.includes("DEF-001") && v.title.includes("расходах"));
      bobSet("rejected", "03_slide2_bob_found_bug.jpg", "Голос за кадром",
        twin ? "Ожидание: ТЗ готово к спринту. Реальность: GET сохраняет настройки, amount то integer, то string, а «в момент покупки» спорит с проведённым клирингом. У ТЗ раздвоение личности, Боб. FATAL GATE его не пропустит."
             : `Ожидание: можно кодить. Реальность: FATAL ${m.fatal_count}, MAJOR ${m.major_count}. Боб, это ТЗ возвращается аналитику — с цитатами и готовыми правками.`,
        `Блокирует: ${v.verdict.blocking.join(", ") || "FATAL-гейт"}. Индекс ${nf(m.readiness_score, 1)} / 100 это не компенсирует.`);
    } else if (s === "CANNOT_PROCEED") {
      bobSet("cannot", "04_slide3_bob_overwhelmed_tz.jpg", "Голос за кадром",
        "Боб, это не ТЗ — это пустота или битый файл. DAIEK не выдумывает отчёт: честно говорит, что проверить нельзя, и объясняет почему.",
        `Статус чтения: ${v.passport.ingestion_status}. Вызовы LLM не тратились.`);
    } else {
      bobSet("pass", "07_slide6_bob_happy_savings.jpg", "Боб",
        s === "APPROVED" ? "Ты не поверишь: это ТЗ можно отдавать в разработку. Спринт стартует без мин."
                         : "Почти идеально: осталась пара замечаний — поправь их, и в спринт.",
        `Индекс готовности ${nf(m.readiness_score, 1)} / 100 · FATAL ${m.fatal_count}.`);
    }
  }

  function renderVerdict(v) {
    const vd = v.verdict, m = v.metrics;
    const el = $("#verdict");
    el.className = `verdict card v-${vd.status}`;
    const score = vd.status === "CANNOT_PROCEED" ? 0 : m.readiness_score;
    const color = vd.status === "REJECTED" ? "var(--fatal)" : vd.status === "APPROVED" ? "var(--pass)" : vd.status === "CANNOT_PROCEED" ? "#64748b" : "var(--major)";
    const src = v.source === "snapshot"
      ? `<div class="note">📌 Снимок реального прогона ${esc(v.passport.git_tag)} (${esc(v.passport.run_id)}): ${v.llm.calls} вызовов LLM, ${nf(v.passport.duration_s, 0)} с, $0. Скачайте оригинальный отчёт ниже.</div>`
      : vd.preliminary && vd.status !== "CANNOT_PROCEED" ? `<div class="note">⚠️ Предварительный вердикт: демо-режим, семантика LLM не проверялась.</div>` : "";
    const diag = vd.status === "CANNOT_PROCEED"
      ? `<ul class="diag">${[...v.passport.warnings].map((w) => `<li>${esc(w)}</li>`).join("") || `<li>${esc(vd.rationale)}</li>`}</ul>` : "";
    el.innerHTML = `
      <div><span class="badge">${esc(vd.status)}</span></div>
      <div><h2>${esc(vd.text)}</h2><p>${esc(vd.rationale)}</p>${diag}
        <div class="chips"><span class="chip ${vd.dev ? "yes" : "no"}">Разработка: ${vd.dev ? "можно" : "нельзя"}</span>
        <span class="chip ${vd.test ? "yes" : "no"}">Тестирование: ${vd.test ? "можно" : "нельзя"}</span>
        ${vd.gate && vd.gate !== "—" ? `<span class="chip">Гейт: ${esc(vd.gate)}</span>` : ""}
        <span class="chip">${esc(v.title)}</span></div>${src}</div>
      <div class="gauge" style="--gauge:${color}"><span data-count="${score}" data-dec="1">0<small>из 100</small></span></div>`;
    const g = $(".gauge", el);
    requestAnimationFrame(() => animateGauge(g, score));
  }
  function animateGauge(g, target) {
    const t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / 1200), e = 1 - Math.pow(1 - k, 3), val = target * e;
      g.style.setProperty("--v", val.toFixed(2));
      g.querySelector("span").innerHTML = `${nf(val, 1)}<small>из 100</small>`;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function renderKpis(v) {
    const m = v.metrics, llm = v.llm;
    const funnel = llm ? llm.funnel : [];
    const cand = funnel.find((f) => f.stage.startsWith("Кандидаты"));
    const ok = funnel.find((f) => f.stage.startsWith("Подтверждено цитатой"));
    const cards = [
      ["", m.total_requirements, "требований выделено"],
      ["gold", m.total_defects, "дефектов в метриках"],
      ["fatal", m.fatal_count, "FATAL · блокеры P0"],
      ["major", m.major_count, "MAJOR · P1"],
      ["minor", m.minor_count, "MINOR · P2"],
      llm && cand ? ["pass", `${ok ? ok.count : 0}/${cand.count}`, "находок LLM подтверждено цитатой"]
                  : ["pass", m.hints_count, "подсказок вне метрик"],
    ];
    $("#kpis").innerHTML = cards.map(([c, n, l]) => `<div class="kpi ${c}"><b data-count="${n}">${typeof n === "number" ? 0 : esc(n)}</b><span>${esc(l)}</span></div>`).join("");
    $$("#kpis [data-count]").forEach((b) => countUp(b));
  }
  function countUp(el) {
    const raw = el.dataset.count;
    if (!/^-?\d+(\.\d+)?$/.test(raw)) { el.textContent = raw; return; }
    const target = Number(raw), t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / 900), e = 1 - Math.pow(1 - k, 3);
      el.textContent = nf(Math.round(target * e));
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // ---- Дефекты ----
  const filters = new Set(["FATAL", "MAJOR", "MINOR"]);
  function renderDefects(v) {
    const pane = $("#pane-defects");
    if (!v.defects.length) {
      pane.innerHTML = `<p class="empty">${v.verdict.status === "CANNOT_PROCEED" ? "Документ не прочитан — проверять нечего. Диагностика — в вердикте и паспорте прогона." : "Дефектов не найдено."}</p>`;
      return;
    }
    const cnt = (s) => v.defects.filter((d) => d.severity === s).length;
    pane.innerHTML = `<div class="filters">${["FATAL", "MAJOR", "MINOR"].map((s) => `<button class="f ${s} ${filters.has(s) ? "on" : ""}" data-f="${s}">${s} · ${cnt(s)}</button>`).join("")}
      <span class="meta">Подсказок вне метрик: ${v.metrics.hints_count} (полный список — в отчёте)</span></div><div class="defects" id="deflist"></div>`;
    $$(".f", pane).forEach((b) => b.addEventListener("click", () => {
      filters.has(b.dataset.f) ? filters.delete(b.dataset.f) : filters.add(b.dataset.f);
      b.classList.toggle("on");
      drawDefects(v);
    }));
    drawDefects(v);
  }
  function drawDefects(v) {
    const list = v.defects.filter((d) => filters.has(d.severity));
    $("#deflist").innerHTML = list.map((d) => `
      <article class="def ${d.severity}">
        <div class="def-head"><span class="sev">${d.severity}</span><b>${esc(d.id)}</b><span class="tag">${esc(d.priority)}</span>
          <span class="tag">${esc(d.rule_id)}</span><span class="tag">${esc(d.req_id)}${d.related_req ? " ↔ " + esc(d.related_req) : ""}</span>
          ${d.original_severity ? `<span class="tag">было ${esc(d.original_severity)}</span>` : ""}</div>
        <h4>${esc(d.criterion)}: ${esc(d.explanation)}</h4>
        <div class="quote"><span class="lbl">Цитата из документа${d.section ? " · " + esc(d.section) : ""}</span>${esc(d.quote_view)}</div>
        ${d.related_view ? `<div class="vs">ПРОТИВОРЕЧИТ</div><div class="quote"><span class="lbl">${esc(d.related_req)}</span>${esc(d.related_view)}</div>` : ""}
        ${d.risk ? `<div class="risk">⚠ ${esc(d.risk)}</div>` : ""}
        <div class="fix"><b>Как исправить</b>${esc(d.suggestion)}</div>
        <div class="meta">Источник: ${esc(d.source_label)} · ${esc(d.review)}</div>
        ${d.review_note ? `<details class="critic"><summary>Обоснование Критика</summary>${esc(d.review_note)}</details>` : ""}
      </article>`).join("") || `<p class="empty">Нет дефектов выбранного уровня.</p>`;
  }

  // ---- Воронка ----
  function renderFunnel(v) {
    const pane = $("#pane-funnel"), llm = v.llm, m = v.metrics;
    const sevBars = [["FATAL", m.fatal_count, "red"], ["MAJOR", m.major_count, "amber"], ["MINOR", m.minor_count, ""]];
    const maxSev = Math.max(1, ...sevBars.map((x) => x[1]));
    const sevHtml = `<div class="panel"><h3>Распределение дефектов</h3>${sevBars.map(([n, c, cl]) => row(n, "", c, c / maxSev, cl)).join("")}
      <p class="meta">Плотность дефектов DD = ${nf(m.defect_density, 2)} · требований с нарушениями: ${m.requirements_with_defects} из ${m.total_requirements}</p></div>`;
    if (!llm) {
      pane.innerHTML = `<div class="two"><div class="panel"><h3>Воронка LLM</h3><p class="empty">Демо-режим: работали только детерминированные правила, LLM не вызывалась — воронки отсева галлюцинаций нет. Каждое замечание правил всё равно привязано к цитате.</p></div>${sevHtml}</div>`;
      animateBars(pane);
      return;
    }
    const top = Math.max(1, ...llm.funnel.map((f) => f.count));
    const color = (s) => /отклон|галлюц/i.test(s) ? "red" : /Подтверждено|подтвердил|Итог/.test(s) ? "green" : /понизил|дубл/.test(s) ? "amber" : "";
    const c = llm.critic;
    pane.innerHTML = `<div class="two">
      <div class="panel"><h3>Воронка качества и отсева галлюцинаций</h3>${llm.funnel.map((f) => row(f.stage, f.note, f.count, f.count / top, color(f.stage))).join("")}</div>
      <div style="display:grid;gap:18px">
        <div class="panel"><h3>Независимый Критик</h3>
          <span class="big-badge">same_model: ${c.same_model}</span><span class="big-badge">галлюцинаций: ${countRejectedQuotes(llm)}</span>
          ${row("Подтвердил", "", c.confirmed, c.confirmed / Math.max(1, c.reviewed), "green")}
          ${row("Понизил", "", c.downgraded, c.downgraded / Math.max(1, c.reviewed), "amber")}
          ${row("Отклонил", "ложные придирки", c.rejected, c.rejected / Math.max(1, c.reviewed), "red")}
          <p class="meta">Критик — модель другого семейства, чем генератор; на проверке ${c.reviewed} находок FATAL/MAJOR.</p></div>
        ${sevHtml}</div></div>`;
    animateBars(pane);
  }
  function countRejectedQuotes(llm) {
    const f = llm.funnel.find((x) => x.stage.startsWith("Отклонено проверкой цитат"));
    return f ? f.count : 0;
  }
  function row(name, note, count, pct, cls) {
    return `<div class="funnel-row"><span>${esc(name)}${note ? `<small>${esc(note)}</small>` : ""}</span><div class="bar ${cls}"><i data-w="${Math.max(count ? 3 : 0, pct * 100)}"></i></div><b>${esc(count)}</b></div>`;
  }
  function animateBars(el) { requestAnimationFrame(() => setTimeout(() => $$(".bar i", el).forEach((i) => (i.style.width = i.dataset.w + "%")), 60)); }

  // ---- ROI ----
  function renderRoi(v) {
    const pane = $("#pane-roi"), roi = v.roi;
    if (!roi) { pane.innerHTML = `<p class="empty">ROI не рассчитывается: документ не прочитан.</p>`; return; }
    const sc = roi.scenarios;
    pane.innerHTML = `
      <div class="slider"><label for="docs">ТЗ в месяц:</label><input type="range" id="docs" min="5" max="100" step="1" value="${roi.docs}"><output id="docs-out"></output></div>
      <div class="roi-cards">${sc.map((s) => `<div class="roi-card" data-h="${s.total_hours}" data-m="${s.money}"><h3>${esc(s.name)} сценарий</h3>
        <div class="h">${nf(s.total_hours, 1)} ч</div><div class="m">${nf(s.money)} ${esc(roi.currency)} на одно ТЗ</div>
        <div class="month">на поток: <b class="mh"></b> ч ≈ <b class="mm"></b> ${esc(roi.currency)}/мес</div></div>`).join("")}</div>
      <table class="t"><thead><tr><th>Статья</th>${sc.map((s) => `<th class="n">${esc(s.name)}</th>`).join("")}<th>Расчёт (базовый)</th></tr></thead><tbody>
        ${sc[0].lines.map((ln, i) => `<tr><td>${esc(ln.item)}</td>${sc.map((s) => `<td class="n">${nf(s.lines[i].hours, 1)} ч</td>`).join("")}<td>${esc((sc.find((s) => s.key === "base") || sc[0]).lines[i].formula)}</td></tr>`).join("")}
      </tbody></table>
      <p class="meta">Стоимость LLM: ${esc(roi.llm_cost)}. Расчёт на Decimal в ядре, без LLM; поток — линейная экстраполяция.</p>
      <details class="critic"><summary>Допущения модели ROI</summary><ul class="assume">${roi.assumptions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></details>`;
    const upd = () => {
      const n = Number($("#docs").value);
      $("#docs-out").textContent = `${n} ТЗ/мес`;
      $$(".roi-card", pane).forEach((c) => {
        $(".mh", c).textContent = nf(Number(c.dataset.h) * n, 1);
        $(".mm", c).textContent = nf(Number(c.dataset.m) * n);
      });
    };
    $("#docs").addEventListener("input", upd);
    upd();
  }

  // ---- Паспорт ----
  function renderPassport(v) {
    const p = v.passport, llm = v.llm;
    const rows = [
      ["Документ", v.title], ["Файл", `${p.file_name} (${nf(p.file_size_bytes)} байт, ${p.format}, ${p.encoding || "—"})`],
      ["SHA-256", p.sha256 || "—"], ["Прогон", `${p.run_id} · ${p.run_date}`], ["Длительность", `${nf(p.duration_s, 2)} с`],
      ["Режим", p.mode === "demo" ? "демо (правила, без LLM и сети)" : "LLM"], ["Модели", p.model],
      ["Версия агента", `${p.agent_version}${p.git_commit ? ", коммит " + p.git_commit : ""}${p.git_tag ? ", тег " + p.git_tag : ""}`],
      ["Статус чтения", p.ingestion_status],
    ];
    if (llm) rows.push(["Вызовов LLM / из кэша", `${llm.calls} / ${llm.cache_hits}`], ["Токены (вход / выход)", `${nf(llm.tokens_in)} / ${nf(llm.tokens_out)}`]);
    const passes = llm ? `<h3 style="margin-top:18px">Проходы пайплайна</h3><table class="t"><thead><tr><th>Проход</th><th>Статус</th><th>Детали</th></tr></thead><tbody>
      ${llm.passes.map((x) => `<tr><td>${esc(x.name)}</td><td>${x.status === "ok" ? "✅ выполнен" : esc(x.status)}</td><td>${esc(x.detail)}</td></tr>`).join("")}</tbody></table>` : "";
    const warn = [...(p.warnings || []), ...(p.notes || [])];
    pane("passport").innerHTML = `<table class="t"><tbody>${rows.map(([k, val]) => `<tr><th>${esc(k)}</th><td>${esc(val)}</td></tr>`).join("")}</tbody></table>
      ${warn.length ? `<h3 style="margin-top:18px">Диагностика</h3><ul class="assume">${warn.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}${passes}`;
  }
  const pane = (n) => $(`#pane-${n}`);

  function renderDownloads(v) {
    const d = v.downloads || {};
    const label = { pdf: "⬇ PDF (Typst)", md: "Markdown", html: "HTML" };
    $("#downloads").innerHTML = ["pdf", "md", "html"].filter((f) => d[f]).map((f, i) => `<a class="dl ${i ? "alt" : ""}" href="${esc(d[f])}" download>${label[f]}</a>`).join("");
  }

  // ---- Вкладки ----
  function activate(name) {
    $$(".tab").forEach((t) => t.classList.toggle("on", t.dataset.tab === name));
    $$(".pane").forEach((p) => p.classList.toggle("on", p.id === `pane-${name}`));
    if (name === "funnel") animateBars(pane("funnel"));
    if (name === "roi" && current && current.roi && current.verdict.status !== "CANNOT_PROCEED") {
      const b = current.roi.scenarios.find((s) => s.key === "base") || current.roi.scenarios[0];
      const c = current.roi.scenarios.find((s) => s.key === "conservative") || b;
      bobSet("pass", "07_slide6_bob_happy_savings.jpg", "Голос за кадром",
        `Боб, это не галлюцинация: ${nf(c.total_hours, 1)}–${nf(b.total_hours, 1)} ч и ${nf(c.money)}–${nf(b.money)} ${current.roi.currency} на одно ТЗ. Посчитано на Decimal, без LLM, при бюджете $0.`,
        "Двигай ползунок — поток ТЗ в месяц.");
    } else if (current && name !== "roi" && bob.el.dataset.state === "pass" && current.verdict.status !== "APPROVED" && current.verdict.status !== "CONDITIONAL_PASS") {
      bobForVerdict(current);
    }
  }
  $$(".tab").forEach((t) => t.addEventListener("click", () => activate(t.dataset.tab)));

  IDLE();
})();
