// Шаблон корпоративного PDF-отчёта. Данные приходят JSON-строкой через sys.inputs.report
// (строки JSON выводятся как текст, а не как разметка — экранирование не требуется).
#let d = json(bytes(sys.inputs.at("report")))
#let p = d.passport
#let m = d.metrics
#let vd = d.verdict

#let sev-color(s) = if s == "FATAL" { rgb("#b42318") } else if s == "MAJOR" { rgb("#b54708") } else { rgb("#475467") }
#let badge(s) = box(fill: sev-color(s), inset: (x: 3pt, y: 1.5pt), radius: 2pt, text(fill: white, size: 7pt, weight: "bold", s))
#let verdict-color = if vd.status == "APPROVED" { rgb("#067647") } else if vd.status == "CONDITIONAL_PASS" { rgb("#b54708") } else { rgb("#b42318") }
#let status-color(s) = if s == "X" { rgb("#b42318") } else if s == "?" { rgb("#b54708") } else if s == "OK" { rgb("#067647") } else { luma(140) }

#set document(title: "Отчёт о проверке требований — " + d.title)
#set text(font: ("Inter", "Helvetica Neue", "Arial", "PT Sans", "Libertinus Serif"), size: 9pt, lang: "ru")
#set page(
  paper: "a4",
  margin: (x: 1.5cm, top: 1.8cm, bottom: 1.6cm),
  header: context if counter(page).get().first() > 1 {
    set text(size: 7.5pt, fill: luma(120))
    [Отчёт о проверке требований — #d.title #h(1fr) #p.run_date]
  },
  footer: context {
    set text(size: 7.5pt, fill: luma(120))
    [req-validator #p.agent_version · #p.git_commit #h(1fr) Стр. #counter(page).display("1 из 1", both: true)]
  },
)
#set par(justify: false, leading: 0.55em)
#set table(stroke: 0.5pt + luma(205), inset: 4pt)
#show table.cell.where(y: 0): set text(weight: "bold")
#show heading.where(level: 1): set text(size: 16pt)
#show heading.where(level: 2): it => block(above: 1.4em, below: 0.7em, text(size: 12.5pt, it))
#show heading.where(level: 3): set text(size: 10.5pt)

= Отчёт о проверке требований
#text(size: 12pt, weight: "bold", d.title)

#block(fill: luma(247), stroke: (left: 3pt + verdict-color), inset: 9pt, width: 100%)[
  #text(weight: "bold", fill: verdict-color, size: 11pt)[Итог: #vd.text] \
  Индекс готовности: *#m.readiness_score / 100* · требований: *#m.total_requirements* · нарушений: *#m.total_defects*
  (FATAL #m.fatal_count / MAJOR #m.major_count / MINOR #m.minor_count) \
  #vd.rationale
]

== Паспорт прогона
#table(columns: (4.2cm, 1fr),
  [Параметр], [Значение],
  [Документ], [#p.document_title],
  [Файл], [#p.file_name (#p.file_size байт, формат: #p.format, кодировка: #p.encoding, страниц: #str(p.pages))],
  [SHA-256], [#text(size: 7.5pt, p.sha256)],
  [Дата и время прогона], [#p.run_date],
  [Идентификатор прогона], [#p.run_id],
  [Режим], [#p.mode],
  [Модель], [#p.model],
  [Версия агента], [req-validator #p.agent_version, коммит #p.git_commit, тег #p.git_tag],
  [Длительность], [#p.duration],
  [Статус чтения], [#p.ingestion_status],
  [Стандарты], [ISO/IEC/IEEE 29148:2018 п. 5.2.5, 5.2.6; IEEE Std 830-1998 п. 4.3],
)
#if p.warnings.len() + p.notes.len() > 0 [
  *Диагностика:*
  #for w in p.warnings [- #w
  ]
  #for n in p.notes [- #n
  ]
]

#if d.llm != none [
  === LLM-аудит
  #table(columns: (4.2cm, 1fr),
    [Параметр], [Значение],
    [Модели], [#d.llm.models_used],
    ..d.llm.routes.map(r => ([Маршрут #r.group], text(size: 8pt, r.chain))).flatten(),
    [Версии промптов], [#text(size: 7.5pt, d.llm.prompts)],
    [Вызовов / из кэша], [#str(d.llm.calls) / #str(d.llm.cache_hits); токены #d.llm.tokens; 429: #str(d.llm.rate_limited); перенаправлений: #str(d.llm.reroutes)],
    [Отклонено без цитаты], [#str(d.llm.unverified)],
    [Критик], [#d.llm.critic],
    [Маскирование ПДн], [#d.llm.pii],
    [Бюджет / реестр], [#d.llm.budget · #d.llm.registry],
  )
  *Воронка качества и отсева галлюцинаций*
  #table(columns: (5cm, 1.2cm, 4cm, 1fr), align: (left, right, left, left),
    [Этап], [Кол-во], [], [Пояснение],
    ..d.llm.funnel.map(f => (f.stage, str(f.count),
      box(width: 4cm * f.pct, height: 7pt, fill: rgb("#3e63dd"), radius: 1pt),
      text(size: 8pt, fill: luma(90), f.note))).flatten(),
  )
  #table(columns: (3.6cm, 1.5cm, 3.2cm, 1.5cm, 1fr),
    [Проход], [Статус], [Модели], [Вызовы / кэш], [Детали],
    ..d.llm.passes.map(ps => (ps.name, ps.raw_status, text(size: 7.5pt, ps.models),
      str(ps.calls) + " / " + str(ps.cache), text(size: 8pt, ps.detail))).flatten(),
  )
]

== 1. Нарезка требований
Выделено *#m.total_requirements* требований; явный ID в документе — у *#m.requirements_with_explicit_id*. Остальным присвоены ID агента REQ-xxx.

#table(columns: (1.5cm, 2.0cm, 3.3cm, 3.0cm, 1fr),
  [ID в отчёте], [ID в документе], [Раздел], [Тип], [Формулировка (цитата)],
  ..d.requirements.map(r => (r.req_id, r.source_id, r.section, r.kind + ", " + r.unit, r.short)).flatten(),
)

== 2. Проверка каждого требования
Легенда: OK — соответствует · ? — требует проверки · X — нарушение · «·» — не проверялось (LLM) · — неприменимо.

#table(columns: (2.6cm,) + (1fr,) * d.criteria.len() + (1.2cm,),
  align: (x, y) => if x == 0 { left } else { center },
  [Требование], ..d.criteria.map(c => text(size: 7pt, c.short)), [Деф.],
  ..d.matrix.map(row => (
    (row.req_id + if row.source_id != "—" { " (" + row.source_id + ")" } else { "" },)
    + row.short.map(s => text(fill: status-color(s), weight: "bold", s))
    + (str(row.n_defects),)
  )).flatten(),
)

=== 2.1. Замечания по требованиям
#for x in d.details [
  #block(breakable: true, above: 1em)[
    *#x.req_id* #if x.source_id != "—" [· #x.source_id] — #x.section
    #block(fill: luma(248), stroke: (left: 2pt + luma(200)), inset: 6pt, width: 100%, text(size: 8.5pt, x.quote))
    #for f in x.defects [
      #badge(f.severity) *#f.id* · #f.criterion (#f.rule_id) · #f.priority · #text(fill: luma(100), f.source_label + "; " + f.review) \
      _Цитата:_ «#f.quote» \
      #if f.related_quote != "" [_Противоречит (#f.related_req):_ «#f.related_quote» \ ]
      #if f.binding_note != "" [#text(fill: luma(100))[_Привязка:_ #f.binding_note] \ ]
      _Проблема:_ #f.explanation \
      _Риск:_ #f.risk \
      _Как исправить:_ #f.suggestion
      #v(3pt)
    ]
    #for h in x.hints [
      #text(fill: luma(100))[? Подсказка (#h.rule_id): #h.message — «#h.quote»] \
    ]
  ]
]

== 3. Проверка набора требований целиком
#table(columns: (3.6cm, 3.2cm, 2.2cm, 1fr),
  [Критерий], [Ссылка], [Статус], [Комментарий],
  ..d.set_checks.map(c => (c.criterion + " (" + c.rule_id + ")", c.reference,
    text(fill: status-color(c.short), weight: "bold", c.status),
    c.comment + if c.defect_ids.len() > 0 { " — " + c.defect_ids.join(", ") } else { "" })).flatten(),
)
#for f in d.set_defects [
  #badge(f.severity) *#f.id* · #f.criterion (#f.rule_id) · #text(fill: luma(100), f.source_label + "; " + f.review) \
  _Цитата:_ «#f.quote» \
  #if f.related_quote != "" [_Противоречит (#f.related_req):_ «#f.related_quote» \ ]
  #if f.binding_note != "" [#text(fill: luma(100))[_Привязка:_ #f.binding_note] \ ]
  _Проблема:_ #f.explanation \
  _Как исправить:_ #f.suggestion
  #v(3pt)
]

== 4. Сводка
#table(columns: (7cm, 1fr),
  [Показатель], [Значение],
  [Требований выделено], [#m.total_requirements],
  [С явным ID в документе], [#m.requirements_with_explicit_id],
  [Требований с нарушениями], [#m.requirements_with_defects],
  [FATAL / MAJOR / MINOR], [#m.fatal_count / #m.major_count / #m.minor_count (всего #m.total_defects)],
  [Подсказок (вне метрик)], [#m.hints_count],
  [Плотность дефектов (DD)], [#str(m.defect_density)],
  [Индекс готовности (S_DoR)], [#str(m.readiness_score) / 100],
  [Сэкономленные трудозатраты], [#str(m.estimated_saved_hours) ч],
)

#if d.roi != none [
  === Бизнес-ценность (ROI)
  #text(size: 8.5pt, fill: luma(90))[Детерминированный расчёт на Decimal, без LLM. Коэффициенты — явные допущения (настраиваются в .env).]
  #table(columns: (6cm, 1fr, 2.2cm), align: (left, left, right),
    [Статья], [Расчёт], [Часы],
    ..d.roi.lines.map(ln => (ln.item, ln.formula, ln.hours)).flatten(),
    [*Итого сэкономлено*], [], [*#d.roi.total_hours ч*],
    [*В деньгах*], [#d.roi.total_hours ч × #d.roi.rate #d.roi.currency/ч], [*#d.roi.money #d.roi.currency*],
    [Стоимость LLM-прогона], [#d.roi.llm_cost], [—],
  )
  #for a in d.roi.assumptions [- #text(size: 8.5pt, a)
  ]
]

=== Критичные дефекты (FATAL и MAJOR)
#if d.critical.len() == 0 [Критичных дефектов не найдено.] else [
  #table(columns: (1.4cm, 1.5cm, 2.2cm, 1.8cm, 1fr, 1fr),
    [ID], [Уровень], [Тип], [Требование], [Цитата], [Суть],
    ..d.critical.map(f => (f.id, badge(f.severity), f.type, f.req_id, "«" + f.quote + "»", f.explanation)).flatten(),
  )
]

== 5. Итог
#block(fill: luma(247), stroke: (left: 3pt + verdict-color), inset: 9pt, width: 100%)[
  #text(weight: "bold", fill: verdict-color, size: 11pt, vd.text) \
  #vd.rationale \
  #text(fill: luma(90))[Решающее условие: #vd.gate#if vd.blocking != "" [ · блокирующие дефекты: #vd.blocking]]
]
#table(columns: (5cm, 1.2cm, 1fr),
  [Вопрос], [Ответ], [Пояснение],
  [Можно начинать разработку?], [*#vd.dev*], [#vd.dev_note],
  [Можно начинать тестирование?], [*#vd.test*], [#vd.test_note],
)

=== Что исправить в первую очередь
#if d.remediation.len() == 0 [Исправлений не требуется.]
#for (i, r) in d.remediation.enumerate() [
  + *#r.priority* — #r.action_item Дефекты: #r.target_defect_ids.join(", "). #text(fill: luma(100), r.business_impact)
]
