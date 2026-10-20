// Type-safe i18n for Memory Travel.
//
// EN is the DEFAULT language (the app boots in English); RU is available
// through the toggle. Both dictionaries are kept in exact parity: `Dict` is
// derived from typeof RU and `EN: Dict` is an object literal, so a key that
// exists in only one dictionary is a BUILD error (missing key -> not
// assignable; extra key -> excess-property check), never a runtime undefined.
// Switching language mid-presentation only swaps strings — the app keeps its
// position (chapter/stage state lives outside this module).

export type Lang = "ru" | "en";

/** {ru, en} pair for inline strings (chapter narrations, captions). */
export interface L10n {
  ru: string;
  en: string;
}

export function pick(l: L10n, lang: Lang): string {
  return l[lang];
}

const RU = {
  // ---- app chrome ----
  brandSub: "· как работает память",
  tabCosmos: "космос",
  tabTravel: "путешествие",
  tabPresentation: "презентация",
  tabHowItWorks: "как это работает",
  tabEvals: "оценки и пруфы",
  howItWorksTitle: "Интерактивное пошаговое объяснение: как работает поиск в памяти",
  evalsTitle: "Рабочая станция оценок Pier Do Less",
  aboutLink: "о приложении · v1",
  // cosmos panel
  cosmosTitle: "Космос памяти",
  cosmosSub:
    "Подключённый граф знаний — {n} сущностей, {m} связей, разложенных по смыслу (3D UMAP по stored-эмбеддингам — настоящая туманность, которую можно вращать). Не мокап: это индекс, который сейчас отдаёт мост.",
  cosmosSubOffline:
    "Граф знаний — {n} сущностей, {m} связей, разложенных по смыслу (3D UMAP по stored-эмбеддингам). Офлайн: мост недоступен — это снапшот экспорта плюс сохранённые фикстуры-трейсы.",
  searchPlaceholder: "найти сущность… (имя, проект, концепт…)",
  searchClear: "очистить поиск",
  searchMetaTop: "{n} совпадений — топ-{k} ниже, совпадения подсвечены на карте",
  searchMetaNone: "ничего не найдено",
  legendTitle: "типы сущностей",
  legendShown: "{a} / {b} показано",
  legendShowAll: "показать все типы",
  resetView: "сбросить вид",
  tourButton: "Презентация: как работает память",
  hintOrbit: "вращение — мышью · колесо — зум · правая кнопка — панорама · клик по звезде — детали",
  flyTo: "полететь",
  clearHighlight: "снять подсветку",
  degreeLinks1: "1 прямая связь",
  degreeLinksN: "{n} прямых связей",
  noDesc: "для этой сущности нет описания",
  loadingStars: "зажигаем звёзды…",
  // ---- data source badge: WHAT the starfield renders (live vs demo) ----
  srcProbing: "источник: подключаюсь…",
  srcLiveBadge: "живые данные: {id}",
  srcDemoBadge: "демо: офлайн-снимок индекса HARS, 2026-10-17",
  srcOfflineBadge: "источник: нет подключения",
  srcBadgeTitle: "какой индекс рендерит космос",
  srcLiveTitle:
    "космос рендерит ПОДКЛЮЧЁННЫЙ индекс: {id} · {docs} док. · {n} узлов · {m} рёбер · ingest {ts}",
  srcDemoTitle: "ЯВНЫЙ демо-датасет: офлайн-снимок нашего индекса HARS от 2026-10-17. Это НЕ память, подключённая к этому развертыванию.",
  srcSampled: "показан топ-{n} из {total} узлов (по связям)",
  notConnectedTitle: "Нет подключения — живой граф недоступен",
  notConnectedBody:
    "Космос показывает граф ТОЙ памяти, к которой подключён визуализатор, и никогда — чужую память. Сейчас экспорт графа (сервис трейсов :8098: /stats, /graph, /coords) не отвечает, поэтому звёздное поле пусто. Поднимите сервис (или прокси /api/graph-stats) и нажмите «Повторить», либо явно откройте демо-снимок.",
  openDemo: "Открыть демо-данные",
  retryConnection: "Повторить подключение",
  backToLive: "вернуться к живым данным",
  demoLoadedNote: "Показан ДЕМО-снимок (офлайн-снимок индекса HARS, 2026-10-17) — не подключённая память.",
  liveLoadFailed: "живой граф не загрузился: {e}",
  // query bar
  qbPlaceholder: "Спроси память… (живой recall, до 120с)",
  qbAsk: "спросить",
  qbCancel: "отменить",
  qbChars: "0/500",
  qbHistory: "недавние:",
  qbClearHistory: "очистить",
  // 2D fallback badge
  fallbackBadge:
    "резервный canvas2d — в этом браузере нет WebGL2, поэтому 3D-туманность недоступна; те же данные, то же путешествие, плоская проекция.",
  hudNodesEdges: "{n} узлов · {m} рёбер",
  // travel panel
  travelTitle: "Путешествие памяти",
  travelSub:
    "Путь одного реального запроса через машину памяти — каждый счёт, задержка и id ниже — сохранённая телеметрия, не мокап.",
  travelLive: "живой",
  travelLoadingTrace: "загружаю трейс…",
  travelSubgraphLoading: "загружаю подграф…",
  travelCanvasUnavailable:
    "Canvas недоступен (WebGL2 и 2D-резерв не запустились) — телеметрия ниже остаётся читаемой.",
  travelLiveNote:
    "Живой recall — тот же формат трейсов, что и фикстуры (сущности чанков по упоминаниям; маркер запроса в центроиде топ-хитов; эмбеддера в браузере нет).",
  cameraJourney: "камера: авто",
  cameraFree: "камера: свободная",
  journeyLogTitle: "журнал путешествия",
  journeyLogEmpty: "пусто — включите воспроизведение",
  journeyLogDropped: "{n} неразрешённых ссылок опущено",
  stagePrev: "предыдущий этап (←)",
  stageNext: "следующий этап (→)",
  stagePlay: "воспроизвести (пробел)",
  stagePause: "пауза (пробел)",
  stageScrubber: "ползунок этапов",
  speedTitle: "скорость воспроизведения",
  inspectTitle: "инспектор этапа",
  chunkTitle: "содержимое чанка",
  inspectClose: "закрыть",
  inspectOpen: "данные этапа",
  scoreDense: "dense",
  scoreSparse: "bm25",
  scoreFused: "fusion",
  scoreRerank: "rerank",
  inFinalContext: "в финальном контексте",
  notInFinalContext: "не вошёл в контекст",
  sourceFile: "источник",
  section: "секция",
  showMore: "показать больше",
  showLess: "свернуть",
  openSourceNote: "открыть исходную заметку",
  rank: "место",
  deltaToTop: "отставание от лидера",
  beforeAfter_in: "порядок fusion (до)",
  beforeAfter_out: "порядок подачи (после)",
  rerankSkipped: "реранк ПРОПУЩЕН — {reason}",
  ringAdded: "кольцо {d}: +{n} сущностей",
  edgesTraversed: "ребёр пройдено: {n}",
  contextBudget: "бюджет: {used} / {total} симв.",
  // presentation
  presLaunch: "Презентация: как работает память",
  presExit: "выйти (Esc)",
  presMenu: "главы",
  presLang: "язык",
  presAutoplay: "автоплей",
  presChapter: "глава {n} / {total}",
  presNext: "дальше (→)",
  presPrev: "назад (←)",
  presFullscreen: "полный экран (F)",
  presLoading: "готовлю данные…",
  // presentation: honest gates for fixture-only chapters outside demo mode
  presGateIngestTitle: "Глава 2 — реплей сохранённой заметки из демо-снимка",
  presGateIngestBody:
    "Эта глава разбирает ОДНУ конкретную заметку с её реальными записями учителя. В живом режиме она не показывается: заметка из снимка — не ваша память. Откройте демо-снимок или вернитесь к живым данным.",
  presGateTraceTitle: "Для этой главы нужен трейс recall",
  presGateTraceBody:
    "Главы 3–8 воспроизводят один реальный recall по этапам. Задайте живой вопрос (строка запроса) — глава переиграет его по подключённой памяти, либо явно откройте демо-снимок с сохранёнными трейсами.",
  // presentation: docked chapter cards (chapters 3-8, real trace data)
  cardsAria: "данные главы",
  cardQuestion: "вопрос",
  cardEmbedding: "эмбеддинг",
  cardVectorDims: "UMAP 768-d → x,y,z",
  cardTopDense: "топ-12 dense-скоров",
  cardPool: "пул кандидатов",
  cardLatencyDense: "dense, мс",
  cardLatencySparse: "bm25, мс",
  cardVectorCh: "векторный канал",
  cardSparseCh: "BM25-канал",
  cardGraphCh: "граф-канал",
  cardHits: "хитов",
  cardTopScore: "топ-скор",
  cardBm25Chunks: "чанков в BM25",
  cardSeeds: "сидов",
  cardEdges: "рёбер пройдено",
  cardNodes: "узлов подграфа",
  cardFused: "слитый рейтинг",
  cardAlpha: "α (вес dense)",
  cardMoved: "сдвинуто",
  cardApplied: "реранк применён",
  cardRerankPool: "пул реранка",
  cardBlocked: "заблокировано инверсий",
  cardEntityFacts: "фактов о сущностях",
  cardContext: "контекст",
  cardSources: "источников",
  cardAnswer: "ответ",
  cardCitations: "цитаты",
  cardQuotes: "цитат в превью",
  // presentation: chapter 2 ingest panel (two panels + split preview)
  ingestExtract: "extraction-записи учителя",
  ingestEntities: "сущности",
  ingestRelations: "связи",
  ingestNoRelations: "в этой выборке связей нет — учитель выпустил только сущности",
  ingestNoEntities: "в этой выборке нет записей",
  ingestDoc: "исходная заметка",
  ingestDocMeta: "{lines} строк · {chars} символов · {chunks} окон в индексе",
  ingestParams: "параметры чанкинга",
  ingestLaneIndex: "как в индексе",
  ingestLaneMirror: "md_struct, по параметрам выше",
  ingestOverlap: "перекрытие",
  ingestSection: "граница секции",
  ingestTokens: "{n} ток.",
  ingestEstTokens: "≈{n} ток. (chars//4)",
  ingestMirrorNote: "оценка без токенизатора: chars//4, как в md_struct.estimate_tokens",
  ingestReset: "значения индекса",
  ingestChunk: "чанк",
  ingestSectionTag: "секция",
  ingestHoverHint: "наведите на окно или на запись — подсветятся обе панели",
  ingestClickHint: "клик по окну — закрепить и прокрутить заметку к его строкам",
  ingestMeta: "{n} записей учителя · {r} связей · {c} окон в индексе",
  ingestCaption:
    "Продакшн-сплиттер (md_structure): секции заголовков → абзацы/таблицы/фенсы, мелкие секции подтягиваются вверх, брейдкрамб [A > B] повторяется в каждой части. Панель повторяет этот контракт для иллюстрации: поменяйте параметры — текст будет нарезан заново.",
  ingestHonest:
    "Честно про образец: окна в индексе нарезаны token-чанкером LightRAG (ровно 2048 токенов, перекрытия 186–217, 476–520, 760–793, 1035–1068 строк), а md_struct не перекрывает окна и выравнивает их по секциям. Поэтому дорожек две.",
  ingestKindSection: "секция целиком",
  ingestKindPart: "часть секции",
  ingestKindMerged: "слито вверх",
  // ---- HUD / legend / hover (cosmos) ----
  hudNodes: "узлов",
  hudEdges: "рёбер",
  linksWord: "связей",
  legendShowFamily: "показать это семейство",
  legendHideFamily: "скрыть это семейство",
  searchAria: "Поиск сущностей",
  // ---- travel: statuses + panel labels ----
  travelCompilingLive: "собираю живой ответ…",
  travelRenderFail: "не удалось отрисовать подграф: {e}",
  travelTraceFail: "не удалось загрузить трейс ({f}): {e}",
  metaMode: "режим",
  metaPool: "пул",
  metaLatency: "задержка",
  kwLow: "ключевые слова low-level",
  kwHigh: "high-level",
  stage2Summary: "{seeds} семенных сущностей × 1 шаг → {nodes} узлов, {edges} рёбер на экране.",
  ctxEntitiesClick: "сущности ({n}) — клик, чтобы улететь",
  ctxSourcesTitle: "источники",
  answerExtractive: "Экстрактивный превью — дословные цитаты. Клик по факту улетает к узлам-источникам.",
  timelineAria: "Управление воспроизведением путешествия",
  // ---- about modal ----
  aboutAria: "О Memory Travel",
  aboutClose: "закрыть (Esc)",
  aboutIntro:
    "Визуализатор того, как помнит машина. Два взгляда на ОДИН индекс — тот, к которому подключён мост (живые счётчики — в шапке, личность и свежесть индекса — в блоке «Подключённая память» ниже):",
  aboutCosmos:
    "Космос — весь граф как звёздное поле, раскрашенное по семействам сущностей. Поиск подсвечивает совпадения, легенда переключает семейства, клик по звезде открывает её карточку.",
  aboutTravel:
    "Путешествие — один recall, воспроизведённый по этапам: вопрос падает, векторные хиты пульсируют, обход графа расходится хоп за хопом, BM25-хиты сыпятся, fusion сливает каналы, реранк переупорядочивает, контекст собирается, и каждый факт ответа связан со своими узлами-источниками.",
  aboutProvenance:
    "Происхождение данных: в живом режиме космос рендерит ответ сервиса экспорта (:8098 /graph + /coords — read-only срезы живого индекса), личность индекса приходит из /stats и показана в бейдже источника. Демо-снимок (data/graph.json + координаты, scripts/export_graph.py + compute_umap.py) и пять фикстур-трейсов (capture_fixtures.py — настоящая телеметрия memory_recall) открываются ТОЛЬКО явной кнопкой демо. «ask live» выполняет реальный recall через мост памяти (context_only — визуализатор показывает дословные цитаты, он не пишет ответы).",
  aboutRendering:
    "Рендер: WebGL2-звёздное поле (Three.js) основное, автоматический Canvas2D-резерв на том же конвейере данных. Клавиши: ←/→ этапы, пробел play/pause, Esc закрывает карточки.",
  aboutBridge:
    "Живой мост: {url} — задайте MEMORY_BRIDGE_URL, чтобы указать другую память (см. docs/SETUP.md). Проксируются только read-only инструменты (memory_recall, memory_status); ничего не записывается.",
  aboutRepo: "Репозиторий: ~/Project/software/memory-travel — схема трейса в docs/trace-api.md.",
  // ---- connection identity (about card + HUD chip) ----
  connConnected: "на связи",
  connConnecting: "подключаюсь…",
  connOffline: "нет связи",
  hudConnTitle: "подключённая память",
  aboutConnTitle: "Подключённая память",
  aboutConnNone:
    "Нет связи с мостом. Космос в этом состоянии не показывает НИКАКИХ данных (чужая память не рендерится никогда); демо-снимок — только явной кнопкой.",
  aboutSourceLive: "Источник данных космоса: ЖИВОЙ индекс {id} ({n} узлов, {m} рёбер — ответ /stats подключённого сервиса).",
  aboutSourceDemo: "Источник данных космоса: ДЕМО-снимок от {date} (явно открытая офлайн-копия, не подключённая память).",
  aboutSourceNone: "Источник данных космоса: не подключён — граф не рендерится.",
  aboutConnBridge: "мост",
  aboutConnIndex: "индекс",
  aboutConnDir: "каталог индекса",
  aboutConnEntities: "сущностей",
  aboutConnRelations: "связей",
  aboutConnIngest: "последний ingest",
  aboutConnStale: "возраст данных",
  aboutStaleDays: "{n} дн.",
  aboutConnModels: "модели",
  // ---- errors ----
  errIndexTitle: "Не удалось загрузить данные индекса",
  errIndexBody:
    "Приложение тянет файлы данных относительными путями (data/graph.json, координаты, traces/manifest.json). Один из них не загрузился — проверьте, что файл ниже есть в app/public/data/ и что dev-сервер его отдаёт.",
  errCanvasTitle: "Не удалось инициализировать canvas",
  errCanvasBody:
    "Ни WebGL2-рендер, ни Canvas2D-резерв не запустились — этот браузер не даёт ни одного GPU-canvas. Текстовые панели остаются читаемыми. (В любом браузере последнего десятилетия этот путь недостижим.)",
  errDismiss: "продолжить (только текст)",
  // ---- app chrome / live notes ----
  commonDismiss: "скрыть",
  langSwitch: "язык",
  loadingIndex: "загружаю индекс…",
  coordCoverage: "покрытие координат {p}%",
  dataLoadFailed: "загрузка данных не удалась: {e}",
  canvasLoadFailed: "загрузка подграфа космоса не удалась: {e}",
  askNoteHits: "живой ответ в Путешествии ({s}с через {via}); подсвечено {n} узлов-хитов в Космосе",
  askNoteNoCanvas: "живой ответ в Путешествии ({s}с); canvas Космоса недоступен (нет рендера), подсветки нет",
  askNoteNoHits: "живой ответ в Путешествии ({s}с); узлов для подсветки нет",
  askViaProxy: "dev-прокси",
  askViaDirect: "напрямую",
  askCancelled: "живой recall отменён.",
  // ---- query bar ----
  qbPhase1: "подключаюсь к мосту памяти…",
  qbPhase2: "эмбеддинг вопроса · извлечение ключевых слов…",
  qbPhase3: "dense + sparse поиск · обход графа · fusion…",
  qbPhase4: "фьюжн + сборка контекста — задержка ~160 мс…",
  qbEstNote: "оценка этапов — мост отвечает разом",
  qbCancelTitle: "Отменить живой recall",
  qbAskTitle: "Выполнить живой memory_recall",
  qbAria: "Живой вопрос к памяти",
  qbClearTitle: "очистить историю вопросов",
  qbHistoryTook: "{q} — заняло {s}с",
  // ---- stage inspector ----
  inspColChunk: "чанк",
  inspMode: "режим",
  inspLlKeywords: "ключевые слова ll",
  inspHlKeywords: "ключевые слова hl",
  inspCandidatePool: "пул кандидатов",
  inspTopK: "top_k",
  inspLatency: "задержки dense/sparse/rerank",
  inspConfidence: "уверенность",
  inspSeeds: "семена",
  inspEntitiesN: "сущности ({n})",
  inspRelationsN: "связи ({n})",
  inspSourcesN: "источники ({n})",
  inspExtractive: "экстрактивный превью — дословные цитаты",
  inspContext: "контекст",
  inspEntitiesShort: "сущности",
  inspVia: "через",
  inspContentMissing: "(текста нет в трейсе — только поданные чанки)",
  inspMoved: "сдвинут",
  inspInversions: "{n} инверсий заблокировано",
  inspPoolShort: "пул {n}",
  // ---- presentation: compact schematics + progressive drill-down ----
  presDetailHint: "Нажмите на элемент для деталей",
  presShowDoc: "Показать документ",
  presHideDoc: "Скрыть документ",
  presShowRecords: "Показать {n} записей учителя",
  presHideRecords: "Скрыть записи учителя",
  presDiagramNote: "заметка",
  presDiagramSections: "секции",
  presDiagramChunks: "чанки",
  presDiagramParams: "параметры",
  presDiagramChunksLine: "{n} чанков · {tok} ток. · перекрытие {ov}",
  presClickDetails: "кликните для деталей",
  presDetailClose: "закрыть детали (Esc)",
  presCapEmbed: "{hits} dense-хитов · {ms} мс",
  presCapChannels: "гибридный поиск (BM25 + вектор + граф + ripgrep)",
  presCapFusion: "{n} чанков в слитом рейтинге",
  presCapRerankOn: "{n} сдвинуто",
  presCapRerankOff: "реранкер выключен (LightGBM/RRF по умолчанию, ~160 мс)",
  presCapContext: "{n} фактов о сущностях",
  presCapAnswer: "{n} цитат",
  presCapIngest: "заметка → секции → чанки → записи учителя",
  presPillVector: "вектор {n}",
  presPillBm25: "bm25 {n}",
  presPillGraph: "граф {n}",
} as const;

export type Dict = Record<keyof typeof RU, string>;

const EN: Dict = {
  brandSub: "· how machine memory works",
  tabCosmos: "cosmos",
  tabTravel: "travel",
  tabPresentation: "presentation",
  tabHowItWorks: "how it works",
  tabEvals: "evals & proofs",
  howItWorksTitle: "Interactive step-by-step walkthrough of memory retrieval",
  evalsTitle: "Live Pier Do Less evaluation workstation",
  aboutLink: "about · v1",
  cosmosTitle: "Memory Cosmos",
  cosmosSub:
    "The connected knowledge graph — {n} entities, {m} relations, laid out by meaning (3D UMAP over the stored embeddings — a true starfield you can orbit). Not a mock: this is the index the bridge is serving right now.",
  cosmosSubOffline:
    "Knowledge graph — {n} entities, {m} relations, laid out by meaning (3D UMAP over the stored embeddings). Offline: the bridge is unreachable — this is the exported snapshot plus saved fixture traces.",
  searchPlaceholder: "find an entity… (a name, a project, a concept…)",
  searchClear: "clear search",
  searchMetaTop: "{n} matches — top {k} below, matches lit on the map",
  searchMetaNone: "no entities match",
  legendTitle: "entity types",
  legendShown: "{a} / {b} shown",
  legendShowAll: "show all types",
  resetView: "reset view",
  tourButton: "Presentation: how the memory works",
  hintOrbit: "drag to orbit · scroll to zoom · right-drag to pan · click a star for details",
  flyTo: "fly to",
  clearHighlight: "clear highlight",
  degreeLinks1: "1 direct relation",
  degreeLinksN: "{n} direct relations",
  noDesc: "no description stored for this entity",
  loadingStars: "lighting up the stars…",
  // ---- data source badge: WHAT the starfield renders (live vs demo) ----
  srcProbing: "source: connecting…",
  srcLiveBadge: "live data: {id}",
  srcDemoBadge: "demo: offline snapshot of the HARS index, 2026-10-17",
  srcOfflineBadge: "source: not connected",
  srcBadgeTitle: "which index the cosmos renders",
  srcLiveTitle:
    "the cosmos renders the CONNECTED index: {id} · {docs} docs · {n} nodes · {m} edges · ingest {ts}",
  srcDemoTitle: "EXPLICIT demo dataset: an offline snapshot of our HARS index taken 2026-10-17. This is NOT the memory this deployment is connected to.",
  srcSampled: "top {n} of {total} nodes shown (by degree)",
  notConnectedTitle: "Not connected — live graph unavailable",
  notConnectedBody:
    "The cosmos renders the graph of the memory this visualiser is connected to — never somebody else's. Right now the graph export (trace service :8098: /stats, /graph, /coords) is unreachable, so the starfield stays empty. Start the service (or the /api/graph-stats proxy) and retry, or explicitly open the demo snapshot.",
  openDemo: "Open demo data",
  retryConnection: "Retry connection",
  backToLive: "back to live data",
  demoLoadedNote: "Showing the DEMO snapshot (offline snapshot of the HARS index, 2026-10-17) — not the connected memory.",
  liveLoadFailed: "live graph failed to load: {e}",
  qbPlaceholder: "Ask the memory… (live recall, up to 120s)",
  qbAsk: "ask",
  qbCancel: "cancel",
  qbChars: "0/500",
  qbHistory: "recent:",
  qbClearHistory: "clear",
  fallbackBadge:
    "canvas2d fallback — no WebGL2 in this browser, so the 3D starfield can't run; same data, same journey, flat projection.",
  hudNodesEdges: "{n} nodes · {m} edges",
  travelTitle: "Memory Travel",
  travelSub:
    "One real query's journey through the memory machinery — every score, latency and id below is saved telemetry, not a mockup.",
  travelLive: "live",
  travelLoadingTrace: "loading trace…",
  travelSubgraphLoading: "loading subgraph…",
  travelCanvasUnavailable:
    "Canvas unavailable (WebGL2 + 2D fallback both failed) — the telemetry text below stays readable.",
  travelLiveNote:
    "Live recall — same trace format as fixtures (per-chunk entities by mention; query marker at the top-hits centroid; no embedder in the browser).",
  cameraJourney: "camera: journey",
  cameraFree: "camera: free",
  journeyLogTitle: "journey log",
  journeyLogEmpty: "empty — press play",
  journeyLogDropped: "{n} unresolved refs omitted",
  stagePrev: "previous stage (←)",
  stageNext: "next stage (→)",
  stagePlay: "play (space)",
  stagePause: "pause (space)",
  stageScrubber: "stage scrubber",
  speedTitle: "playback speed",
  inspectTitle: "stage inspector",
  chunkTitle: "chunk content",
  inspectClose: "close",
  inspectOpen: "stage data",
  scoreDense: "dense",
  scoreSparse: "bm25",
  scoreFused: "fusion",
  scoreRerank: "rerank",
  inFinalContext: "in final context",
  notInFinalContext: "not served to the LLM",
  sourceFile: "source",
  section: "section",
  showMore: "show more",
  showLess: "show less",
  openSourceNote: "open source note",
  rank: "rank",
  deltaToTop: "delta to top",
  beforeAfter_in: "fusion order (in)",
  beforeAfter_out: "served order (out)",
  rerankSkipped: "rerank SKIPPED — {reason}",
  ringAdded: "ring {d}: +{n} entities",
  edgesTraversed: "edges traversed: {n}",
  contextBudget: "budget: {used} / {total} chars",
  presLaunch: "Presentation: how the memory works",
  presExit: "exit (Esc)",
  presMenu: "chapters",
  presLang: "language",
  presAutoplay: "autoplay",
  presChapter: "chapter {n} / {total}",
  presNext: "next (→)",
  presPrev: "back (←)",
  presFullscreen: "fullscreen (F)",
  presLoading: "preparing data…",
  // presentation: honest gates for fixture-only chapters outside demo mode
  presGateIngestTitle: "Chapter 2 replays one captured note from the demo snapshot",
  presGateIngestBody:
    "This chapter dissects ONE specific note with its real teacher records. In live mode it is not shown: a note from the snapshot is not your memory. Open the demo snapshot, or go back to live data.",
  presGateTraceTitle: "This chapter needs a recall trace",
  presGateTraceBody:
    "Chapters 3-8 replay one real recall stage by stage. Ask a live question and the chapter replays it against the connected memory, or explicitly open the demo snapshot with its saved traces.",
  cardsAria: "chapter data",
  cardQuestion: "question",
  cardEmbedding: "embedding",
  cardVectorDims: "UMAP 768-d → x,y,z",
  cardTopDense: "top-12 dense scores",
  cardPool: "candidate pool",
  cardLatencyDense: "dense, ms",
  cardLatencySparse: "bm25, ms",
  cardVectorCh: "vector channel",
  cardSparseCh: "BM25 channel",
  cardGraphCh: "graph channel",
  cardHits: "hits",
  cardTopScore: "top score",
  cardBm25Chunks: "BM25 chunks",
  cardSeeds: "seeds",
  cardEdges: "edges traversed",
  cardNodes: "subgraph nodes",
  cardFused: "fused ranking",
  cardAlpha: "α (dense weight)",
  cardMoved: "moved",
  cardApplied: "rerank applied",
  cardRerankPool: "rerank pool",
  cardBlocked: "inversions blocked",
  cardEntityFacts: "entity facts",
  cardContext: "context",
  cardSources: "sources",
  cardAnswer: "answer",
  cardCitations: "citations",
  cardQuotes: "quotes in preview",
  ingestExtract: "teacher extraction records",
  ingestEntities: "entities",
  ingestRelations: "relations",
  ingestNoRelations: "no relations in this sample — the teacher emitted entities only",
  ingestNoEntities: "no records in this sample",
  ingestDoc: "source document",
  ingestDocMeta: "{lines} lines · {chars} chars · {chunks} index windows",
  ingestParams: "chunking parameters",
  ingestLaneIndex: "as ingested",
  ingestLaneMirror: "md_struct, re-split from the parameters above",
  ingestOverlap: "overlap",
  ingestSection: "section boundary",
  ingestTokens: "{n} tok",
  ingestEstTokens: "≈{n} tok (chars//4)",
  ingestMirrorNote: "tokenizer-free estimate chars//4, as in md_struct.estimate_tokens",
  ingestReset: "index values",
  ingestChunk: "chunk",
  ingestSectionTag: "section",
  ingestHoverHint: "hover a window or a record — both panels light up",
  ingestClickHint: "click a window to pin it and scroll the note to its lines",
  ingestMeta: "{n} teacher records · {r} relations · {c} index windows",
  ingestCaption:
    "The production splitter (md_structure): heading sections → paragraphs/tables/fences, tiny sections merge up, the [A > B] breadcrumb repeats in every part. This panel mirrors that contract for illustration: change the parameters and the text is cut again.",
  ingestHonest:
    "Honest about the sample: the index windows were cut by LightRAG's token chunker (exactly 2048 tokens, overlaps at lines 186–217, 476–520, 760–793, 1035–1068), while md_struct never overlaps and aligns to sections. Hence two lanes.",
  ingestKindSection: "whole section",
  ingestKindPart: "part of a section",
  ingestKindMerged: "merged up",
  // ---- HUD / legend / hover (cosmos) ----
  hudNodes: "nodes",
  hudEdges: "edges",
  linksWord: "links",
  legendShowFamily: "show this family",
  legendHideFamily: "hide this family",
  searchAria: "Search entities",
  // ---- travel: statuses + panel labels ----
  travelCompilingLive: "compiling live answer…",
  travelRenderFail: "subgraph render failed: {e}",
  travelTraceFail: "trace load failed ({f}): {e}",
  metaMode: "mode",
  metaPool: "pool",
  metaLatency: "latency",
  kwLow: "low-level keywords",
  kwHigh: "high-level keywords",
  stage2Summary: "{seeds} seed entities × 1 hop → {nodes} nodes, {edges} edges on screen.",
  ctxEntitiesClick: "entities ({n}) — click to fly",
  ctxSourcesTitle: "sources",
  answerExtractive: "Extractive preview — verbatim quotes. Click a fact to fly to its source nodes.",
  timelineAria: "Journey playback controls",
  // ---- about modal ----
  aboutAria: "About Memory Travel",
  aboutClose: "close (Esc)",
  aboutIntro:
    "A visualiser for how machine memory remembers. Two views over ONE index — the one the bridge is connected to (live counts in the header; the index's identity and freshness in the Connected memory block below):",
  aboutCosmos:
    "Cosmos — the whole graph as a starfield, coloured by entity family. Search spotlights matches, the legend toggles families, clicking a star opens its card.",
  aboutTravel:
    "Travel — one recall, replayed stage by stage: question lands, vector hits pulse in, the graph walk spreads hop by hop, BM25 hits rain in, fusion merges both channels, rerank reorders, context assembles, and every answer fact links back to its source nodes.",
  aboutProvenance:
    "Data provenance: in live mode the cosmos renders the export service's answer (:8098 /graph + /coords — read-only slices of the connected index); the index identity comes from /stats and is shown in the source badge. The demo snapshot (data/graph.json + coords, scripts/export_graph.py + compute_umap.py) and the five fixture traces (capture_fixtures.py — real memory_recall telemetry) open ONLY behind the explicit demo button. \"ask live\" runs a real recall through the memory bridge (context_only — the visualiser shows extractive quotes, it does not write answers).",
  aboutRendering:
    "Rendering: WebGL2 starfield (Three.js) primary, automatic Canvas2D fallback on the same data pipeline. Keyboard: ←/→ stages, space play/pause, Esc closes cards.",
  aboutBridge:
    "Live bridge: {url} — set MEMORY_BRIDGE_URL to point at a different memory (see docs/SETUP.md). Only read-only tools are proxied (memory_recall, memory_status); nothing is written.",
  aboutRepo: "Repo: ~/Project/software/memory-travel — docs/trace-api.md has the trace schema.",
  // ---- connection identity (about card + HUD chip) ----
  connConnected: "connected",
  connConnecting: "connecting…",
  connOffline: "offline",
  hudConnTitle: "connected memory",
  aboutConnTitle: "Connected memory",
  aboutConnNone:
    "No bridge connection. In this state the cosmos renders NO data at all (another memory's data is never shown); the demo snapshot exists only behind its explicit button.",
  aboutSourceLive: "Cosmos data source: LIVE index {id} ({n} nodes, {m} edges — the connected service's /stats answer).",
  aboutSourceDemo: "Cosmos data source: DEMO snapshot from {date} (an explicitly opened offline copy, not the connected memory).",
  aboutSourceNone: "Cosmos data source: not connected — no graph is rendered.",
  aboutConnBridge: "bridge",
  aboutConnIndex: "index",
  aboutConnDir: "index dir",
  aboutConnEntities: "entities",
  aboutConnRelations: "relations",
  aboutConnIngest: "last ingest",
  aboutConnStale: "data age",
  aboutStaleDays: "{n} d",
  aboutConnModels: "models",
  // ---- errors ----
  errIndexTitle: "Index data failed to load",
  errIndexBody:
    "The app fetches its data files with relative paths (data/graph.json, coords, traces/manifest.json). One of them failed — check the file named below exists in app/public/data/ and the dev server is serving it.",
  errCanvasTitle: "Canvas initialisation failed",
  errCanvasBody:
    "Neither the WebGL2 renderer nor the Canvas2D fallback could start — this browser cannot provide any GPU canvas at all. The text panels stay readable. (On any browser from the last decade this path should be unreachable.)",
  errDismiss: "continue anyway (text only)",
  // ---- app chrome / live notes ----
  commonDismiss: "dismiss",
  langSwitch: "language",
  loadingIndex: "loading index…",
  coordCoverage: "coord coverage {p}%",
  dataLoadFailed: "data load failed: {e}",
  canvasLoadFailed: "cosmos subgraph load failed: {e}",
  askNoteHits: "live answer in Travel ({s}s via {via}); {n} hit nodes lit in Cosmos",
  askNoteNoCanvas: "live answer in Travel ({s}s); Cosmos canvas unavailable (no renderer), so no highlight",
  askNoteNoHits: "live answer in Travel ({s}s); no graph hits to highlight",
  askViaProxy: "dev proxy",
  askViaDirect: "direct",
  askCancelled: "live recall cancelled.",
  // ---- query bar ----
  qbPhase1: "connecting to the memory bridge…",
  qbPhase2: "embedding the question · extracting keywords…",
  qbPhase3: "dense + sparse retrieval · graph walk · fusion…",
  qbPhase4: "fusion + context assembly — typical latency ~160ms…",
  qbEstNote: "estimated stages, the bridge answers in one shot",
  qbCancelTitle: "Cancel the live recall",
  qbAskTitle: "Run a live memory_recall",
  qbAria: "Ask a live memory question",
  qbClearTitle: "clear recent questions",
  qbHistoryTook: "{q} — took {s}s",
  // ---- stage inspector ----
  inspColChunk: "chunk",
  inspMode: "mode",
  inspLlKeywords: "ll keywords",
  inspHlKeywords: "hl keywords",
  inspCandidatePool: "candidate pool",
  inspTopK: "top_k",
  inspLatency: "latency dense/sparse/rerank",
  inspConfidence: "confidence",
  inspSeeds: "seeds",
  inspEntitiesN: "entities ({n})",
  inspRelationsN: "relations ({n})",
  inspSourcesN: "sources ({n})",
  inspExtractive: "extractive preview — verbatim quotes",
  inspContext: "context",
  inspEntitiesShort: "entities",
  inspVia: "via",
  inspContentMissing: "(content not in trace — served chunks only)",
  inspMoved: "moved",
  inspInversions: "{n} inversions blocked",
  inspPoolShort: "pool {n}",
  // ---- presentation: compact schematics + progressive drill-down ----
  presDetailHint: "Click any element for details",
  presShowDoc: "Show document",
  presHideDoc: "Hide document",
  presShowRecords: "Show {n} extracted records",
  presHideRecords: "Hide extracted records",
  presDiagramNote: "note",
  presDiagramSections: "sections",
  presDiagramChunks: "chunks",
  presDiagramParams: "params",
  presDiagramChunksLine: "{n} chunks · {tok} tok · overlap {ov}",
  presClickDetails: "click for details",
  presDetailClose: "close details (Esc)",
  presCapEmbed: "{hits} dense hits · {ms} ms",
  presCapChannels: "hybrid search (BM25 + vector + graph + ripgrep)",
  presCapFusion: "{n} chunks in the fused ranking",
  presCapRerankOn: "{n} moved",
  presCapRerankOff: "reranker OFF (LightGBM/RRF by default, ~160ms)",
  presCapContext: "{n} entity facts",
  presCapAnswer: "{n} citations",
  presCapIngest: "note → sections → chunks → teacher records",
  presPillVector: "vector {n}",
  presPillBm25: "bm25 {n}",
  presPillGraph: "graph {n}",
};

export const STRINGS: Record<Lang, Dict> = { ru: RU, en: EN };

const LANG_KEY = "mt.lang.v1";

export function loadLang(): Lang {
  try {
    const v = localStorage.getItem(LANG_KEY);
    // EN is the default; only an explicit stored "ru" switches to Russian.
    return v === "ru" ? "ru" : "en";
  } catch {
    return "en";
  }
}

export function saveLang(l: Lang): void {
  try {
    localStorage.setItem(LANG_KEY, l);
  } catch { /* private mode */ }
}

/** Interpolate {a}/{b} placeholders into a template. */
export function fmt(tpl: string, vars: Record<string, string | number>): string {
  return tpl.replace(/\{(\w+)\}/g, (_, k: string) => String(vars[k] ?? `{${k}}`));
}

/** Tiny React hook (no context — one module-level listener set). */
let listeners: (() => void)[] = [];
let currentLang: Lang = loadLang();

export function getLang(): Lang {
  return currentLang;
}

export function setLang(l: Lang): void {
  currentLang = l;
  saveLang(l);
  listeners.forEach((fn) => fn());
}

export function subscribeLang(fn: () => void): () => void {
  listeners.push(fn);
  return () => {
    listeners = listeners.filter((f) => f !== fn);
  };
}
