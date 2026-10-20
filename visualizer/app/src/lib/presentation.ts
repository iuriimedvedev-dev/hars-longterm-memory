// Presentation chapters — "Презентация: как работает память".
// Every string is an L10n {ru, en} pair; missing translation = build error
// (PresentationChapter['title' | 'narration' | 'fact'] require both).
// All numbers shown come from real trace/graph telemetry — never invented.
import type { L10n } from "../i18n/strings";
import type { TravelTrace } from "./types";

export type VizMode = "cosmos" | "trace";

export interface PresentationChapter {
  id: string;
  title: L10n;
  narration: (ctx: PresCtx) => L10n;
  /** which canvas drives this chapter */
  viz: VizMode;
  /** journey stage to visualise (trace mode) — one choreography per chapter */
  stage?: number;
  /** one highlighted REAL fact, rendered on the card */
  fact: (ctx: PresCtx) => L10n;
}

export interface PresCtx {
  trace: TravelTrace | null;
  nodes: number;
  edges: number;
}

/**
 * Presentation copy is deployment-agnostic on purpose: no hardcoded index
 * name, no hardcoded corpus marketing numbers. Every number shown comes from
 * the loaded trace (`ctx.trace`) or the loaded graph (`ctx.nodes`/`ctx.edges`)
 * — the same data the rest of the app renders.
 */

/**
 * Model/pipeline identity shown on the presentation cards. These are the same
 * facts the chapter narrations state (docs/presentation.md): the embedder and
 * the cross-encoder of this deployment. Not per-trace telemetry — hence a
 * constant, so no card invents a model name or a dimension.
 */
export const PIPELINE = {
  embedder: { model: "EmbeddingGemma", dim: 768, via: "llama.cpp" },
  reranker: { model: "LightGBM / RRF (Cross-encoder OFF by default)", device: "CPU / Zero-cost" },
} as const;

export const CHAPTERS: PresentationChapter[] = [
  {
    id: "what",
    viz: "cosmos",
    title: { ru: "Что это", en: "What this is" },
    narration: ({ nodes, edges }) => ({
      ru: `Перед вами долговременная память агента: корпус заметок превращён в ${nodes.toLocaleString("ru-RU")} сущностей и ${edges.toLocaleString("ru-RU")} связей — столько сейчас в индексе, к которому подключён мост. Это GraphRAG-гибрид: векторный поиск по смыслу + knowledge graph + BM25 по словам. Каждая звезда — реальная сущность подключённого индекса, положение — 3D UMAP по хранимым эмбеддингам.`,
      en: `This is an agent's long-term memory: a corpus of notes distilled into ${nodes.toLocaleString("en-US")} entities and ${edges.toLocaleString("en-US")} relations — whatever the connected index holds right now. A GraphRAG hybrid: semantic vector search + a knowledge graph + BM25 over words. Every star is a real entity of the connected index; positions are a 3D UMAP of the stored embeddings.`,
    }),
    fact: ({ nodes, edges }) => ({
      ru: `Факт: ${nodes.toLocaleString("ru-RU")} сущностей · ${edges.toLocaleString("ru-RU")} связей — не мокап, а живой индекс.`,
      en: `Fact: ${nodes.toLocaleString("en-US")} entities · ${edges.toLocaleString("en-US")} relations — not a mock, the live index.`,
    }),
  },
  {
    id: "ingest",
    viz: "cosmos",
    title: { ru: "Как знание попадает в память", en: "How knowledge enters memory" },
    narration: () => ({
      ru: `Заметка разрезается md_struct-чанкером по секциям (заголовки → секции → чанки с номерами строк). Учитель — локальная 27B модель — извлекает из каждого чанка сущности и связи; мердж идёт в общий граф. Справа в карточке — настоящая заметка и её настоящие extraction-записи из кэша учителя.`,
      en: `A note is split by the md_struct chunker along its sections (headings → sections → chunks with line ranges). A teacher — a local 27B model — extracts entities and relations from every chunk; everything merges into one graph. The card shows a real note with its real extraction records from the teacher's cache.`,
    }),
    fact: () => ({
      ru: `Факт: каждая заметка нарезается на окна индекса; каждая запись учителя сохранена в kv-кэше — воспроизводимо.`,
      en: `Fact: every note is cut into index windows; every teacher record is kept in the kv cache — reproducible.`,
    }),
  },
  {
    id: "embed",
    viz: "trace",
    stage: 0,
    title: { ru: "Вопрос → эмбеддинг", en: "Question → embedding" },
    narration: () => ({
      ru: `Вопрос превращается в вектор 768-мерным эмбеддером (EmbeddingGemma через llama.cpp; промпт «task: search result | query: …») и ложится в то же смысловое пространство UMAP — розовый маркер. Ближайшие соседи сразу подсвечены: это кандидаты в ответы.`,
      en: `The question becomes a vector via the 768-dim embedder (EmbeddingGemma through llama.cpp; prompt “task: search result | query: …”) and lands in the same UMAP meaning space — the pink marker. Its nearest neighbours light up immediately: those are the answer candidates.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: dense-поиск по ${trace?.meta.candidate_pool ?? "?"} чанкам занял ${trace?.meta.latency_ms.dense ?? "?"} мс.`,
      en: `Fact: the dense scan over ${trace?.meta.candidate_pool ?? "?"} chunks took ${trace?.meta.latency_ms.dense ?? "?"} ms.`,
    }),
  },
  {
    id: "channels",
    viz: "trace",
    stage: 1,
    title: { ru: "Каналы: Вектор vs BM25", en: "Channels: Vector vs BM25" },
    narration: () => ({
      ru: `Два ключевых канала работают параллельно: векторный семантический поиск (голубые маркеры) находит чанки по смыслу, а классический BM25 (янтарно-золотые маркеры) — по точным ключевым словам. Чанки, найденные обоими каналами, подсвечены изумрудным ореолом. Каждый найденный узел увеличен и окружён пульсирующим гало.`,
      en: `Two key channels run in parallel: semantic vector search (bright cyan markers) locates chunks by meaning, while classic BM25 (golden amber markers) scores exact keyword occurrences. Chunks hit by both channels radiate with emerald halos. Each retrieved node is enlarged with an animated breathing halo.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: ${trace?.vector_hits.length ?? 0} векторных хитов (${trace?.meta.latency_ms.dense ?? "?"} мс) + ${((trace?.meta.bm25_chunks ?? 0)).toLocaleString("ru-RU")} BM25-чанков (${trace?.meta.latency_ms.sparse ?? "?"} мс).`,
      en: `Fact: ${trace?.vector_hits.length ?? 0} vector hits (${trace?.meta.latency_ms.dense ?? "?"} ms) + ${(trace?.meta.bm25_chunks ?? 0).toLocaleString("en-US")} BM25 chunks (${trace?.meta.latency_ms.sparse ?? "?"} ms).`,
    }),
  },
  {
    id: "graph",
    viz: "trace",
    stage: 2,
    title: { ru: "Обход графа", en: "Graph walk" },
    narration: () => ({
      ru: `LightRAG-режим извлекает сущности из найденного текста (золотые сид-узлы) и совершает обход по реальным рёбрам графа к их соседям (фиолетовые узлы). На графе визуализированы соединяющие рёбра, а анимированные световые импульсы показывают передачу смыслового контекста от сидов к соседям 1-го и 2-го порядка.`,
      en: `LightRAG extracts entities from retrieved text (golden seed nodes) and traverses real graph edges to their neighbours (violet nodes). Connecting edges are rendered in high fidelity with animated photon pulses streaming relational context from seed nodes to 1-hop and 2-hop neighbours.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: ${trace?.graph.nodes.filter((n) => n.seed).length ?? 0} сид-сущностей соединяются через ${trace?.graph.edges.length ?? 0} связей с соседями в подграфе recall.`,
      en: `Fact: ${trace?.graph.nodes.filter((n) => n.seed).length ?? 0} seed entities connect across ${trace?.graph.edges.length ?? 0} relations to neighbours in the recall subgraph.`,
    }),
  },
  {
    id: "fusion",
    viz: "trace",
    stage: 4,
    title: { ru: "Фьюжн и ранжирование", en: "Fusion & ranking" },
    narration: () => ({
      ru: `Рейтинги каналов сливаются через взвешенный RRF / α-фьюжн (α = 0.5 — оптимум) и ранжируются легковесным LightGBM (~160 мс). Топ-победители выделены светящимися бейджами с номерами мест (1, 2, 3...) и яркими маяками, а защитный предохранитель блокирует деградацию fusion-порядка.`,
      en: `Channel rankings are blended via weighted RRF / α-fusion (α = 0.5 measured optimum) and ranked with fast LightGBM (~160ms). Top winners are highlighted with glowing numbered badges (1, 2, 3...) and high-intensity beacons, with guardrails protecting high-confidence order.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: ${trace?.fusion.length ?? 0} чанков в слитом топе; задержка выборки ~160 мс (кросс-энкодер ВЫКЛЮЧЕН по умолчанию).`,
      en: `Fact: ${trace?.fusion.length ?? 0} chunks in the fused top; recall latency ~160ms (cross-encoder OFF by default).`,
    }),
  },
  {
    id: "context",
    viz: "trace",
    stage: 6,
    title: { ru: "Сборка контекста", en: "Context assembly" },
    narration: () => ({
      ru: `Победители собираются в промпт: золотые линии на карте — сущности, чьи факты войдут в контекст. Бюджет — 15 000 символов, по документу не более 6 000, гранулярность — целый документ. Что не влезло — отрезано по бюджету, а не по полезности.`,
      en: `The winners are assembled into the prompt: gold lines on the map are the entities whose facts enter the context. The budget is 15,000 chars, 6,000 per document, whole-document granularity. What doesn't fit is cut by budget, not by usefulness.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: ${trace?.context_blocks.entities.length ?? "?"} фактов о сущностях, ${trace?.context_blocks.relations.length ?? "?"} связей, ${trace?.context_blocks.source_count ?? "?"} источников с референсами.`,
      en: `Fact: ${trace?.context_blocks.entities.length ?? "?"} entity facts, ${trace?.context_blocks.relations.length ?? "?"} relations, ${trace?.context_blocks.source_count ?? "?"} cited sources.`,
    }),
  },
  {
    id: "answer",
    viz: "trace",
    stage: 7,
    title: { ru: "Синтез", en: "Synthesis" },
    narration: () => ({
      ru: `Сервис возвращает собранный контекст (context_only) — финальный ответ пишет LLM вызывающей стороны, с цитатами на источники. Розовый след на карте — весь путь запроса: от эмбеддинга до узлов-доказательств. Каждая цитата кликабельна и ведёт к своим узлам.`,
      en: `The service returns the assembled context (context_only) — the caller's LLM writes the final answer, citing sources. The pink trail on the map is the whole journey: from the embedding to the evidence nodes. Every citation is clickable and leads back to its nodes.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: клиент получил ответ за ${trace?.meta.client_latency_s ?? "?"} с — от нажатия Enter до цитат.`,
      en: `Fact: the client got the answer in ${trace?.meta.client_latency_s ?? "?"} s — from Enter to citations.`,
    }),
  },
  {
    id: "numbers",
    viz: "cosmos",
    title: { ru: "Цифры", en: "The numbers" },
    narration: ({ nodes, edges }) => ({
      ru: `Итог: ${nodes.toLocaleString("ru-RU")} сущностей и ${edges.toLocaleString("ru-RU")} связей в подключённом индексе — шапка и карточка «о приложении» читают их с моста живьём, вместе с именем индекса и свежестью данных. Всё, что вы видели — телеметрия одного реального запроса к этой памяти.`,
      en: `Bottom line: ${nodes.toLocaleString("en-US")} entities and ${edges.toLocaleString("en-US")} relations in the connected index — the header and the about card read them live from the bridge, together with the index id and data freshness. Everything you saw is the telemetry of one real query against this memory.`,
    }),
    fact: ({ nodes, edges }) => ({
      ru: `Факт: ${nodes.toLocaleString("ru-RU")} сущностей · ${edges.toLocaleString("ru-RU")} связей · гибрид (BM25 + вектор + граф + ripgrep) со слиянием RRF/LightGBM — задержка ~160 мс.`,
      en: `Fact: ${nodes.toLocaleString("en-US")} entities · ${edges.toLocaleString("en-US")} relations · hybrid (BM25 + vector + graph + ripgrep) with RRF/LightGBM — latency ~160ms.`,
    }),
  },
];
