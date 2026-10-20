// Presentation mode — guided eight-chapter walkthrough of how the memory works.
// Three columns edge to edge: step data (left), the live 3D graph (centre),
// the presenter's script + derivation chain + final answer (right). Chapters
// 1-2 + the answer trail use the full cosmos / trace canvas; 3-7 replay the
// journey choreography on the trace subgraph.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CosmosCanvas from "./CosmosCanvas";
import { INK, toWorld3, type RGBA, type SelectedNode, type XYZ } from "../lib/cosmos";
import type { AnyCosmos } from "../lib/cosmos2d";
import { asJourney3D, type Journey3D } from "../lib/cosmos3d";
import { loadIngestFixture, loadTrace } from "../lib/data";
import { compileLiveTrace } from "../lib/compileLive";
import { fetchLiveTraceJson, CANONICAL_QUESTION, CANONICAL_KEYWORDS } from "../lib/live";
import type { LoadedCosmos, SourcePhase } from "../lib/cosmosSource";
import { makeTimeline, type Timeline } from "../lib/reveal";
import { CHAPTERS, type PresCtx } from "../lib/presentation";
import { buildChain, type AnswerPart } from "../lib/presChain";
import { getUiScale } from "../lib/uiScale";
import type { ChunkHit, CosmosGraph, IngestFixture, TravelTrace } from "../lib/types";
import type { Dict } from "../i18n/strings";
import { useI18n } from "../i18n/useI18n";
import { IngestPanel } from "./IngestPanel";
import { PresDataPanel } from "./PresDataPanel";
import { PresNarrationPanel } from "./PresNarrationPanel";
import { XIcon } from "./icons";
import { Inspector, type Inspection } from "./Inspector";

/** Chapters whose data is owed the trace-driven left column. */
const TRACE_CHAPTERS = new Set(["embed", "channels", "graph", "fusion", "context", "answer"]);

interface BadgePin {
  rank?: number;
  citation?: string;
  name: string;
  x: number;
  y: number;
  color: string;
  idx: number;
}

interface PinDef {
  rank?: number;
  citation?: string;
  name: string;
  idx: number;
  color: string;
}

function flyCameraToPoints(
  j3d: Journey3D | null,
  points: (XYZ | null | undefined)[],
  options?: {
    minDist?: number;
    padFactor?: number;
    elevation?: number;
    azimuth?: number;
    duration?: number;
  },
) {
  if (!j3d) return;
  const valid = points.filter((p): p is XYZ => p != null && !isNaN(p[0]) && !isNaN(p[1]) && !isNaN(p[2]));
  if (!valid.length) return;

  let sx = 0, sy = 0, sz = 0;
  for (const p of valid) {
    sx += p[0];
    sy += p[1];
    sz += p[2];
  }
  const cx = sx / valid.length;
  const cy = sy / valid.length;
  const cz = sz / valid.length;

  let maxR = 0;
  for (const p of valid) {
    const d = Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz);
    if (d > maxR) maxR = d;
  }

  const pad = options?.padFactor ?? 2.1;
  const minDist = options?.minDist ?? 22;
  const dist = Math.max(minDist, maxR * pad);
  const elev = options?.elevation ?? 0.36;
  const azim = options?.azimuth ?? 0.24;
  const zOffset = Math.sqrt(Math.max(0.1, 1 - elev * elev - azim * azim));

  const pos: XYZ = [
    cx + dist * azim,
    cy + dist * elev,
    cz + dist * zOffset,
  ];
  const target: XYZ = [cx, cy, cz];
  j3d.cameraFly(pos, target, options?.duration ?? 1300, "inout");
}

interface PresProps {
  onExit: () => void;
  /** The app-level cosmos data (live or demo) — the SAME source the main
   *  views render; the presentation never loads its own graph. */
  data: LoadedCosmos | null;
  /** App-level source phase; unreachable shows the honest gate, demo gates
   *  the fixture-only chapters (ingest note, trace_01). */
  source: SourcePhase;
  /** Latest live recall (chapters 3-8 replay it in live mode when present). */
  liveTrace: TravelTrace | null;
  onOpenDemo: () => void;
  onRetryLive: () => void;
}

/** Fixture-only chapters (ingest, demo trace replay) in live/unreachable
 *  mode: an honest gate instead of silently showing another index's data. */
function PresGate({ source, onOpenDemo, onRetryLive, what }: {
  source: SourcePhase;
  onOpenDemo: () => void;
  onRetryLive: () => void;
  what: "ingest" | "trace";
}) {
  const { t } = useI18n();
  return (
    <div className="presgate" role="status">
      <b>{what === "ingest" ? t.presGateIngestTitle : t.presGateTraceTitle}</b>
      <p className="dim">{what === "ingest" ? t.presGateIngestBody : t.presGateTraceBody}</p>
      <div className="overlay-actions">
        {source.phase !== "demo" && <button className="primary" onClick={onOpenDemo}>{t.openDemo}</button>}
        {(source.phase === "unreachable" || source.phase === "demo") && (
          <button onClick={onRetryLive}>{source.phase === "demo" ? t.backToLive : t.retryConnection}</button>
        )}
      </div>
    </div>
  );
}

export default function PresentationView({ onExit, data, source, liveTrace, onOpenDemo, onRetryLive }: PresProps) {
  const { t, lang, setLangL10n } = useI18n();
  const [chapter, setChapter] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [insp, setInsp] = useState<Inspection | null>(null);
  const [progress, setProgress] = useState<string | null>(t.presLoading);
  const [fixture, setFixture] = useState<IngestFixture | null>(null);
  /** true while the full-note overlay is open (Esc closes it first). */
  const [detailOpen, setDetailOpen] = useState(false);
  /** left data column: collapsible below the three-column breakpoint. */
  const [dataOpen, setDataOpen] = useState(false);
  /** citation focused from the final-answer panel (1-based). */
  const [activeCite, setActiveCite] = useState<number | null>(null);
  /** node under the cursor / selected on the centre canvas (hover tooltip). */
  const [hoverNode, setHoverNode] = useState<SelectedNode | null>(null);
  const [pickedNode, setPickedNode] = useState<SelectedNode | null>(null);

  // two canvases: full cosmos (ch 1-2, 9) + trace subgraph (ch 3-8)
  const [cosmosFull, setCosmosFull] = useState<AnyCosmos | null>(null);
  const [cosmosTrace, setCosmosTrace] = useState<AnyCosmos | null>(null);
  const [graph, setGraph] = useState<CosmosGraph | null>(null);
  const [posByIdx, setPosByIdx] = useState<(XYZ | null)[] | null>(null);
  const [trace, setTrace] = useState<TravelTrace | null>(null);
  const timeline = useRef<Timeline | null>(null);
  const animateRef = useRef(true);
  const jFull = cosmosFull ? asJourney3D(cosmosFull) : null;
  const jTrace = cosmosTrace ? asJourney3D(cosmosTrace) : null;

  const ch = CHAPTERS[chapter];
  const ctx: PresCtx = useMemo(
    () => ({
      trace,
      nodes: data?.identity?.nodes ?? graph?.nodes.length ?? 0,
      edges: data?.identity?.edges ?? graph?.edges.length ?? 0,
    }),
    [trace, graph, data],
  );

  /** The derivation chain, rebuilt from the same trace on every chapter/lang switch. */
  const chain = useMemo(() => buildChain(trace, ch.id, t, lang), [trace, ch.id, t, lang]);

  /** name -> index in the trace subgraph (citations, chunk focus). */
  const traceIdxOfName = useMemo(() => {
    const m = new Map<string, number>();
    trace?.graph.nodes.forEach((n, i) => {
      m.set(n.name, i);
      m.set(n.name.toLowerCase(), i);
    });
    return m;
  }, [trace]);
  const lookupNode = useCallback(
    (name: string): number | undefined =>
      traceIdxOfName.get(name) ?? traceIdxOfName.get(name.toLowerCase()),
    [traceIdxOfName],
  );

  // ---- data: the app-level LoadedCosmos (live OR explicitly-opened demo).
  // The presentation loads NO data of its own: whatever the app is connected
  // to (or the explicit demo snapshot) is what renders here too.
  const dataKey = `${data?.source ?? "none"}:${data?.identity?.index_id ?? "-"}`;
  useEffect(() => {
    if (!data) return;
    setGraph(data.graph);
    const m = new Map<string, XYZ>();
    data.coords.ids.forEach((id, i) => {
      const v = data.coords.xyz?.[i];
      if (v) m.set(id, v);
      else {
        const xy = data.coords.xy[i];
        if (xy) m.set(id, [xy[0], xy[1], 0]);
      }
    });
    setPosByIdx(
      data.graph.nodes.map((n) => {
        const v = m.get(n.id);
        return v ? toWorld3(v) : null;
      }),
    );
  }, [dataKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- Chapter 2: Ingest panel fixture (real SRE note with extraction records)
  useEffect(() => {
    let dead = false;
    loadIngestFixture()
      .then((fx) => !dead && setFixture(fx as IngestFixture))
      .catch(() => undefined);
    return () => {
      dead = true;
    };
  }, []);

  // ---- Chapters 3-8 trace:
  // Live mode: fetch REAL trace from the live HARS memory trace API (/travel/api/trace/trace)
  // compiled on the real live index; fallback to real production trace snapshot.
  const demoMode = source.phase === "demo";
  useEffect(() => {
    let dead = false;

    // 1. If liveTrace was already provided by caller (e.g. from query bar), use it
    if (liveTrace) {
      setTrace(liveTrace);
      return;
    }

    // 2. Explicit demo mode: load baked real trace
    if (demoMode) {
      loadTrace("traces/trace_01.json")
        .then((tr) => !dead && setTrace(tr))
        .catch(() => undefined);
      return;
    }

    // 3. Live mode: fetch real trace from the live HARS memory trace API!
    let cancelled = false;
    async function loadRealTrace() {
      try {
        const { result, events, clientLatencyS } = await fetchLiveTraceJson(CANONICAL_QUESTION, {
          keywords: CANONICAL_KEYWORDS,
          timeoutMs: 15_000,
        });
        if (cancelled || dead) return;

        if (data) {
          const idxOf = new Map(data.names.map((n, i) => [n, i] as const));
          const entityXy = new Map<string, [number, number]>();
          const entityXyz = new Map<string, XYZ>();
          data.graph.nodes.forEach((n, i) => {
            const v = posByIdx ? posByIdx[i] : null;
            if (v) {
              entityXyz.set(n.id, v);
              entityXy.set(n.id, [v[0], v[1]]);
            }
          });
          const compiled = compileLiveTrace(
            CANONICAL_QUESTION,
            result,
            clientLatencyS,
            {
              graph: data.graph,
              idxOf,
              entityXy,
              entityXyz,
              chunkXyz: data.chunkCoords ?? {},
            },
            events,
          );
          if (!cancelled && !dead) {
            setTrace(compiled);
            return;
          }
        }
      } catch (e) {
        console.warn("Live trace API fetch failed, falling back to canonical trace:", e);
      }

      // Offline fallback: load real canonical production trace from trace_01.json
      if (!cancelled && !dead) {
        loadTrace("traces/trace_01.json")
          .then((tr) => !cancelled && !dead && setTrace(tr))
          .catch(() => undefined);
      }
    }

    loadRealTrace();

    return () => {
      cancelled = true;
      dead = true;
    };
  }, [demoMode, liveTrace, dataKey, posByIdx]);

  // honest progress when there is no data yet (probing / not connected)
  useEffect(() => {
    if (data) return;
    setProgress(source.phase === "unreachable" ? t.notConnectedTitle : t.srcProbing);
  }, [source.phase, data, t]);

  // ---- load canvases when they first become visible
  const chViz = ch.viz;
  useEffect(() => {
    if (chViz !== "cosmos" || !cosmosFull || !graph || !posByIdx) return;
    let dead = false;
    setProgress(t.loadingStars);
    cosmosFull
      .load(graph, posByIdx, (done, total) => !dead && setProgress(`${t.loadingStars} ${Math.round((done / total) * 100)}%`))
      .then(() => !dead && setProgress(null))
      .catch((e) => !dead && setProgress(String(e)));
    return () => {
      dead = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cosmosFull, graph, posByIdx, chViz]);

  const [traceLoaded, setTraceLoaded] = useState(0);
  const loadedTraceRef = useRef<TravelTrace | null>(null);
  useEffect(() => {
    if (chViz !== "trace" || !cosmosTrace || !trace) return;
    if (loadedTraceRef.current === trace) return;
    let dead = false;
    const g2g = new Map<number, number>();
    trace.graph.nodes.forEach((n, li) => g2g.set(n.i, li));
    const cg: CosmosGraph = {
      nodes: trace.graph.nodes.map((n) => ({ id: n.name, t: n.type, d: n.desc, deg: n.deg })),
      edges: trace.graph.edges
        .map(([s, e]) => [g2g.get(s), g2g.get(e)] as [number, number])
        .filter(([s, e]) => s !== undefined && e !== undefined),
    };
    cosmosTrace.clear();
    cosmosTrace
      .load(
        cg,
        trace.graph.nodes.map((n) => {
          const p = n.xyz ?? (n.xy ? [n.xy[0], n.xy[1], 0] : null);
          return p ? toWorld3(p) : null;
        }),
      )
      .then(() => {
        if (dead) return;
        loadedTraceRef.current = trace;
        setProgress(null);
        // load() ends with fit(): re-run the chapter choreography so the
        // chapter camera keyframe wins over the fit position
        setTraceLoaded((v) => v + 1);
      })
      .catch((e) => !dead && setProgress(String(e)));
    return () => {
      dead = true;
    };
  }, [cosmosTrace, chViz, trace]);

  const traceCanvasActive = chViz === "trace";

  // ---- floating badges / pins projected into screen-space ----
  const [pinDefs, setPinDefs] = useState<PinDef[]>([]);
  const [pins, setPins] = useState<BadgePin[]>([]);

  useEffect(() => {
    if (!pinDefs.length || !cosmosTrace || !traceCanvasActive) {
      setPins([]);
      return;
    }
    let raf = 0;
    const tick = () => {
      const updated: BadgePin[] = [];
      for (const def of pinDefs) {
        const p = cosmosTrace.projectNode(def.idx);
        if (p && p.visible) {
          updated.push({
            ...def,
            x: p.x,
            y: p.y,
          });
        }
      }
      setPins(updated);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pinDefs, cosmosTrace, traceCanvasActive]);

  const playChapter = useCallback(() => {
    const cosmos = traceCanvasActive ? cosmosTrace : cosmosFull;
    const j3d = traceCanvasActive ? jTrace : jFull;
    timeline.current?.cancel();
    setPinDefs([]);
    if (!cosmos) return;
    cosmos.paint(null);
    cosmos.spotlight(null);
    cosmos.clearHalos();

    if (j3d) {
      j3d.clearContextLines();
      j3d.clearTrail();
      j3d.clearQueryBeacon();
      j3d.clearGraphWalkEdges();
      j3d.clearClusterHull();
      j3d.clearCitationBeams();
      j3d.setEdgeRingProgress(1e9);
      j3d.setEdgeOpacity(0.25);
      j3d.setControlsEnabled(false);
    }

    const tl = makeTimeline(1, false);
    timeline.current = tl;

    if (!traceCanvasActive) {
      // full-cosmos chapters (1: What this is, 2: Ingest, 9: Numbers)
      if (j3d) {
        j3d.cameraSet([0, 60, 145], [0, 0, 0]);
        tl.tween(400, 12000, (tt) => j3d.orbitAround([0, 0, 0], 148, tt * 0.9, 0.36));
      }
    } else if (trace) {
      const stage = ch.stage ?? 0;
      const qpos = trace.query.xyz ?? null;
      const wQ = qpos ? toWorld3(qpos) : null;

      const nodeW = (n: typeof trace.graph.nodes[0]): XYZ | null => {
        const p = n.xyz ?? (n.xy ? [n.xy[0], n.xy[1], 0] : null);
        return p ? toWorld3(p) : null;
      };

      const nameToIdx = new Map<string, number>();
      const lowNameToIdx = new Map<string, number>();
      trace.graph.nodes.forEach((n, i) => {
        nameToIdx.set(n.name, i);
        lowNameToIdx.set(n.name.toLowerCase(), i);
      });
      const getIdx = (name: string): number | undefined =>
        nameToIdx.get(name) ?? lowNameToIdx.get(name.toLowerCase());

      // ---------------------------------------------------------------
      // Chapter 3: Embedding (stage 0)
      // ---------------------------------------------------------------
      if (stage === 0 && wQ) {
        j3d?.setQueryBeacon(wQ, INK.query);

        // Highlight nearest candidate neighbors around the query
        const candHits = trace.vector_hits.slice(0, 5);
        const candIndices = candHits
          .flatMap((h) => h.entities.map((e) => getIdx(e)))
          .filter((i): i is number => i !== undefined);

        const candPoints: XYZ[] = [];
        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];

        candIndices.forEach((idx) => {
          paints.set(idx, { color: INK.cyan, size: 3.4 });
          halos.push({ idx, color: INK.cyan, size: 1.6 });
          const p = nodeW(trace.graph.nodes[idx]);
          if (p) candPoints.push(p);
        });

        tl.at(100, () => {
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          if (j3d && candPoints.length) {
            j3d.setContextLines(wQ, candPoints, [0.0, 0.95, 1.0, 0.55]);
          }
          flyCameraToPoints(j3d, [wQ], { minDist: 22, padFactor: 2.0, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 4: Channels - Vector vs BM25 (stage 1)
      // ---------------------------------------------------------------
      else if (stage === 1) {
        const vecNames = new Set(trace.vector_hits.slice(0, 10).flatMap((h) => h.entities.map((e) => e.toLowerCase())));
        const bm25Names = new Set(trace.bm25_hits.slice(0, 10).flatMap((h) => h.entities.map((e) => e.toLowerCase())));
        // Also map doc names so hits without explicit entity strings still highlight their relevant node
        trace.vector_hits.slice(0, 8).forEach((h) => {
          const docBase = h.doc.replace(/\.md$/, "").toLowerCase();
          vecNames.add(docBase);
          docBase.split(/[-_.]/).forEach((part) => { if (part.length > 3) vecNames.add(part); });
        });
        trace.bm25_hits.slice(0, 8).forEach((h) => {
          const docBase = h.doc.replace(/\.md$/, "").toLowerCase();
          bm25Names.add(docBase);
          docBase.split(/[-_.]/).forEach((part) => { if (part.length > 3) bm25Names.add(part); });
        });

        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const activeIndices = new Set<number>();
        const activePts: XYZ[] = [];

        trace.graph.nodes.forEach((n, idx) => {
          const low = n.name.toLowerCase();
          const isVec = vecNames.has(low);
          const isBm25 = bm25Names.has(low);
          if (!isVec && !isBm25) return;
          activeIndices.add(idx);
          const p = nodeW(n);
          if (p) activePts.push(p);

          if (isVec && isBm25) {
            paints.set(idx, { color: INK.overlap, size: 4.4 });
            halos.push({ idx, color: INK.overlap, size: 2.2 });
          } else if (isVec) {
            paints.set(idx, { color: INK.cyan, size: 3.6 });
            halos.push({ idx, color: INK.cyan, size: 1.8 });
          } else {
            paints.set(idx, { color: INK.bm25, size: 3.6 });
            halos.push({ idx, color: INK.bm25, size: 1.8 });
          }
        });

        tl.at(50, () => {
          cosmos.spotlight(activeIndices);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          j3d?.setEdgeOpacity(0.18);
          flyCameraToPoints(j3d, activePts, { minDist: 24, padFactor: 1.8, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 5: Graph Walk (stage 2)
      // ---------------------------------------------------------------
      else if (stage === 2) {
        const g2g = new Map<number, number>();
        trace.graph.nodes.forEach((n, li) => g2g.set(n.i, li));

        const seedIndices = new Set<number>();
        const neighborIndices = new Set<number>();
        const walkEdges: { from: XYZ; to: XYZ; color?: RGBA }[] = [];
        const activePts: XYZ[] = [];

        trace.graph.nodes.forEach((n, idx) => {
          if (n.seed) seedIndices.add(idx);
        });

        trace.graph.edges.forEach(([si, ti]) => {
          const sIdx = g2g.get(si);
          const tIdx = g2g.get(ti);
          if (sIdx === undefined || tIdx === undefined) return;
          const sNode = trace.graph.nodes[sIdx];
          const tNode = trace.graph.nodes[tIdx];
          if (sNode.seed || tNode.seed) {
            const p1 = nodeW(sNode);
            const p2 = nodeW(tNode);
            if (p1 && p2) {
              const from = sNode.seed ? p1 : p2;
              const to = sNode.seed ? p2 : p1;
              walkEdges.push({ from, to, color: INK.gold });
              if (!sNode.seed) neighborIndices.add(sIdx);
              if (!tNode.seed) neighborIndices.add(tIdx);
            }
          }
        });

        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const allWalk = new Set([...seedIndices, ...neighborIndices]);

        seedIndices.forEach((idx) => {
          paints.set(idx, { color: INK.gold, size: 4.4 });
          halos.push({ idx, color: INK.gold, size: 2.4 });
          const p = nodeW(trace.graph.nodes[idx]);
          if (p) activePts.push(p);
        });

        neighborIndices.forEach((idx) => {
          paints.set(idx, { color: INK.violet, size: 3.2 });
          halos.push({ idx, color: INK.violet, size: 1.7 });
          const p = nodeW(trace.graph.nodes[idx]);
          if (p) activePts.push(p);
        });

        tl.at(50, () => {
          cosmos.spotlight(allWalk);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          j3d?.setGraphWalkEdges(walkEdges, { intensity: 1.0 });
          j3d?.setEdgeOpacity(0.35);
          flyCameraToPoints(j3d, activePts, { minDist: 25, padFactor: 1.8, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 6: Fusion & Ranking (stage 4)
      // ---------------------------------------------------------------
      else if (stage === 4) {
        const topHits = trace.fusion.slice(0, 6);
        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const activeIndices = new Set<number>();
        const activePts: XYZ[] = [];
        const pinsList: PinDef[] = [];

        topHits.forEach((h, rankIdx) => {
          const rank = rankIdx + 1;
          const entities = [...h.entities];
          if (!entities.length) {
            const docBase = h.doc.replace(/\.md$/, "");
            entities.push(docBase);
            docBase.split(/[-_.]/).forEach((part) => { if (part.length > 3) entities.push(part); });
          }

          entities.forEach((name) => {
            const idx = getIdx(name);
            if (idx === undefined || activeIndices.has(idx)) return;
            activeIndices.add(idx);
            const p = nodeW(trace.graph.nodes[idx]);
            if (p) activePts.push(p);

            const col = rank === 1 ? INK.gold : rank === 2 ? INK.amber : INK.orange;
            const size = Math.max(2.8, 5.0 - rankIdx * 0.4);
            const haloSize = Math.max(1.6, 3.2 - rankIdx * 0.3);

            paints.set(idx, { color: col, size });
            halos.push({ idx, color: col, size: haloSize });

            pinsList.push({
              rank,
              name,
              idx,
              color: rank === 1 ? "#ffc83b" : rank === 2 ? "#ffa726" : "#ff7043",
            });
          });
        });

        tl.at(50, () => {
          cosmos.spotlight(activeIndices);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          setPinDefs(pinsList);
          j3d?.setEdgeOpacity(0.15);
          flyCameraToPoints(j3d, activePts, { minDist: 22, padFactor: 1.8, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 7: Context Assembly (stage 6)
      // ---------------------------------------------------------------
      else if (stage === 6) {
        const ents = trace.context_blocks.entities;
        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const activeIndices = new Set<number>();
        const ctxPoints: XYZ[] = [];

        ents.forEach((e) => {
          const idx = getIdx(e.entity);
          if (idx === undefined) return;
          activeIndices.add(idx);
          const p = nodeW(trace.graph.nodes[idx]);
          if (p) ctxPoints.push(p);

          paints.set(idx, { color: INK.gold, size: 3.8 });
          halos.push({ idx, color: INK.gold, size: 2.0 });
        });

        tl.at(50, () => {
          cosmos.spotlight(activeIndices);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          if (j3d) {
            const hullPoints = ctxPoints.length >= 1 ? ctxPoints : (wQ ? [wQ] : []);
            if (hullPoints.length >= 1) j3d.setClusterHull(hullPoints, INK.gold);
            if (wQ && ctxPoints.length) j3d.setContextLines(wQ, ctxPoints, [1.0, 0.8, 0.25, 0.7]);
          }
          flyCameraToPoints(j3d, wQ ? [wQ, ...ctxPoints] : ctxPoints, { minDist: 26, padFactor: 1.9, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 8: Grounded Answer (stage 7)
      // ---------------------------------------------------------------
      else if (stage === 7) {
        const trail: XYZ[] = [];
        if (wQ) trail.push(wQ);

        trace.fusion.slice(0, 5).forEach((h) => {
          if (h.xyz) trail.push(toWorld3(h.xyz));
        });

        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const activeIndices = new Set<number>();
        const citedPoints: XYZ[] = [];
        const citationPins: PinDef[] = [];
        const pinned = new Set<number>();
        const beamItems: { from: XYZ; to: XYZ; color?: RGBA }[] = [];

        trace.answer_preview.forEach((b, bIdx) => {
          const cNum = String(bIdx + 1);
          const entities = [...b.entities];
          if (!entities.length && b.doc) {
            const docBase = b.doc.replace(/\.md$/, "");
            entities.push(docBase);
            docBase.split(/[-_.]/).forEach((part) => { if (part.length > 3) entities.push(part); });
          }

          entities.forEach((name) => {
            const idx = getIdx(name);
            if (idx === undefined) return;
            activeIndices.add(idx);
            const p = nodeW(trace.graph.nodes[idx]);
            if (p) {
              citedPoints.push(p);
              trail.push(p);
              if (wQ) beamItems.push({ from: wQ, to: p, color: INK.gold });
            }

            paints.set(idx, { color: INK.gold, size: 4.2 });
            halos.push({ idx, color: INK.gold, size: 2.2 });

            // one pin per node: several citations can name the same entity
            if (!pinned.has(idx)) {
              pinned.add(idx);
              citationPins.push({
                citation: cNum,
                name,
                idx,
                color: "#ffd54f",
              });
            }
          });
        });

        tl.at(50, () => {
          cosmos.spotlight(activeIndices);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          setPinDefs(citationPins);
          if (j3d) {
            j3d.setTrail(trail, [1.0, 0.35, 0.65, 1.0]);
            if (beamItems.length) j3d.setCitationBeams(beamItems);
          }
          flyCameraToPoints(j3d, citedPoints.length ? citedPoints : trail, { minDist: 26, padFactor: 2.0, duration: 1400 });
        });
      }
    }

    tl.start();
    return () => {
      tl.cancel();
      setPinDefs([]);
    };
  }, [chapter, trace, cosmosFull, cosmosTrace, traceCanvasActive, jTrace, jFull, ch.stage, traceLoaded]);

  useEffect(() => {
    if (progress) return; // wait for data
    if (playing) playChapter();
    else timeline.current?.cancel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, progress, playChapter]);

  // a chapter switch always drops the focused citation and node hover
  useEffect(() => {
    setActiveCite(null);
    setHoverNode(null);
    setPickedNode(null);
  }, [chapter]);

  const nodeInfo = pickedNode ?? hoverNode;

  /** Light the given trace-subgraph nodes and fly the camera to them. */
  const lightNodes = useCallback(
    (idxs: number[], duration = 900) => {
      if (!cosmosTrace || !trace || !idxs.length) return;
      const paints = new Map<number, { color: RGBA; size?: number }>();
      const halos: { idx: number; color?: RGBA; size?: number }[] = [];
      const pts: XYZ[] = [];
      for (const idx of idxs) {
        const n = trace.graph.nodes[idx];
        if (!n) continue;
        const p = n.xyz ?? (n.xy ? [n.xy[0], n.xy[1], 0] : null);
        if (p) pts.push(toWorld3(p));
        paints.set(idx, { color: INK.gold, size: 5.4 });
        halos.push({ idx, color: INK.gold, size: 3.0 });
      }
      cosmosTrace.paint(paints);
      cosmosTrace.spotlight(new Set(idxs));
      cosmosTrace.setHalos(halos);
      flyCameraToPoints(jTrace, pts, { minDist: 18, padFactor: 1.7, duration });
    },
    [cosmosTrace, jTrace, trace],
  );

  /** Click on an answer citation: focus its evidence nodes (or its chunk). */
  const focusAnswerPart = useCallback(
    (part: AnswerPart) => {
      setActiveCite(part.n);
      const idxs = part.entities.map(lookupNode).filter((i): i is number => i !== undefined);
      if (idxs.length) {
        lightNodes(idxs);
        return;
      }
      // no named entity on the quote: fall back to the chunk that produced it
      const hit = trace?.fusion.find((h) => h.chunk_id === part.chunkId);
      const hitIdxs = (hit?.entities ?? []).map(lookupNode).filter((i): i is number => i !== undefined);
      if (hitIdxs.length) {
        lightNodes(hitIdxs);
        return;
      }
      const xyz = part.xyz ?? hit?.xyz ?? null;
      if (xyz && jTrace) {
        jTrace.cameraFly(toWorld3(xyz), toWorld3(xyz), 900, "inout");
      }
    },
    [lightNodes, lookupNode, trace, jTrace],
  );

  /** Click on a chunk row in the data column: focus its nodes. */
  const focusChunk = useCallback(
    (hit: ChunkHit) => {
      const idxs = (hit.entities ?? []).map(lookupNode).filter((i): i is number => i !== undefined);
      if (idxs.length) lightNodes(idxs);
      else if (hit.xyz && jTrace) jTrace.cameraFly(toWorld3(hit.xyz), toWorld3(hit.xyz), 900, "inout");
    },
    [lightNodes, lookupNode, jTrace],
  );

  // ---- presenter controls
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") { animateRef.current = true; setChapter((c) => Math.min(CHAPTERS.length - 1, c + 1)); }
      else if (e.key === "ArrowLeft") { setChapter((c) => Math.max(0, c - 1)); }
      else if (e.key === " ") { setPlaying((p) => !p); e.preventDefault(); }
      else if (e.key.toLowerCase() === "f") {
        if (document.fullscreenElement) void document.exitFullscreen();
        else void document.documentElement.requestFullscreen();
      } else if (e.key === "Escape") {
        if (detailOpen) setDetailOpen(false);
        else onExit();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [detailOpen, onExit]);

  const goto = (i: number) => {
    setInsp(null);
    setDetailOpen(false);
    setPlaying(true);
    setChapter((c) => {
      void c;
      animateRef.current = true;
      return i;
    });
  };

  const uiScale = getUiScale();
  const legend = legendFor(ch.id, t);

  return (
    <div className="pres">
      <div className="pres-progress" aria-hidden="true">
        {CHAPTERS.map((c, i) => (
          <span key={c.id} className={i <= chapter ? "on" : ""} />
        ))}
      </div>

      <div className="pres-top">
        <span className="pres-brand">MEMORY TRAVEL · {t.presLaunch}</span>
        <a className="doclink" href="/travel/presentation.html" target="_blank" rel="noopener">{t.tabPresentation}</a>
        <a className="doclink" href="/evals/" target="_blank" rel="noopener" title={t.evalsTitle}>{t.tabEvals}</a>
        <button onClick={onExit} title={t.presExit}>
          <XIcon size={11} /> {t.presExit}
        </button>
      </div>

      <div className="pres-grid">
        <PresDataPanel
          chapterId={ch.id}
          trace={trace}
          fixture={fixture}
          identity={data?.identity ?? null}
          graph={graph}
          nodes={ctx.nodes}
          edges={ctx.edges}
          demo={data?.source === "demo"}
          chapterIndex={chapter}
          chapterTotal={CHAPTERS.length}
          open={dataOpen}
          t={t}
          lang={lang}
          activeCite={activeCite}
          onFocusAnswer={focusAnswerPart}
          onFocusChunk={focusChunk}
          onOpenNote={() => setDetailOpen(true)}
        />

        <main className="pres-stage">
          <div className="pres-canvas-cell">
            <div className={traceCanvasActive ? "canvaswrap" : "canvaswrap hidden"}>
              <CosmosCanvas label="pres-trace" onCosmos={setCosmosTrace} onSelect={setPickedNode} onHover={setHoverNode} onError={(e) => setProgress(String(e))} />
            </div>
            <div className={traceCanvasActive ? "canvaswrap hidden" : "canvaswrap"}>
              <CosmosCanvas label="pres-full" onCosmos={setCosmosFull} onSelect={setPickedNode} onHover={setHoverNode} onError={(e) => setProgress(String(e))} />
            </div>

            {/* Floating 3D badge pins (ranked winners, grounded citations).
                Projected coordinates are canvas client px; the DOM layer below
                is zoomed with #root, so divide by the UI scale to stay aligned. */}
            {pins.length > 0 && (
              <div className="pres-floating-pins" aria-hidden="true">
                {pins.map((pin) => (
                  <div
                    key={`${pin.idx}-${pin.rank ?? pin.citation}`}
                    className="pres-pin"
                    style={{
                      left: `${pin.x / uiScale}px`,
                      top: `${pin.y / uiScale}px`,
                      borderColor: pin.color,
                      boxShadow: `0 0 16px rgba(0,0,0,0.85), 0 0 10px ${pin.color}`,
                    }}
                    onClick={() => {
                      if (pin.citation) setActiveCite(Number(pin.citation));
                      const node = trace?.graph.nodes[pin.idx];
                      const p = node?.xyz ?? (node?.xy ? [node.xy[0], node.xy[1], 0] : null);
                      if (p && jTrace) flyCameraToPoints(jTrace, [toWorld3(p)], { minDist: 18, duration: 900 });
                    }}
                  >
                    <span className="pres-pin-num" style={{ background: pin.color }}>
                      {pin.rank ? `#${pin.rank}` : `[${pin.citation}]`}
                    </span>
                    <span className="pres-pin-label">{pin.name}</span>
                  </div>
                ))}
              </div>
            )}

            {nodeInfo && (
              <div className="pres-nodecard" role="status">
                <b>{nodeInfo.name}</b>
                {traceCanvasActive && trace?.graph.nodes[nodeInfo.graphIdx] ? (
                  <>
                    <span>
                      {trace.graph.nodes[nodeInfo.graphIdx].type || t.presNotRecorded}
                      {" · "}
                      {t.hiwDegree} {trace.graph.nodes[nodeInfo.graphIdx].deg}
                    </span>
                    {trace.graph.nodes[nodeInfo.graphIdx].desc && (
                      <p>{trace.graph.nodes[nodeInfo.graphIdx].desc.slice(0, 150)}</p>
                    )}
                  </>
                ) : graph?.nodes[nodeInfo.graphIdx] ? (
                  <span>
                    {graph.nodes[nodeInfo.graphIdx].t || t.presNotRecorded}
                    {" · "}
                    {t.hiwDegree} {graph.nodes[nodeInfo.graphIdx].deg}
                  </span>
                ) : null}
              </div>
            )}

            <div className="pres-stage-bar">
              <span className="pres-kicker">{ch.viz === "trace" ? t.presVizTrace : t.presVizCosmos}</span>
              {legend.length > 0 && (
                <div className="pres-legend" aria-label={t.presGraphLegend}>
                  {legend.map((l) => (
                    <span className="pres-legend-item" key={l.label}>
                      <i style={{ background: l.color, boxShadow: `0 0 8px ${l.color}` }} />
                      {l.label}
                    </span>
                  ))}
                </div>
              )}
              <span className="spacer" />
              {traceCanvasActive && trace && (
                <button
                  type="button"
                  className="pres-stage-detail"
                  onClick={() => setInsp({ kind: "stage", stage: ch.stage ?? 0 })}
                >
                  {t.inspectOpen}
                </button>
              )}
            </div>

            {TRACE_CHAPTERS.has(ch.id) && !trace && (
              <PresGate source={source} onOpenDemo={onOpenDemo} onRetryLive={onRetryLive} what="trace" />
            )}
            {ch.id === "ingest" && !fixture && (
              <PresGate source={source} onOpenDemo={onOpenDemo} onRetryLive={onRetryLive} what="ingest" />
            )}
            {progress && <p className="loading pres-loading">{progress}</p>}
          </div>
        </main>

        <PresNarrationPanel
          chapter={ch}
          chapterIndex={chapter}
          chapterTotal={CHAPTERS.length}
          ctx={ctx}
          trace={trace}
          chain={chain}
          lang={lang}
          t={t}
          playing={playing}
          activeCite={activeCite}
          onGoto={goto}
          onTogglePlay={() => setPlaying((p) => !p)}
          onToggleLang={() => setLangL10n(lang === "ru" ? "en" : "ru")}
          onFocusAnswer={focusAnswerPart}
          onToggleData={() => setDataOpen((v) => !v)}
          dataOpen={dataOpen}
        />
      </div>

      {detailOpen && ch.id === "ingest" && fixture && (
        <div className="pres-note-modal" role="dialog" aria-modal="true" aria-label={t.presNoteTitle}>
          <div className="pres-note-shell">
            <button type="button" className="pres-note-close" onClick={() => setDetailOpen(false)}>
              <XIcon size={11} /> {t.presCloseNote}
            </button>
            <IngestPanel fixture={fixture} lang={lang} t={t} initialDocOpen />
          </div>
        </div>
      )}

      {insp && (
        <Inspector
          insp={insp}
          trace={trace}
          onClose={() => setInsp(null)}
          onOpenChunk={(id) => setInsp({ kind: "chunk", chunkId: id })}
          onFocusPoint={() => undefined}
          t={t}
        />
      )}
    </div>
  );
}

/** Per-chapter colour legend for the centre graph (same INK palette). */
function legendFor(id: string, t: Dict): { color: string; label: string }[] {
  const base = [
    { color: "#7df3ff", label: t.presLegendVector },
    { color: "#ffb724", label: t.presLegendBm25 },
    { color: "#38ff94", label: t.presLegendBoth },
  ];
  switch (id) {
    case "embed":
      return [
        { color: "#ff477e", label: t.presLegendQuery },
        { color: "#7df3ff", label: t.presChainCandidates },
      ];
    case "channels":
      return base;
    case "graph":
      return [
        { color: "#ffd166", label: t.presLegendSeeds },
        { color: "#bf66ff", label: t.presLegendNeighbours },
      ];
    case "fusion":
      return [
        { color: "#ffc83b", label: "#1" },
        { color: "#ffa726", label: "#2" },
        { color: "#ff7043", label: "#3+" },
      ];
    case "context":
    case "answer":
      return [{ color: "#ffd166", label: t.presLegendCitations }];
    default:
      return [];
  }
}
