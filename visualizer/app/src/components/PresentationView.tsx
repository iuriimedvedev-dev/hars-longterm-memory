// Presentation mode — guided 9-chapter walkthrough of how the memory works.
// Dark fullscreen stage; narration card (RU/EN) over the LIVE 3D starfield
// driven by real data (full graph + the enriched trace). Chapters 1-2 + 9 use
// the full cosmos; 3-8 replay the journey choreography on the trace subgraph.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CosmosCanvas from "./CosmosCanvas";
import { INK, toWorld3, type RGBA, type XYZ } from "../lib/cosmos";
import type { AnyCosmos } from "../lib/cosmos2d";
import { asJourney3D, type Journey3D } from "../lib/cosmos3d";
import { loadIngestFixture, loadTrace } from "../lib/data";
import { compileLiveTrace } from "../lib/compileLive";
import { fetchLiveTraceJson, CANONICAL_QUESTION, CANONICAL_KEYWORDS } from "../lib/live";
import type { LoadedCosmos, SourcePhase } from "../lib/cosmosSource";
import { makeTimeline, type Timeline } from "../lib/reveal";
import { CHAPTERS, type PresCtx } from "../lib/presentation";
import type { CosmosGraph, IngestFixture, TravelTrace } from "../lib/types";
import { pick } from "../i18n/strings";
import { useI18n } from "../i18n/useI18n";
import { AnimatedCard } from "./AnimatedCard";
import { IngestPanel } from "./IngestPanel";
import { PresStageCards } from "./PresStageCards";
import { ChevronLeftIcon, ChevronRightIcon, GearIcon, PauseIcon, PlayIcon, XIcon } from "./icons";
import { Inspector, type Inspection } from "./Inspector";

/** Chapters whose data is owed a docked card row (3–8 = the transformation). */
const CARD_CHAPTERS = new Set(["embed", "channels", "graph", "fusion", "context", "answer"]);

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
  const [menuOpen, setMenuOpen] = useState(false);
  const [insp, setInsp] = useState<Inspection | null>(null);
  const [progress, setProgress] = useState<string | null>(t.presLoading);
  const [fixture, setFixture] = useState<IngestFixture | null>(null);
  /** true while a chapter's drill-down detail card is open (Esc closes it first). */
  const [detailOpen, setDetailOpen] = useState(false);

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

            citationPins.push({
              citation: cNum,
              name,
              idx,
              color: "#ffd54f",
            });
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
        if (menuOpen) setMenuOpen(false);
        else if (!detailOpen) onExit();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menuOpen, detailOpen, onExit]);

  const goto = (i: number) => {
    setMenuOpen(false);
    setInsp(null);
    setDetailOpen(false);
    setPlaying(true);
    setChapter((c) => {
      void c;
      animateRef.current = true;
      return i;
    });
  };

  return (
    <div className="pres">
      <div className={traceCanvasActive ? "canvaswrap" : "canvaswrap hidden"}>
        <CosmosCanvas label="pres-trace" onCosmos={setCosmosTrace} onSelect={() => {}} onHover={() => {}} onError={(e) => setProgress(String(e))} />
      </div>
      <div className={traceCanvasActive ? "canvaswrap hidden" : "canvaswrap"}>
        <CosmosCanvas label="pres-full" onCosmos={setCosmosFull} onSelect={() => {}} onHover={() => {}} onError={(e) => setProgress(String(e))} />
      </div>

      {/* Floating 3D badge pins (ranked winners, grounded citations) */}
      {pins.length > 0 && (
        <div className="pres-floating-pins" aria-hidden="true">
          {pins.map((pin) => (
            <div
              key={`${pin.idx}-${pin.rank ?? pin.citation}`}
              className="pres-pin"
              style={{
                left: `${pin.x}px`,
                top: `${pin.y}px`,
                borderColor: pin.color,
                boxShadow: `0 0 16px rgba(0,0,0,0.85), 0 0 10px ${pin.color}`,
              }}
              onClick={() => {
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

      <div className="pres-top">
        <span className="pres-brand">MEMORY TRAVEL · {t.presLaunch}</span>
        <a className="doclink" href="/travel/presentation.html" target="_blank" rel="noopener">{t.tabPresentation}</a>
        <a className="doclink" href="/evals/" target="_blank" rel="noopener" title={t.evalsTitle}>{t.tabEvals}</a>
        <button className="langtoggle" onClick={() => setLangL10n(lang === "ru" ? "en" : "ru")} title={t.presLang}>
          {lang.toUpperCase()}
        </button>
        <button onClick={() => setMenuOpen((o) => !o)}>{t.presMenu}</button>
        <button onClick={onExit} title={t.presExit}>
          <XIcon size={11} /> {t.presExit}
        </button>
      </div>

      {menuOpen && (
        <div className="pres-menu">
          {CHAPTERS.map((c, i) => (
            <button key={c.id} className={i === chapter ? "on" : ""} onClick={() => goto(i)}>
              <span className="n">{i + 1}</span> {pick(c.title, lang)}
            </button>
          ))}
        </div>
      )}

      {/* chapter 2: compact ingest schematic with optional drill-downs.
          The fixture replays ONE captured note from the demo snapshot, so it
          only exists in demo mode; otherwise the honest gate shows. */}
      {ch.id === "ingest" && (fixture
        ? <IngestPanel fixture={fixture} lang={lang} t={t} onDetailOpen={setDetailOpen} />
        : <PresGate source={source} onOpenDemo={onOpenDemo} onRetryLive={onRetryLive} what="ingest" />)}

      {/* chapters 3-8: a compact schematic strip + one optional detail card */}
      {CARD_CHAPTERS.has(ch.id) && (
        trace
          ? <PresStageCards chapterId={ch.id} trace={trace} t={t} lang={lang} onDetailOpen={setDetailOpen} />
          : <PresGate source={source} onOpenDemo={onOpenDemo} onRetryLive={onRetryLive} what="trace" />
      )}

      <div className="pres-bottom">
        <div className="pres-progress" aria-hidden="true">
          {CHAPTERS.map((c, i) => (
            <span key={c.id} className={i <= chapter ? "on" : ""} />
          ))}
        </div>
        <div className="pres-chapternum">
          {fmtT(t.presChapter, { n: chapter + 1, total: CHAPTERS.length })} · {ch.viz === "trace" ? "trace" : "cosmos"}
        </div>
        {/* narration + fact render through AnimatedCard (remount per chapter = replay) */}
        <AnimatedCard
          key={ch.id}
          open
          variant="plain"
          size="lg"
          className="pres-narrcard"
          title={pick(ch.title, lang)}
        >
          <p className="pres-narr">{pick(ch.narration(ctx), lang)}</p>
        </AnimatedCard>
        <AnimatedCard key={`${ch.id}-fact`} open variant="accent" size="md" className="pres-factcard" delayMs={120}>
          <p className="pres-fact">{pick(ch.fact(ctx), lang)}</p>
        </AnimatedCard>
        <div className="pres-controls">
          <button onClick={() => goto(chapter - 1)} disabled={chapter === 0} title={t.presPrev}>
            <ChevronLeftIcon size={12} />
          </button>
          <button className="primary" onClick={() => setPlaying((p) => !p)} title={playing ? t.stagePause : t.stagePlay}>
            {playing ? <PauseIcon size={12} /> : <PlayIcon size={12} />}
          </button>
          <button onClick={() => goto(chapter + 1)} disabled={chapter === CHAPTERS.length - 1} title={t.presNext}>
            <ChevronRightIcon size={12} />
          </button>
          <span className="spacer" />
          {traceCanvasActive && trace && (
            <button onClick={() => setInsp({ kind: "stage", stage: ch.stage ?? 0 })}>
              {t.inspectOpen} <GearIcon size={13} />
            </button>
          )}
          <span className="dim pres-hint" style={{ fontSize: 11 }}>
            {t.presDetailHint}
          </span>
          <span className="dim" style={{ fontSize: 11 }}>
            ←/→ · space · F · Esc
          </span>
        </div>
        {progress && <p className="loading">{progress}</p>}
      </div>

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

function fmtT(tpl: string, vars: Record<string, string | number>): string {
  return tpl.replace(/\{(\w+)\}/g, (_, k: string) => String(vars[k] ?? `{${k}}`));
}

