import React, { useEffect, useMemo, useRef, useState } from 'react'

/* ------------------------------------------------------------------ */
/* event stream                                                        */
/* ------------------------------------------------------------------ */

function useCacophony() {
  const [state, setState] = useState({
    running: false, agents: [], seed: '', turns: [], speaking: null,
    metrics: {}, artifact: [],
  })
  const [connected, setConnected] = useState(false)
  const [history, setHistory] = useState([]) // metrics samples for the trace

  useEffect(() => {
    const es = new EventSource('/api/events')
    es.onopen = () => setConnected(true)
    es.onerror = () => setConnected(false)
    es.onmessage = (e) => {
      let ev
      try { ev = JSON.parse(e.data) } catch { return }

      // The server replays a snapshot on connect so a refresh mid-run does
      // not land on an empty screen.
      if (ev.kind === 'snapshot') { setState(ev.state); return }

      setState((s) => {
        const n = { ...s }
        switch (ev.kind) {
          case 'started': n.running = true; n.turns = []; n.speaking = null; break
          case 'done': n.running = false; n.speaking = null; break
          case 'agents': n.agents = ev.agents || []; break
          case 'seed': n.seed = ev.text || ''; break
          case 'said':
            n.turns = [...s.turns, { turn: ev.turn, speaker: ev.speaker, text: ev.text, llm_s: ev.llm_s }]
            break
          case 'heard':
            n.turns = s.turns.map((t) => t.turn === ev.turn
              ? { ...t, heard: ev.text, misheard: ev.misheard, asr_s: ev.asr_s } : t)
            break
          case 'speaking': n.speaking = { speaker: ev.speaker, turn: ev.turn, duration: ev.duration }; break
          case 'spoke': n.speaking = null; break
          case 'metrics': n.metrics = ev.data || {}; break
          case 'artifact': n.artifact = [...s.artifact, { speaker: ev.speaker, text: ev.text }]; break
          default: break
        }
        return n
      })

      if (ev.kind === 'metrics' && ev.data) {
        setHistory((h) => [...h.slice(-119), {
          t: ev.t,
          vram: ev.data.vram_used_mb || 0,
          vramTotal: ev.data.vram_total_mb || 12288,
          util: ev.data.gpu_util || 0,
          ram: ev.data.ram_total_mb || 0,
        }])
      }
    }
    return () => es.close()
  }, [])

  return { state, connected, history }
}

/* ------------------------------------------------------------------ */
/* pieces                                                              */
/* ------------------------------------------------------------------ */

const PALETTE = [
  { ring: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/40', dot: 'bg-orange-400', text: 'text-orange-300' },
  { ring: 'text-cyan-400', bg: 'bg-cyan-500/10', border: 'border-cyan-500/40', dot: 'bg-cyan-400', text: 'text-cyan-300' },
]

function Avatar({ agent, idx, speaking, lastLine }) {
  const c = PALETTE[idx % PALETTE.length]
  const initials = (agent?.name || '?').slice(0, 2).toUpperCase()
  return (
    <div className={`flex-1 rounded-2xl border ${c.border} ${speaking ? c.bg : 'bg-zinc-900/60'} p-5 transition-colors duration-300`}>
      <div className="flex items-center gap-4">
        <div className={`relative grid h-16 w-16 shrink-0 place-items-center rounded-full border-2 ${c.border} ${c.ring} ${speaking ? 'ring-speak' : ''}`}>
          <span className={`text-lg font-semibold ${c.text}`}>{initials}</span>
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate text-lg font-semibold text-zinc-100">{agent?.name || '—'}</span>
            {speaking && (
              <span className={`inline-flex items-center gap-1 rounded-full ${c.bg} px-2 py-0.5 text-[11px] font-medium ${c.text}`}>
                <span className={`h-1.5 w-1.5 animate-pulse rounded-full ${c.dot}`} />speaking
              </span>
            )}
          </div>
          <div className="mt-0.5 truncate font-mono text-xs text-zinc-500">
            {agent?.model} · voice {agent?.voice}
          </div>
        </div>
      </div>
      <p className="mt-4 line-clamp-3 min-h-[3.5rem] text-sm leading-relaxed text-zinc-400">
        {lastLine || <span className="text-zinc-600">…</span>}
      </p>
    </div>
  )
}

function Sparkline({ points, max, className = '' }) {
  if (!points.length) return <div className="h-10" />
  const w = 240, h = 40
  const d = points.map((v, i) => {
    const x = (i / Math.max(1, points.length - 1)) * w
    const y = h - (Math.min(v, max) / (max || 1)) * h
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className={`h-10 w-full ${className}`} preserveAspectRatio="none">
      <path d={`${d} L${w},${h} L0,${h} Z`} fill="currentColor" opacity="0.12" />
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

function Meter({ label, value, sub, points, max, tone }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">{label}</span>
        <span className="font-mono text-sm text-zinc-200">{value}</span>
      </div>
      <div className={tone}><Sparkline points={points} max={max} /></div>
      <div className="font-mono text-[11px] text-zinc-600">{sub}</div>
    </div>
  )
}

function Transcript({ turns, agents }) {
  const ref = useRef(null)
  useEffect(() => { ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: 'smooth' }) }, [turns.length])
  const idxOf = (name) => Math.max(0, agents.findIndex((a) => a.name === name))
  return (
    <div ref={ref} className="h-full space-y-3 overflow-y-auto pr-1">
      {turns.length === 0 && (
        <p className="py-10 text-center text-sm text-zinc-600">waiting for the first turn…</p>
      )}
      {turns.map((t) => {
        const c = PALETTE[idxOf(t.speaker) % PALETTE.length]
        return (
          <div key={t.turn} className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${c.dot}`} />
              <span className={`text-xs font-semibold ${c.text}`}>{t.speaker}</span>
              <span className="font-mono text-[11px] text-zinc-600">#{t.turn}</span>
              <span className="ml-auto font-mono text-[11px] text-zinc-600">
                {t.llm_s != null && `llm ${t.llm_s}s`}{t.asr_s != null && ` · asr ${t.asr_s}s`}
              </span>
            </div>
            <p className="mt-1.5 text-sm leading-relaxed text-zinc-200">{t.text}</p>
            {/* The mishearing is the interesting part of a speech-to-speech loop:
                it is what the *other* agent actually received. */}
            {t.misheard && t.heard && (
              <p className="mt-1.5 border-l-2 border-amber-500/40 pl-2 text-xs italic text-amber-300/70">
                heard as: {t.heard}
              </p>
            )}
          </div>
        )
      })}
    </div>
  )
}

/* ------------------------------------------------------------------ */

export default function App() {
  const { state, connected, history } = useCacophony()
  const [busy, setBusy] = useState(false)

  const vramPts = useMemo(() => history.map((h) => h.vram), [history])
  const ramPts = useMemo(() => history.map((h) => h.ram), [history])
  const utilPts = useMemo(() => history.map((h) => h.util), [history])
  const m = state.metrics || {}

  const lastBy = (name) => [...state.turns].reverse().find((t) => t.speaker === name)?.text

  const start = async () => {
    setBusy(true)
    try {
      await fetch('/api/control', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd: 'start' }),
      })
    } finally { setBusy(false) }
  }

  return (
    <div className="min-h-full bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-7xl p-6">

        <header className="mb-6 flex flex-wrap items-center gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              vault<span className="text-zinc-500">-</span>cacophony
            </h1>
            <p className="mt-0.5 text-sm text-zinc-500">
              two local models talking to each other through speech
            </p>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${
              connected ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                        : 'border-zinc-700 bg-zinc-900 text-zinc-500'}`}>
              <span className={`h-1.5 w-1.5 rounded-full ${connected ? 'animate-pulse bg-emerald-400' : 'bg-zinc-600'}`} />
              {connected ? 'connected' : 'offline'}
            </span>
            <button
              onClick={start}
              disabled={busy || state.running}
              className="rounded-lg bg-zinc-100 px-4 py-2 text-sm font-semibold text-zinc-900 transition hover:bg-white disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500"
            >
              {state.running ? 'running…' : busy ? 'starting…' : 'Start conversation'}
            </button>
          </div>
        </header>

        {state.seed && (
          <div className="mb-5 rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3">
            <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">seed</span>
            <p className="mt-1 text-sm text-zinc-300">{state.seed}</p>
          </div>
        )}

        <div className="mb-5 flex flex-col gap-4 sm:flex-row">
          {(state.agents.length ? state.agents : [null, null]).map((a, i) => (
            <Avatar
              key={a?.name || i} agent={a} idx={i}
              speaking={state.speaking?.speaker === a?.name}
              lastLine={a ? lastBy(a.name) : null}
            />
          ))}
        </div>

        <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
          <section className="flex h-[46vh] flex-col rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4 lg:h-[52vh]">
            <h2 className="mb-3 text-xs font-medium uppercase tracking-wider text-zinc-500">transcript</h2>
            <div className="min-h-0 flex-1">
              <Transcript turns={state.turns} agents={state.agents} />
            </div>
          </section>

          <aside className="space-y-4">
            <Meter
              label="VRAM" tone="text-violet-400"
              value={m.vram_used_mb != null ? `${(m.vram_used_mb / 1024).toFixed(1)} GB` : '—'}
              sub={m.vram_total_mb ? `of ${(m.vram_total_mb / 1024).toFixed(0)} GB` : ''}
              points={vramPts} max={m.vram_total_mb || 12288}
            />
            <Meter
              label="GPU" tone="text-emerald-400"
              value={m.gpu_util != null ? `${m.gpu_util}%` : '—'}
              sub={m.gpu_temp != null ? `${m.gpu_temp}°C` : ''}
              points={utilPts} max={100}
            />
            <Meter
              label="Stack RAM" tone="text-sky-400"
              value={m.ram_total_mb != null ? `${(m.ram_total_mb / 1024).toFixed(1)} GB` : '—'}
              sub={[m.crispasr && `crispasr ${m.crispasr}M`, m.llama_server && `llm ${m.llama_server}M`]
                .filter(Boolean).join(' · ')}
              points={ramPts} max={Math.max(4096, ...ramPts)}
            />
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
              <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">turns</span>
              <div className="mt-1 font-mono text-2xl text-zinc-100">{state.turns.length}</div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  )
}
