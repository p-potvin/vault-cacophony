import React, { useEffect, useMemo, useRef, useState } from 'react'

/* ------------------------------------------------------------------ */
/* live run stream (SSE)                                               */
/* ------------------------------------------------------------------ */

function useRunStream() {
  const [connected, setConnected] = useState(false)
  const [running, setRunning] = useState(false)
  const [log, setLog] = useState([])
  const [result, setResult] = useState(null)
  const [runId, setRunId] = useState(null)
  const [outDir, setOutDir] = useState(null)

  useEffect(() => {
    const es = new EventSource('/api/events')
    es.onopen = () => setConnected(true)
    es.onerror = () => setConnected(false)
    es.onmessage = (e) => {
      let ev
      try { ev = JSON.parse(e.data) } catch { return }
      if (ev.kind === 'line') {
        setLog((l) => [...l.slice(-1499), ev.text])
      } else if (ev.kind === 'started') {
        setRunning(true); setLog([]); setResult(null)
        setRunId(ev.run_id); setOutDir(ev.out_dir)
      } else if (ev.kind === 'done') {
        setRunning(false); setResult(ev)
      }
    }
    return () => es.close()
  }, [])
  return { connected, running, log, result, runId, outDir }
}

/* ------------------------------------------------------------------ */
/* small ui pieces                                                     */
/* ------------------------------------------------------------------ */

function Field({ label, children, hint }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-zinc-500">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-zinc-600">{hint}</span>}
    </label>
  )
}

const inputCls =
  'w-full rounded-lg border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-100 outline-none ' +
  'focus:border-cyan-600/60 focus:ring-1 focus:ring-cyan-600/40 disabled:opacity-40'

const selectCls = inputCls + ' [&>option]:bg-zinc-900'

function Section({ title, children }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/60 p-4">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-widest text-zinc-500">{title}</h2>
      <div className="space-y-3">{children}</div>
    </div>
  )
}

function PersonaRow({ p, globalEngine, engines, kokoroVoices, magpieVoices, cosyRefs, cosyBaked, onChange }) {
  const engine = p.engine || ''
  const effectiveEngine = engine || globalEngine || 'omnivoice'
  const voiceOptions = useMemo(() => {
    const base = [{ value: 'default', label: `default (${p.defaultVoice || 'preset'})` }]
    if (effectiveEngine === 'omnivoice' || effectiveEngine === 'fish_audio')
      return [
        ...base,
        ...cosyRefs.map((r) => ({ value: r.name, label: `${r.source === 'voices-store' ? 'store' : 'sample'} · ${r.name}` })),
      ]
    if (effectiveEngine === 'kokoro') return [...base, ...kokoroVoices.map((v) => ({ value: v, label: v }))]
    if (effectiveEngine === 'magpie') return [...base, ...magpieVoices.map((v) => ({ value: v, label: v }))]
    if (effectiveEngine === 'cosyvoice')
      return [
        ...base,
        ...cosyRefs.map((r) => ({ value: r.name, label: `clone · ${r.name}` })),
        ...cosyBaked.map((v) => ({ value: v, label: `baked · ${v}` })),
      ]
    return base
  }, [effectiveEngine, p.defaultVoice, kokoroVoices, magpieVoices, cosyRefs, cosyBaked])

  const voice = p.voice === '' ? 'default' : p.voice
  return (
    <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-semibold text-zinc-200">{p.name}</span>
        <span className="font-mono text-[10px] text-zinc-600">#{p.index}</span>
      </div>
      <div className="grid grid-cols-[1fr_1.4fr_64px] gap-2">
        <select
          className={selectCls}
          value={engine}
          onChange={(e) => onChange({ ...p, engine: e.target.value, voice: e.target.value === engine ? p.voice : '' })}
        >
          <option value="">inherit (global)</option>
          {engines.map((en) => <option key={en} value={en}>{en}</option>)}
        </select>
        <select
          className={selectCls}
          value={voice}
          onChange={(e) => onChange({ ...p, voice: e.target.value === 'default' ? '' : e.target.value })}
        >
          {voiceOptions.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        <input
          type="number" step="0.05" min="0.6" max="1.6"
          className={inputCls} title="speaking pace"
          value={p.speed}
          onChange={(e) => onChange({ ...p, speed: parseFloat(e.target.value) || 1.0 })}
        />
      </div>
    </div>
  )
}

function LogView({ log, result }) {
  const ref = useRef(null)
  useEffect(() => { ref.current?.scrollTo({ top: ref.current.scrollHeight }) }, [log.length])
  return (
    <div ref={ref} className="h-72 overflow-y-auto rounded-xl border border-zinc-800 bg-black/50 p-3 font-mono text-[11.5px] leading-relaxed">
      {log.length === 0 && <p className="text-zinc-600">waiting for a run…</p>}
      {log.map((line, i) => {
        let cls = 'text-zinc-500'
        if (line.startsWith('[ui]')) cls = 'text-zinc-600'
        else if (line.includes('SUCCESS') || line.includes('[+]')) cls = 'text-emerald-400'
        else if (line.startsWith('[Turn') || line.includes('] [')) cls = 'text-cyan-300'
        else if (line.startsWith('  >')) cls = 'text-zinc-300'
        else if (line.includes('[!]')) cls = 'text-amber-400'
        return <div key={i} className={cls}>{line || '\u00a0'}</div>
      })}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

export default function App() {
  const { connected, running, log, result, runId, outDir } = useRunStream()
  const [opts, setOpts] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  // form state
  const [script, setScript] = useState('testbench')
  const [seed, setSeed] = useState('')
  const [model, setModel] = useState('openai/gpt-oss-120b:cheapest')
  const [engineType, setEngineType] = useState('omnivoice')
  const [numSpeakers, setNumSpeakers] = useState(5)
  const [turns, setTurns] = useState(30)
  const [durationSec, setDurationSec] = useState(180)
  const [rt60, setRt60] = useState(0.2)
  const [ramblers, setRamblers] = useState(false)
  const [ramblerGain, setRamblerGain] = useState(-18)
  const [gpuDevice, setGpuDevice] = useState(-1)
  const [outDirName, setOutDirName] = useState('cacophony_output/ui_run')
  const [personas, setPersonas] = useState([])

  useEffect(() => {
    fetch('/api/options')
      .then((r) => r.json())
      .then((o) => {
        setOpts(o)
        const d = o.defaults
        setModel(d.model)
        setEngineType(d.engine_type)
        setNumSpeakers(d.num_speakers)
        setTurns(d.turns)
        setDurationSec(d.duration_sec)
        setRt60(d.rt60)
        setRamblerGain(d.rambler_gain_db)
        setGpuDevice(d.gpu_device)
        setOutDirName(d.out_dir)
        const st = o.scripts[0]
        setScript(st.id)
        setSeed(st.seed)
        setPersonas(o.personas.map((p) => {
          const refStem = p.voice_ref_wav ? p.voice_ref_wav.split(/[\\/]/).pop().replace(/\.wav$/i, '') : p.voice_name
          return {
            index: p.index, name: p.name, engine: '', voice: '',
            speed: d.engine_type === 'fish_audio' ? 1.0 : p.speed_ratio,
            defaultVoice: d.engine_type === 'fish_audio' ? refStem : p.voice_name,
          }
        }))
      })
      .catch(() => setError('could not load options — is the python server running on :8753?'))
  }, [])

  const onScriptChange = (id) => {
    setScript(id)
    const s = (opts?.scripts || []).find((x) => x.id === id)
    if (s) setSeed(s.seed)
  }

  const refsByName = useMemo(() => {
    const m = {}
    for (const r of opts?.cosyvoice_refs || []) m[r.name] = r
    return m
  }, [opts])

  const buildPayload = () => {
    const payload = {
      script, model, engine_type: engineType, num_speakers: numSpeakers,
      turns, duration_sec: durationSec, rt60, enable_ramblers: ramblers,
      rambler_gain_db: ramblerGain, gpu_device: gpuDevice,
      out_dir: outDirName, seed_topic: seed,
    }
    const overrides = []
    for (const p of personas.slice(0, numSpeakers)) {
      const defaultSpeed = opts?.personas[p.index]?.speed_ratio ?? 1.0
      const hasSpeedChange = p.speed !== undefined && Math.abs(p.speed - defaultSpeed) > 0.001
      if (!p.engine && !p.voice && !hasSpeedChange) continue // untouched -> preset defaults
      const entry = { index: p.index }
      if (p.engine) entry.engine = p.engine
      const eng = p.engine || engineType
      if (p.voice) {
        entry.voice_name = p.voice
        if ((eng === 'cosyvoice' || eng === 'fish_audio' || eng === 'omnivoice') && refsByName[p.voice]) {
          entry.voice_ref = { wav: refsByName[p.voice].wav, text: refsByName[p.voice].text }
        }
      }
      if (hasSpeedChange) {
        entry.speed_ratio = p.speed
      }
      overrides.push(entry)
    }
    payload.personas = overrides
    return payload
  }

  const start = async () => {
    setBusy(true); setError(null)
    try {
      const resp = await fetch('/api/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload()),
      })
      const data = await resp.json()
      if (!resp.ok) setError(data.error || 'start failed')
    } catch (e) {
      setError(String(e))
    } finally { setBusy(false) }
  }

  const stop = async () => {
    await fetch('/api/stop', { method: 'POST' })
  }

  const fileUrl = (p) => (p ? `/api/file?path=${encodeURIComponent(p.replace(/\\/g, '/'))}` : null)

  return (
    <div className="min-h-full bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-[1500px] p-6">
        <header className="mb-6 flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-3">
            <div className={`h-2.5 w-2.5 rounded-full ${connected ? 'bg-emerald-400' : 'bg-red-500'}`} />
            <h1 className="text-2xl font-semibold tracking-tight">cacophony studio</h1>
          </div>
          <div className="flex items-center gap-2 text-xs">
            {running ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-orange-500/10 px-3 py-1 font-medium text-orange-300">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-orange-400" /> running{runId ? ` · ${runId}` : ''}
              </span>
            ) : (
              <span className="rounded-full bg-zinc-800/80 px-3 py-1 text-zinc-400">idle</span>
            )}
            {outDir && <span className="font-mono text-zinc-600">{outDir}</span>}
          </div>
          <div className="ml-auto flex gap-2">
            {running && (
              <button onClick={stop} className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-1.5 text-sm font-medium text-red-300 hover:bg-red-500/20">
                stop
              </button>
            )}
            <button
              onClick={start}
              disabled={busy || running || !opts}
              className="rounded-lg bg-cyan-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-cyan-500 disabled:opacity-40"
            >
              {busy ? 'starting…' : running ? 'generating…' : 'generate conversation'}
            </button>
          </div>
        </header>

        {error && (
          <div className="mb-4 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">{error}</div>
        )}

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[430px_1fr]">
          {/* ------------------------------ config ------------------------------ */}
          <div className="space-y-4">
            <Section title="story">
              <Field label="script">
                <select className={selectCls} value={script} onChange={(e) => onScriptChange(e.target.value)}>
                  {(opts?.scripts || []).map((s) => <option key={s.id} value={s.id}>{s.label} ({s.id})</option>)}
                </select>
              </Field>
              <Field label="seed topic" hint="first line of the conversation">
                <textarea className={inputCls + ' h-20 resize-y'} value={seed} onChange={(e) => setSeed(e.target.value)} />
              </Field>
            </Section>

            <Section title="brain & voice">
              <Field label="HF model">
                <select className={selectCls} value={model} onChange={(e) => setModel(e.target.value)}>
                  {(opts?.models || []).map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </Field>
              <div className="grid grid-cols-2 gap-2">
                <Field label="tts engine">
                  <select className={selectCls} value={engineType} onChange={(e) => setEngineType(e.target.value)}>
                    {(opts?.engines || []).map((en) => <option key={en} value={en}>{en}</option>)}
                  </select>
                </Field>
                <Field label="speakers">
                  <select className={selectCls} value={numSpeakers} onChange={(e) => setNumSpeakers(parseInt(e.target.value))}>
                    {(opts?.num_speakers || []).map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </Field>
              </div>
            </Section>

            <Section title="timing & room">
              <div className="grid grid-cols-2 gap-2">
                <Field label="max turns">
                  <input type="number" min="1" className={inputCls} value={turns} onChange={(e) => setTurns(parseInt(e.target.value) || 1)} />
                </Field>
                <Field label="duration (s)">
                  <input type="number" min="10" className={inputCls} value={durationSec} onChange={(e) => setDurationSec(parseFloat(e.target.value) || 0)} />
                </Field>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Field label="rt60 (s)">
                  <input type="number" step="0.05" min="0" className={inputCls} value={rt60} onChange={(e) => setRt60(parseFloat(e.target.value) || 0)} />
                </Field>
                <Field label="gpu device">
                  <input type="number" className={inputCls} value={gpuDevice} onChange={(e) => setGpuDevice(Number.isNaN(parseInt(e.target.value, 10)) ? -1 : parseInt(e.target.value, 10))} />
                </Field>
              </div>
              <div className="flex items-center gap-3">
                <label className="flex cursor-pointer items-center gap-2 text-sm text-zinc-300">
                  <input type="checkbox" className="h-4 w-4 accent-cyan-600" checked={ramblers} onChange={(e) => setRamblers(e.target.checked)} />
                  background ramblers
                </label>
                <Field label="gain dB">
                  <input type="number" className={inputCls + ' !w-24'} disabled={!ramblers} value={ramblerGain} onChange={(e) => setRamblerGain(parseFloat(e.target.value) || 0)} />
                </Field>
              </div>
            </Section>

            <Section title="output">
              <Field label="output dir" hint="relative to the repo root">
                <input className={inputCls} value={outDirName} onChange={(e) => setOutDirName(e.target.value)} />
              </Field>
            </Section>

            <Section title="voices per persona">
              <p className="mb-2 -mt-1 text-[11px] text-zinc-600">
                pick an engine and voice for each speaker; <span className="text-zinc-400">inherit</span> uses the global engine.
              </p>
              <div className="space-y-2">
                {personas.slice(0, numSpeakers).map((p) => (
                  <PersonaRow
                    key={p.index} p={p}
                    globalEngine={engineType}
                    engines={opts?.engines || []}
                    kokoroVoices={opts?.kokoro_voices || []}
                    magpieVoices={opts?.magpie_voices || []}
                    cosyRefs={opts?.cosyvoice_refs || []}
                    cosyBaked={opts?.cosyvoice_baked || []}
                    onChange={(np) => setPersonas((ps) => ps.map((x) => (x.index === np.index ? np : x)))}
                  />
                ))}
              </div>
            </Section>
          </div>

          {/* ------------------------------ results ------------------------------ */}
          <div className="space-y-4">
            <Section title="console">
              <LogView log={log} result={result} />
            </Section>

            {result && (
              <>
                <Section title="output">
                  <div className="flex flex-wrap items-center gap-3 text-sm">
                    <span className={`rounded-full px-3 py-1 font-medium ${result.run_status?.status === 'partial' ? 'bg-amber-500/10 text-amber-300' : result.code === 0 ? 'bg-emerald-500/10 text-emerald-300' : 'bg-red-500/10 text-red-300'}`}>
                      {result.run_status?.status === 'partial'
                        ? `partial · ${result.run_status.completed_turns}/${result.run_status.requested_turns} turns`
                        : result.code === 0 ? 'success' : `exit ${result.code}`}
                    </span>
                    <span className="font-mono text-xs text-zinc-500">{result.out_dir}</span>
                    <div className="ml-auto flex gap-2">
                      {result.master_wav && (
                        <a className="rounded-lg border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:border-cyan-600/60" href={fileUrl(result.master_wav)} download>
                          download wav
                        </a>
                      )}
                      {result.transcript && (
                        <a className="rounded-lg border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:border-cyan-600/60" href={fileUrl(result.transcript)} download>
                          transcript
                        </a>
                      )}
                      {result.manifest && (
                        <a className="rounded-lg border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:border-cyan-600/60" href={fileUrl(result.manifest)} download>
                          manifest
                        </a>
                      )}
                    </div>
                  </div>
                  {result.master_wav && (
                    <audio controls className="mt-3 w-full" src={fileUrl(result.master_wav)} />
                  )}
                </Section>

                {result.transcript_text && (
                  <Section title="transcript">
                    <pre className="max-h-96 overflow-y-auto whitespace-pre-wrap rounded-xl border border-zinc-800 bg-black/40 p-3 font-mono text-xs leading-relaxed text-zinc-300">
                      {result.transcript_text}
                    </pre>
                  </Section>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}