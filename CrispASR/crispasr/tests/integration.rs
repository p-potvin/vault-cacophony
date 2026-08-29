//! Integration tests for the CrispASR Rust wrapper.
//!
//! Requires:
//!   - whisper-tiny model at CRISPASR_MODEL env var (or ../models/ggml-tiny.en.bin)
//!   - parakeet model at PARAKEET_MODEL env var (optional, skipped if absent)
//!   - jfk.wav at ../samples/jfk.wav

use std::path::Path;

fn jfk_pcm() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../samples/jfk.wav");
    let mut reader = hound::WavReader::open(path).expect("failed to open jfk.wav");
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn whisper_model() -> String {
    std::env::var("CRISPASR_MODEL").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/ggml-tiny.en.bin").to_string()
    })
}

fn parakeet_model() -> Option<String> {
    let p = std::env::var("PARAKEET_MODEL").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test_cohere/parakeet-tdt-0.6b-v3.gguf"
        )
        .to_string()
    });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn omni_ctc_model() -> Option<String> {
    let p = std::env::var("OMNI_CTC_MODEL").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/omniasr-ctc.gguf").to_string()
    });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn canary_ctc_model() -> Option<String> {
    let p = std::env::var("CANARY_CTC_MODEL").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/canary-ctc.gguf").to_string()
    });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn wav2vec2_model() -> Option<String> {
    let p = std::env::var("WAV2VEC2_MODEL").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/wav2vec2-ctc.gguf").to_string()
    });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

/// Backend-agnostic sanity for an exposed CTC grid: correctly shaped, finite,
/// and carrying real per-frame acoustic structure (the argmax varies across
/// the clip but isn't noise every frame). Makes no assumption about which id
/// is the CTC blank, so it holds for Omni (blank 0), canary-ctc, and wav2vec2.
fn assert_real_ctc_grid(lg: &crispasr::CtcLogits) {
    assert!(lg.n_vocab > 0 && lg.n_frames > 0);
    assert_eq!(lg.data.len(), lg.n_vocab * lg.n_frames);
    assert!(
        lg.data.iter().all(|x| x.is_finite()),
        "logits must be finite"
    );

    let v = lg.n_vocab;
    let argmax: Vec<usize> = (0..lg.n_frames)
        .map(|t| {
            let frame = &lg.data[t * v..(t + 1) * v];
            (0..v)
                .max_by(|&a, &b| frame[a].partial_cmp(&frame[b]).unwrap())
                .unwrap()
        })
        .collect();
    let transitions = (1..lg.n_frames)
        .filter(|&t| argmax[t] != argmax[t - 1])
        .count();
    assert!(
        transitions > 0,
        "degenerate grid: constant argmax across all {} frames",
        lg.n_frames
    );
    assert!(
        transitions < lg.n_frames,
        "argmax changes every frame ({transitions}/{}): suspect noise, not a real decode",
        lg.n_frames
    );
}

// ---- CrispASR (whisper-only) tests ----

#[test]
#[ignore = "CrispASR (whisper-direct) API crashes in Rust — use Session API instead"]
fn whisper_load_and_transcribe() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        eprintln!("SKIP: whisper model not found at {model_path}");
        return;
    }
    let model = crispasr::CrispASR::new(&model_path).expect("load whisper-tiny");
    let pcm = jfk_pcm();
    let segs = model.transcribe_pcm(&pcm).expect("transcribe");
    assert!(!segs.is_empty(), "should produce segments");
    let full = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    assert!(
        full.contains("fellow americans"),
        "text should mention 'fellow americans': {full}"
    );
    assert!(
        full.contains("country"),
        "text should mention 'country': {full}"
    );
}

#[test]
#[ignore = "CrispASR (whisper-direct) API crashes in Rust — use Session API instead"]
fn whisper_timestamps_valid() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        return;
    }
    let model = crispasr::CrispASR::new(&model_path).unwrap();
    let segs = model.transcribe_pcm(&jfk_pcm()).unwrap();
    for seg in &segs {
        assert!(seg.start >= 0.0, "start >= 0");
        assert!(
            seg.end > seg.start,
            "end > start: {} vs {}",
            seg.end,
            seg.start
        );
        assert!(seg.end < 15.0, "end < 15s (audio is ~11s)");
    }
}

#[test]
#[ignore = "CrispASR (whisper-direct) API crashes in Rust — use Session API instead"]
fn whisper_empty_audio() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        return;
    }
    let model = crispasr::CrispASR::new(&model_path).unwrap();
    let silence = vec![0.0f32; 16000]; // 1s silence
    let segs = model.transcribe_pcm(&silence).unwrap();
    // Should not crash; may produce empty or whitespace-only segments
    let _ = segs;
}

// ---- Session (unified, any backend) tests ----

#[test]
fn session_whisper_auto_detect() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        return;
    }
    let sess = crispasr::Session::open(&model_path).expect("session open whisper");
    assert_eq!(sess.backend(), "whisper");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty());
    let full = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    assert!(full.contains("country"));
}

#[test]
fn session_whisper_no_speech_prob() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        eprintln!("SKIP: whisper model not found at {model_path}");
        return;
    }
    let sess = crispasr::Session::open(&model_path).expect("session open whisper");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty());

    // Every whisper segment carries a real no-speech probability in [0, 1] —
    // not the -1.0 "no data" sentinel other backends leave. JFK is clean
    // speech, so the values should also sit well below the 0.6 suspect
    // threshold, confirming it is the true posterior and not a placeholder.
    for s in &segs {
        assert!(
            (0.0..=1.0).contains(&s.no_speech_prob),
            "no_speech_prob {} out of [0,1] for segment {:?}",
            s.no_speech_prob,
            s.text
        );
        assert!(
            s.no_speech_prob < 0.6,
            "unexpected high no_speech_prob {} on clean speech {:?}",
            s.no_speech_prob,
            s.text
        );
    }
}

#[test]
fn session_whisper_detected_language() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        eprintln!("SKIP: whisper model not found at {model_path}");
        return;
    }
    let sess = crispasr::Session::open(&model_path).expect("session open whisper");
    // Whisper's in-decode acoustic language detection surfaces on the
    // exception-safe session (JFK is English).
    sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert_eq!(sess.detected_language(), "en");
}

#[test]
fn session_available_backends() {
    let backends = crispasr::Session::available_backends();
    assert!(backends.contains(&"whisper".to_string()));
    assert!(backends.contains(&"parakeet".to_string()));
}

#[test]
fn session_parakeet_word_timestamps() {
    let model_path = match parakeet_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: parakeet model not found");
            return;
        }
    };
    let sess = crispasr::Session::open(&model_path).expect("session open parakeet");
    assert_eq!(sess.backend(), "parakeet");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty());

    // Parakeet should produce word-level timestamps
    let words = &segs[0].words;
    assert!(!words.is_empty(), "parakeet should produce words");
    for w in words {
        assert!(w.start >= 0.0);
        assert!(w.end >= w.start);
        assert!(!w.text.is_empty());
    }

    // Monotonicity
    let mut prev_end = 0.0f64;
    for w in words {
        assert!(
            w.start >= prev_end - 0.02,
            "word '{}' starts at {} before prev end {}",
            w.text,
            w.start,
            prev_end
        );
        prev_end = w.end;
    }
}

#[test]
fn session_omni_ctc_logits() {
    let model_path = match omni_ctc_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: omni CTC model not found (set OMNI_CTC_MODEL)");
            return;
        }
    };
    // Auto-detect doesn't recognise every Omni GGUF on this pinned release;
    // the generic "omniasr" backend routes all CTC/LLM variants.
    let sess = crispasr::Session::open_with_backend(&model_path, "omniasr", 4)
        .expect("session open omniasr");

    // The 300M CTC model has a ~5 s positional-encoding limit (per its HF
    // card), so decode only the first ~4 s of the ~11 s clip.
    let pcm: Vec<f32> = jfk_pcm().into_iter().take(16_000 * 4).collect();

    let (segs, logits) = sess
        .transcribe_with_logits(&pcm)
        .expect("transcribe_with_logits");
    let text = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!text.trim().is_empty(), "expected a transcript");

    // Accessor contract: a dense [n_vocab × n_frames] grid, correctly shaped
    // and finite.
    let lg = logits.expect("CTC backend should return Some(CtcLogits)");
    assert!(lg.n_vocab > 0 && lg.n_frames > 0);
    assert_eq!(lg.data.len(), lg.n_vocab * lg.n_frames);
    assert!(
        lg.data.iter().all(|x| x.is_finite()),
        "logits must be finite"
    );

    // Greedy CTC over the exposed logits (argmax per frame, collapse repeats,
    // drop blank id 0) must yield a non-degenerate token stream — evidence the
    // grid is the real decode input, not zeros/garbage.
    let v = lg.n_vocab;
    let mut prev: i32 = -1;
    let mut n_tokens = 0usize;
    for t in 0..lg.n_frames {
        let frame = &lg.data[t * v..(t + 1) * v];
        let best = (0..v)
            .max_by(|&a, &b| frame[a].partial_cmp(&frame[b]).unwrap())
            .unwrap() as i32;
        if best != 0 && best != prev {
            n_tokens += 1;
        }
        prev = best;
    }
    assert!(
        n_tokens > 0 && n_tokens < lg.n_frames,
        "degenerate greedy decode: {n_tokens} tokens over {} frames",
        lg.n_frames
    );

    // Capturing logits must not perturb the transcript.
    let plain = sess.transcribe(&pcm).expect("transcribe");
    let ptext = plain
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(ptext, text, "logits capture changed the transcript");
}

#[test]
fn session_omni_ctc_vocab() {
    let model_path = match omni_ctc_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: omni CTC model not found (set OMNI_CTC_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "omniasr", 4)
        .expect("session open omniasr");

    // Accessor contract: a non-empty vocab of raw SentencePiece pieces.
    let vocab = sess.ctc_vocab().expect("CTC backend should expose a vocab");
    assert!(
        vocab.len() > 1000,
        "unexpectedly small vocab: {}",
        vocab.len()
    );
    // Real pieces carry a word-boundary marker. The v2 Omni CTC vocab is built
    // verbatim from vocab.json and uses a literal ASCII space; v1 (SentencePiece)
    // uses U+2581 (▁). Accept either so the accessor test isn't tied to one
    // tokenizer flavour.
    assert!(
        vocab.iter().any(|p| p.contains('\u{2581}') || p == " "),
        "no word-boundary token (U+2581 piece or literal space) — not a real vocab"
    );

    // End-to-end: a greedy CTC decode over the exposed logits, detokenized via
    // the exposed vocab, must reproduce the backend's built-in transcript. This
    // proves the vocab indexing aligns with the logits argmax (same id space).
    let pcm: Vec<f32> = jfk_pcm().into_iter().take(16_000 * 4).collect();
    let (segs, logits) = sess
        .transcribe_with_logits(&pcm)
        .expect("transcribe_with_logits");
    let text = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!text.trim().is_empty(), "expected a transcript");
    let lg = logits.expect("CTC backend should return Some(CtcLogits)");
    assert_eq!(lg.n_vocab, vocab.len(), "logit vocab dim != vocab len");

    // Greedy CTC: argmax per frame, collapse repeats, drop blank (id 0);
    // detokenize SentencePiece pieces with U+2581 → space, then trim.
    let v = lg.n_vocab;
    let mut prev: i32 = -1;
    let mut decoded = String::new();
    for t in 0..lg.n_frames {
        let frame = &lg.data[t * v..(t + 1) * v];
        let best = (0..v)
            .max_by(|&a, &b| frame[a].partial_cmp(&frame[b]).unwrap())
            .unwrap() as i32;
        if best != 0 && best != prev {
            decoded.push_str(&vocab[best as usize].replace('\u{2581}', " "));
        }
        prev = best;
    }
    let decoded = decoded.trim();
    assert_eq!(
        decoded, text,
        "vocab-detokenized greedy decode != built-in transcript"
    );
}

#[test]
fn session_canary_ctc_logits() {
    let model_path = match canary_ctc_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: canary-ctc model not found (set CANARY_CTC_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "canary-ctc", 4)
        .expect("session open canary-ctc");
    let pcm = jfk_pcm();

    let (segs, logits) = sess
        .transcribe_with_logits(&pcm)
        .expect("transcribe_with_logits");
    let text = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!text.trim().is_empty(), "expected a transcript");

    // canary_ctc_compute_logits returns per-frame log-probabilities; the grid
    // sanity is normalization-agnostic (argmax only).
    let lg = logits.expect("canary-ctc should return Some(CtcLogits)");
    assert_real_ctc_grid(&lg);

    // Capturing logits must not perturb the transcript.
    let plain = sess.transcribe(&pcm).expect("transcribe");
    let ptext = plain
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(ptext, text, "logits capture changed the transcript");
}

#[test]
fn session_wav2vec2_ctc_logits() {
    let model_path = match wav2vec2_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: wav2vec2 model not found (set WAV2VEC2_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "wav2vec2", 4)
        .expect("session open wav2vec2");
    let pcm = jfk_pcm();

    let (segs, logits) = sess
        .transcribe_with_logits(&pcm)
        .expect("transcribe_with_logits");
    let text = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!text.trim().is_empty(), "expected a transcript");

    // wav2vec2_compute_logits returns raw pre-softmax logits.
    let lg = logits.expect("wav2vec2 should return Some(CtcLogits)");
    assert_real_ctc_grid(&lg);

    // Capturing logits must not perturb the transcript.
    let plain = sess.transcribe(&pcm).expect("transcribe");
    let ptext = plain
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(ptext, text, "logits capture changed the transcript");
}

// Shared vocab-accessor contract for a CTC backend (PR #259 made ctc_vocab
// comprehensive across omni-ctc / canary-ctc / wav2vec2 / data2vec, but only
// omni-ctc had a vocab test). Asserts: a Some, non-empty vocab of valid C
// strings, and that its length lines up with the exposed logit grid — equal
// when blank is an in-vocab id (wav2vec2 <pad>), or one less when blank is a
// separate appended index (canary-ctc blank_id). No tokenizer-flavour
// assumptions, so it stays green across backends.
fn assert_ctc_vocab_contract(sess: &crispasr::Session, pcm: &[f32]) {
    let vocab = sess
        .ctc_vocab()
        .expect("CTC backend should expose Some(vocab)");
    assert!(
        vocab.len() > 1,
        "unexpectedly small CTC vocab: {}",
        vocab.len()
    );
    // token_text must always yield a valid (possibly empty) string, never panic
    // — including an out-of-range id, which the accessor guards to "".
    assert!(
        vocab.iter().any(|p| !p.is_empty()),
        "every vocab piece was empty — accessor returned no token strings"
    );

    let (_segs, logits) = sess
        .transcribe_with_logits(pcm)
        .expect("transcribe_with_logits");
    let lg = logits.expect("CTC backend should return Some(CtcLogits)");
    // The logit grid is either the vocab (blank in-vocab) or vocab + blank.
    assert!(
        lg.n_vocab == vocab.len() || lg.n_vocab == vocab.len() + 1,
        "logit dim {} inconsistent with vocab len {} (expected == or +1 for blank)",
        lg.n_vocab,
        vocab.len()
    );
}

#[test]
fn session_canary_ctc_vocab() {
    let model_path = match canary_ctc_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: canary-ctc model not found (set CANARY_CTC_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "canary-ctc", 4)
        .expect("session open canary-ctc");
    // canary appends the blank as a separate index (blank_id), so the exposed
    // vocab is one shorter than the logit grid — covered by the shared contract.
    assert_ctc_vocab_contract(&sess, &jfk_pcm());
}

#[test]
fn session_wav2vec2_ctc_vocab() {
    let model_path = match wav2vec2_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: wav2vec2 model not found (set WAV2VEC2_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "wav2vec2", 4)
        .expect("session open wav2vec2");
    assert_ctc_vocab_contract(&sess, &jfk_pcm());
}

#[test]
fn session_ctc_backend_no_speech_sentinel() {
    // The no_speech_prob / detected_language sentinels must hold on a real
    // NON-whisper session: CTC backends never populate the <|nospeech|>
    // posterior, so every segment must carry the -1.0 "no data" sentinel (not a
    // bogus in-[0,1] value), and detected_language must not crash — it falls
    // back to the source-language hint or "unknown". Prefer canary-ctc, else
    // wav2vec2; skip if neither model is present.
    let (model_path, backend) = match (canary_ctc_model(), wav2vec2_model()) {
        (Some(p), _) => (p, "canary-ctc"),
        (None, Some(p)) => (p, "wav2vec2"),
        (None, None) => {
            eprintln!("SKIP: no CTC model found (set CANARY_CTC_MODEL or WAV2VEC2_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, backend, 4)
        .expect("session open CTC backend");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty(), "expected a transcript");
    for s in &segs {
        assert_eq!(
            s.no_speech_prob, -1.0,
            "non-whisper backend must leave the -1.0 no_speech_prob sentinel, got {}",
            s.no_speech_prob
        );
    }
    // Fallback path: never a whisper acoustic code here; a non-empty string
    // (source hint or "unknown"), never a panic.
    let lang = sess.detected_language();
    assert!(
        !lang.is_empty(),
        "detected_language fallback must be non-empty"
    );
}

// ---- Registry + cache ----

#[test]
fn registry_lookup_parakeet() {
    let entry = crispasr::registry_lookup("parakeet").expect("registry call");
    if let Some(e) = entry {
        assert!(!e.filename.is_empty());
        assert!(!e.url.is_empty());
    }
}

#[test]
fn registry_default_bundle_omnivoice() {
    let bundle = crispasr::registry_default_bundle("omnivoice")
        .expect("bundle call")
        .expect("omnivoice bundle");
    assert_eq!(bundle.backend, "omnivoice");
    assert_eq!(bundle.artifacts.len(), 2);
    assert_eq!(
        bundle.artifacts[0].kind,
        crispasr::RegistryArtifactKind::Primary
    );
    assert_eq!(bundle.artifacts[0].filename, "omnivoice-f16.gguf");
    assert_eq!(
        bundle.artifacts[1].kind,
        crispasr::RegistryArtifactKind::Companion
    );
    assert_eq!(bundle.artifacts[1].filename, "omnivoice-tokenizer-f16.gguf");
}

#[test]
fn cache_dir_exists() {
    let dir = crispasr::cache_dir(None).expect("cache_dir");
    if let Some(d) = dir {
        assert!(!d.is_empty());
    }
}

// ---- C-ABI parity: new types from bindings-parity milestone ----

#[test]
fn lcs_dedup_empty_inputs() {
    assert_eq!(crispasr::lcs_dedup_prefix_count(&[], &[], 1), 0);
    assert_eq!(crispasr::lcs_dedup_prefix_count(&[1, 2, 3], &[], 1), 0);
    assert_eq!(crispasr::lcs_dedup_prefix_count(&[], &[1, 2, 3], 1), 0);
}

#[test]
fn lcs_dedup_overlap() {
    // prev ends with [3, 4, 5], curr starts with [4, 5, 6] -> drop 2 leading
    let prev = vec![1, 2, 3, 4, 5];
    let curr = vec![4, 5, 6, 7];
    let drop = crispasr::lcs_dedup_prefix_count(&prev, &curr, 1);
    assert!(drop >= 0, "should return non-negative");
}

#[test]
fn titanet_cosine_sim_identical() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![1.0f32, 0.0, 0.0];
    let sim = crispasr::titanet_cosine_sim(&a, &b);
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "identical vectors should have sim ~1.0, got {sim}"
    );
}

#[test]
fn titanet_cosine_sim_orthogonal() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    let sim = crispasr::titanet_cosine_sim(&a, &b);
    assert!(
        sim.abs() < 1e-5,
        "orthogonal vectors should have sim ~0, got {sim}"
    );
}

#[test]
fn kokoro_lang_helpers() {
    assert!(crispasr::kokoro_lang_is_german("de"));
    assert!(crispasr::kokoro_lang_is_german("deu"));
    assert!(!crispasr::kokoro_lang_is_german("en"));
    // "en" always has a native Kokoro voice
    assert!(crispasr::kokoro_lang_has_native_voice("en"));
}

#[test]
fn speaker_db_missing_dir() {
    // Loading from a non-existent directory should return an error
    let result = crispasr::SpeakerDB::load("/nonexistent/speaker_db_dir_12345");
    assert!(result.is_err());
}

#[test]
fn vad_segments_null_model() {
    // Passing a nonsense model path should return an error
    let pcm = vec![0.0f32; 16000];
    let result = crispasr::vad_segments(
        "/nonexistent/vad.gguf",
        &pcm,
        16000,
        0.5,
        250,
        100,
        1,
        false,
    );
    assert!(result.is_err());
}

#[test]
fn vad_slices_null_model() {
    let pcm = vec![0.0f32; 16000];
    let result = crispasr::vad_slices(
        "/nonexistent/vad.gguf",
        &pcm,
        16000,
        0.5,
        250,
        100,
        30,
        30.0,
        1,
    );
    assert!(result.is_err());
}
