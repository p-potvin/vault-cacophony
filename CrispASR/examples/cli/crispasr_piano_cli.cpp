// crispasr_piano_cli.cpp — see crispasr_piano_cli.h.

#include "crispasr_piano_cli.h"

#include "common-crispasr.h" // read_audio_data
#include "crispasr_model_mgr_cli.h"
#include "whisper_params.h"

#include "core/gguf_loader.h" // core_gguf::open_metadata / kv_str
#include "piano_transcription.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

// MIDI note -> name. Sharps only, matching the convention the chord vocabulary
// already uses (btc_chord_vocab.h), so the two music surfaces agree.
std::string midi_note_name(int midi) {
    static const char* kNames[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    if (midi < 0)
        return "?";
    const int pc = midi % 12;
    const int octave = midi / 12 - 1; // MIDI 60 = C4
    return std::string(kNames[pc]) + std::to_string(octave);
}

// One line per note, tab-separated and greppable:
// "onset_sec  offset_sec  midi  name  velocity".
// Deliberately NOT the .lab shape used by --chords: a chord timeline is
// contiguous non-overlapping spans, whereas piano notes overlap freely, so
// reusing that layout would imply a structure the data does not have.
void print_notes_text(const piano_transcription_result& r) {
    for (int i = 0; i < r.n_notes; i++) {
        const piano_note_event& n = r.note_events[i];
        printf("%.3f\t%.3f\t%d\t%s\t%d\n", n.onset_time, n.offset_time, n.midi_note,
               midi_note_name(n.midi_note).c_str(), n.velocity);
    }
}

void print_notes_json(const piano_transcription_result& r, const std::string& fname) {
    printf("{\n");
    printf("  \"file\": \"%s\",\n", fname.c_str());
    printf("  \"n_notes\": %d,\n", r.n_notes);
    printf("  \"n_pedals\": %d,\n", r.n_pedals);
    printf("  \"notes\": [\n");
    for (int i = 0; i < r.n_notes; i++) {
        const piano_note_event& n = r.note_events[i];
        printf("    {\"onset\": %.3f, \"offset\": %.3f, \"midi\": %d, \"name\": \"%s\", \"velocity\": %d}%s\n",
               n.onset_time, n.offset_time, n.midi_note, midi_note_name(n.midi_note).c_str(), n.velocity,
               i + 1 < r.n_notes ? "," : "");
    }
    printf("  ],\n");
    printf("  \"pedals\": [\n");
    for (int i = 0; i < r.n_pedals; i++) {
        const piano_pedal_event& p = r.pedal_events[i];
        printf("    {\"onset\": %.3f, \"offset\": %.3f}%s\n", p.onset_time, p.offset_time,
               i + 1 < r.n_pedals ? "," : "");
    }
    printf("  ]\n}\n");
}

} // namespace

int crispasr_run_piano(const whisper_params& params) {
    if (params.fname_inp.empty()) {
        fprintf(stderr, "crispasr: --piano needs an input file (-f)\n");
        return 2;
    }

    const bool json = params.piano_format == "json";
    if (!json && !params.piano_format.empty() && params.piano_format != "text") {
        fprintf(stderr, "crispasr: --piano-format: unknown format '%s' (expected text or json)\n",
                params.piano_format.c_str());
        return 2;
    }

    const std::string backend_key = params.backend.empty() ? "piano-transcription" : params.backend;
    const std::string model = crispasr_resolve_model_cli(params.model, backend_key, params.no_prints, params.cache_dir,
                                                         params.auto_download, "", params.accept_license);
    if (model.empty()) {
        fprintf(stderr, "crispasr: --piano: could not resolve a model.\n");
        return 2;
    }

    gguf_context* meta = core_gguf::open_metadata(model.c_str());
    if (!meta) {
        fprintf(stderr, "crispasr: --piano: cannot open '%s'\n", model.c_str());
        return 2;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    core_gguf::free_metadata(meta);
    if (arch != "piano-transcription" && arch != "piano_transcription") {
        fprintf(stderr, "crispasr: --piano: '%s' is not a piano model (arch='%s').\n", model.c_str(), arch.c_str());
        return 2;
    }

    piano_transcription_params pp = piano_transcription_default_params();
    pp.n_threads = params.n_threads;
    pp.verbosity = params.no_prints ? 0 : 1;
    piano_transcription_ctx* ctx = piano_transcription_init_from_file(model.c_str(), pp);
    if (!ctx) {
        fprintf(stderr, "crispasr: --piano: failed to load '%s'\n", model.c_str());
        return 2;
    }
    const int sr = (int)piano_transcription_sample_rate(ctx);

    int rc = 0;
    for (const auto& fname : params.fname_inp) {
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(fname, mono, stereo, /*stereo=*/false, /*target_rate=*/sr)) {
            fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
            rc = 20;
            continue;
        }

        piano_transcription_result res{};
        if (piano_transcription_transcribe(ctx, mono.data(), (int)mono.size(), &res) != 0) {
            fprintf(stderr, "crispasr: --piano failed on '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }

        if (json) {
            print_notes_json(res, fname);
        } else {
            if (!params.no_prints && params.fname_inp.size() > 1)
                printf("# %s\n", fname.c_str());
            print_notes_text(res);
        }
        if (!params.no_prints)
            fprintf(stderr, "crispasr: %s: %d notes, %d pedal events\n", fname.c_str(), res.n_notes, res.n_pedals);
        piano_transcription_result_free(&res);
    }

    piano_transcription_free(ctx);
    return rc;
}
