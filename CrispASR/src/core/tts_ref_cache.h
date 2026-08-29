// core/tts_ref_cache.h — on-disk cache for expensive TTS reference
// conditioning (voice cloning). Encoding a reference — a codec encoder, a
// Conformer/Perceiver, or an ASR pass over the clip — is slow and produces a
// small, reusable blob. Cache it so repeated runs skip the encode entirely
// (#221). Lives in core/ so BOTH the runtime backends and every consumer
// (CLI adapters, C ABI, server, wrappers) can share it.
//
// Two keying modes:
//   1. Path-keyed (path_for + load*/save*): cache next to a voice file,
//      invalidated by that file's mtime. Convenient for the CLI (`<voice>.iro32latent`).
//   2. Content-addressed (content_* + get_floats/put_floats over a key buffer):
//      key = hash of the raw reference (e.g. the PCM), stored in a cache dir.
//      No file path needed, so a backend can cache in its RUNTIME set_reference
//      and EVERY caller — including the C ABI and wrappers that only pass raw
//      PCM — benefits, and any runtime adopts it in a few lines.
//
// Format (little-endian):
//   magic "CRC1" | u32 version | u32 tag_len | tag bytes |
//   u32 nshape | nshape * u32 shape | u32 payload_bytes | payload bytes
//
// The tag identifies the backend + representation (e.g. "irodori-latent",
// "indextts-cond", "f5-reftext"); a mismatching tag rejects the cache so two
// backends never read each other's blob. Payload is opaque bytes — callers
// interpret it as float32 latents, UTF-8 text, etc.
//
// Header-only; no backend dependencies.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifdef _WIN32
#include <direct.h> // _mkdir
#endif

namespace crispasr_ref_cache {

inline const char kMagic[4] = {'C', 'R', 'C', '1'};
inline constexpr uint32_t kVersion = 1;

// Globally disable via CRISPASR_TTS_REF_CACHE=0.
inline bool disabled() {
    const char* e = std::getenv("CRISPASR_TTS_REF_CACHE");
    return e && std::strcmp(e, "0") == 0;
}

// "<voice><suffix>", e.g. suffix ".iro32latent".
inline std::string path_for(const std::string& voice, const char* suffix) {
    return voice + suffix;
}

inline bool mtime(const std::string& path, time_t& out) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return false;
    out = st.st_mtime;
    return true;
}

// Load a cache if present, well-formed, tagged `tag`, and at least as new as
// `voice_path`. Fills `shape` + `payload` and returns true on success.
inline bool load(const std::string& cache_path, const std::string& voice_path, const char* tag,
                 std::vector<uint32_t>& shape, std::vector<uint8_t>& payload) {
    time_t ct = 0, vt = 0;
    if (!mtime(cache_path, ct))
        return false;
    if (mtime(voice_path, vt) && ct < vt)
        return false; // stale
    FILE* f = std::fopen(cache_path.c_str(), "rb");
    if (!f)
        return false;
    auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
    auto rdu32 = [&](uint32_t& v) { return rd(&v, sizeof(uint32_t)); };
    char magic[4];
    uint32_t ver = 0, tag_len = 0, nshape = 0, pbytes = 0;
    bool ok = rd(magic, 4) && std::memcmp(magic, kMagic, 4) == 0 && rdu32(ver) && ver == kVersion && rdu32(tag_len) &&
              tag_len < 256;
    if (ok) {
        std::string got(tag_len, '\0');
        ok = (tag_len == 0 || rd(&got[0], tag_len)) && got == tag && rdu32(nshape) && nshape < 16;
    }
    if (ok) {
        shape.resize(nshape);
        for (uint32_t i = 0; ok && i < nshape; i++)
            ok = rdu32(shape[i]);
        ok = ok && rdu32(pbytes) && pbytes <= (256u << 20); // 256 MB sanity cap
    }
    if (ok) {
        payload.resize(pbytes);
        ok = (pbytes == 0 || rd(payload.data(), pbytes));
    }
    std::fclose(f);
    return ok;
}

inline void save(const std::string& cache_path, const char* tag, const std::vector<uint32_t>& shape,
                 const void* payload, size_t nbytes) {
    FILE* f = std::fopen(cache_path.c_str(), "wb");
    if (!f)
        return;
    auto wr = [&](const void* p, size_t n) { std::fwrite(p, 1, n, f); };
    auto wru32 = [&](uint32_t v) { wr(&v, sizeof(uint32_t)); };
    uint32_t tag_len = (uint32_t)std::strlen(tag);
    wr(kMagic, 4);
    wru32(kVersion);
    wru32(tag_len);
    wr(tag, tag_len);
    wru32((uint32_t)shape.size());
    for (uint32_t d : shape)
        wru32(d);
    wru32((uint32_t)nbytes);
    wr(payload, nbytes);
    std::fclose(f);
}

// ── float-blob convenience (latents / conditioning) ──
inline bool load_floats(const std::string& cache_path, const std::string& voice_path, const char* tag,
                        std::vector<uint32_t>& shape, std::vector<float>& out) {
    std::vector<uint8_t> payload;
    if (!load(cache_path, voice_path, tag, shape, payload) || (payload.size() % sizeof(float)) != 0)
        return false;
    out.resize(payload.size() / sizeof(float));
    std::memcpy(out.data(), payload.data(), payload.size());
    return true;
}

inline void save_floats(const std::string& cache_path, const char* tag, const std::vector<uint32_t>& shape,
                        const float* data, size_t count) {
    save(cache_path, tag, shape, data, count * sizeof(float));
}

// ── content-addressed cache (runtime; benefits every consumer) ──
// FNV-1a 64-bit over a byte range.
inline uint64_t fnv1a(const void* data, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Cache directory: CRISPASR_TTS_REF_CACHE_DIR, else <temp>/crispasr-tts-refcache.
inline std::string cache_dir() {
    if (const char* d = std::getenv("CRISPASR_TTS_REF_CACHE_DIR"))
        if (*d)
            return d;
    const char* tmp = std::getenv("TMPDIR");
#ifdef _WIN32
    if (!(tmp && *tmp))
        tmp = std::getenv("TEMP");
    if (!(tmp && *tmp))
        tmp = std::getenv("TMP");
    std::string base = (tmp && *tmp) ? tmp : ".";
#else
    std::string base = (tmp && *tmp) ? tmp : "/tmp";
#endif
    while (!base.empty() && (base.back() == '/' || base.back() == '\\'))
        base.pop_back();
    std::string dir = base + "/crispasr-tts-refcache";
    // best-effort; ignored if it exists
#ifdef _WIN32
    _mkdir(dir.c_str());
#else
    ::mkdir(dir.c_str(), 0700);
#endif
    return dir;
}

// Content path for a (tag, key-bytes) pair: <cache_dir>/<tag>-<hash>.ref
inline std::string content_path(const char* tag, const void* key, size_t key_len) {
    uint64_t h = fnv1a(tag, std::strlen(tag)) ^ fnv1a(key, key_len);
    char name[80];
    std::snprintf(name, sizeof(name), "/%s-%016llx.ref", tag, (unsigned long long)h);
    return cache_dir() + name;
}

// Content-addressed float-blob get/put keyed on `key` bytes (e.g. the raw
// reference PCM). Returns true on a cache hit. Disabled by CRISPASR_TTS_REF_CACHE=0.
inline bool get_floats(const char* tag, const void* key, size_t key_len, std::vector<uint32_t>& shape,
                       std::vector<float>& out) {
    if (disabled())
        return false;
    return load_floats(content_path(tag, key, key_len), /*voice_path=*/"", tag, shape, out);
}

inline void put_floats(const char* tag, const void* key, size_t key_len, const std::vector<uint32_t>& shape,
                       const float* data, size_t count) {
    if (disabled())
        return;
    save_floats(content_path(tag, key, key_len), tag, shape, data, count);
}

// ── content-addressed raw-byte blob get/put ──
// Same as get_floats/put_floats but for opaque payloads (e.g. int32 RVQ codes)
// that aren't float32. Returns true on a cache hit. Disabled by CRISPASR_TTS_REF_CACHE=0.
inline bool get_bytes(const char* tag, const void* key, size_t key_len, std::vector<uint32_t>& shape,
                      std::vector<uint8_t>& out) {
    if (disabled())
        return false;
    return load(content_path(tag, key, key_len), /*voice_path=*/"", tag, shape, out);
}

inline void put_bytes(const char* tag, const void* key, size_t key_len, const std::vector<uint32_t>& shape,
                      const void* data, size_t nbytes) {
    if (disabled())
        return;
    save(content_path(tag, key, key_len), tag, shape, data, nbytes);
}

} // namespace crispasr_ref_cache
