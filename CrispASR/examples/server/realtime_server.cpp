#include "realtime_server.h"
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../json.hpp"
#include "crispasr.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define CLOSE_SOCKET closesocket
#define SOCKET_ERRNO WSAGetLastError()
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int socket_t;
#define INVALID_SOCKET (-1)
#define CLOSE_SOCKET close
#define SOCKET_ERRNO errno
#endif

static std::atomic<bool> g_rt_running{false};
static std::thread g_rt_thread;
static socket_t g_rt_listen_fd = INVALID_SOCKET;

// SHA-1 for WebSocket handshake (minimal, RFC 3174 compliant)
namespace {
struct sha1_ctx {
    uint32_t h[5];
    uint64_t len;
    uint8_t buf[64];
    int buf_len;
};
static void sha1_init(sha1_ctx& c) {
    c.h[0] = 0x67452301;
    c.h[1] = 0xEFCDAB89;
    c.h[2] = 0x98BADCFE;
    c.h[3] = 0x10325476;
    c.h[4] = 0xC3D2E1F0;
    c.len = 0;
    c.buf_len = 0;
}
static uint32_t rotl(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}
static void sha1_block(sha1_ctx& c, const uint8_t* d) {
    uint32_t w[80], a = c.h[0], b = c.h[1], cc = c.h[2], dd = c.h[3], e = c.h[4];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)d[i * 4] << 24 | (uint32_t)d[i * 4 + 1] << 16 | (uint32_t)d[i * 4 + 2] << 8 | d[i * 4 + 3];
    for (int i = 16; i < 80; i++)
        w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & dd);
            k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ cc ^ dd;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & cc) | (b & dd) | (cc & dd);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ cc ^ dd;
            k = 0xCA62C1D6;
        }
        uint32_t t = rotl(a, 5) + f + e + k + w[i];
        e = dd;
        dd = cc;
        cc = rotl(b, 30);
        b = a;
        a = t;
    }
    c.h[0] += a;
    c.h[1] += b;
    c.h[2] += cc;
    c.h[3] += dd;
    c.h[4] += e;
}
static void sha1_update(sha1_ctx& c, const void* data, size_t len) {
    auto* p = (const uint8_t*)data;
    c.len += len;
    while (len > 0) {
        int space = 64 - c.buf_len;
        int take = (int)len < space ? (int)len : space;
        memcpy(c.buf + c.buf_len, p, take);
        c.buf_len += take;
        p += take;
        len -= take;
        if (c.buf_len == 64) {
            sha1_block(c, c.buf);
            c.buf_len = 0;
        }
    }
}
static void sha1_final(sha1_ctx& c, uint8_t out[20]) {
    uint64_t bits = c.len * 8;
    uint8_t pad = 0x80;
    sha1_update(c, &pad, 1);
    pad = 0;
    while (c.buf_len != 56)
        sha1_update(c, &pad, 1);
    uint8_t be[8];
    for (int i = 0; i < 8; i++)
        be[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha1_update(c, be, 8);
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c.h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c.h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c.h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c.h[i];
    }
}

static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::string base64_encode(const uint8_t* d, int n) {
    std::string r;
    for (int i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)d[i] << 16;
        if (i + 1 < n)
            v |= (uint32_t)d[i + 1] << 8;
        if (i + 2 < n)
            v |= d[i + 2];
        r += b64[(v >> 18) & 63];
        r += b64[(v >> 12) & 63];
        r += (i + 1 < n) ? b64[(v >> 6) & 63] : '=';
        r += (i + 2 < n) ? b64[v & 63] : '=';
    }
    return r;
}

static std::string ws_accept_key(const std::string& client_key) {
    std::string cat = client_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    sha1_ctx c;
    sha1_init(c);
    sha1_update(c, cat.data(), cat.size());
    uint8_t hash[20];
    sha1_final(c, hash);
    return base64_encode(hash, 20);
}

static bool recv_exact(socket_t fd, void* buf, size_t n) {
    auto* p = (uint8_t*)buf;
    while (n > 0) {
        auto r = recv(fd, (char*)p, (int)n, 0);
        if (r <= 0)
            return false;
        p += r;
        n -= r;
    }
    return true;
}

static bool send_all(socket_t fd, const void* buf, size_t n) {
    auto* p = (const uint8_t*)buf;
    while (n > 0) {
        auto r = send(fd, (const char*)p, (int)n, 0);
        if (r <= 0)
            return false;
        p += r;
        n -= r;
    }
    return true;
}

static bool ws_send_text(socket_t fd, const std::string& text) {
    size_t n = text.size();
    uint8_t head[10];
    int hlen = 0;
    head[0] = 0x81; // FIN + text
    if (n < 126) {
        head[1] = (uint8_t)n;
        hlen = 2;
    } else if (n <= 65535) {
        head[1] = 126;
        head[2] = (uint8_t)(n >> 8);
        head[3] = (uint8_t)n;
        hlen = 4;
    } else {
        head[1] = 127;
        for (int i = 0; i < 8; i++)
            head[2 + i] = (uint8_t)(n >> (56 - 8 * i));
        hlen = 10;
    }
    if (!send_all(fd, head, hlen))
        return false;
    return send_all(fd, text.data(), n);
}

static void ws_send_close(socket_t fd) {
    uint8_t f[2] = {0x88, 0x00};
    send_all(fd, f, 2);
}

static int ws_read_frame(socket_t fd, std::vector<uint8_t>& payload, uint8_t* out_opcode) {
    uint8_t h[2];
    if (!recv_exact(fd, h, 2))
        return -1;
    bool fin = (h[0] & 0x80) != 0;
    uint8_t opcode = h[0] & 0x0F;
    bool masked = (h[1] & 0x80) != 0;
    uint64_t len = h[1] & 0x7F;
    if (len == 126) {
        uint8_t e[2];
        if (!recv_exact(fd, e, 2))
            return -1;
        len = ((uint64_t)e[0] << 8) | e[1];
    } else if (len == 127) {
        uint8_t e[8];
        if (!recv_exact(fd, e, 8))
            return -1;
        len = 0;
        for (int i = 0; i < 8; i++)
            len = (len << 8) | e[i];
    }
    uint8_t mask[4] = {0, 0, 0, 0};
    if (masked) {
        if (!recv_exact(fd, mask, 4))
            return -1;
    }
    if (len > 1024 * 1024 * 10)
        return -1; // 10MB limit
    payload.resize((size_t)len);
    if (len > 0) {
        if (!recv_exact(fd, payload.data(), (size_t)len))
            return -1;
        if (masked) {
            for (size_t i = 0; i < len; i++)
                payload[i] ^= mask[i % 4];
        }
    }
    if (out_opcode)
        *out_opcode = opcode;
    return (int)len;
}

static std::vector<uint8_t> base64_decode(const std::string& in) {
    std::vector<uint8_t> out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++)
        T[(int)b64[i]] = i;

    int val = 0, valb = -8;
    for (char c : in) {
        if (T[(unsigned char)c] == -1)
            break;
        val = (val << 6) + T[(unsigned char)c];
        valb += 6;
        if (valb >= 0) {
            out.push_back((uint8_t)((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

} // namespace

struct rt_session {
    socket_t client_fd;
    CrispasrBackend* backend;
    std::mutex* model_mutex;
    whisper_params rp;
    std::vector<float> audio_buffer;
    std::string text_sent;

    rt_session(socket_t fd, CrispasrBackend* b, std::mutex* m, whisper_params p)
        : client_fd(fd), backend(b), model_mutex(m), rp(std::move(p)) {}

    void process_audio() {
        if (audio_buffer.empty())
            return;
        std::lock_guard<std::mutex> lock(*model_mutex);
        std::string current_buffer_text;

        backend->transcribe_streaming(
            audio_buffer.data(), audio_buffer.size(), 0, rp,
            [&](const std::string& partial, bool is_final) { current_buffer_text = partial; });

        // Simple sliding string diffing for JSON SSE
        std::string diff;
        if (current_buffer_text.size() > text_sent.size() &&
            current_buffer_text.compare(0, text_sent.size(), text_sent) == 0) {
            diff = current_buffer_text.substr(text_sent.size());
        } else {
            diff = current_buffer_text; // full replace
        }

        if (!diff.empty()) {
            nlohmann::json evt;
            evt["type"] = "conversation.item.input_audio_transcription.delta";
            evt["delta"] = diff;
            ws_send_text(client_fd, evt.dump());
            text_sent = current_buffer_text;
        }
    }

    void handle_commit() {
        // flush
        process_audio();
        nlohmann::json evt;
        evt["type"] = "conversation.item.input_audio_transcription.completed";
        evt["transcript"] = text_sent;
        ws_send_text(client_fd, evt.dump());

        // Reset for next utterance
        audio_buffer.clear();
        text_sent.clear();
    }
};

static void rt_handle_connection(rt_session* sess) {
    char req_buf[8192];
    int req_len = 0;
    while (req_len < (int)sizeof(req_buf) - 1) {
        int n = recv(sess->client_fd, req_buf + req_len, sizeof(req_buf) - 1 - req_len, 0);
        if (n <= 0)
            break;
        req_len += n;
        req_buf[req_len] = '\0';
        if (strstr(req_buf, "\r\n\r\n"))
            break;
    }
    if (req_len <= 0) {
        CLOSE_SOCKET(sess->client_fd);
        delete sess;
        return;
    }
    req_buf[req_len] = '\0';

    std::string req(req_buf);
    std::string ws_key;
    auto pos = req.find("Sec-WebSocket-Key:");
    if (pos == std::string::npos)
        pos = req.find("sec-websocket-key:");
    if (pos != std::string::npos) {
        auto start = req.find_first_not_of(" \t", pos + 18);
        auto end = req.find("\r\n", start);
        if (start != std::string::npos && end != std::string::npos)
            ws_key = req.substr(start, end - start);
    }
    if (ws_key.empty()) {
        const char* bad = "HTTP/1.1 400 Bad Request\r\n\r\n";
        send_all(sess->client_fd, bad, strlen(bad));
        CLOSE_SOCKET(sess->client_fd);
        delete sess;
        return;
    }

    std::string accept = ws_accept_key(ws_key);
    std::string resp = "HTTP/1.1 101 Switching Protocols\r\n"
                       "Upgrade: websocket\r\n"
                       "Connection: Upgrade\r\n"
                       "Sec-WebSocket-Accept: " +
                       accept + "\r\n\r\n";
    if (!send_all(sess->client_fd, resp.data(), resp.size())) {
        CLOSE_SOCKET(sess->client_fd);
        delete sess;
        return;
    }

    nlohmann::json created;
    created["type"] = "session.created";
    ws_send_text(sess->client_fd, created.dump());

    std::vector<uint8_t> payload;
    while (g_rt_running.load()) {
        uint8_t opcode = 0;
        int len = ws_read_frame(sess->client_fd, payload, &opcode);
        if (len < 0)
            break;
        if (opcode == 0x08)
            break; // close

        if (opcode == 0x01 && len > 0) { // text
            std::string msg(payload.begin(), payload.end());
            try {
                auto j = nlohmann::json::parse(msg);
                std::string type = j.value("type", "");
                if (type == "input_audio_buffer.append") {
                    std::string b64 = j.value("audio", "");
                    if (!b64.empty()) {
                        auto pcm16 = base64_decode(b64);
                        int n_samples = pcm16.size() / 2;
                        const int16_t* p = (const int16_t*)pcm16.data();
                        for (int i = 0; i < n_samples; i++) {
                            sess->audio_buffer.push_back((float)p[i] / 32768.0f);
                        }
                        if (sess->audio_buffer.size() > 16000 * 30) {
                            // Too long! Force commit if we reach 30 seconds
                            sess->handle_commit();
                        } else {
                            // Process incrementally every half second
                            if (sess->audio_buffer.size() % 8000 < n_samples) {
                                sess->process_audio();
                            }
                        }
                    }
                } else if (type == "input_audio_buffer.commit") {
                    sess->handle_commit();
                }
            } catch (...) {
            }
        }
    }

    ws_send_close(sess->client_fd);
    CLOSE_SOCKET(sess->client_fd);
    delete sess;
}

static void rt_listener_thread(CrispasrBackend* backend, std::mutex* model_mutex, whisper_params base_params) {
    while (g_rt_running.load()) {
        struct sockaddr_in addr;
        socklen_t addr_len = sizeof(addr);
        socket_t client = accept(g_rt_listen_fd, (struct sockaddr*)&addr, &addr_len);
        if (client == INVALID_SOCKET) {
            continue;
        }
        auto* sess = new rt_session(client, backend, model_mutex, base_params);
        std::thread(rt_handle_connection, sess).detach();
    }
}

int realtime_server_start(CrispasrBackend* backend, std::mutex& model_mutex, const whisper_params& base_params,
                          int port) {
    if (g_rt_running.load())
        return 0;
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    g_rt_listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_rt_listen_fd == INVALID_SOCKET)
        return -1;

    int opt = 1;
    setsockopt(g_rt_listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(g_rt_listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        CLOSE_SOCKET(g_rt_listen_fd);
        g_rt_listen_fd = INVALID_SOCKET;
        return -1;
    }

    if (listen(g_rt_listen_fd, 4) < 0) {
        CLOSE_SOCKET(g_rt_listen_fd);
        g_rt_listen_fd = INVALID_SOCKET;
        return -1;
    }

    g_rt_running.store(true);
    g_rt_thread = std::thread(rt_listener_thread, backend, &model_mutex, base_params);

    fprintf(stderr, "realtime: vLLM Realtime WebSocket listening on ws://0.0.0.0:%d/v1/realtime\n", port);
    return 0;
}

void realtime_server_stop() {
    if (!g_rt_running.load())
        return;
    g_rt_running.store(false);
    if (g_rt_listen_fd != INVALID_SOCKET) {
        CLOSE_SOCKET(g_rt_listen_fd);
        g_rt_listen_fd = INVALID_SOCKET;
    }
    if (g_rt_thread.joinable())
        g_rt_thread.join();
}
