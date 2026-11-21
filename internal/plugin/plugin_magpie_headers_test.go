package plugin

import (
	"bufio"
	"context"
	"crypto/sha1"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// magpieOnly are the headers magpie tells a plugin, and a vendor must
// never be told (MAGPIE_ONLY in host.js).
var magpieOnly = []string{
	"x-magpie-work-dir",
	"x-magpie-work-git",
	"x-magpie-fallback",
	"x-magpie-session",
	"x-magpie-subagent",
	"x-magpie-effort",
	"x-magpie-caller-agent",
	"x-magpie-caller-name",
	"x-magpie-caller-via",
}

// #7415: unsaid took those headers off globalThis.fetch only. A plugin
// that forwards every header it was given on a WebSocket, on node:http or
// node:https (object, array, setHeader, appendHeader, a named import or
// require), or on node:http2, sent them to its vendor. The same list has
// to come off a Map, an iterator and inherited WebSocket options, off
// Bun.FetchSession, off an object handed to _storeHeader or addTrailers,
// and off require("ws") and require("undici"). Bun.fetch cannot be
// wrapped. A redirect followed by fetch (#5508, #5520) keeps the same
// rule: the hop carries the plugin's own headers and none of magpie's.
// The plugin also checks the caller's Request, Headers, list, Map and
// options still hold every magpie name after the call.
func TestPluginSendsNoMagpieHeaderToTheVendor(t *testing.T) {
	sandbox(t)
	for _, k := range []string{"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"} {
		t.Setenv(k, "")
	}
	plain := listenHeaders(t, nil)
	cert, _ := certFor(t, net.ParseIP("127.0.0.1"))
	secure := listenHeaders(t, &tls.Config{Certificates: []tls.Certificate{cert}})
	var mu sync.Mutex
	h2seen := map[string]http.Header{}
	h2trails := map[string]http.Header{}
	h2srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		mu.Lock()
		h2seen[r.URL.Path] = r.Header.Clone()
		h2trails[r.URL.Path] = r.Trailer.Clone()
		mu.Unlock()
		fmt.Fprint(w, "ok")
	}))
	h2srv.EnableHTTP2 = true
	h2srv.StartTLS()
	t.Cleanup(h2srv.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	abs, err := filepath.Abs("testdata/strip/index.js")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Add(ctx, abs); err != nil {
		t.Fatal(err)
	}
	if _, err := Providers(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := APIKey(ctx, "stripco", 0, nil, "k-strip", NewAccount); err != nil {
		t.Fatal(err)
	}
	headers := map[string]string{"x-own": "relay-secret"}
	for _, k := range magpieOnly {
		headers[k] = "secret-" + strings.TrimPrefix(k, "x-magpie-")
	}
	call := func(t *testing.T, url string) {
		t.Helper()
		fctx, cancel := context.WithTimeout(ctx, 20*time.Second)
		defer cancel()
		res, err := Fetch(fctx, FetchRequest{
			Provider: "stripco", Model: "strip-1", NPM: "@ai-sdk/openai-compatible",
			URL: url, Method: "GET", Headers: headers,
		})
		if err != nil {
			t.Fatalf("fetch: %v", err)
		}
		defer res.Body.Close()
		b, _ := io.ReadAll(res.Body)
		if res.StatusCode != 200 {
			t.Fatalf("status %d, body %q", res.StatusCode, b)
		}
	}
	check := func(t *testing.T, h http.Header) {
		t.Helper()
		if h == nil {
			t.Fatal("the vendor saw no request")
		}
		if got := h.Get("X-Own"); got != "relay-secret" {
			t.Errorf("x-own = %q, want relay-secret", got)
		}
		if got := h.Get("Authorization"); !strings.HasPrefix(got, "Bearer ") {
			t.Errorf("authorization = %q, want the plugin's key", got)
		}
		for _, k := range magpieOnly {
			if v := h.Values(k); len(v) > 0 {
				t.Errorf("the vendor got %s: %q", k, v)
			}
		}
	}
	plainModes := []string{
		"fetch-object", "fetch-headers", "fetch-tuples", "fetch-request", "fetch-url", "fetch-case",
		"fetch-oneshot-clean", "fetch-proto-init", "fetch-frozen-proto", "fetch-request-init", "fetch-request-undef-init",
		"fetch-post-override-get-req", "fetch-req-obj", "fetch-req-obj-manual", "fetch-iter-tuples-clean", "fetch-consumed-req-body", "fetch-consumed-req-headers",
		"ws-object", "ws-headers", "ws-tuples", "ws-case", "ws-map", "ws-entries", "ws-iterable", "ws-dup-iterable", "ws-inherited", "ws-setter-proto", "ws-buffer-names", "ws-boxed-string-names", "ws-iter-tuples", "ws-mixed-tuples", "ws-3arg-undef", "ws-http",
		"ws-require", "ws-require-3", "ws-require-direct", "ws-require-3-undef", "ws-esm-named",
		"http-url", "http-opts", "http-get", "http-set", "http-append", "http-setheaders",
		"http-tuples", "http-flat", "http-flat-array-val", "http-named", "http-get-named", "http-require", "http-case",
		"http-client", "http-store", "http-trailers", "http-agent",
		"bun-session", "bun-session-detached", "bun-session-oneshot-clean", "bun-session-proto-init", "bun-session-req-obj", "bun-session-consumed-req-body",
		"undici-fetch", "undici-fetch-oneshot-clean", "undici-fetch-proto-init", "undici-fetch-req-obj", "undici-fetch-consumed-req-body",
		"undici-request", "undici-request-proto-init", "undici-request-setter-proto", "undici-request-buffer-names", "undici-request-iter-tuples", "undici-request-dup-iterable",
		"undici-ws", "undici-ws-3-undef",
		"undici-default-fetch", "undici-default-request", "undici-default-ws",
		"undici-esm-fetch", "undici-esm-request", "undici-esm-ws",
	}
	for _, mode := range plainModes {
		t.Run(mode, func(t *testing.T) {
			scheme := "http"
			if strings.HasPrefix(mode, "ws-") {
				scheme = "ws"
			}
			call(t, scheme+"://"+plain.addr+"/"+mode)
			check(t, plain.header("/"+mode))
			if mode == "ws-inherited" || mode == "ws-setter-proto" || mode == "ws-require-3" {
				if got := plain.header("/" + mode).Get("Sec-WebSocket-Protocol"); got != "chat" {
					t.Errorf("Sec-WebSocket-Protocol = %q, want chat", got)
				}
			}
			if mode == "ws-dup-iterable" || mode == "undici-request-dup-iterable" {
				if got := plain.header("/" + mode).Get("X-Dup"); got != "a, b" {
					t.Errorf("x-dup = %q, want \"a, b\"", got)
				}
			}
			if strings.HasSuffix(mode, "-proto-init") || strings.HasSuffix(mode, "-consumed-req-body") || mode == "fetch-consumed-req-headers" || mode == "fetch-frozen-proto" || mode == "fetch-request-init" || mode == "fetch-request-undef-init" || mode == "fetch-req-obj" || mode == "undici-request-setter-proto" {
				if got := plain.method("/" + mode); got != "POST" {
					t.Errorf("method = %q, want POST", got)
				}
				if got := plain.body("/" + mode); got != "proto-body" {
					t.Errorf("body = %q, want proto-body", got)
				}
			}
			if mode == "http-flat-array-val" {
				if got := strings.Join(plain.header("/"+mode).Values("X-Multi"), ","); got != "a,b" {
					t.Errorf("x-multi = %q, want a,b", got)
				}
			}
			if mode == "fetch-post-override-get-req" {
				if got := plain.method("/" + mode); got != "GET" {
					t.Errorf("method = %q, want GET", got)
				}
				if got := plain.body("/" + mode); got != "" {
					t.Errorf("body = %q, want empty", got)
				}
			}
			if mode == "http-trailers" {
				tr := plain.trailer("/" + mode)
				if tr.Get("X-Trail") != "kept" {
					t.Errorf("x-trail = %q, want kept", tr.Get("X-Trail"))
				}
				for _, k := range magpieOnly {
					if v := tr.Values(k); len(v) > 0 {
						t.Errorf("the vendor got trailer %s: %q", k, v)
					}
				}
			}
		})
	}
	t.Run("fetch-redirect", func(t *testing.T) {
		call(t, "http://"+plain.addr+"/fetch-redirect")
		check(t, plain.header("/fetch-redirect"))
		check(t, plain.header("/landed"))
	})
	for _, mode := range []string{"https-url", "https-named", "https-set", "https-case", "https-get"} {
		t.Run(mode, func(t *testing.T) {
			call(t, "https://"+secure.addr+"/"+mode)
			check(t, secure.header("/"+mode))
		})
	}
	for _, mode := range []string{"http2-object", "http2-flat", "http2-flat-array-val", "http2-case", "http2-sensitive", "http2-sensitive-flat", "http2-trailers"} {
		t.Run(mode, func(t *testing.T) {
			call(t, h2srv.URL+"/"+mode)
			mu.Lock()
			h := h2seen["/"+mode]
			tr := h2trails["/"+mode]
			mu.Unlock()
			check(t, h)
			if mode == "http2-flat-array-val" {
				if got := strings.Join(h.Values("X-Multi"), ","); got != "a,b" {
					t.Errorf("x-multi = %q, want a,b", got)
				}
			}
			if mode == "http2-trailers" {
				if tr.Get("X-Trail") != "kept" {
					t.Errorf("x-trail = %q, want kept", tr.Get("X-Trail"))
				}
				for _, k := range magpieOnly {
					if v := tr.Values(k); len(v) > 0 {
						t.Errorf("the vendor got h2 trailer %s: %q", k, v)
					}
				}
			}
		})
	}
	t.Run("bun-fetch", func(t *testing.T) {
		fctx, cancel := context.WithTimeout(ctx, 20*time.Second)
		defer cancel()
		res, err := Fetch(fctx, FetchRequest{
			Provider: "stripco", Model: "strip-1", NPM: "@ai-sdk/openai-compatible",
			URL: "http://127.0.0.1/bun-desc", Method: "GET",
		})
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		var d struct {
			Configurable bool `json:"configurable"`
			Writable     bool `json:"writable"`
			Same         bool `json:"same"`
		}
		if err := json.NewDecoder(res.Body).Decode(&d); err != nil {
			t.Fatal(err)
		}
		if d.Configurable || d.Writable || d.Same {
			t.Fatalf("Bun.fetch descriptor = %+v, want non-configurable, non-writable, distinct from globalThis.fetch", d)
		}
	})
}

// headerSink accepts HTTP/1.1, and a WebSocket upgrade, and keeps the
// headers of each path. A TLS config makes it HTTPS. It answers 200, a
// same-origin 302 for /fetch-redirect, or 101 for a WebSocket. Chunked
// requests also keep their trailers. Every accept and every connection
// is waited on at cleanup, so none of them outlives the test.
type headerSink struct {
	addr    string
	mu      sync.Mutex
	got     map[string]http.Header
	trails  map[string]http.Header
	methods map[string]string
	bodies  map[string]string
	conns   map[net.Conn]struct{}
	wg      sync.WaitGroup
}

func (s *headerSink) header(path string) http.Header {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.got[path]
}

func (s *headerSink) trailer(path string) http.Header {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.trails[path]
}

func (s *headerSink) method(path string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.methods[path]
}

func (s *headerSink) body(path string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bodies[path]
}

func listenHeaders(t *testing.T, cfg *tls.Config) *headerSink {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	if cfg != nil {
		ln = tls.NewListener(ln, cfg)
	}
	s := &headerSink{
		addr:    ln.Addr().String(),
		got:     map[string]http.Header{},
		trails:  map[string]http.Header{},
		methods: map[string]string{},
		bodies:  map[string]string{},
		conns:   map[net.Conn]struct{}{},
	}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			s.mu.Lock()
			s.conns[c] = struct{}{}
			s.mu.Unlock()
			s.wg.Add(1)
			go func() {
				defer s.wg.Done()
				defer func() {
					s.mu.Lock()
					delete(s.conns, c)
					s.mu.Unlock()
					c.Close()
				}()
				s.serve(c)
			}()
		}
	}()
	t.Cleanup(func() {
		ln.Close()
		s.mu.Lock()
		open := make([]net.Conn, 0, len(s.conns))
		for c := range s.conns {
			open = append(open, c)
		}
		s.mu.Unlock()
		for _, c := range open {
			c.Close()
		}
		s.wg.Wait()
	})
	return s
}

func (s *headerSink) serve(c net.Conn) {
	_ = c.SetDeadline(time.Now().Add(5 * time.Second))
	br := bufio.NewReader(c)
	line, err := br.ReadString('\n')
	if err != nil {
		return
	}
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return
	}
	path := fields[1]
	h := make(http.Header)
	for {
		l, err := br.ReadString('\n')
		if err != nil || l == "\r\n" || l == "\n" {
			break
		}
		k, v, ok := strings.Cut(strings.TrimRight(l, "\r\n"), ":")
		if !ok {
			continue
		}
		h.Add(textproto.CanonicalMIMEHeaderKey(strings.TrimSpace(k)), strings.TrimSpace(v))
	}
	s.mu.Lock()
	s.got[path] = h
	s.methods[path] = fields[0]
	s.mu.Unlock()
	if path == "/fetch-redirect" {
		fmt.Fprintf(c, "HTTP/1.1 302 Found\r\nLocation: /landed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
		return
	}
	if key := h.Get("Sec-WebSocket-Key"); key != "" && strings.EqualFold(h.Get("Upgrade"), "websocket") {
		sum := sha1.Sum([]byte(strings.TrimSpace(key) + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
		proto := ""
		if p := h.Get("Sec-WebSocket-Protocol"); p != "" {
			first, _, _ := strings.Cut(p, ",")
			proto = fmt.Sprintf("Sec-WebSocket-Protocol: %s\r\n", strings.TrimSpace(first))
		}
		fmt.Fprintf(c, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n%s\r\n", base64.StdEncoding.EncodeToString(sum[:]), proto)
		// the client closes once it is open; answer that frame and return
		// so this goroutine does not sit out the read deadline
		_ = c.SetReadDeadline(time.Now().Add(2 * time.Second))
		buf := make([]byte, 256)
		_, _ = br.Read(buf)
		_, _ = c.Write([]byte{0x88, 0x00})
		return
	}
	if strings.Contains(strings.ToLower(h.Get("Transfer-Encoding")), "chunked") {
		s.mu.Lock()
		s.trails[path] = readChunkedTrailers(br)
		s.mu.Unlock()
	} else if cl, err := strconv.Atoi(strings.TrimSpace(h.Get("Content-Length"))); err == nil && cl > 0 {
		body := make([]byte, cl)
		if _, err := io.ReadFull(br, body); err == nil {
			s.mu.Lock()
			s.bodies[path] = string(body)
			s.mu.Unlock()
		}
	}
	fmt.Fprintf(c, "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
}

// readChunkedTrailers drops a chunked body and returns the trailers that
// follow the last chunk. A short or broken body is whatever arrived.
func readChunkedTrailers(br *bufio.Reader) http.Header {
	tr := make(http.Header)
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return tr
		}
		hex, _, _ := strings.Cut(strings.TrimRight(line, "\r\n"), ";")
		n, err := strconv.ParseInt(strings.TrimSpace(hex), 16, 64)
		if err != nil {
			return tr
		}
		if n > 0 {
			if _, err := io.CopyN(io.Discard, br, n); err != nil {
				return tr
			}
			_, _ = br.ReadString('\n')
			continue
		}
		for {
			l, err := br.ReadString('\n')
			if err != nil || l == "\r\n" || l == "\n" {
				return tr
			}
			k, v, ok := strings.Cut(strings.TrimRight(l, "\r\n"), ":")
			if !ok {
				continue
			}
			tr.Add(textproto.CanonicalMIMEHeaderKey(strings.TrimSpace(k)), strings.TrimSpace(v))
		}
	}
}
