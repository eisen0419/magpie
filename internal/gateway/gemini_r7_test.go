package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/settings"
)

// An uncataloged Gemini 4 that a SetUpstreamName points at has no recorded
// efforts. The listed model's xhigh must not be borrowed for it: the level
// stays unknown and the request goes at high.
func TestUncatalogedGemini4RenameStaysAtHigh(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var paths []string
	var levels []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		paths = append(paths, r.URL.Path)
		levels = append(levels, gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String())
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen"},
	}); err != nil {
		t.Fatal(err)
	}
	p, err := provider.Find("gem")
	if err != nil {
		t.Fatal(err)
	}
	if got := geminiGateEfforts(*p, "gemini-4-hydrogen", "gemini-4-quartz"); len(got) != 0 {
		t.Fatalf("uncataloged quartz efforts %v, want none", got)
	}
	if err := provider.SetUpstreamName("gem/gemini-4-hydrogen", "gemini-4-quartz"); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 1 || !strings.Contains(paths[0], "gemini-4-quartz") {
		t.Fatalf("path %v, want quartz", paths)
	}
	if len(levels) != 1 || levels[0] != "high" {
		t.Fatalf("thinkingLevel %v, want high", levels)
	}
}

// SetUpstreamName("gem/*", "gemini-4-argon") names one model for every
// listing. The request already goes to Argon; the level has to be Argon's
// too, not Hydrogen's xhigh. A "*" that still wraps the id (peta-*) is not
// a model the levels are read from.
func TestLiteralWildcardTargetUsesItsOwnEfforts(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", Gemini: "http://127.0.0.1:9/v1beta",
		Models: []string{"gemini-4-hydrogen", "gemini-4-argon"},
	}); err != nil {
		t.Fatal(err)
	}
	p, err := provider.Find("gem")
	if err != nil {
		t.Fatal(err)
	}
	wires := map[string]string{"gem/*": "gemini-4-argon"}
	if as := geminiCheckedModel(*p, "gemini-4-hydrogen", wires); as != "gemini-4-argon" {
		t.Fatalf("checked %q, want gemini-4-argon", as)
	}
	if as := geminiCheckedModel(*p, "gemini-4-hydrogen", map[string]string{"gem/*": "peta-*"}); as != "gemini-4-hydrogen" {
		t.Fatalf("wrapping wildcard checked %q, want the listed model", as)
	}
	// a name on one model wins over the provider's literal "*"
	if as := geminiCheckedModel(*p, "gemini-4-hydrogen", map[string]string{"gem/*": "gemini-4-argon", "gem/gemini-4-hydrogen": "gemini-4-neon"}); as != "gemini-4-neon" {
		t.Fatalf("specific rename checked %q, want gemini-4-neon", as)
	}

	var mu sync.Mutex
	var paths []string
	var levels []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		paths = append(paths, r.URL.Path)
		levels = append(levels, gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String())
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen", "gemini-4-argon"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("gem/*", "gemini-4-argon"); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 1 || !strings.Contains(paths[0], "gemini-4-argon") || strings.Contains(paths[0], "gemini-4-hydrogen") {
		t.Fatalf("path %v, want argon", paths)
	}
	if len(levels) != 1 || levels[0] != "high" {
		t.Fatalf("thinkingLevel %v, want high", levels)
	}
}

// A translated request that Gemini, Code Assist or Vertex answers with
// "Thinking level XHIGH is not supported" is asked again at high, and the
// next request starts at high. thinking_config is not marked unfit.
func TestTranslatedXHighRefusalRetriesAtHigh(t *testing.T) {
	t.Run("gemini", func(t *testing.T) {
		fresh(t)
		loadGeminiXHighCatalog(t)
		var mu sync.Mutex
		var levels []string
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			lv := gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String()
			mu.Lock()
			levels = append(levels, lv)
			mu.Unlock()
			if strings.EqualFold(lv, "xhigh") {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, aiStudioRefusal("xhigh"))
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, vertexPong)
		}))
		t.Cleanup(up.Close)
		if err := provider.Save(provider.Provider{
			ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
			Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen", "gemini-4-argon"},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		ask := func() int {
			t.Helper()
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
				`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
			return rec.Code
		}
		if code := ask(); code != http.StatusOK {
			t.Fatalf("first status %d", code)
		}
		mu.Lock()
		got := append([]string(nil), levels...)
		mu.Unlock()
		if len(got) != 2 || got[0] != "xhigh" || got[1] != "high" {
			t.Fatalf("retry levels %v, want xhigh then high", got)
		}
		if code := ask(); code != http.StatusOK {
			t.Fatalf("second status %d", code)
		}
		mu.Lock()
		got = append([]string(nil), levels...)
		mu.Unlock()
		if len(got) != 3 || got[2] != "high" {
			t.Fatalf("remembered levels %v, want the next request to start at high", got)
		}
		if code := askArgon(t, s); code != http.StatusOK {
			t.Fatalf("argon after the refusal: %d", code)
		}
	})

	t.Run("vertex", func(t *testing.T) {
		v := &vertexUp{}
		v.reply = func(_ int, body map[string]any) (int, string) {
			gen, _ := body["generationConfig"].(map[string]any)
			tc, _ := gen["thinkingConfig"].(map[string]any)
			lv, _ := tc["thinkingLevel"].(string)
			if strings.EqualFold(lv, "xhigh") {
				return http.StatusBadRequest, aiStudioRefusal("xhigh")
			}
			return http.StatusOK, vertexPong
		}
		s := vertexServe(t, v, geminiXHighCatalog)
		vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
		if n := v.count(); n != 2 {
			t.Fatalf("vertex calls %d, want 2", n)
		}
		v.mu.Lock()
		first, second := thinkingOf(v.asked[0])["thinkingLevel"], thinkingOf(v.asked[1])["thinkingLevel"]
		v.mu.Unlock()
		if first != "xhigh" || second != "high" {
			t.Fatalf("vertex thinking %v then %v, want xhigh then high", first, second)
		}
		before := v.count()
		vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
		if v.count() != before+1 || thinkingOf(v.asked[len(v.asked)-1])["thinkingLevel"] != "high" {
			t.Fatalf("remembered vertex calls %d last %v", v.count()-before, thinkingOf(v.last(t)))
		}
	})

	t.Run("codeassist", func(t *testing.T) {
		levels := codeAssistXHighRetry(t)
		if len(levels) != 2 || levels[0] != "xhigh" || levels[1] != "high" {
			t.Fatalf("code assist retry levels %v, want xhigh then high", levels)
		}
	})
}

// A SetModelID plus a provider "*" is not a SetUpstreamName. A ModelWires
// entry added after the snapshot was read does not make the synthesized
// name one: the effort variant still goes out.
func TestSnapshotKeepsASetModelIDWhenAWireIsAdded(t *testing.T) {
	fresh(t)
	if err := os.MkdirAll(filepath.Dir(catalog.LivePath("antigravity")), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalog.LivePath("antigravity"), []byte(`{"models":[{"id":"flash-9-low"},{"id":"flash-9-high"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog.Reset()
	t.Cleanup(catalog.Reset)
	s := settings.Load()
	s.ModelIDs = map[string]string{"antigravity/my-flash": "flash-9"}
	s.ModelWires = map[string]string{"antigravity/*": "vendor/*"}
	if err := settings.Save(s); err != nil {
		t.Fatal(err)
	}
	wires := settings.Load().Wires()
	raw := maps.Clone(settings.Load().ModelWires)
	s = settings.Load()
	s.ModelWires["antigravity/my-flash"] = "other-model"
	if err := settings.Save(s); err != nil {
		t.Fatal(err)
	}
	ctx := context.WithValue(context.Background(), wiresKey{}, func() requestWires {
		return requestWires{computed: wires, raw: raw}
	})
	p := provider.Provider{ID: "antigravity", Account: &provider.Account{Agent: "antigravity", User: "u"}}
	r := &Request{Effort: "low", Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}
	r.geminiAs = "flash-9"
	var env struct{ Model string }
	body := codeAssistBodyAt(ctx, p, r, "my-flash", wires)
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Model != "vendor/flash-9-low" {
		t.Fatalf("model %q, want vendor/flash-9-low", env.Model)
	}
}

func askArgon(t *testing.T, s *Server) int {
	t.Helper()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/gemini-4-argon","stream":true,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}`)))
	return rec.Code
}

// codeAssistXHighRetry is one Gemini CLI sign-in asked xhigh on Hydrogen,
// and the thinking levels that reached Code Assist.
func codeAssistXHighRetry(t *testing.T) []string {
	t.Helper()
	fresh(t)
	loadGeminiXHighCatalog(t)
	dir := filepath.Dir(provider.Path())
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	logins := []map[string]any{{"agent": "gemini", "user": "u@example.com", "on": true,
		"auth": map[string]any{"access_token": "tok", "refresh_token": "ref", "project": "p1", "expiry_date": time.Now().Add(time.Hour).UnixMilli()}}}
	if err := os.WriteFile(filepath.Join(dir, "logins.json"), mustJSON(logins), 0o600); err != nil {
		t.Fatal(err)
	}
	old := http.DefaultClient
	http.DefaultClient = &http.Client{Transport: countTransport(func(r *http.Request) (*http.Response, error) {
		if strings.Contains(r.URL.Path, ":loadCodeAssist") {
			return &http.Response{StatusCode: 200, Header: make(http.Header),
				Body: io.NopCloser(strings.NewReader(`{"currentTier":{"id":"standard-tier"},"cloudaicompanionProject":"p1"}`))}, nil
		}
		return nil, fmt.Errorf("unexpected request to %s", r.URL)
	})}
	t.Cleanup(func() { http.DefaultClient = old })
	var mu sync.Mutex
	var levels []string
	s := New()
	s.client = &http.Client{Transport: countTransport(func(r *http.Request) (*http.Response, error) {
		b, _ := io.ReadAll(r.Body)
		if !strings.Contains(r.URL.Path, "GenerateContent") {
			return nil, fmt.Errorf("unexpected generate request to %s", r.URL)
		}
		lv := gjson.GetBytes(b, "request.generationConfig.thinkingConfig.thinkingLevel").String()
		mu.Lock()
		levels = append(levels, lv)
		mu.Unlock()
		rec := httptest.NewRecorder()
		if strings.EqualFold(lv, "xhigh") {
			rec.Header().Set("Content-Type", "application/json")
			rec.WriteHeader(http.StatusBadRequest)
			io.WriteString(rec, aiStudioRefusal("xhigh"))
			return rec.Result(), nil
		}
		rec.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(rec, sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1}}}`))
		return rec.Result(), nil
	})}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gemini/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	return append([]string(nil), levels...)
}

// A wire saved while a request is in flight is the next request's. The
// retry of this one still asks the model the snapshot named.
func TestWireSnapshotIgnoresAMappingAddedMidRequest(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var paths []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		lv := gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String()
		mu.Lock()
		paths = append(paths, r.URL.Path)
		n := len(paths)
		mu.Unlock()
		if n == 1 {
			if err := provider.SetUpstreamName("gem/gemini-4-hydrogen", "gemini-4-argon"); err != nil {
				t.Errorf("set wire: %v", err)
			}
		}
		if strings.EqualFold(lv, "xhigh") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, aiStudioRefusal("xhigh"))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen", "gemini-4-argon"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got := append([]string(nil), paths...)
	mu.Unlock()
	if len(got) != 2 || !strings.Contains(got[0], "gemini-4-hydrogen") || !strings.Contains(got[1], "gemini-4-hydrogen") || strings.Contains(got[1], "argon") {
		t.Fatalf("in-flight paths %v, want both hydrogen", got)
	}
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("next status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 3 || !strings.Contains(paths[2], "gemini-4-argon") {
		t.Fatalf("next request path %v, want argon", paths)
	}
}

// A plain Chat relay is not Gemini's own API, so a nested
// extra_body.google.thinking_config level is fitted there too, in either
// spelling, including one that arrived beside a budget. A model that lists
// xhigh keeps it. Another vendor's model is left as sent.
func TestChatRelayFitsNestedThinkingLevel(t *testing.T) {
	var mu sync.Mutex
	var last []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = append([]byte(nil), b...)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(up.Close)
	addr := strings.TrimPrefix(up.URL, "http://")
	for _, c := range []struct {
		name, model, body, path, want string
	}{
		{"argon snake", "gemini-4-argon", `{"extra_body":{"google":{"thinking_config":{"thinking_level":"xhigh","include_thoughts":true}}}}`, "extra_body.google.thinking_config.thinking_level", "high"},
		{"argon camel", "gemini-4-argon", `{"extra_body":{"google":{"thinkingConfig":{"thinkingLevel":"xhigh","includeThoughts":true}}}}`, "extra_body.google.thinkingConfig.thinkingLevel", "high"},
		{"argon level beside a budget", "gemini-4-argon", `{"extra_body":{"google":{"thinking_config":{"thinking_budget":2000,"thinking_level":"xhigh","include_thoughts":true}}}}`, "extra_body.google.thinking_config.thinking_level", "high"},
		{"hydrogen keeps xhigh", "gemini-4-hydrogen", `{"extra_body":{"google":{"thinking_config":{"thinking_level":"xhigh","include_thoughts":true}}}}`, "extra_body.google.thinking_config.thinking_level", "xhigh"},
		{"other vendor", "gpt-5.5", `{"extra_body":{"google":{"thinking_config":{"thinking_level":"xhigh"}}}}`, "extra_body.google.thinking_config.thinking_level", "xhigh"},
	} {
		t.Run(c.name, func(t *testing.T) {
			fresh(t)
			loadGeminiXHighCatalog(t)
			if err := provider.Save(provider.Provider{
				ID: "peta", Name: "peta", Key: "k", Chat: "http://api.peta.example.com/v1",
				Models: []string{c.model},
			}); err != nil {
				t.Fatal(err)
			}
			s := New()
			s.client = &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, network, addr)
			}}}
			body := `{"model":"peta/` + c.model + `","stream":true,"messages":[{"role":"user","content":"hi"}],` + strings.TrimPrefix(c.body, "{")
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
			if rec.Code != http.StatusOK {
				t.Fatalf("status %d %s", rec.Code, rec.Body.String())
			}
			mu.Lock()
			got := string(last)
			mu.Unlock()
			if lv := gjson.Get(got, c.path).String(); lv != c.want {
				t.Fatalf("level %q, want %q: %s", lv, c.want, got)
			}
			if c.name == "argon snake" && !gjson.Get(got, "extra_body.google.thinking_config.include_thoughts").Bool() {
				t.Fatalf("include_thoughts dropped: %s", got)
			}
			if c.name == "argon level beside a budget" && (gjson.Get(got, "extra_body.google.thinking_config.thinking_budget").Exists() || gjson.Get(got, "extra_body.google.thinking_config.thinkingBudget").Exists()) {
				t.Fatalf("budget kept: %s", got)
			}
		})
	}
}

// A literal provider "*" (gem/* → gemini-4-argon) must not chain into a
// second per-model rename (gem/gemini-4-argon → gemini-4-neon). Requesting
// Hydrogen must go to Argon at Argon's fitted high level, not Neon.
func TestLiteralWildcardTargetDoesNotChainSecondRename(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var paths []string
	var levels []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		paths = append(paths, r.URL.Path)
		levels = append(levels, gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String())
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen", "gemini-4-argon", "gemini-4-neon"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("gem/*", "gemini-4-argon"); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("gem/gemini-4-argon", "gemini-4-neon"); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 1 || !strings.Contains(paths[0], "gemini-4-argon") || strings.Contains(paths[0], "gemini-4-neon") {
		t.Fatalf("path %v, want gemini-4-argon (not neon)", paths)
	}
	if len(levels) != 1 || levels[0] != "high" {
		t.Fatalf("thinkingLevel %v, want high", levels)
	}
}

// A wrapping single-model rename (gem/gemini-4-hydrogen → peta-*) expands
// to peta-gemini-4-hydrogen, whose prefix hides the Gemini generation. It must
// not be treated as an identified model with generation 0 (which sent a
// thinkingBudget to Gemini 4); it keeps the listed model's generation and level.
func TestWrappingSingleModelRenameKeepsListedGeneration(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var paths []string
	var bodies []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		paths = append(paths, r.URL.Path)
		bodies = append(bodies, string(b))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("gem/gemini-4-hydrogen", "peta-*"); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 1 || !strings.Contains(paths[0], "peta-gemini-4-hydrogen") {
		t.Fatalf("path %v, want peta-gemini-4-hydrogen", paths)
	}
	if len(bodies) != 1 {
		t.Fatalf("bodies %d, want 1", len(bodies))
	}
	if gjson.Get(bodies[0], "generationConfig.thinkingConfig.thinkingBudget").Exists() {
		t.Fatalf("sent thinkingBudget to Gemini 4: %s", bodies[0])
	}
	if lv := gjson.Get(bodies[0], "generationConfig.thinkingConfig.thinkingLevel").String(); lv != "xhigh" {
		t.Fatalf("thinkingLevel %q, want xhigh: %s", lv, bodies[0])
	}
}

// An xhigh refusal is keyed on the upstream target model, not the local
// alias. When Hydrogen refuses xhigh and the same local id is later repointed
// to Neon, Neon receives xhigh.
func TestXHighRefusalFollowsUpstreamTarget(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	type call struct{ path, level string }
	var calls []call
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		lv := gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String()
		mu.Lock()
		calls = append(calls, call{r.URL.Path, lv})
		mu.Unlock()
		if strings.Contains(r.URL.Path, "gemini-4-hydrogen") && strings.EqualFold(lv, "xhigh") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, aiStudioRefusal("xhigh"))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"my-gem", "gemini-4-hydrogen", "gemini-4-neon"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("gem/my-gem", "gemini-4-hydrogen"); err != nil {
		t.Fatal(err)
	}
	s := New()
	ask := func() int {
		t.Helper()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
			`{"model":"gem/my-gem","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
		return rec.Code
	}
	if code := ask(); code != http.StatusOK {
		t.Fatalf("hydrogen request status %d", code)
	}
	if err := provider.SetUpstreamName("gem/my-gem", "gemini-4-neon"); err != nil {
		t.Fatal(err)
	}
	if code := ask(); code != http.StatusOK {
		t.Fatalf("neon request status %d", code)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(calls) != 3 {
		t.Fatalf("calls %v, want 3 (hydrogen xhigh, hydrogen high, neon xhigh)", calls)
	}
	if !strings.Contains(calls[0].path, "gemini-4-hydrogen") || calls[0].level != "xhigh" ||
		!strings.Contains(calls[1].path, "gemini-4-hydrogen") || calls[1].level != "high" ||
		!strings.Contains(calls[2].path, "gemini-4-neon") || calls[2].level != "xhigh" {
		t.Fatalf("calls %v, want hydrogen xhigh→high then neon xhigh", calls)
	}
}

// A 422 saying "Thinking level XHIGH is not supported" on Gemini's
// OpenAI-compatible API must retry at high and remember the level refusal
// without marking thinking_config unfit for the provider.
func TestGeminiCompat422XHighRefusalKeepsThinkingConfig(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var bodies []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(b))
		mu.Unlock()
		lv := gjson.GetBytes(b, "extra_body.google.thinking_config.thinking_level").String()
		if strings.EqualFold(lv, "xhigh") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnprocessableEntity)
			io.WriteString(w, aiStudioRefusal("xhigh"))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "chat",
		Chat:   up.URL + "/v1beta/openai",
		Models: []string{"gemini-4-hydrogen", "gemini-4-argon"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(
		`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning":{"effort":"xhigh"},"input":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("hydrogen 422 retry status %d %s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(
		`{"model":"gem/gemini-4-argon","stream":true,"reasoning":{"effort":"high"},"input":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("argon status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 3 {
		t.Fatalf("calls %d, want 3 (hydrogen xhigh, hydrogen high, argon high)", len(bodies))
	}
	if lv := gjson.Get(bodies[0], "extra_body.google.thinking_config.thinking_level").String(); lv != "xhigh" {
		t.Fatalf("first body %s, want xhigh", bodies[0])
	}
	if lv := gjson.Get(bodies[1], "extra_body.google.thinking_config.thinking_level").String(); lv != "high" {
		t.Fatalf("retry body %s, want thinking_config high", bodies[1])
	}
	if lv := gjson.Get(bodies[2], "extra_body.google.thinking_config.thinking_level").String(); lv != "high" {
		t.Fatalf("argon body %s, want thinking_config preserved", bodies[2])
	}
}

// Fitting top-level reasoning_effort on a non-Gemini Chat model must
// leave nested extra_body.google.thinking_config untouched.
func TestNonGeminiChatFitPreservesExtraBodyThinkingConfig(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	var last []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = append([]byte(nil), b...)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(up.Close)
	addr := strings.TrimPrefix(up.URL, "http://")
	if err := provider.Save(provider.Provider{
		ID: "peta", Name: "peta", Key: "k", Chat: "http://api.peta.example.com/v1",
		Models: []string{"gpt-5.5"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("peta/gpt-5.5", []string{"low", "medium", "high"}); err != nil {
		t.Fatal(err)
	}
	s := New()
	s.client = &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, addr)
	}}}
	body := `{"model":"peta/gpt-5.5","stream":true,"reasoning_effort":"xhigh","extra_body":{"google":{"thinking_config":{"thinking_level":"xhigh"}}},"messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got := string(last)
	mu.Unlock()
	if re := gjson.Get(got, "reasoning_effort").String(); re != "high" {
		t.Fatalf("reasoning_effort %q, want high: %s", re, got)
	}
	if nested := gjson.Get(got, "extra_body.google.thinking_config.thinking_level").String(); nested != "xhigh" {
		t.Fatalf("extra_body.google.thinking_config.thinking_level %q, want xhigh preserved: %s", nested, got)
	}
	if rt := lastRoute(s); tryEffort(rt) != "high" {
		t.Fatalf("logged wire effort %q, want high (not nested xhigh)", tryEffort(rt))
	}
}

// Rebinding a local alias with SetModelID while a request is in flight
// must not redirect that request's xhigh-refusal retry to the new model.
func TestSetModelIDChangeMidRequestDoesNotRedirectRetry(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	type call struct{ path, level string }
	var calls []call
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		lv := gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String()
		mu.Lock()
		calls = append(calls, call{r.URL.Path, lv})
		n := len(calls)
		mu.Unlock()
		if n == 1 {
			if _, _, err := provider.SetModelID("gem/gemini-4-hydrogen", "my-gem"); err != nil {
				t.Errorf("SetModelID mid-request: %v", err)
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, aiStudioRefusal("xhigh"))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-neon", "gemini-4-hydrogen"},
	}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := provider.SetModelID("gem/gemini-4-neon", "my-gem"); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/my-gem","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got := append([]call(nil), calls...)
	mu.Unlock()
	if len(got) != 2 ||
		!strings.Contains(got[0].path, "gemini-4-neon") || got[0].level != "xhigh" ||
		!strings.Contains(got[1].path, "gemini-4-neon") || got[1].level != "high" {
		t.Fatalf("in-flight calls %v, want neon xhigh then neon high", got)
	}
	// Next request sees the new SetModelID binding (Hydrogen) and sends xhigh.
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/my-gem","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("next status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(calls) != 3 || !strings.Contains(calls[2].path, "gemini-4-hydrogen") || calls[2].level != "xhigh" {
		t.Fatalf("next call %v, want hydrogen xhigh", calls)
	}
}

// When a local alias for Hydrogen has an explicit opaque SetUpstreamName
// (ep-hydrogen) and Hydrogen itself has a separate rule to Argon, the alias's
// own rule wins: the wire goes to ep-hydrogen and capabilities stay on Hydrogen
// (xhigh), not Argon (high).
func TestOpaqueAliasRuleDoesNotInheritVendorRenameCapabilities(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var paths []string
	var levels []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		paths = append(paths, r.URL.Path)
		levels = append(levels, gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String())
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen", "gemini-4-argon"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("gem/gemini-4-hydrogen", "gemini-4-argon"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := provider.SetModelID("gem/gemini-4-hydrogen", "my-h"); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("gem/my-h", "ep-hydrogen"); err != nil {
		t.Fatal(err)
	}
	s := New()
	// Translated path
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/my-h","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("translated status %d %s", rec.Code, rec.Body.String())
	}
	// Passthrough path
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1beta/models/gem/my-h:streamGenerateContent?alt=sse", strings.NewReader(
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingLevel":"xhigh"}}}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("passthrough status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 2 || !strings.Contains(paths[0], "ep-hydrogen") || !strings.Contains(paths[1], "ep-hydrogen") {
		t.Fatalf("paths %v, want both ep-hydrogen", paths)
	}
	if len(levels) != 2 || levels[0] != "xhigh" || levels[1] != "xhigh" {
		t.Fatalf("levels %v, want both xhigh (not argon high)", levels)
	}
}

// A 422 whose details echo "Thinking level XHIGH is not supported" on a
// request that actually sent high must not retry or record xhigh as refused.
func TestEchoedXHighErrorOnHighRequestDoesNotRecordRefusal(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var levels []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		lv := gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String()
		mu.Lock()
		levels = append(levels, lv)
		n := len(levels)
		mu.Unlock()
		if n == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnprocessableEntity)
			io.WriteString(w, `{"error":{"code":422,"message":"invalid content","details":[{"echo":"Thinking level XHIGH is not supported"}]}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"high","messages":[{"role":"user","content":"Thinking level XHIGH is not supported"}]}`)))
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("first status %d, want 422", rec.Code)
	}
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("second status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(levels) != 2 || levels[0] != "high" || levels[1] != "xhigh" {
		t.Fatalf("levels %v, want [high xhigh] (no spurious retry or refusal)", levels)
	}
}

// On a non-Gemini Chat upstream, a nested extra_body.google.thinking_config
// must not mask top-level reasoning_effort: none when retrying an off-refusal.
func TestNonGeminiChatOffRefusalIgnoresNestedThinkingLevel(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	var efforts []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		re := gjson.GetBytes(b, "reasoning_effort").String()
		mu.Lock()
		efforts = append(efforts, re)
		mu.Unlock()
		if re == "none" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"error":{"message":"reasoning_effort must be one of \"low\", \"medium\", \"high\""}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(up.Close)
	addr := strings.TrimPrefix(up.URL, "http://")
	if err := provider.Save(provider.Provider{
		ID: "peta", Name: "peta", Key: "k", Chat: "http://api.peta.example.com/v1",
		Models: []string{"gpt-5.5"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	s.client = &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, addr)
	}}}
	body := `{"model":"peta/gpt-5.5","stream":true,"reasoning_effort":"none","extra_body":{"google":{"thinking_config":{"thinking_level":"high"}}},"messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(efforts) != 2 || efforts[0] != "none" || efforts[1] != "low" {
		t.Fatalf("efforts %v, want [none low]", efforts)
	}
}

// On GeminiCompat, a 422 echoing an xhigh refusal in details or
// rejecting an unsent xhigh level in message must neither retry nor poison
// thinking_config for the provider.
func TestGeminiCompatUnsentOrEchoedXHigh422KeepsThinkingConfig(t *testing.T) {
	for _, tc := range []struct {
		name string
		resp string
	}{
		{"echoed in details", `{"error":{"code":422,"message":"invalid content","details":[{"echo":"Thinking level XHIGH is not supported"}]}}`},
		{"unsent level in message", `{"error":{"code":422,"message":"Thinking level XHIGH is not supported for this model."}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fresh(t)
			loadGeminiXHighCatalog(t)
			var mu sync.Mutex
			var bodies []string
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				mu.Lock()
				bodies = append(bodies, string(b))
				n := len(bodies)
				mu.Unlock()
				if n == 1 {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusUnprocessableEntity)
					io.WriteString(w, tc.resp)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, chatHi)
			}))
			t.Cleanup(up.Close)
			if err := provider.Save(provider.Provider{
				ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "chat",
				Chat:   up.URL + "/v1beta/openai",
				Models: []string{"gemini-4-hydrogen"},
			}); err != nil {
				t.Fatal(err)
			}
			s := New()
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(
				`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning":{"effort":"high"},"input":[{"role":"user","content":"hi"}]}`)))
			if rec.Code != http.StatusUnprocessableEntity {
				t.Fatalf("first status %d, want 422 (no retry)", rec.Code)
			}
			rec = httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(
				`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning":{"effort":"high"},"input":[{"role":"user","content":"hi"}]}`)))
			if rec.Code != http.StatusOK {
				t.Fatalf("second status %d %s", rec.Code, rec.Body.String())
			}
			mu.Lock()
			defer mu.Unlock()
			if len(bodies) != 2 {
				t.Fatalf("calls %d, want 2 (no retry on first)", len(bodies))
			}
			if lv := gjson.Get(bodies[1], "extra_body.google.thinking_config.thinking_level").String(); lv != "high" {
				t.Fatalf("thinking_config lost after 422: %s", bodies[1])
			}
		})
	}
}

// Native Gemini refusal matching checks the explicit outgoing
// thinkingLevel (after ExtraBody), never a thinkingBudget converted to xhigh
// or a pre-ExtraBody xhigh overwritten to high.
func TestGeminiLevelRefusalRequiresExplicitOutgoingXHigh(t *testing.T) {
	t.Run("passthrough budget 32000", func(t *testing.T) {
		fresh(t)
		loadGeminiXHighCatalog(t)
		var mu sync.Mutex
		var bodies []string
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			bodies = append(bodies, string(b))
			n := len(bodies)
			mu.Unlock()
			if n == 1 {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, aiStudioRefusal("xhigh"))
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, vertexPong)
		}))
		t.Cleanup(up.Close)
		if err := provider.Save(provider.Provider{
			ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
			Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen"},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1beta/models/gem/gemini-4-hydrogen:streamGenerateContent?alt=sse", strings.NewReader(
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":32000}}}`)))
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("budget request status %d, want 400 without retry", rec.Code)
		}
		// Subsequent explicit xhigh must NOT be downgraded to high.
		rec = httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1beta/models/gem/gemini-4-hydrogen:streamGenerateContent?alt=sse", strings.NewReader(
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingLevel":"xhigh"}}}`)))
		if rec.Code != http.StatusOK {
			t.Fatalf("explicit xhigh status %d %s", rec.Code, rec.Body.String())
		}
		mu.Lock()
		defer mu.Unlock()
		if len(bodies) != 2 {
			t.Fatalf("calls %d, want 2 (no budget retry)", len(bodies))
		}
		if lv := gjson.Get(bodies[1], "generationConfig.thinkingConfig.thinkingLevel").String(); lv != "xhigh" {
			t.Fatalf("second request thinkingLevel %q, want xhigh: %s", lv, bodies[1])
		}
	})

	t.Run("translated extra_body overrides xhigh to high", func(t *testing.T) {
		fresh(t)
		loadGeminiXHighCatalog(t)
		var mu sync.Mutex
		var levels []string
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			lv := gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String()
			mu.Lock()
			levels = append(levels, lv)
			n := len(levels)
			mu.Unlock()
			if n == 1 {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, aiStudioRefusal("xhigh"))
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, vertexPong)
		}))
		t.Cleanup(up.Close)
		if err := provider.Save(provider.Provider{
			ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
			Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen"},
			ExtraBody: provider.ExtraBody{
				"!generationConfig": json.RawMessage(`{"thinkingConfig":{"thinkingLevel":"high"}}`),
			},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
			`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("overridden request status %d, want 400 (no retry)", rec.Code)
		}
		// Remove the ExtraBody override and confirm xhigh was NOT marked refused.
		if err := provider.Save(provider.Provider{
			ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
			Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-hydrogen"},
		}); err != nil {
			t.Fatal(err)
		}
		rec = httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
			`{"model":"gem/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != http.StatusOK {
			t.Fatalf("second request status %d %s", rec.Code, rec.Body.String())
		}
		mu.Lock()
		defer mu.Unlock()
		if len(levels) != 2 || levels[0] != "high" || levels[1] != "xhigh" {
			t.Fatalf("levels %v, want [high xhigh]", levels)
		}
	})
}

// A wrapping wildcard gem/* → gemini-4-* turning an unidentified alias
// ("hydrogen") into a recognized Gemini 4 ("gemini-4-hydrogen") must shape as
// Gemini 4 (thinkingLevel: xhigh), never a 2.x thinkingBudget.
func TestWildcardExpansionIdentifyingGemini4SendsLevel(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var paths []string
	var bodies []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		paths = append(paths, r.URL.Path)
		bodies = append(bodies, string(b))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"hydrogen"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("gem/*", "gemini-4-*"); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gem/hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 1 || !strings.Contains(paths[0], "gemini-4-hydrogen") {
		t.Fatalf("paths %v, want gemini-4-hydrogen", paths)
	}
	if gjson.Get(bodies[0], "generationConfig.thinkingConfig.thinkingBudget").Exists() {
		t.Fatalf("sent thinkingBudget to Gemini 4: %s", bodies[0])
	}
	if lv := gjson.Get(bodies[0], "generationConfig.thinkingConfig.thinkingLevel").String(); lv != "xhigh" {
		t.Fatalf("thinkingLevel %q, want xhigh: %s", lv, bodies[0])
	}
}

// Non-object extra_body.google.thinking_config on a non-Gemini Chat
// body must not break decoding of top-level reasoning_effort for wire logging.
func TestChatWireEffortIgnoresMalformedNestedThinkingConfig(t *testing.T) {
	body := []byte(`{"model":"gpt-5.5","reasoning_effort":"high","extra_body":{"google":{"thinking_config":"high"}}}`)
	if e, ok := chatWireEffort(body, false); !ok || e != "high" {
		t.Fatalf("chatWireEffort = (%q, %v), want (high, true)", e, ok)
	}
}

// A JSON null error is absent. It must not hide the envelope's message,
// and a thinking-level refusal in that message is still a level refusal.
func TestNullErrorKeepsTopLevelMessage(t *testing.T) {
	body := []byte(`{"error":null,"message":"Thinking level XHIGH is not supported for this model."}`)
	msg, ok := geminiErrorMessage(body)
	if !ok || msg != "Thinking level XHIGH is not supported for this model." {
		t.Fatalf("geminiErrorMessage = (%q, %v), want the top-level message", msg, ok)
	}
	if geminiRefusedLevel(body) != "xhigh" {
		t.Fatalf("geminiRefusedLevel = %q, want xhigh", geminiRefusedLevel(body))
	}
	if refusesThinkingConfig(http.StatusBadRequest, body) {
		t.Fatal("a level refusal was treated as thinking_config")
	}
	plain := []byte(`{"error":null,"message":"thinking_config is not allowed"}`)
	if msg, ok := geminiErrorMessage(plain); !ok || msg != "thinking_config is not allowed" {
		t.Fatalf("plain message = (%q, %v)", msg, ok)
	}
	if !refusesThinkingConfig(http.StatusBadRequest, plain) {
		t.Fatal("null error hid thinking_config")
	}
	if refusesThinkingConfig(http.StatusOK, plain) {
		t.Fatal("a 200 is not a refusal")
	}
}

// OpenAI's error.param and a validation error's detail loc and msg name
// the rejected field. The message alone does not, and a local geminiCompat
// proxy still has to fall back to reasoning_effort. A level refusal in the
// message stays a level refusal even when param is extra_body, and an
// echoed request outside those fields is not a level refusal.
func TestStructuredFieldRejectionRefusesThinkingConfig(t *testing.T) {
	openai := []byte(`{"error":{"message":"Unrecognized request argument supplied","type":"invalid_request_error","param":"extra_body"}}`)
	if !refusesThinkingConfig(http.StatusBadRequest, openai) {
		t.Fatal("error.param extra_body was not a thinking_config refusal")
	}
	fastapi := []byte(`{"message":"Validation error","detail":[{"type":"extra_forbidden","loc":["body","extra_body"],"msg":"Extra inputs are not permitted"}]}`)
	if !refusesThinkingConfig(http.StatusUnprocessableEntity, fastapi) {
		t.Fatal("detail loc extra_body was not a thinking_config refusal")
	}
	nullDetail := []byte(`{"error":null,"message":"Validation error","detail":[{"type":"extra_forbidden","loc":["body","extra_body"],"msg":"Extra inputs are not permitted"}]}`)
	if msg, ok := geminiErrorMessage(nullDetail); !ok || msg != "Validation error" {
		t.Fatalf("null error message = (%q, %v), want Validation error", msg, ok)
	}
	if !refusesThinkingConfig(http.StatusBadRequest, nullDetail) {
		t.Fatal("null error hid detail loc extra_body")
	}
	level := []byte(`{"error":{"message":"Thinking level XHIGH is not supported for this model.","param":"extra_body"}}`)
	if geminiRefusedLevel(level) != "xhigh" {
		t.Fatalf("geminiRefusedLevel = %q, want xhigh from the message", geminiRefusedLevel(level))
	}
	if refusesThinkingConfig(http.StatusBadRequest, level) {
		t.Fatal("a level refusal with param extra_body was treated as thinking_config")
	}
	echo := []byte(`{"error":{"message":"invalid content","details":[{"echo":"Thinking level XHIGH is not supported"}]}}`)
	if geminiRefusedLevel(echo) != "" {
		t.Fatalf("echoed detail triggered level %q", geminiRefusedLevel(echo))
	}
	if refusesThinkingConfig(http.StatusUnprocessableEntity, echo) {
		t.Fatal("an echoed level in details was a thinking_config refusal")
	}
}

// Antigravity's effort variant stays on the wire when a wrapping "*"
// expands it. geminiAs is the canonical model and must not drop the -low.
func TestAntigravityWildcardKeepsEffortVariant(t *testing.T) {
	fresh(t)
	if err := os.MkdirAll(filepath.Dir(catalog.LivePath("antigravity")), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalog.LivePath("antigravity"), []byte(`{"models":[{"id":"hydrogen-low"},{"id":"hydrogen-high"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog.Reset()
	t.Cleanup(catalog.Reset)
	st := settings.Load()
	st.ModelWires = map[string]string{"antigravity/*": "gemini-4-*"}
	if err := settings.Save(st); err != nil {
		t.Fatal(err)
	}
	p := provider.Provider{ID: "antigravity", Account: &provider.Account{Agent: "antigravity", User: "u"}}
	r := &Request{Thinking: true, Effort: "low", Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}
	r.geminiAs = "gemini-4-hydrogen"
	body := codeAssistBody(p, r, "hydrogen", settings.Load().Wires())
	if m := gjson.GetBytes(body, "model").String(); m != "gemini-4-hydrogen-low" {
		t.Fatalf("model %q, want gemini-4-hydrogen-low", m)
	}
}

// codeAssistBody and literalProviderTarget read a saved provider "*" when
// wires is nil, the same way wildcardWireAt does. The level is Argon's,
// not Hydrogen's xhigh.
func TestNilWiresLiteralStarShapesTheTarget(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	st := settings.Load()
	st.ModelWires = map[string]string{"gem/*": "gemini-4-argon"}
	if err := settings.Save(st); err != nil {
		t.Fatal(err)
	}
	p := provider.Provider{ID: "gem", Catalog: "google", Account: &provider.Account{Agent: "gemini"}}
	if w, ok := literalProviderTarget(nil, "gem"); !ok || w != "gemini-4-argon" {
		t.Fatalf("literalProviderTarget(nil) = %q %v, want gemini-4-argon", w, ok)
	}
	if as := geminiCheckedModel(p, "gemini-4-hydrogen", nil); as != "gemini-4-argon" {
		t.Fatalf("checked %q, want gemini-4-argon", as)
	}
	r := &Request{Thinking: true, Effort: "xhigh", Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}
	r.geminiAs = "gemini-4-hydrogen"
	body := codeAssistBody(p, r, "gemini-4-hydrogen", nil)
	if m := gjson.GetBytes(body, "model").String(); m != "gemini-4-argon" {
		t.Fatalf("model %q, want gemini-4-argon: %s", m, body)
	}
	lv := codeAssistThinkingLevel(body)
	if lv != "high" {
		t.Fatalf("thinkingLevel %q, want high: %s", lv, body)
	}
}

func codeAssistThinkingLevel(body []byte) string {
	for _, p := range []string{
		"generationConfig.thinkingConfig.thinkingLevel",
		"request.generationConfig.thinkingConfig.thinkingLevel",
	} {
		if v := gjson.GetBytes(body, p).String(); v != "" {
			return v
		}
	}
	return ""
}

// A wrapping "*" that only spells an unidentified alias keeps that alias's
// SetModelEfforts and Reasonless. A rename of an identified model onto
// another uses the target, and does not borrow the listed model's xhigh.
func TestWrappingExpansionKeepsListedEfforts(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google",
		Gemini: "http://127.0.0.1:9/v1beta",
		Models: []string{"quartz", "alias", "gemini-4-hydrogen"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("gem/quartz", []string{"low", "high"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelReasonless("gem/alias", true); err != nil {
		t.Fatal(err)
	}
	p, err := provider.Find("gem")
	if err != nil {
		t.Fatal(err)
	}
	if got := geminiGateEfforts(*p, "quartz", "gemini-4-quartz"); strings.Join(got, ",") != "low,high" {
		t.Fatalf("quartz efforts %v, want low,high", got)
	}
	if fit := effortFitModel(*p, "quartz", "gemini-4-quartz"); fit != "quartz" {
		t.Fatalf("quartz fit %q, want quartz", fit)
	}
	if fit := effortFitModel(*p, "alias", "gemini-4-alias"); fit != "alias" {
		t.Fatalf("reasonless fit model %q, want alias", fit)
	}
	if got := fitFor(*p, "alias", "xhigh"); got != "high" {
		t.Fatalf("reasonless xhigh fitted to %q, want high", got)
	}
	if got := geminiGateEfforts(*p, "gemini-4-hydrogen", "gemini-4-quartz"); len(got) != 0 {
		t.Fatalf("hydrogen→quartz efforts %v, want none of hydrogen's", got)
	}
	if fit := effortFitModel(*p, "gemini-4-hydrogen", "gemini-4-argon"); fit != "gemini-4-argon" {
		t.Fatalf("hydrogen→argon fit %q, want gemini-4-argon", fit)
	}
}

// A non-object or non-string sibling is not a level, and it must not hide
// the valid spelling beside it.
func TestWireLevelIgnoresMistypedSibling(t *testing.T) {
	cases := []struct {
		name, body, want string
		gemini           bool
		fn               string
	}{
		{"request string", `{"request":"id-1","generationConfig":{"thinkingConfig":{"thinkingLevel":"high"}}}`, "high", false, "gemini"},
		{"generation_config string", `{"generation_config":"bad","generationConfig":{"thinkingConfig":{"thinkingLevel":"low"}}}`, "low", false, "gemini"},
		{"thinking_config object", `{"generationConfig":{"thinking_config":"bad","thinkingConfig":{"thinkingLevel":"medium"}}}`, "medium", false, "gemini"},
		{"thinking_level number", `{"generationConfig":{"thinkingConfig":{"thinking_level":1,"thinkingLevel":"high"}}}`, "high", false, "gemini"},
		{"reasoning_effort number", `{"model":"gemini-4-hydrogen","reasoning_effort":1,"extra_body":{"google":{"thinking_config":{"thinking_level":"high"}}}}`, "high", true, "chat"},
		{"thinking_config string beside thinkingConfig", `{"model":"gemini-4-hydrogen","reasoning_effort":1,"extra_body":{"google":{"thinking_config":"bad","thinkingConfig":{"thinkingLevel":"medium"}}}}`, "medium", true, "chat"},
		{"compat extra", `{"reasoning_effort":1,"extra_body":{"google":{"thinking_config":{"thinking_level":"low"}}}}`, "low", false, "compat"},
	}
	for _, c := range cases {
		var got string
		var ok bool
		switch c.fn {
		case "gemini":
			got, ok = geminiWireLevel([]byte(c.body))
		case "chat":
			got, ok = chatWireEffort([]byte(c.body), c.gemini)
		case "compat":
			got, ok = geminiCompatWireLevel([]byte(c.body))
		}
		if !ok || got != c.want {
			t.Errorf("%s = (%q, %v), want (%q, true)", c.name, got, ok, c.want)
		}
	}
	if _, ok := geminiWireLevel([]byte(`{"generationConfig":{"thinkingConfig":{"thinkingLevel":1}}}`)); ok {
		t.Error("a non-string thinkingLevel was a level")
	}
}

// Gemini passthrough never sends max. A custom list that includes it goes
// out as xhigh where the model has xhigh, and high where it does not.
func TestGeminiPassthroughMaxNeverReachesTheWire(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var levels []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		levels = append(levels, gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String())
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-custom", "gemini-4-plain"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("gem/gemini-4-custom", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("gem/gemini-4-plain", []string{"low", "medium", "high", "max"}); err != nil {
		t.Fatal(err)
	}
	s := New()
	ask := func(model string) int {
		t.Helper()
		rec := httptest.NewRecorder()
		body := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingLevel":"max","includeThoughts":true}}}`
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1beta/models/gem/"+model+":streamGenerateContent?alt=sse", strings.NewReader(body)))
		return rec.Code
	}
	if code := ask("gemini-4-custom"); code != http.StatusOK {
		t.Fatalf("custom status %d", code)
	}
	if code := ask("gemini-4-plain"); code != http.StatusOK {
		t.Fatalf("plain status %d", code)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(levels) != 2 || levels[0] != "xhigh" || levels[1] != "high" {
		t.Fatalf("thinkingLevel %v, want [xhigh high]", levels)
	}
	for _, lv := range levels {
		if lv == "max" {
			t.Fatal("max reached the wire")
		}
	}
}

// Custom SetModelEfforts containing max (or xhigh on Gemini 3) must never
// leak max onto Gemini, Vertex, Code Assist, GeminiCompat (including after
// thinking_config fallback), or Chat/Anthropic relays, and an xhigh refusal
// must downgrade subsequent max requests to high on both translated and
// passthrough routes.
func TestGeminiCustomMaxNeverReachesWireAcrossPaths(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k", Catalog: "google",
		Chat:   "http://api.relay.example.com/v1",
		Models: []string{"gemini-4-custom", "gemini-4-plain", "gemini-3-custom"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/gemini-4-custom", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/gemini-4-plain", []string{"low", "medium", "high", "max"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/gemini-3-custom", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	p, err := provider.Find("relay")
	if err != nil {
		t.Fatal(err)
	}

	// geminiLevelBody (Chat & Anthropic relays) and withGeminiEffortFor
	for _, c := range []struct {
		model, want string
	}{
		{"gemini-4-custom", "xhigh"},
		{"gemini-4-plain", "high"},
		{"gemini-3-custom", "high"},
	} {
		chatIn := []byte(`{"model":"` + c.model + `","reasoning_effort":"max","extra_body":{"google":{"thinking_config":{"thinking_level":"max"}}}}`)
		chatOut := geminiLevelBody(*p, provider.Chat, c.model, c.model, chatIn)
		if re := gjson.GetBytes(chatOut, "reasoning_effort").String(); re != c.want {
			t.Errorf("geminiLevelBody chat %s reasoning_effort = %q, want %q", c.model, re, c.want)
		}
		if lv := gjson.GetBytes(chatOut, "extra_body.google.thinking_config.thinking_level").String(); lv != c.want {
			t.Errorf("geminiLevelBody chat %s thinking_level = %q, want %q", c.model, lv, c.want)
		}
		anthIn := []byte(`{"model":"` + c.model + `","thinking":{"type":"adaptive"},"output_config":{"effort":"max"}}`)
		anthOut := geminiLevelBody(*p, provider.Anthropic, c.model, c.model, anthIn)
		if ef := gjson.GetBytes(anthOut, "output_config.effort").String(); ef != c.want {
			t.Errorf("geminiLevelBody anthropic %s effort = %q, want %q", c.model, ef, c.want)
		}
		gemOut := withGeminiEffortFor([]byte(`{"contents":[]}`), c.model, "max", p.Efforts(c.model))
		if lv := gjson.GetBytes(gemOut, "generationConfig.thinkingConfig.thinkingLevel").String(); lv != c.want {
			t.Errorf("withGeminiEffortFor %s = %q, want %q", c.model, lv, c.want)
		}
	}

	// vertexEffort strips max/ultra from custom lists
	if got := vertexEffort(&Request{Effort: "max"}, []string{"low", "high", "xhigh", "max"}).Effort; got != "xhigh" {
		t.Fatalf("vertexEffort max with xhigh = %q, want xhigh", got)
	}
	if got := vertexEffort(&Request{Effort: "max"}, []string{"low", "high", "max"}).Effort; got != "high" {
		t.Fatalf("vertexEffort max without xhigh = %q, want high", got)
	}
	if got := vertexEffort(&Request{Effort: "ultra"}, []string{"low", "high", "max", "ultra"}).Effort; got != "high" {
		t.Fatalf("vertexEffort ultra without xhigh = %q, want high", got)
	}

	// Translated Gemini + Passthrough Gemini after xhigh refusal: max clamps to xhigh,
	// retries at high on xhigh refusal, and remembers high for subsequent max calls.
	var mu sync.Mutex
	var levels []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		lv := gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String()
		mu.Lock()
		levels = append(levels, lv)
		mu.Unlock()
		if strings.EqualFold(lv, "xhigh") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, aiStudioRefusal("xhigh"))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gemmax", Name: "GemMax", Key: "k", Catalog: "google", BaseAPI: "gemini",
		Gemini: up.URL + "/v1beta", Models: []string{"gemini-4-custom"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("gemmax/gemini-4-custom", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gemmax/gemini-4-custom","stream":true,"reasoning_effort":"max","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("translated max status %d %s", rec.Code, rec.Body.String())
	}
	// Second translated max call must go straight to high (remembered refusal).
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"gemmax/gemini-4-custom","stream":true,"reasoning_effort":"max","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("second translated max status %d %s", rec.Code, rec.Body.String())
	}
	// Passthrough max call on the same provider/model must also observe the remembered xhigh refusal.
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1beta/models/gemmax/gemini-4-custom:streamGenerateContent?alt=sse", strings.NewReader(
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingLevel":"max","includeThoughts":true}}}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("passthrough max status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	gotLevels := append([]string(nil), levels...)
	mu.Unlock()
	if len(gotLevels) != 4 || gotLevels[0] != "xhigh" || gotLevels[1] != "high" || gotLevels[2] != "high" || gotLevels[3] != "high" {
		t.Fatalf("gemmax levels = %v, want [xhigh high high high]", gotLevels)
	}

	// GeminiCompat local proxy: when thinking_config is refused and falls back
	// to reasoning_effort, custom max still clamps to high (not max).
	var compatMu sync.Mutex
	var compatBodies []string
	compatUp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		compatMu.Lock()
		compatBodies = append(compatBodies, string(b))
		compatMu.Unlock()
		if gjson.GetBytes(b, "extra_body").Exists() {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"error":{"message":"Unknown name \"extra_body\""}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(compatUp.Close)
	if err := provider.Save(provider.Provider{
		ID: "locgem", Name: "LocGem", Key: "k", Catalog: "google", BaseAPI: "chat",
		Chat: compatUp.URL + "/v1", Models: []string{"gemini-4-plain"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("locgem/gemini-4-plain", []string{"low", "medium", "high", "max"}); err != nil {
		t.Fatal(err)
	}
	s2 := New()
	for range 2 {
		rec := httptest.NewRecorder()
		s2.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
			`{"model":"locgem/gemini-4-plain","max_tokens":4096,"stream":true,"thinking":{"type":"adaptive"},"output_config":{"effort":"max"},"messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != http.StatusOK {
			t.Fatalf("locgem status %d %s", rec.Code, rec.Body.String())
		}
	}
	compatMu.Lock()
	defer compatMu.Unlock()
	if len(compatBodies) != 3 {
		t.Fatalf("locgem calls = %d, want 3 (extra_body refused, fallback retry, remembered fallback)", len(compatBodies))
	}
	if lv := gjson.Get(compatBodies[0], "extra_body.google.thinking_config.thinking_level").String(); lv != "high" {
		t.Fatalf("first locgem thinking_level = %q, want high", lv)
	}
	for i := 1; i < len(compatBodies); i++ {
		if re := gjson.Get(compatBodies[i], "reasoning_effort").String(); re != "high" {
			t.Fatalf("fallback locgem[%d] reasoning_effort = %q, want high: %s", i, re, compatBodies[i])
		}
	}
}

// FastAPI detail-only 422s (no top-level message) and message-less error
// objects must never fall back to scanning echoed request inputs, while real
// extra_body rejections in detail[].loc and string detail messages still match.
func TestFastAPIDetailOnlyAndStringDetailErrors(t *testing.T) {
	echoedFastAPI := []byte(`{"detail":[{"type":"value_error","loc":["body","messages",0],"msg":"bad content","input":{"extra_body":{"google":{"thinking_config":{"thinking_level":"xhigh"}}},"text":"Thinking level XHIGH is not supported"}}]}`)
	if msg, ok := geminiErrorMessage(echoedFastAPI); !ok || msg != "" {
		t.Fatalf("echoedFastAPI geminiErrorMessage = (%q, %v), want (\"\", true)", msg, ok)
	}
	if lv := geminiRefusedLevel(echoedFastAPI); lv != "" {
		t.Fatalf("echoedFastAPI geminiRefusedLevel = %q, want empty", lv)
	}
	if refusesThinkingConfig(http.StatusUnprocessableEntity, echoedFastAPI) {
		t.Fatal("echoedFastAPI input triggered refusesThinkingConfig")
	}

	realFastAPI := []byte(`{"detail":[{"type":"extra_forbidden","loc":["body","extra_body"],"msg":"Extra inputs are not permitted"}]}`)
	if !refusesThinkingConfig(http.StatusUnprocessableEntity, realFastAPI) {
		t.Fatal("realFastAPI detail[].loc extra_body did not trigger refusesThinkingConfig")
	}
	if lv := geminiRefusedLevel(realFastAPI); lv != "" {
		t.Fatalf("realFastAPI geminiRefusedLevel = %q, want empty", lv)
	}

	messagelessErr := []byte(`{"error":{"code":422,"details":[{"echo":"Thinking level XHIGH is not supported","field":"extra_body"}]}}`)
	if msg, ok := geminiErrorMessage(messagelessErr); !ok || msg != "" {
		t.Fatalf("messagelessErr geminiErrorMessage = (%q, %v), want (\"\", true)", msg, ok)
	}
	if lv := geminiRefusedLevel(messagelessErr); lv != "" {
		t.Fatalf("messagelessErr geminiRefusedLevel = %q, want empty", lv)
	}
	if refusesThinkingConfig(http.StatusUnprocessableEntity, messagelessErr) {
		t.Fatal("messagelessErr details triggered refusesThinkingConfig")
	}

	strDetailLevel := []byte(`{"detail":"Thinking level XHIGH is not supported for this model."}`)
	if msg, ok := geminiErrorMessage(strDetailLevel); !ok || !strings.Contains(msg, "XHIGH") {
		t.Fatalf("strDetailLevel geminiErrorMessage = (%q, %v)", msg, ok)
	}
	if lv := geminiRefusedLevel(strDetailLevel); lv != "xhigh" {
		t.Fatalf("strDetailLevel geminiRefusedLevel = %q, want xhigh", lv)
	}
	if refusesThinkingConfig(http.StatusBadRequest, strDetailLevel) {
		t.Fatal("strDetailLevel level refusal triggered refusesThinkingConfig")
	}

	msgAndStrDetail := []byte(`{"message":"Bad request","detail":"extra_body is not allowed"}`)
	if !refusesThinkingConfig(http.StatusBadRequest, msgAndStrDetail) {
		t.Fatal("string detail extra_body beside message did not trigger refusesThinkingConfig")
	}
}

// A wrapping "*" on Antigravity (antigravity/* -> gemini-4-* or gemini-3-*)
// that expands an unidentified variant id (hydrogen-low, flash-minimal,
// flash-9-xhigh) into a recognized Gemini id must preserve that variant's
// level for thinkingConfig, while a literal target (antigravity/* ->
// gemini-4-argon) must NOT inherit the source variant's level.
func TestAntigravityWrappingStarKeepsVariantLevel(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	if err := os.MkdirAll(filepath.Dir(catalog.LivePath("antigravity")), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalog.LivePath("antigravity"), []byte(`{"models":[{"id":"hydrogen-low"},{"id":"hydrogen-high"},{"id":"flash-minimal"},{"id":"flash-high"},{"id":"flash-9-low"},{"id":"flash-9-xhigh"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog.Reset()
	t.Cleanup(catalog.Reset)

	p := provider.Provider{ID: "antigravity", Account: &provider.Account{Agent: "antigravity", User: "u"}}
	cases := []struct {
		name, star, model, effort, wantModel, wantLevel string
	}{
		{"direct hydrogen-low pick", "gemini-4-*", "hydrogen-low", "", "gemini-4-hydrogen-low", "low"},
		{"family hydrogen with low effort", "gemini-4-*", "hydrogen", "low", "gemini-4-hydrogen-low", "low"},
		{"direct flash-minimal pick", "gemini-3-*", "flash-minimal", "", "gemini-3-flash-minimal", "minimal"},
		{"direct flash-9-xhigh pick", "gemini-4-*", "flash-9-xhigh", "", "gemini-4-flash-9-xhigh", "xhigh"},
		{"literal target does not inherit -low", "gemini-4-argon", "hydrogen-low", "high", "gemini-4-argon", "high"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			st := settings.Load()
			st.ModelWires = map[string]string{"antigravity/*": c.star}
			if err := settings.Save(st); err != nil {
				t.Fatal(err)
			}
			r := &Request{Thinking: true, Effort: c.effort, Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}
			body := codeAssistBody(p, r, c.model, settings.Load().Wires())
			if gotModel := gjson.GetBytes(body, "model").String(); gotModel != c.wantModel {
				t.Errorf("model = %q, want %q (body: %s)", gotModel, c.wantModel, body)
			}
			if gotLevel := codeAssistThinkingLevel(body); gotLevel != c.wantLevel {
				t.Errorf("thinkingLevel = %q, want %q (body: %s)", gotLevel, c.wantLevel, body)
			}
		})
	}
}
