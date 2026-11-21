package gateway

import (
	"encoding/json"
	"io"
	"maps"
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
	"github.com/yetone/magpie/internal/usage"
)

// geminiXHighCatalog is models.dev's google list for the models #7400
// names: Hydrogen and Neon take xhigh, Argon and Gemini 3 stop at high,
// and 2.5 is still a budget.
const geminiXHighCatalog = `{"google":{"models":{
	"gemini-4-hydrogen":{"id":"gemini-4-hydrogen","reasoning_options":[{"type":"effort","values":["low","medium","high","xhigh"]}]},
	"gemini-4-neon":{"id":"gemini-4-neon","reasoning_options":[{"type":"effort","values":["low","medium","high","xhigh"]}]},
	"gemini-4-argon":{"id":"gemini-4-argon","reasoning_options":[{"type":"effort","values":["low","medium","high"]}]},
	"gemini-3.8-flash":{"id":"gemini-3.8-flash","reasoning_options":[{"type":"effort","values":["low","medium","high"]}]},
	"gemini-3-flash-min":{"id":"gemini-3-flash-min","reasoning_options":[{"type":"effort","values":["minimal","low","medium","high"]}]},
	"gemini-3-pro-preview":{"id":"gemini-3-pro-preview","reasoning_options":[{"type":"effort","values":["low","medium","high"]}]},
	"gemini-2.5-flash":{"id":"gemini-2.5-flash","reasoning_options":[{"type":"effort","values":["low","medium","high"]}]}}}}`

// #7400: a Gemini 4 model whose levels include xhigh, asked xhigh on a
// non-Vertex Gemini route, is sent thinkingLevel xhigh. A Gemini 3, and a
// Gemini 4 whose levels stop at high, stay at high. Gemini 3 Pro's medium
// is still sent as high, and the try says that level rather than the one
// asked. 2.5 stays a budget, logged as asked.
func TestGeminiXHighKeptOnNativeRoutes(t *testing.T) {
	fresh(t)
	if err := os.MkdirAll(filepath.Dir(catalog.CachePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalog.CachePath(), []byte(geminiXHighCatalog), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog.Reset()
	t.Cleanup(catalog.Reset)

	var mu sync.Mutex
	var last struct {
		path string
		tc   map[string]any
	}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var body map[string]any
		json.Unmarshal(b, &body)
		gen, _ := body["generationConfig"].(map[string]any)
		tc, _ := gen["thinkingConfig"].(map[string]any)
		mu.Lock()
		last.path, last.tc = r.URL.Path, tc
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)

	models := []string{"gemini-4-hydrogen", "gemini-4-neon", "gemini-4-argon", "gemini-3.8-flash", "gemini-3-pro-preview", "gemini-2.5-flash"}
	level := func(l string) map[string]any {
		return map[string]any{"thinkingLevel": l, "includeThoughts": true}
	}
	budget := map[string]any{"thinkingBudget": 24000.0, "includeThoughts": true}
	type ask struct {
		path, model, effort, went string
		want                      map[string]any
	}
	asks := []ask{
		{"/v1/chat/completions", "gemini-4-hydrogen", "xhigh", "xhigh", level("xhigh")},
		{"/v1/responses", "gemini-4-hydrogen", "xhigh", "xhigh", level("xhigh")},
		{"/v1/chat/completions", "gemini-4-neon", "xhigh", "xhigh", level("xhigh")},
		{"/v1/chat/completions", "gemini-4-argon", "xhigh", "high", level("high")},
		{"/v1/chat/completions", "gemini-3.8-flash", "xhigh", "high", level("high")},
		{"/v1/chat/completions", "gemini-3-pro-preview", "xhigh", "high", level("high")},
		{"/v1/chat/completions", "gemini-3-pro-preview", "medium", "high", level("high")},
		{"/v1/chat/completions", "gemini-2.5-flash", "high", "high", budget},
	}

	for _, p := range []provider.Provider{
		{ID: "custom", Name: "Custom", Key: "k", BaseAPI: "gemini", Catalog: "google", Gemini: up.URL + "/v1beta", Models: models},
		{ID: "google", Name: "Google Gemini", Preset: "google", Key: "k", Catalog: "google", Chat: up.URL + "/v1", Gemini: up.URL + "/v1beta", Models: models},
	} {
		if err := provider.Save(p); err != nil {
			t.Fatal(err)
		}
		if p.ID == "google" {
			for _, m := range models {
				if err := provider.SetModelAPI(p.ID+"/"+m, "gemini"); err != nil {
					t.Fatal(err)
				}
			}
		}
		s := New()
		for _, c := range asks {
			body := `{"model":"` + p.ID + `/` + c.model + `","stream":true,"reasoning_effort":"` + c.effort + `","messages":[{"role":"user","content":"hi"}]}`
			if c.path == "/v1/responses" {
				body = `{"model":"` + p.ID + `/` + c.model + `","stream":true,"reasoning":{"effort":"` + c.effort + `"},"input":"hi"}`
			}
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", c.path, strings.NewReader(body)))
			name := p.ID + " " + c.model + " " + c.effort
			if rec.Code != http.StatusOK {
				t.Errorf("%s: status %d: %s", name, rec.Code, rec.Body.String())
				continue
			}
			mu.Lock()
			got, path := last.tc, last.path
			mu.Unlock()
			if !strings.Contains(path, ":streamGenerateContent") {
				t.Errorf("%s: sent to %s, want Gemini generateContent", name, path)
			}
			if !maps.Equal(got, c.want) {
				t.Errorf("%s: thinkingConfig %v, want %v", name, got, c.want)
			}
			rt := lastRoute(s)
			if rt.Effort != c.effort || len(rt.Tries) != 1 || rt.Tries[0].Effort != c.went {
				t.Errorf("%s: route asked %q, try %q; want asked %q, went %q", name, rt.Effort, tryEffort(rt), c.effort, c.went)
			}
			recs := usage.Load(time.Time{})
			if len(recs) == 0 || recs[len(recs)-1].Effort != c.went {
				t.Errorf("%s: logged %q, want %q", name, loggedEffort(recs), c.went)
			}
		}
	}
}

func tryEffort(rt Route) string {
	if len(rt.Tries) == 0 {
		return ""
	}
	return rt.Tries[0].Effort
}

func loggedEffort(recs []usage.Record) string {
	if len(recs) == 0 {
		return ""
	}
	return recs[len(recs)-1].Effort
}

// Bare-id Code Assist (Gemini CLI, not an Antigravity variant) sends the
// same thinkingLevel: xhigh on a Gemini 4 that lists it, high on a Gemini
// 3 and on a Gemini 4 that does not. The envelope's thinkingLevel is what
// the log reads.
func TestCodeAssistBareIDKeepsXHigh(t *testing.T) {
	fresh(t)
	if err := os.MkdirAll(filepath.Dir(catalog.CachePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalog.CachePath(), []byte(geminiXHighCatalog), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog.Reset()
	t.Cleanup(catalog.Reset)
	p := provider.Provider{ID: "gemini", Catalog: "google"}
	for _, c := range []struct{ model, effort, want string }{
		{"gemini-4-hydrogen", "xhigh", "xhigh"},
		{"gemini-4-neon", "xhigh", "xhigh"},
		{"gemini-4-argon", "xhigh", "high"},
		{"gemini-3.8-flash", "xhigh", "high"},
		{"gemini-3-pro-preview", "medium", "high"},
		{"gemini-3-pro-preview", "xhigh", "high"},
	} {
		r := &Request{Thinking: true, Effort: fitFor(p, c.model, c.effort),
			Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}
		body := buildCodeAssist(r, c.model, "gemini")
		if got := gjson.GetBytes(body, "request.generationConfig.thinkingConfig.thinkingLevel").String(); got != c.want {
			t.Errorf("%s at %s: thinkingLevel %q, want %q", c.model, c.effort, got, c.want)
		}
		ctx := withWireAt(t.Context(), &wireAt{})
		noteWireEffort(ctx, provider.CodeAssist, r, body)
		w, _ := ctx.Value(wireAtKey{}).(*wireAt)
		if w == nil || !w.set || w.effort != c.want {
			t.Errorf("%s at %s: log %q set %v, want %q", c.model, c.effort, wireEffortOf(w), w != nil && w.set, c.want)
		}
		// the same level on Gemini's own body, not only the envelope
		own, err := buildGemini(r, c.model)
		if err != nil {
			t.Fatal(err)
		}
		if got := gjson.GetBytes(own, "generationConfig.thinkingConfig.thinkingLevel").String(); got != c.want {
			t.Errorf("%s at %s: Gemini thinkingLevel %q, want %q", c.model, c.effort, got, c.want)
		}
		ctx = withWireAt(t.Context(), &wireAt{})
		noteWireEffort(ctx, provider.Gemini, r, own)
		w, _ = ctx.Value(wireAtKey{}).(*wireAt)
		if w == nil || !w.set || w.effort != c.want {
			t.Errorf("%s at %s: Gemini log %q set %v, want %q", c.model, c.effort, wireEffortOf(w), w != nil && w.set, c.want)
		}
	}
	// Gemini 3 stays at high even when the effort was not fitted first
	raw := &Request{Thinking: true, Effort: "xhigh", Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}
	if got := gjson.GetBytes(buildCodeAssist(raw, "gemini-3.8-flash", "gemini"), "request.generationConfig.thinkingConfig.thinkingLevel").String(); got != "high" {
		t.Errorf("gemini 3 asked xhigh with no fitting: thinkingLevel %q, want high", got)
	}
	// a budget, and a request that asks for no thinking, leave the log as
	// asked: there is no thinkingLevel to report
	for _, c := range []struct {
		name, model string
		req         *Request
	}{
		{"budget", "gemini-2.5-flash", &Request{Thinking: true, Effort: "high", Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}},
		{"none", "gemini-4-hydrogen", &Request{Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}},
	} {
		body := buildCodeAssist(c.req, c.model, "gemini")
		ctx := withWireAt(t.Context(), &wireAt{})
		noteWireEffort(ctx, provider.CodeAssist, c.req, body)
		w, _ := ctx.Value(wireAtKey{}).(*wireAt)
		if w == nil || w.set {
			t.Errorf("%s: log set to %q, want it left as asked", c.name, wireEffortOf(w))
		}
		own, err := buildGemini(c.req, c.model)
		if err != nil {
			t.Fatal(err)
		}
		ctx = withWireAt(t.Context(), &wireAt{})
		noteWireEffort(ctx, provider.Gemini, c.req, own)
		w, _ = ctx.Value(wireAtKey{}).(*wireAt)
		if w == nil || w.set {
			t.Errorf("%s Gemini: log set to %q, want it left as asked", c.name, wireEffortOf(w))
		}
	}
}

func wireEffortOf(w *wireAt) string {
	if w == nil {
		return ""
	}
	return w.effort
}

func loadGeminiXHighCatalog(t *testing.T) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(catalog.CachePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalog.CachePath(), []byte(geminiXHighCatalog), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog.Reset()
	t.Cleanup(catalog.Reset)
}

// The default Google preset speaks AI Studio's OpenAI-compatible API, with
// no per-model API set. xhigh is sent when the model lists it, and a 400
// that names xhigh retries at high for that model without marking
// thinking_config unfit for the provider.
func TestGeminiCompatPresetKeepsXHighAndRefitsARefusal(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var bodies []map[string]any
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		bodies = append(bodies, b)
		level := geminiLevel(b)
		mu.Unlock()
		if level == "xhigh" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, aiStudioRefusal("xhigh"))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, sse(
			`data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
			`data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}`,
			`data: [DONE]`,
		))
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "google", Name: "Google Gemini", Preset: "google", Key: "k", Catalog: "google",
		Chat: up.URL + "/v1", Models: []string{"gemini-4-hydrogen", "gemini-3.8-flash", "gemini-4-argon"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	// Messages is translated onto the preset's Chat API, which is where
	// GeminiCompat's thinking_config is built. A Chat client is relayed.
	ask := func(model, effort string) *httptest.ResponseRecorder {
		t.Helper()
		rec := httptest.NewRecorder()
		body := `{"model":"google/` + model + `","max_tokens":32000,"stream":true,"thinking":{"type":"adaptive"},"output_config":{"effort":"` + effort + `"},"messages":[{"role":"user","content":"hi"}]}`
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		return rec
	}
	rec := ask("gemini-4-hydrogen", "xhigh")
	if rec.Code != http.StatusOK {
		t.Fatalf("hydrogen: %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got := slicesCloneMaps(bodies)
	mu.Unlock()
	if len(got) != 2 || geminiLevel(got[0]) != "xhigh" || geminiLevel(got[1]) != "high" {
		t.Fatalf("refusal retry sent %d bodies, levels %q then %q", len(got), geminiLevel(atMap(got, 0)), geminiLevel(atMap(got, 1)))
	}
	if geminiThinkingConfig(got[0])["include_thoughts"] != true || geminiThinkingConfig(got[1])["include_thoughts"] != true {
		t.Fatalf("thinking_config dropped: %v then %v", got[0], got[1])
	}
	rt := lastRoute(s)
	if len(rt.Tries) != 1 || rt.Tries[0].Effort != "high" {
		t.Fatalf("try effort %q, want high (%d tries)", tryEffort(rt), len(rt.Tries))
	}
	recs := usage.Load(time.Time{})
	if loggedEffort(recs) != "high" {
		t.Fatalf("logged %q, want high", loggedEffort(recs))
	}
	// a later model still gets thinking_config: the xhigh refusal did not
	// mark the field unfit for the provider
	before := len(got)
	rec = ask("gemini-3.8-flash", "high")
	if rec.Code != http.StatusOK {
		t.Fatalf("flash: %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got = slicesCloneMaps(bodies)
	mu.Unlock()
	if len(got) != before+1 || geminiThinkingConfig(got[len(got)-1])["thinking_level"] != "high" {
		t.Fatalf("flash after the refusal: %v", got[len(got)-1])
	}
	// argon never lists xhigh, so it is asked at high once
	rec = ask("gemini-4-argon", "xhigh")
	if rec.Code != http.StatusOK {
		t.Fatalf("argon: %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	last := bodies[len(bodies)-1]
	mu.Unlock()
	if geminiLevel(last) != "high" || len(bodies) != before+2 {
		t.Fatalf("argon xhigh went as %q after %d requests", geminiLevel(last), len(bodies)-before)
	}
	if rt = lastRoute(s); tryEffort(rt) != "high" {
		t.Fatalf("argon try %q, want high", tryEffort(rt))
	}
}

func atMap(ms []map[string]any, i int) map[string]any {
	if i < 0 || i >= len(ms) {
		return nil
	}
	return ms[i]
}

func slicesCloneMaps(ms []map[string]any) []map[string]any {
	out := make([]map[string]any, len(ms))
	copy(out, ms)
	return out
}

// A native Gemini client is relayed on provider.Gemini. xhigh is rewritten
// to high on Argon and Gemini 3.8 Flash, kept on Hydrogen, and minimal stays
// minimal on a Flash that lists it. The try says the level that went out.
func TestGeminiPassthroughClampsThinkingLevel(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var last []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = append([]byte(nil), b...)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-4-hydrogen", "gemini-4-argon", "gemini-3.8-flash", "gemini-3-flash-min"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	ask := func(model, level string) {
		t.Helper()
		body := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"temperature":0.2,"thinkingConfig":{"thinkingLevel":"` + level + `","includeThoughts":true}}}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1beta/models/gem/"+model+":streamGenerateContent?alt=sse", strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s %s: %d %s", model, level, rec.Code, rec.Body.String())
		}
	}
	bodyOf := func() []byte {
		t.Helper()
		mu.Lock()
		defer mu.Unlock()
		return append([]byte(nil), last...)
	}
	levelOf := func() string {
		t.Helper()
		return gjson.GetBytes(bodyOf(), "generationConfig.thinkingConfig.thinkingLevel").String()
	}
	ask("gemini-4-argon", "xhigh")
	if got := levelOf(); got != "high" {
		t.Fatalf("argon xhigh went as %q", got)
	}
	if rt := lastRoute(s); tryEffort(rt) != "high" {
		t.Fatalf("argon try %q, want high", tryEffort(rt))
	}
	ask("gemini-4-argon", "XHIGH")
	if b := bodyOf(); gjson.GetBytes(b, "generationConfig.thinkingConfig.thinkingLevel").String() != "high" || !gjson.GetBytes(b, "generationConfig.thinkingConfig.includeThoughts").Bool() || gjson.GetBytes(b, "generationConfig.temperature").Float() != 0.2 {
		t.Fatalf("XHIGH rewrite lost the rest of the config: %s", b)
	}
	ask("gemini-3.8-flash", "xhigh")
	if got := levelOf(); got != "high" {
		t.Fatalf("3.8 flash xhigh went as %q", got)
	}
	ask("gemini-4-hydrogen", "xhigh")
	if got := levelOf(); got != "xhigh" {
		t.Fatalf("hydrogen xhigh went as %q", got)
	}
	if rt := lastRoute(s); tryEffort(rt) != "xhigh" {
		t.Fatalf("hydrogen try %q, want xhigh", tryEffort(rt))
	}
	ask("gemini-4-hydrogen", "XHIGH")
	if got := levelOf(); got != "XHIGH" {
		t.Fatalf("hydrogen kept case %q, want XHIGH", got)
	}
	if rt := lastRoute(s); tryEffort(rt) != "xhigh" {
		t.Fatalf("hydrogen XHIGH try %q, want xhigh", tryEffort(rt))
	}
	ask("gemini-3-flash-min", "minimal")
	if got := levelOf(); got != "minimal" {
		t.Fatalf("minimal went as %q", got)
	}
	if rt := lastRoute(s); tryEffort(rt) != "minimal" {
		t.Fatalf("minimal try %q, want minimal", tryEffort(rt))
	}
	// a group fitting the level rewrites thinkingLevel and keeps the rest
	fitted := withEffort(provider.Gemini, []byte(`{"generationConfig":{"temperature":0.2,"thinkingConfig":{"thinkingLevel":"xhigh","includeThoughts":true}}}`), "high")
	if gjson.GetBytes(fitted, "generationConfig.thinkingConfig.thinkingLevel").String() != "high" ||
		!gjson.GetBytes(fitted, "generationConfig.thinkingConfig.includeThoughts").Bool() ||
		gjson.GetBytes(fitted, "generationConfig.temperature").Float() != 0.2 {
		t.Fatalf("withEffort: %s", fitted)
	}
}

// Messages translated onto Gemini and Code Assist send xhigh where the
// model lists it and high for max on a model that stops at high. max on
// Hydrogen is the nearest listed level, xhigh.
func TestMessagesToGeminiKeepsXHigh(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var last []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = append([]byte(nil), b...)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-4-hydrogen", "gemini-4-argon"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	ask := func(model, effort string) string {
		t.Helper()
		rec := httptest.NewRecorder()
		body := `{"model":"gem/` + model + `","max_tokens":32000,"stream":true,"thinking":{"type":"adaptive"},"output_config":{"effort":"` + effort + `"},"messages":[{"role":"user","content":"hi"}]}`
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s %s: %d %s", model, effort, rec.Code, rec.Body.String())
		}
		mu.Lock()
		b := string(last)
		mu.Unlock()
		return gjson.Get(b, "generationConfig.thinkingConfig.thinkingLevel").String()
	}
	if got := ask("gemini-4-hydrogen", "xhigh"); got != "xhigh" {
		t.Fatalf("hydrogen xhigh: %q", got)
	}
	if got := ask("gemini-4-hydrogen", "max"); got != "xhigh" {
		t.Fatalf("hydrogen max: %q, want xhigh", got)
	}
	if got := ask("gemini-4-argon", "xhigh"); got != "high" {
		t.Fatalf("argon xhigh: %q", got)
	}
	if got := ask("gemini-4-argon", "max"); got != "high" {
		t.Fatalf("argon max: %q", got)
	}
	p := provider.Provider{ID: "gemini", Catalog: "google", Account: &provider.Account{Agent: "gemini"}}
	for _, c := range []struct{ model, effort, want string }{
		{"gemini-4-hydrogen", "xhigh", "xhigh"},
		{"gemini-4-hydrogen", "max", "xhigh"},
		{"gemini-4-argon", "xhigh", "high"},
		{"gemini-4-argon", "max", "high"},
	} {
		r := &Request{Thinking: true, Effort: c.effort, Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}
		body := codeAssistBody(p, r, c.model, nil)
		if got := gjson.GetBytes(body, "request.generationConfig.thinkingConfig.thinkingLevel").String(); got != c.want {
			t.Errorf("code assist %s %s: %q, want %q", c.model, c.effort, got, c.want)
		}
	}
}

// SetModelID and SetUpstreamName are the model generation and levels are
// read from, and the effort is fitted to that model. my-hydrogen is
// gemini-4-hydrogen, so xhigh is a level and not a budget, and max is
// hydrogen's xhigh. gemini-4-hydrogen renamed to gemini-4-argon is not
// sent xhigh, and max on that rename is high. An argon id renamed onto
// hydrogen is fitted to hydrogen, so max is xhigh rather than argon's high.
func TestGeminiAliasUsesTheUpstreamModel(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	type sent struct {
		path string
		body []byte
	}
	var last sent
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = sent{r.URL.Path, append([]byte(nil), b...)}
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "alias", Name: "Alias", Key: "k", Catalog: "google", Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-4-hydrogen", "gemini-4-argon"},
	}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := provider.SetModelID("alias/gemini-4-hydrogen", "my-hydrogen"); err != nil {
		t.Fatal(err)
	}
	s := New()
	ask := func(model, effort string) sent {
		t.Helper()
		rec := httptest.NewRecorder()
		body := `{"model":"alias/` + model + `","stream":true,"reasoning_effort":"` + effort + `","messages":[{"role":"user","content":"hi"}]}`
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", model, rec.Code, rec.Body.String())
		}
		mu.Lock()
		got := last
		mu.Unlock()
		return got
	}
	got := ask("my-hydrogen", "xhigh")
	if !strings.Contains(got.path, "gemini-4-hydrogen") {
		t.Fatalf("my-hydrogen asked %s", got.path)
	}
	if gjson.GetBytes(got.body, "generationConfig.thinkingConfig.thinkingBudget").Exists() {
		t.Fatalf("my-hydrogen sent a budget: %s", got.body)
	}
	if lv := gjson.GetBytes(got.body, "generationConfig.thinkingConfig.thinkingLevel").String(); lv != "xhigh" {
		t.Fatalf("my-hydrogen thinkingLevel %q", lv)
	}
	got = ask("my-hydrogen", "max")
	if lv := gjson.GetBytes(got.body, "generationConfig.thinkingConfig.thinkingLevel").String(); lv != "xhigh" {
		t.Fatalf("my-hydrogen max thinkingLevel %q, want xhigh", lv)
	}
	if err := provider.SetUpstreamName("alias/gemini-4-argon", "gemini-4-hydrogen"); err != nil {
		t.Fatal(err)
	}
	got = ask("gemini-4-argon", "max")
	if !strings.Contains(got.path, "gemini-4-hydrogen") {
		t.Fatalf("argon renamed onto hydrogen asked %s", got.path)
	}
	if lv := gjson.GetBytes(got.body, "generationConfig.thinkingConfig.thinkingLevel").String(); lv != "xhigh" {
		t.Fatalf("argon renamed onto hydrogen at max: thinkingLevel %q, want xhigh", lv)
	}
	if err := provider.SetUpstreamName("alias/my-hydrogen", "gemini-4-argon"); err != nil {
		t.Fatal(err)
	}
	got = ask("my-hydrogen", "xhigh")
	if !strings.Contains(got.path, "gemini-4-argon") {
		t.Fatalf("renamed asked %s", got.path)
	}
	if lv := gjson.GetBytes(got.body, "generationConfig.thinkingConfig.thinkingLevel").String(); lv != "high" {
		t.Fatalf("argon alias thinkingLevel %q, want high", lv)
	}
}

// An unlisted gemini-4-* model is not sent xhigh on the native route or
// on Code Assist. Nothing lists the level, so it stays at high.
func TestUnlistedGemini4ClampsXHigh(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var last []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = append([]byte(nil), b...)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Catalog: "google", Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-4-quartz"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	body := `{"model":"gem/gemini-4-quartz","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got := last
	mu.Unlock()
	if lv := gjson.GetBytes(got, "generationConfig.thinkingConfig.thinkingLevel").String(); lv != "high" {
		t.Fatalf("unlisted thinkingLevel %q, body %s", lv, got)
	}
	if gjson.GetBytes(got, "generationConfig.thinkingConfig.thinkingBudget").Exists() {
		t.Fatalf("unlisted sent a budget: %s", got)
	}
	raw := &Request{Thinking: true, Effort: "xhigh", Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}
	env := buildCodeAssist(raw, "gemini-4-quartz", "gemini")
	if lv := gjson.GetBytes(env, "request.generationConfig.thinkingConfig.thinkingLevel").String(); lv != "high" {
		t.Fatalf("code assist unlisted %q", lv)
	}
}
