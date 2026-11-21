package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/settings"
)

// A Code Assist plugin (@ai-sdk/google) is told the effort in
// X-Magpie-Effort. The body's thinkingLevel is only what Gemini 3 can
// say, so the log keeps the header — xhigh — rather than the high the
// body was built with.
func TestCodeAssistPluginEffortIsTheHeader(t *testing.T) {
	t.Setenv("FAKE_GEMINI_DRAWS", "1")
	var mu sync.Mutex
	var body []byte
	var saw string
	pid := besideFake(t, "fakeco", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		body = append([]byte(nil), b...)
		saw = r.Header.Get("X-Saw-Effort")
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	s := New()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"`+pid+`/gemini-3-flash-canvas","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`))
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got, effort := string(body), saw
	mu.Unlock()
	if effort != "xhigh" {
		t.Fatalf("plugin header %q, want xhigh", effort)
	}
	level := gjson.Get(got, "request.generationConfig.thinkingConfig.thinkingLevel").String()
	if level == "" {
		level = gjson.Get(got, "generationConfig.thinkingConfig.thinkingLevel").String()
	}
	if level != "high" {
		t.Fatalf("body thinkingLevel %q in %s, want high", level, got)
	}
	if got := tryEffort(lastRoute(s)); got != "xhigh" {
		t.Fatalf("try effort %q, want xhigh", got)
	}
}

// An opaque SetUpstreamName is the model field only. Generation, the
// thinking level and compat stay on the model magpie lists.
func TestOpaqueUpstreamNameKeepsGeminiGeneration(t *testing.T) {
	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var last []byte
	var path string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = append([]byte(nil), b...)
		path = r.URL.RequestURI()
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Gemini: up.URL + "/v1beta",
		Models: []string{"gemini-4-hydrogen"}, Catalog: "google",
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("gem/gemini-4-hydrogen", "ep-123"); err != nil {
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
	got, at := string(last), path
	mu.Unlock()
	if !strings.Contains(at, "ep-123") {
		t.Fatalf("path %q, want the opaque id", at)
	}
	if lv := gjson.Get(got, "generationConfig.thinkingConfig.thinkingLevel").String(); lv != "xhigh" {
		t.Fatalf("thinkingLevel %q, want xhigh (a budget means generation fell back to 2.x): %s", lv, got)
	}
	if gjson.Get(got, "generationConfig.thinkingConfig.thinkingBudget").Exists() {
		t.Fatalf("opaque id was sent a budget: %s", got)
	}
}

// A deployment id on a non-Gemini model has no levels. Fitting falls
// back to the listed model, so mimo's max still becomes high.
func TestOpaqueUpstreamNameFitsListedEfforts(t *testing.T) {
	fresh(t)
	if err := writeMimoCatalog(t); err != nil {
		t.Fatal(err)
	}
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
	if err := provider.Save(provider.Provider{
		ID: "volc", Name: "Volc", Key: "k", Chat: up.URL + "/v1",
		Models: []string{"mimo-v2.6-flash"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("volc/mimo-v2.6-flash", "ep-123"); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"volc/mimo-v2.6-flash","stream":true,"reasoning_effort":"max","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got := string(last)
	mu.Unlock()
	if m := gjson.Get(got, "model").String(); m != "ep-123" {
		t.Fatalf("model %q, want ep-123", m)
	}
	if e := gjson.Get(got, "reasoning_effort").String(); e != "high" {
		t.Fatalf("effort %q, want high (%s)", e, got)
	}
}

// SetModelID moves the user's efforts onto the listed id. A request
// fitted against the vendor id, which no longer has them, would keep max.
func TestModelIDKeepsListedEfforts(t *testing.T) {
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
	if err := provider.Save(provider.Provider{
		ID: "box", Name: "Box", Key: "k", Chat: up.URL + "/v1",
		Models: []string{"custom-thinker"},
	}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := provider.SetModelID("box/custom-thinker", "my-thinker"); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("box/my-thinker", []string{"low", "high"}); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
		`{"model":"box/my-thinker","stream":true,"reasoning_effort":"max","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got := string(last)
	mu.Unlock()
	if e := gjson.Get(got, "reasoning_effort").String(); e != "high" {
		t.Fatalf("effort %q, want high (efforts live on the listed id): %s", e, got)
	}
}

// upstreamWire reads a SetUpstreamName on the vendor id, and
// specificUpstream honors a wires snapshot that is not the live map.
func TestUpstreamWireReadsOwnAndSnapshot(t *testing.T) {
	fresh(t)
	if err := provider.Save(provider.Provider{
		ID: "gem", Name: "Gem", Key: "k", Chat: "http://127.0.0.1",
		Models: []string{"gemini-4-hydrogen"},
	}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := provider.SetModelID("gem/gemini-4-hydrogen", "my-hydrogen"); err != nil {
		t.Fatal(err)
	}
	// the rename lives on the vendor id. SetUpstreamName refuses that id
	// once the model is listed by another, which is how a hand-edited
	// file and a request's wires snapshot still name it.
	s := settings.Load()
	if s.ModelWires == nil {
		s.ModelWires = map[string]string{}
	}
	s.ModelWires["gem/gemini-4-hydrogen"] = "ep-own"
	if err := settings.Save(s); err != nil {
		t.Fatal(err)
	}
	p, err := provider.Find("gem")
	if err != nil {
		t.Fatal(err)
	}
	wires := settings.Load().Wires()
	if w := upstreamWire(wires, *p, "my-hydrogen", nil); w != "ep-own" {
		t.Fatalf("own rename %q, want ep-own", w)
	}
	snap := map[string]string{"gem/gemini-4-hydrogen": "ep-snap"}
	if w, ok := specificUpstream(snap, *p, "gemini-4-hydrogen"); !ok || w != "ep-snap" {
		t.Fatalf("snapshot %q %v, want ep-snap", w, ok)
	}
	if as := geminiCheckedModel(*p, "my-hydrogen", snap); as != "gemini-4-hydrogen" {
		t.Fatalf("opaque snapshot checked %q, want the vendor id", as)
	}
}

func TestResponsesConfigurationUpdateWinsTheLog(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, responsesHi)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "codexr", Name: "CodexR", Key: "k", Responses: up.URL + "/v1",
		Models: []string{"gpt-5"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	body := `{"model":"codexr/gpt-5","stream":true,"reasoning":{"effort":"low"},"input":[{"type":"message","role":"user","content":"hi"},{"type":"configuration_update","reasoning":{"effort":"xhigh"}}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	rt := lastRoute(s)
	if rt.Effort != "xhigh" || tryEffort(rt) != "xhigh" {
		t.Fatalf("route %q try %q, want xhigh/xhigh", rt.Effort, tryEffort(rt))
	}
	if e, ok := bodyWireEffort(provider.Responses, []byte(body)); !ok || e != "xhigh" {
		t.Fatalf("bodyWireEffort %q %v, want xhigh", e, ok)
	}
}

func TestAnthropicWireEffortRegressions(t *testing.T) {
	// a session title: output_config.effort and no thinking is not an ask
	if e, ok := anthropicWireEffort([]byte(`{"model":"claude-sonnet-4-5","output_config":{"effort":"high"},"messages":[{"role":"user","content":"title"}]}`)); ok {
		t.Fatalf("session title logged %q", e)
	}
	// a model that cannot turn thinking off still logs the effort
	if e, ok := anthropicWireEffort([]byte(`{"model":"claude-opus-5-5","output_config":{"effort":"high"}}`)); !ok || e != "high" {
		t.Fatalf("offNever %q %v, want high", e, ok)
	}
	// a budget that replaced disabled is that budget's level
	if e, ok := anthropicWireEffort([]byte(`{"thinking":{"type":"enabled","budget_tokens":24000}}`)); !ok || e != "high" {
		t.Fatalf("budget %q %v, want high", e, ok)
	}
	// disabled the client asked for is still none
	if e, ok := anthropicWireEffort([]byte(`{"thinking":{"type":"disabled"},"output_config":{"effort":"high"}}`)); !ok || e != "none" {
		t.Fatalf("asked off %q %v, want none", e, ok)
	}

	fresh(t)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, anthropicHi)
	}))
	t.Cleanup(up.Close)
	base := provider.Provider{ID: "anth", Name: "Anth", Key: "k", Anthropic: up.URL, Models: []string{"m1", "claude-sonnet-4-5"}}
	if err := provider.Save(base); err != nil {
		t.Fatal(err)
	}
	s := New()
	post := func(body string) {
		t.Helper()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("%d %s", rec.Code, rec.Body.String())
		}
	}
	post(`{"model":"anth/m1","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"title"}]}`)
	if got := tryEffort(lastRoute(s)); got != "" {
		t.Fatalf("appended disabled logged %q, want empty", got)
	}
	post(`{"model":"anth/claude-sonnet-4-5","max_tokens":64,"stream":true,"output_config":{"effort":"high"},"messages":[{"role":"user","content":"title"}]}`)
	if got := tryEffort(lastRoute(s)); got != "" {
		t.Fatalf("session title logged %q, want empty", got)
	}
	post(`{"model":"anth/m1","max_tokens":64,"stream":true,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`)
	if got := tryEffort(lastRoute(s)); got != "none" {
		t.Fatalf("asked disabled logged %q, want none", got)
	}
	base.ExtraBody = provider.ExtraBody{"!thinking": json.RawMessage(`{"type":"enabled","budget_tokens":24000}`)}
	if err := provider.Save(base); err != nil {
		t.Fatal(err)
	}
	post(`{"model":"anth/m1","max_tokens":64,"stream":true,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`)
	if got := tryEffort(lastRoute(s)); got != "high" {
		t.Fatalf("budget reopened logged %q, want high", got)
	}
}

func TestGeminiPassthroughEdges(t *testing.T) {
	if got := (&gThinking{ThinkingLevel: "none"}).effort(); got != "none" {
		t.Fatalf("thinkingLevel none read as %q", got)
	}
	if got := geminiRefusedLevel([]byte(`Thinking level NONE is not supported for this model`)); got != "none" {
		t.Fatalf("refused none %q", got)
	}
	if got := geminiBodyEffort([]byte(`{"generationConfig":{"thinkingConfig":{"thinkingLevel":"none"}}}`)); got != "none" {
		t.Fatalf("body none %q", got)
	}

	fresh(t)
	loadGeminiXHighCatalog(t)
	var mu sync.Mutex
	var bodies []map[string]any
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		json.NewDecoder(r.Body).Decode(&b)
		mu.Lock()
		bodies = append(bodies, b)
		level, _ := b["reasoning_effort"].(string)
		if level == "" {
			level = thinkingLevelOf(b)
		}
		mu.Unlock()
		if strings.EqualFold(level, "xhigh") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, aiStudioRefusal("xhigh"))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{
		ID: "studio", Name: "Studio", Key: "k", Catalog: "google", Chat: up.URL + "/v1",
		Models: []string{"gemini-4-quartz", "gemini-4-hydrogen"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	ask := func(body string) {
		t.Helper()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("%d %s", rec.Code, rec.Body.String())
		}
	}
	ask(`{"model":"studio/gemini-4-quartz","stream":true,"reasoning_effort":"max","messages":[{"role":"user","content":"hi"}]}`)
	mu.Lock()
	if len(bodies) != 1 || bodies[0]["reasoning_effort"] != "high" {
		t.Fatalf("max on unlisted: %+v", bodies)
	}
	mu.Unlock()

	before := 0
	mu.Lock()
	before = len(bodies)
	mu.Unlock()
	ask(`{"model":"studio/gemini-4-quartz","stream":true,"messages":[{"role":"user","content":"hi"}],"extra_body":{"google":{"thinking_config":{"thinking_level":"xhigh","include_thoughts":true}}}}`)
	mu.Lock()
	clamped := append([]map[string]any{}, bodies[before:]...)
	mu.Unlock()
	if len(clamped) != 1 || thinkingLevelOf(clamped[0]) != "high" || clamped[0]["reasoning_effort"] != nil {
		t.Fatalf("unlisted extra_body xhigh clamped, reasoning_effort absent: %+v", clamped)
	}

	before = 0
	mu.Lock()
	before = len(bodies)
	mu.Unlock()
	ask(`{"model":"studio/gemini-4-hydrogen","stream":true,"messages":[{"role":"user","content":"hi"}],"extra_body":{"google":{"thinking_config":{"thinking_level":"xhigh","include_thoughts":true}}}}`)
	mu.Lock()
	got := append([]map[string]any{}, bodies[before:]...)
	mu.Unlock()
	if len(got) != 2 || thinkingLevelOf(got[0]) != "xhigh" || thinkingLevelOf(got[1]) != "high" {
		t.Fatalf("extra_body xhigh retry: %+v", got)
	}
	if got[0]["reasoning_effort"] != nil || got[1]["reasoning_effort"] != nil {
		t.Fatalf("extra_body retry carried reasoning_effort: %+v", got)
	}
}

func thinkingLevelOf(b map[string]any) string {
	eb, _ := b["extra_body"].(map[string]any)
	g, _ := eb["google"].(map[string]any)
	tc, _ := g["thinking_config"].(map[string]any)
	lv, _ := tc["thinking_level"].(string)
	return lv
}

func writeMimoCatalog(t *testing.T) error {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(catalog.CachePath()), 0o755); err != nil {
		return err
	}
	err := os.WriteFile(catalog.CachePath(), []byte(`{
	  "xiaomi":{"models":{"mimo-v2.6-flash":{"id":"mimo-v2.6-flash","reasoning":true,"reasoning_options":[{"type":"toggle"}]}}},
	  "llmgateway":{"models":{"mimo-v2.6-flash":{"id":"mimo-v2.6-flash","reasoning_options":[{"type":"effort","values":["none","low","medium","high","xhigh","max"]}]}}}}`), 0o644)
	if err != nil {
		return err
	}
	catalog.Reset()
	t.Cleanup(catalog.Reset)
	return nil
}

// SetModelID("…/ep-123", "gemini-4-hydrogen") keeps generation and the
// thinking level on the listed model. The vendor id is what is sent,
// unless a ModelWires entry on that id names another model, which is
// then both the name on the wire and the model the level is shaped for.
func TestGeminiOpaqueDeploymentKeepsTheGeneration(t *testing.T) {
	level := func(body []byte) (lv string, budget bool) {
		t.Helper()
		for _, p := range []string{
			"generationConfig.thinkingConfig.thinkingLevel",
			"request.generationConfig.thinkingConfig.thinkingLevel",
		} {
			if v := gjson.GetBytes(body, p).String(); v != "" {
				lv = v
			}
		}
		budget = gjson.GetBytes(body, "generationConfig.thinkingConfig.thinkingBudget").Exists() ||
			gjson.GetBytes(body, "request.generationConfig.thinkingConfig.thinkingBudget").Exists()
		return lv, budget
	}

	t.Run("codeassist", func(t *testing.T) {
		fresh(t)
		loadGeminiXHighCatalog(t)
		st := settings.Load()
		st.ModelIDs = map[string]string{"gem/gemini-4-hydrogen": "ep-123"}
		if err := settings.Save(st); err != nil {
			t.Fatal(err)
		}
		p := provider.Provider{ID: "gem", Catalog: "google", Account: &provider.Account{Agent: "gemini"}}
		r := &Request{Thinking: true, Effort: "xhigh", Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}}}
		body := codeAssistBody(p, r, "gemini-4-hydrogen", settings.Load().Wires())
		if m := gjson.GetBytes(body, "model").String(); m != "ep-123" {
			t.Fatalf("model %q, want ep-123", m)
		}
		if lv, budget := level(body); lv != "xhigh" || budget {
			t.Fatalf("code assist level %q budget %v: %s", lv, budget, body)
		}
		if as := geminiCheckedModel(p, "gemini-4-hydrogen", settings.Load().Wires()); as != "gemini-4-hydrogen" {
			t.Fatalf("checked %q, want the listed model", as)
		}
		st = settings.Load()
		st.ModelWires = map[string]string{"gem/ep-123": "gemini-4-argon"}
		if err := settings.Save(st); err != nil {
			t.Fatal(err)
		}
		body = codeAssistBody(p, r, "gemini-4-hydrogen", settings.Load().Wires())
		if m := gjson.GetBytes(body, "model").String(); m != "gemini-4-argon" {
			t.Fatalf("renamed model %q, want gemini-4-argon", m)
		}
		if lv, _ := level(body); lv != "high" {
			t.Fatalf("argon level %q, want high: %s", lv, body)
		}
	})

	t.Run("gemini", func(t *testing.T) {
		fresh(t)
		loadGeminiXHighCatalog(t)
		var mu sync.Mutex
		var path string
		var last []byte
		up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			path, last = r.URL.Path, append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, vertexPong)
		}))
		t.Cleanup(up.Close)
		if err := provider.Save(provider.Provider{
			ID: "gem", Name: "Gem", Key: "k", Catalog: "google", Gemini: up.URL + "/v1beta",
			Models: []string{"ep-123"},
		}); err != nil {
			t.Fatal(err)
		}
		if _, _, err := provider.SetModelID("gem/ep-123", "gemini-4-hydrogen"); err != nil {
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
		gotPath, got := path, last
		mu.Unlock()
		if !strings.Contains(gotPath, "ep-123") || strings.Contains(gotPath, "gemini-4-hydrogen") {
			t.Fatalf("path %s, want the vendor id", gotPath)
		}
		if lv, budget := level(got); lv != "xhigh" || budget {
			t.Fatalf("gemini level %q budget %v: %s", lv, budget, got)
		}
	})

	t.Run("vertex", func(t *testing.T) {
		v := &vertexUp{}
		s := vertexServe(t, v, geminiXHighCatalog)
		st := settings.Load()
		if st.ModelIDs == nil {
			st.ModelIDs = map[string]string{}
		}
		st.ModelIDs["google-vertex/gemini-4-hydrogen"] = "ep-123"
		if err := settings.Save(st); err != nil {
			t.Fatal(err)
		}
		vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
		got := v.last(t)
		if !strings.Contains(got.path, "ep-123") || strings.Contains(got.path, "gemini-4-hydrogen") {
			t.Fatalf("path %s, want the vendor id", got.path)
		}
		if lv, _ := thinkingOf(got)["thinkingLevel"].(string); lv != "xhigh" || thinkingOf(got)["thinkingBudget"] != nil {
			t.Fatalf("vertex thinking %v", thinkingOf(got))
		}
		st = settings.Load()
		if st.ModelWires == nil {
			st.ModelWires = map[string]string{}
		}
		st.ModelWires["google-vertex/ep-123"] = "gemini-4-argon"
		if err := settings.Save(st); err != nil {
			t.Fatal(err)
		}
		vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
		got = v.last(t)
		if !strings.Contains(got.path, "gemini-4-argon") {
			t.Fatalf("rename path %s", got.path)
		}
		if lv, _ := thinkingOf(got)["thinkingLevel"].(string); lv != "high" {
			t.Fatalf("argon thinking %v", thinkingOf(got))
		}
	})

	t.Run("aistudio and passthrough", func(t *testing.T) {
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
			io.WriteString(w, chatHi)
		}))
		t.Cleanup(up.Close)
		if err := provider.Save(provider.Provider{
			ID: "studio", Name: "Studio", Key: "k", Catalog: "google", Chat: up.URL + "/v1",
			Models: []string{"ep-123"},
		}); err != nil {
			t.Fatal(err)
		}
		if _, _, err := provider.SetModelID("studio/ep-123", "gemini-4-hydrogen"); err != nil {
			t.Fatal(err)
		}
		s := New()
		post := func(path, body string) {
			t.Helper()
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", path, strings.NewReader(body)))
			if rec.Code != http.StatusOK {
				t.Fatalf("%s: %d %s", path, rec.Code, rec.Body.String())
			}
		}
		post("/v1/messages", `{"model":"studio/gemini-4-hydrogen","max_tokens":32000,"stream":true,"thinking":{"type":"adaptive"},"output_config":{"effort":"xhigh"},"messages":[{"role":"user","content":"hi"}]}`)
		mu.Lock()
		got := string(last)
		mu.Unlock()
		if m := gjson.Get(got, "model").String(); m != "ep-123" {
			t.Fatalf("aistudio model %q, want ep-123: %s", m, got)
		}
		if lv := gjson.Get(got, "extra_body.google.thinking_config.thinking_level").String(); lv != "xhigh" {
			t.Fatalf("aistudio level %q: %s", lv, got)
		}
		if gjson.Get(got, "reasoning_effort").Exists() || gjson.Get(got, "extra_body.google.thinking_config.thinking_budget").Exists() {
			t.Fatalf("aistudio carried reasoning_effort or a budget: %s", got)
		}
		post("/v1/chat/completions", `{"model":"studio/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
		mu.Lock()
		got = string(last)
		mu.Unlock()
		if m := gjson.Get(got, "model").String(); m != "ep-123" {
			t.Fatalf("passthrough model %q, want ep-123: %s", m, got)
		}
		if e := gjson.Get(got, "reasoning_effort").String(); e != "xhigh" {
			t.Fatalf("passthrough effort %q, want xhigh: %s", e, got)
		}
		st := settings.Load()
		if st.ModelWires == nil {
			st.ModelWires = map[string]string{}
		}
		st.ModelWires["studio/ep-123"] = "gemini-4-argon"
		if err := settings.Save(st); err != nil {
			t.Fatal(err)
		}
		post("/v1/chat/completions", `{"model":"studio/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
		mu.Lock()
		got = string(last)
		mu.Unlock()
		if m := gjson.Get(got, "model").String(); m != "gemini-4-argon" {
			t.Fatalf("passthrough rename %q, want gemini-4-argon: %s", m, got)
		}
		if e := gjson.Get(got, "reasoning_effort").String(); e != "high" {
			t.Fatalf("passthrough argon effort %q, want high: %s", e, got)
		}
	})
}

// A turn's output_config.effort is the level that went out, ahead of the
// budget beside it (#5379). A translated max stays max: its budget is
// 32000, which effortOfBudget would read as xhigh.
func TestAnthropicTurnEffortAndTranslatedMax(t *testing.T) {
	turn := []byte(`{"model":"claude-opus-5-5","max_tokens":40000,"thinking":{"type":"enabled","budget_tokens":31999},"messages":[{"role":"user","content":"a"},{"role":"system","content":[],"output_config":{"effort":"high"}}]}`)
	if e, ok := anthropicWireEffort(turn); !ok || e != "high" {
		t.Fatalf("turn effort %q %v, want high", e, ok)
	}
	adaptive := []byte(`{"model":"claude-opus-5-5","thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"a"},{"role":"system","output_config":{"effort":"max"}}]}`)
	if e, ok := anthropicWireEffort(adaptive); !ok || e != "max" {
		t.Fatalf("adaptive turn %q %v, want max", e, ok)
	}
	bare := []byte(`{"model":"claude-opus-5-5","messages":[{"role":"system","output_config":{"effort":"low"}}]}`)
	if e, ok := anthropicWireEffort(bare); !ok || e != "low" {
		t.Fatalf("offNever turn %q %v, want low", e, ok)
	}
	title := []byte(`{"model":"claude-sonnet-4-5","messages":[{"role":"system","output_config":{"effort":"high"}}]}`)
	if e, ok := anthropicWireEffort(title); ok {
		t.Fatalf("session title logged %q", e)
	}
	// adaptive on a model that can turn thinking off still keeps the
	// turn's level when the request names none of its own (#5379).
	sonnet := []byte(`{"model":"claude-sonnet-4-5","thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"a"},{"role":"system","output_config":{"effort":"high"}}]}`)
	if e, ok := anthropicWireEffort(sonnet); !ok || e != "high" {
		t.Fatalf("adaptive sonnet turn %q %v, want high", e, ok)
	}
	noneTurn := []byte(`{"model":"claude-sonnet-4-5","thinking":{"type":"enabled","budget_tokens":4096},"messages":[{"role":"system","output_config":{"effort":"none"}},{"role":"system","output_config":{"effort":"minimal"}}]}`)
	if e, ok := anthropicWireEffort(noneTurn); !ok || e != "minimal" {
		t.Fatalf("last turn %q %v, want minimal, not low", e, ok)
	}
	ctx := withWireAt(t.Context(), &wireAt{})
	noteAnthropicAsked(ctx, []byte(`{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"system","output_config":{"effort":"none"}}]}`))
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w == nil || !w.askedOff {
		t.Fatal("turn none was not asked off")
	}
	ctx = withWireAt(t.Context(), &wireAt{})
	noteAnthropicAsked(ctx, []byte(`{"model":"m","output_config":{"effort":"high"},"messages":[{"role":"system","output_config":{"effort":"none"}}]}`))
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w == nil || w.askedOff {
		t.Fatal("top-level high was overridden by an earlier turn's none")
	}

	maxBody := []byte(`{"model":"claude-sonnet-4-5","thinking":{"type":"enabled","budget_tokens":32000},"messages":[{"role":"user","content":"hi"}]}`)
	req := &Request{Effort: "max", Thinking: true}
	if e, ok := wireEffort(provider.Anthropic, req, maxBody); !ok || e != "max" {
		t.Fatalf("translated max %q %v, want max", e, ok)
	}
	ctx = withWireAt(t.Context(), &wireAt{})
	noteWireEffort(ctx, provider.Anthropic, req, maxBody)
	noteWireBody(ctx, provider.Anthropic, maxBody)
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w == nil || !w.set || w.effort != "max" {
		t.Fatalf("logged max as %q", wireEffortOf(w))
	}
	// max_tokens 32000 clamps the 32000 budget to 30976, which
	// effortOfBudget reads as xhigh. The translation stays max.
	clamped := []byte(`{"model":"claude-sonnet-4-5","max_tokens":32000,"thinking":{"type":"enabled","budget_tokens":30976},"messages":[{"role":"user","content":"hi"}]}`)
	if e, ok := wireEffort(provider.Anthropic, req, clamped); !ok || e != "max" {
		t.Fatalf("clamped max %q %v, want max", e, ok)
	}
	ctx = withWireAt(t.Context(), &wireAt{})
	noteWireEffort(ctx, provider.Anthropic, req, clamped)
	noteWireBody(ctx, provider.Anthropic, clamped)
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w == nil || !w.set || w.effort != "max" {
		t.Fatalf("logged clamped max as %q", wireEffortOf(w))
	}
}

// extra_body that disables thinking still counts after thinkingOffUnlessAsked
// has already appended disabled. A retry that strips the field, and a
// fallback onto another endpoint, drop the failed attempt's effort.
func TestThinkingExtraDisabledAndWireAtReset(t *testing.T) {
	ctx := withWireAt(t.Context(), &wireAt{effort: "none", set: true})
	noteWireBody(ctx, provider.Anthropic, []byte(`{"model":"m1","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`))
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); !w.set || w.effort != "" || w.fromBody {
		t.Fatalf("stripped none logged %q set %v fromBody %v", w.effort, w.set, w.fromBody)
	}

	ctx = withWireAt(t.Context(), &wireAt{})
	noteWireEffort(ctx, provider.Anthropic, &Request{ThinkOff: true, Effort: "none"}, []byte(`{"thinking":{"type":"disabled"}}`))
	noteWireEffort(ctx, provider.Chat, &Request{}, []byte(`{"messages":[{"role":"user","content":"hi"}]}`))
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w.effort != "" || w.askedOff || w.extraDisabled || w.pluginEffort {
		t.Fatalf("fallback leaked %+v", w)
	}
	noteWireEffort(ctx, provider.Anthropic, &Request{Effort: "high", Thinking: true}, []byte(`{"thinking":{"type":"enabled","budget_tokens":24000}}`))
	noteWireEffort(ctx, provider.Anthropic, &Request{}, []byte(`{"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`))
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w.effort != "" {
		t.Fatalf("cleared effort stuck %q", w.effort)
	}

	fresh(t)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, anthropicHi)
	}))
	t.Cleanup(up.Close)
	base := provider.Provider{
		ID: "anth", Name: "Anth", Key: "k", Anthropic: up.URL, Models: []string{"m1"},
		ExtraBody: provider.ExtraBody{"!thinking": json.RawMessage(`{"type":"disabled"}`)},
	}
	if err := provider.Save(base); err != nil {
		t.Fatal(err)
	}
	s := New()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
		`{"model":"anth/m1","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"title"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d %s", rec.Code, rec.Body.String())
	}
	if got := tryEffort(lastRoute(s)); got != "none" {
		t.Fatalf("extra disabled after thinkingOffUnlessAsked logged %q, want none", got)
	}

	off := provider.ExtraBody{"thinking": json.RawMessage(`{"type":"disabled"}`)}
	if extraDisablesThinking(provider.Provider{Account: &provider.Account{Agent: "claude"}, ExtraBody: off}) {
		t.Fatal("unused extra body on a claude account counted as thinking off")
	}
	if !extraDisablesThinking(provider.Provider{Account: &provider.Account{Agent: "codex"}, ExtraBody: off}) {
		t.Fatal("codex extra body did not count as thinking off")
	}
	if !extraDisablesThinking(provider.Provider{ExtraBody: off}) {
		t.Fatal("a provider's extra body did not count as thinking off")
	}
}

// A retry that strips the effort the first attempt put on the wire must
// log that nothing went, not the level the client asked. OpenRouter's
// "reasoning is mandatory" drops reasoning_effort none
// (withoutReasoningOff); a turn's output_config dropped after a 400
// (withoutTurnEffort) is the same for an Anthropic body (#7400).
func TestStrippedRetryDropsNotedEffort(t *testing.T) {
	t.Run("withoutReasoningOff", func(t *testing.T) {
		f := &fake{t: t, reply: chatHi, refuse: func(body []byte) (int, string) {
			if gjson.GetBytes(body, "reasoning_effort").String() == "none" {
				return 400, `{"error":{"message":"reasoning is mandatory for this endpoint and cannot be disabled"}}`
			}
			return 0, ""
		}}
		setup(t, provider.Chat, f)
		s := New()
		code, out := postTo(t, s, "/v1/chat/completions", `{"model":"fake/m1","stream":true,"reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]}`)
		if code != 200 || f.calls != 2 {
			t.Fatalf("%d after %d calls: %s", code, f.calls, out)
		}
		if gjson.GetBytes(f.got, "reasoning_effort").Exists() {
			t.Fatalf("retry kept reasoning_effort: %s", f.got)
		}
		rt := lastRoute(s)
		if rt.Effort != "none" || tryEffort(rt) != "" {
			t.Fatalf("asked %q went %q, want asked none and went empty", rt.Effort, tryEffort(rt))
		}
	})
	t.Run("withoutTurnEffort", func(t *testing.T) {
		f := &fake{t: t, reply: anthropicHi, refuse: func(body []byte) (int, string) {
			if hasTurnEffort(body) {
				return 400, `{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.output_config: Extra inputs are not permitted"}}`
			}
			return 0, ""
		}}
		setup(t, provider.Anthropic, f)
		s := New()
		body := `{"model":"fake/m1","max_tokens":64,"stream":true,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"hi"},{"role":"system","content":"ctx","output_config":{"effort":"high"}}]}`
		code, out := postTo(t, s, "/v1/messages", body)
		if code != 200 || f.calls != 2 {
			t.Fatalf("%d after %d calls: %s", code, f.calls, out)
		}
		if hasTurnEffort(f.got) {
			t.Fatalf("retry kept the turn effort: %s", f.got)
		}
		rt := lastRoute(s)
		if rt.Effort != "high" || tryEffort(rt) != "" {
			t.Fatalf("asked %q went %q, want asked high and went empty", rt.Effort, tryEffort(rt))
		}
		// Turn 2 hits the learned shapeTurnEffort strip before noteWireBody
		// ever sees the inbound body: the log still clears the effort.
		f.calls = 0
		f.got = nil
		code, out = postTo(t, s, "/v1/messages", body)
		if code != 200 || f.calls != 1 {
			t.Fatalf("learned turn %d after %d calls: %s", code, f.calls, out)
		}
		if hasTurnEffort(f.got) {
			t.Fatalf("learned turn kept the turn effort: %s", f.got)
		}
		rt = lastRoute(s)
		if rt.Effort != "high" || tryEffort(rt) != "" {
			t.Fatalf("learned turn asked %q went %q, want asked high and went empty", rt.Effort, tryEffort(rt))
		}
	})
	t.Run("extraBodyClearsRelayEffort", func(t *testing.T) {
		cases := []struct {
			name  string
			proto provider.Protocol
			path  string
			reply string
			extra map[string]json.RawMessage
			body  string
		}{
			{
				name:  "chat",
				proto: provider.Chat,
				path:  "/v1/chat/completions",
				reply: chatHi,
				extra: map[string]json.RawMessage{"!reasoning_effort": json.RawMessage(`""`)},
				body:  `{"model":"fake/m1","stream":true,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}`,
			},
			{
				name:  "responses",
				proto: provider.Responses,
				path:  "/v1/responses",
				reply: responsesHi,
				extra: map[string]json.RawMessage{"!reasoning": json.RawMessage(`{"summary":"auto"}`)},
				body:  `{"model":"fake/m1","stream":true,"reasoning":{"effort":"high"},"input":"hi"}`,
			},
			{
				name:  "anthropic",
				proto: provider.Anthropic,
				path:  "/v1/messages",
				reply: anthropicHi,
				extra: map[string]json.RawMessage{"!output_config": json.RawMessage(`{}`)},
				body:  `{"model":"fake/m1","max_tokens":64,"stream":true,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`,
			},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				fresh(t)
				f := &fake{t: t, reply: tc.reply}
				srv := httptest.NewServer(f)
				t.Cleanup(srv.Close)
				p := provider.Provider{
					ID:        "fake",
					Name:      "Fake",
					Key:       "k",
					Models:    []string{"m1"},
					ExtraBody: provider.ExtraBody(tc.extra),
				}
				switch tc.proto {
				case provider.Chat:
					p.Chat = srv.URL
				case provider.Responses:
					p.Responses = srv.URL
				case provider.Anthropic:
					p.Anthropic = srv.URL
				}
				if err := provider.Save(p); err != nil {
					t.Fatal(err)
				}
				s := New()
				code, out := postTo(t, s, tc.path, tc.body)
				if code != 200 || f.calls != 1 {
					t.Fatalf("%d after %d calls: %s", code, f.calls, out)
				}
				rt := lastRoute(s)
				if rt.Effort != "high" || tryEffort(rt) != "" {
					t.Fatalf("asked %q went %q, want asked high and went empty (sent %s)", rt.Effort, tryEffort(rt), f.got)
				}
			})
		}
	})
}

// The word reasoning_effort inside a prompt is not a top-level field.
// Treating it as one made withBodyEffort inject a second level beside
// extra_body.google.thinking_config, which AI Studio turns away (#7400).
func TestChatGeminiLevelOnlyIgnoresPromptText(t *testing.T) {
	body := []byte(`{"model":"gemini-4-hydrogen","messages":[{"role":"user","content":"mention reasoning_effort in the prompt"}],"extra_body":{"google":{"thinking_config":{"thinking_level":"high","include_thoughts":true}}}}`)
	if !chatGeminiLevelOnly(body) {
		t.Fatal("prompt text hid a gemini-only level")
	}
	out := withBodyEffort(provider.Chat, body, "low")
	if gjson.GetBytes(out, "reasoning_effort").Exists() {
		t.Fatalf("injected reasoning_effort beside thinking_config: %s", out)
	}
	if lv := gjson.GetBytes(out, "extra_body.google.thinking_config.thinking_level").String(); lv != "low" {
		t.Fatalf("thinking_level %q, want low: %s", lv, out)
	}
	both := []byte(`{"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}],"extra_body":{"google":{"thinking_config":{"thinking_level":"high"}}}}`)
	if chatGeminiLevelOnly(both) {
		t.Fatal("a real top-level reasoning_effort was ignored")
	}
}

// Gemini's thinkingLevel none is thinking off, and minimal is a 4096
// budget. Neither word is written into Anthropic's output_config.effort,
// which rejects both; Chat and Responses keep minimal and do not receive
// none with thinking left on (#7400).
func TestGeminiNoneAndMinimalTranslate(t *testing.T) {
	if budgetOf("minimal") != 4096 {
		t.Fatalf("budgetOf(minimal) = %d, want 4096", budgetOf("minimal"))
	}
	parse := func(level string) *Request {
		t.Helper()
		r, err := parseGemini([]byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingLevel":"` + level + `"}}}`))
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	none := parse("none")
	if !none.ThinkOff || none.Effort != "low" || !none.Thinking {
		t.Fatalf("none parsed ThinkOff %v effort %q thinking %v", none.ThinkOff, none.Effort, none.Thinking)
	}
	minimal := parse("minimal")
	if minimal.ThinkOff || minimal.Effort != "minimal" || !minimal.Thinking {
		t.Fatalf("minimal parsed ThinkOff %v effort %q thinking %v", minimal.ThinkOff, minimal.Effort, minimal.Thinking)
	}
	checkAnthropic := func(r *Request, model, wantType, wantEffort string, wantBudget int) {
		t.Helper()
		got := buildAnthropic(r, model)
		if typ := gjson.GetBytes(got, "thinking.type").String(); typ != wantType {
			t.Errorf("%s %s: thinking %q, want %q: %s", r.Effort, model, typ, wantType, got)
		}
		if e := gjson.GetBytes(got, "output_config.effort").String(); e != wantEffort {
			t.Errorf("%s %s: output_config.effort %q, want %q: %s", r.Effort, model, e, wantEffort, got)
		}
		if wantBudget > 0 {
			if b := int(gjson.GetBytes(got, "thinking.budget_tokens").Int()); b != wantBudget {
				t.Errorf("%s %s: budget %d, want %d: %s", r.Effort, model, b, wantBudget, got)
			}
		}
		if strings.Contains(string(got), `"minimal"`) || strings.Contains(string(got), `"none"`) {
			t.Errorf("%s %s leaked none or minimal: %s", r.Effort, model, got)
		}
	}
	checkAnthropic(none, "claude-sonnet-4-5", "disabled", "", 0)
	checkAnthropic(none, "claude-opus-4-6", "disabled", "", 0)
	checkAnthropic(minimal, "claude-sonnet-4-5", "enabled", "", 4096)
	checkAnthropic(minimal, "claude-opus-4-6", "adaptive", "low", 0)
	checkAnthropic(minimal, "glm-5.2", "enabled", "low", 4096)

	for _, level := range []string{"none", "minimal"} {
		r := parse(level)
		chat := buildChat(r, "m1", "", false)
		resp := buildResponses(r, "m1", "", false)
		want := level
		if level == "none" {
			want = "low"
		}
		if e := gjson.GetBytes(chat, "reasoning_effort").String(); e != want {
			t.Errorf("chat %s: reasoning_effort %q, want %q: %s", level, e, want, chat)
		}
		if e := gjson.GetBytes(resp, "reasoning.effort").String(); e != want {
			t.Errorf("responses %s: effort %q, want %q: %s", level, e, want, resp)
		}
		if level == "none" && (strings.Contains(string(chat), `"none"`) || strings.Contains(string(resp), `"none"`)) {
			t.Errorf("none leaked: chat %s responses %s", chat, resp)
		}
	}
}
