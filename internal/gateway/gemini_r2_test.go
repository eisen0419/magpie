package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"os"
	"path/filepath"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/settings"
)

// A SetModelID is not a SetUpstreamName. Antigravity still sends the
// variant the effort picked, and a provider "*" wraps that variant once.
// geminiAs, the family's own id, does not replace it.
func TestAntigravityModelIDKeepsTheEffortVariant(t *testing.T) {
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
	p := provider.Provider{ID: "antigravity", Account: &provider.Account{Agent: "antigravity", User: "u"}}
	r, err := parseChat([]byte(`{"model":"antigravity/my-flash","stream":true,"reasoning_effort":"low","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	r.geminiAs = "flash-9"
	var env struct{ Model string }
	body := codeAssistBody(p, r, "my-flash", settings.Load().Wires())
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatal(err)
	}
	if env.Model != "vendor/flash-9-low" {
		t.Fatalf("model %q, want vendor/flash-9-low", env.Model)
	}
	if as := geminiCheckedModel(p, "my-flash", settings.Load().Wires()); as != "flash-9" {
		t.Fatalf("geminiCheckedModel = %q, want flash-9", as)
	}
}

// SetModelID plus a provider "*" is applied once. The canonical model
// stays unprefixed, so a later lookup does not send openai/openai/gpt-5
// or peta-gemini-4-hydrogen.
func TestModelIDWildcardAppliedOnce(t *testing.T) {
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
		ID: "relay", Name: "Relay", Key: "k", Chat: up.URL + "/v1",
		Models: []string{"gpt-5", "claude-sonnet-5", "gemini-4-hydrogen"},
	}); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ own, listed string }{
		{"gpt-5", "my-gpt"},
		{"claude-sonnet-5", "my-claude"},
	} {
		if _, _, err := provider.SetModelID("relay/"+c.own, c.listed); err != nil {
			t.Fatal(err)
		}
	}
	if err := provider.SetUpstreamName("relay/*", "openai/*"); err != nil {
		t.Fatal(err)
	}
	p, err := provider.Find("relay")
	if err != nil {
		t.Fatal(err)
	}
	wires := settings.Load().Wires()
	if as := geminiCheckedModel(*p, "my-gpt", wires); as != "gpt-5" {
		t.Fatalf("gpt checked %q, want gpt-5", as)
	}
	s := New()
	ask := func(model, want string) {
		t.Helper()
		rec := httptest.NewRecorder()
		body := `{"model":"relay/` + model + `","stream":true,"messages":[{"role":"user","content":"hi"}]}`
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", model, rec.Code, rec.Body.String())
		}
		mu.Lock()
		got := string(last)
		mu.Unlock()
		if m := gjson.Get(got, "model").String(); m != want {
			t.Fatalf("%s went out as %q, want %q (%s)", model, m, want, got)
		}
	}
	ask("my-gpt", "openai/gpt-5")
	ask("my-claude", "openai/claude-sonnet-5")

	if err := provider.SetUpstreamName("relay/*", "peta-*"); err != nil {
		t.Fatal(err)
	}
	wires = settings.Load().Wires()
	if as := geminiCheckedModel(*p, "gemini-4-hydrogen", wires); as != "gemini-4-hydrogen" {
		t.Fatalf("peta checked %q, want gemini-4-hydrogen", as)
	}
	req := &Request{Effort: "xhigh"}
	if w := upstreamWire(wires, *p, "gemini-4-hydrogen", req); w != "peta-gemini-4-hydrogen" {
		t.Fatalf("peta wire %q", w)
	}
}

// Vertex AI fits and builds thinking for the upstream model: a rename onto
// argon stays at high, an alias onto hydrogen is sent xhigh, and an alias
// onto 2.5 is a budget.
func TestVertexAliasUsesTheUpstreamModel(t *testing.T) {
	levels := `{"google":{"models":{
		"gemini-4-hydrogen":{"id":"gemini-4-hydrogen","reasoning_options":[{"type":"effort","values":["low","medium","high","xhigh"]}]},
		"gemini-4-argon":{"id":"gemini-4-argon","reasoning_options":[{"type":"effort","values":["low","medium","high"]}]},
		"gemini-2.5-flash":{"id":"gemini-2.5-flash","reasoning_options":[{"type":"effort","values":["low","medium","high"]}]}}}}`
	v := &vertexUp{}
	s := vertexServe(t, v, levels)
	if err := provider.SetUpstreamName("google-vertex/gemini-4-hydrogen", "gemini-4-argon"); err != nil {
		t.Fatal(err)
	}
	vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
	got := v.last(t)
	if !strings.Contains(got.path, "gemini-4-argon") || strings.Contains(got.path, "gemini-4-hydrogen") {
		t.Fatalf("rename path %s", got.path)
	}
	if lv, _ := thinkingOf(got)["thinkingLevel"].(string); lv != "high" {
		t.Fatalf("argon rename thinking %v", thinkingOf(got))
	}

	v2 := &vertexUp{}
	s = vertexServe(t, v2, levels)
	if _, _, err := provider.SetModelID("google-vertex/gemini-4-hydrogen", "my-hydrogen"); err != nil {
		t.Fatal(err)
	}
	vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/my-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
	got = v2.last(t)
	if !strings.Contains(got.path, "gemini-4-hydrogen") {
		t.Fatalf("hydrogen alias path %s", got.path)
	}
	if lv, _ := thinkingOf(got)["thinkingLevel"].(string); lv != "xhigh" {
		t.Fatalf("hydrogen alias thinking %v", thinkingOf(got))
	}

	v3 := &vertexUp{}
	s = vertexServe(t, v3, levels)
	if _, _, err := provider.SetModelID("google-vertex/gemini-2.5-flash", "my-25"); err != nil {
		t.Fatal(err)
	}
	vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/my-25","stream":true,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}`)
	got = v3.last(t)
	if !strings.Contains(got.path, "gemini-2.5-flash") {
		t.Fatalf("2.5 alias path %s", got.path)
	}
	tc := thinkingOf(got)
	if _, ok := tc["thinkingLevel"]; ok || tc["thinkingBudget"] == nil {
		t.Fatalf("2.5 alias thinking %v, want a budget", tc)
	}
}

// Gemini's OpenAI-compatible Chat passthrough clamps an unlisted Gemini 4,
// fits a rename onto argon, and retries a refused xhigh at high.
func TestGeminiCompatChatPassthrough(t *testing.T) {
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
		Models: []string{"gemini-4-quartz", "gemini-4-hydrogen", "gemini-4-argon"},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	ask := func(model, effort string) *httptest.ResponseRecorder {
		t.Helper()
		rec := httptest.NewRecorder()
		body := `{"model":"studio/` + model + `","stream":true,"reasoning_effort":"` + effort + `","messages":[{"role":"user","content":"hi"}]}`
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		return rec
	}
	rec := ask("gemini-4-quartz", "xhigh")
	if rec.Code != http.StatusOK {
		t.Fatalf("quartz: %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	if len(bodies) != 1 || bodies[0]["reasoning_effort"] != "high" || bodies[0]["model"] != "gemini-4-quartz" {
		t.Fatalf("unlisted passthrough: %+v", bodies)
	}
	mu.Unlock()

	if err := provider.SetUpstreamName("studio/gemini-4-hydrogen", "gemini-4-argon"); err != nil {
		t.Fatal(err)
	}
	rec = ask("gemini-4-hydrogen", "xhigh")
	if rec.Code != http.StatusOK {
		t.Fatalf("argon rename: %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	last := bodies[len(bodies)-1]
	mu.Unlock()
	if last["model"] != "gemini-4-argon" || last["reasoning_effort"] != "high" {
		t.Fatalf("argon passthrough: %+v", last)
	}

	if _, err := provider.DropUpstreamName("studio/gemini-4-hydrogen"); err != nil {
		t.Fatal(err)
	}
	before := 0
	mu.Lock()
	before = len(bodies)
	mu.Unlock()
	rec = ask("gemini-4-hydrogen", "xhigh")
	if rec.Code != http.StatusOK {
		t.Fatalf("refusal: %d %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	got := bodies[before:]
	mu.Unlock()
	if len(got) != 2 || got[0]["reasoning_effort"] != "xhigh" || got[1]["reasoning_effort"] != "high" {
		t.Fatalf("xhigh refusal retry: %+v", got)
	}
}

// The effort the log keeps is the one in the body that left: snake_case
// Gemini levels, a Responses effort, thinking turned back on, and a
// disabled Anthropic request whose output_config still names a level.
func TestWireEffortFollowsTheFinalBody(t *testing.T) {
	for _, c := range []struct{ body, want string }{
		{`{"generationConfig":{"thinkingConfig":{"thinkingLevel":"xhigh"}}}`, "xhigh"},
		{`{"generation_config":{"thinking_config":{"thinking_level":"low"}}}`, "low"},
		{`{"request":{"generation_config":{"thinkingConfig":{"thinking_level":"high"}}}}`, "high"},
		{`{"generationConfig":{"thinking_config":{"thinkingLevel":"medium"}}}`, "medium"},
	} {
		got, ok := geminiWireLevel([]byte(c.body))
		if !ok || got != c.want {
			t.Errorf("geminiWireLevel %s = %q %v, want %q", c.body, got, ok, c.want)
		}
	}
	if _, ok := geminiWireLevel([]byte(`{"generationConfig":{"thinkingConfig":{"thinkingLevel":""}}}`)); ok {
		t.Error("empty thinkingLevel was a level")
	}
	if _, ok := geminiWireLevel([]byte(`{"generationConfig":{"thinkingConfig":{"thinkingLevel":1}}}`)); ok {
		t.Error("a non-string thinkingLevel was a level")
	}
	for _, c := range []struct{ body, want string }{
		{`{"thinking":{"type":"disabled"},"output_config":{"effort":"high"}}`, "none"},
		{`{"thinking":{"type":"between_tools"},"output_config":{"effort":"max"}}`, "none"},
		{`{"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`, "high"},
	} {
		got, ok := anthropicWireEffort([]byte(c.body))
		if !ok || got != c.want {
			t.Errorf("anthropic %s = %q %v, want %q", c.body, got, ok, c.want)
		}
	}
	ctx := withWireAt(t.Context(), &wireAt{effort: "none", set: true})
	noteWireBody(ctx, provider.Anthropic, []byte(`{"thinking":{"type":"disabled"},"output_config":{"effort":"high"}}`))
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w.effort != "none" {
		t.Fatalf("disabled leftover effort logged %q", w.effort)
	}
	noteWireBody(ctx, provider.Anthropic, []byte(`{"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`))
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w.effort != "high" {
		t.Fatalf("thinking turned back on logged %q", w.effort)
	}
	if e, ok := bodyWireEffort(provider.Responses, []byte(`{"reasoning":{"effort":"xhigh","summary":"auto"}}`)); !ok || e != "xhigh" {
		t.Fatalf("responses effort %q %v", e, ok)
	}

	fresh(t)
	var mu sync.Mutex
	var last []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = append([]byte(nil), b...)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		switch {
		case strings.Contains(r.URL.Path, "generateContent"):
			io.WriteString(w, vertexPong)
		case strings.Contains(r.URL.Path, "/messages"):
			io.WriteString(w, anthropicHi)
		case strings.Contains(r.URL.Path, "/responses"):
			io.WriteString(w, responsesHi)
		default:
			io.WriteString(w, chatHi)
		}
	}))
	t.Cleanup(up.Close)
	save := func(p provider.Provider) {
		t.Helper()
		if err := provider.Save(p); err != nil {
			t.Fatal(err)
		}
	}
	base := provider.Provider{ID: "wire", Name: "Wire", Key: "k", Models: []string{"m1", "gemini-4-hydrogen"}, Catalog: "google"}
	loadGeminiXHighCatalog(t)

	base.Chat = up.URL + "/v1"
	base.ExtraBody = provider.ExtraBody{"!reasoning_effort": json.RawMessage(`"low"`)}
	save(base)
	s := New()
	post := func(path, body string) {
		t.Helper()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", path, strings.NewReader(body)))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", path, rec.Code, rec.Body.String())
		}
	}
	post("/v1/chat/completions", `{"model":"wire/m1","stream":true,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}`)
	if got := tryEffort(lastRoute(s)); got != "low" {
		t.Fatalf("chat extra logged %q, want low", got)
	}

	base.ExtraBody = nil
	base.Chat = ""
	base.Anthropic = up.URL
	save(base)
	post("/v1/messages", `{"model":"wire/m1","max_tokens":64,"stream":true,"thinking":{"type":"disabled"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`)
	if got := tryEffort(lastRoute(s)); got != "none" {
		t.Fatalf("disabled thinking logged %q, want none", got)
	}
	base.ExtraBody = provider.ExtraBody{
		"!thinking":      json.RawMessage(`{"type":"adaptive"}`),
		"!output_config": json.RawMessage(`{"effort":"high"}`),
	}
	save(base)
	post("/v1/messages", `{"model":"wire/m1","max_tokens":64,"stream":true,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`)
	if got := tryEffort(lastRoute(s)); got != "high" {
		t.Fatalf("thinking turned back on logged %q, want high", got)
	}

	base.ExtraBody = provider.ExtraBody{"!reasoning": json.RawMessage(`{"effort":"low"}`)}
	base.Anthropic = ""
	base.Responses = up.URL + "/v1"
	save(base)
	post("/v1/responses", `{"model":"wire/m1","stream":true,"reasoning":{"effort":"high"},"input":"hi"}`)
	if got := tryEffort(lastRoute(s)); got != "low" {
		t.Fatalf("responses extra logged %q, want low", got)
	}

	base.Responses = ""
	base.Gemini = up.URL + "/v1beta"
	base.ExtraBody = provider.ExtraBody{"!generationConfig": json.RawMessage(`{"thinkingConfig":{"thinking_level":"low","includeThoughts":true}}`)}
	save(base)
	post("/v1/chat/completions", `{"model":"wire/gemini-4-hydrogen","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
	mu.Lock()
	sent := string(last)
	mu.Unlock()
	if lv := gjson.Get(sent, "generationConfig.thinkingConfig.thinking_level").String(); lv != "low" {
		t.Fatalf("snake_case level %q in %s", lv, sent)
	}
	if got := tryEffort(lastRoute(s)); got != "low" {
		t.Fatalf("snake_case extra logged %q, want low", got)
	}
}

// A Gemini plugin is told the effort in a header. The thinkingLevel the
// body can carry does not replace minimal or xhigh unless the final body
// names another level.
func TestPluginGeminiEffortKeepsTheHeader(t *testing.T) {
	body := []byte(`{"generationConfig":{"thinkingConfig":{"thinkingLevel":"low","includeThoughts":true}}}`)
	for _, effort := range []string{"minimal", "xhigh"} {
		ctx := withWireAt(t.Context(), &wireAt{})
		noteWireEffort(ctx, provider.Gemini, &Request{Effort: effort, Thinking: true}, body)
		noteGeminiPluginEffort(ctx, effort, body)
		noteWireBody(ctx, provider.Gemini, body)
		w, _ := ctx.Value(wireAtKey{}).(*wireAt)
		if w == nil || w.effort != effort {
			t.Fatalf("%s logged %q", effort, wireEffortOf(w))
		}
		noteWireBody(ctx, provider.Gemini, []byte(`{"generationConfig":{"thinkingConfig":{"thinkingLevel":"high"}}}`))
		if w.effort != "high" {
			t.Fatalf("%s after extra logged %q, want high", effort, w.effort)
		}
	}
}
