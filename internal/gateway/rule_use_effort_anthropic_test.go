package gateway

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/settings"
	"github.com/yetone/magpie/internal/testenv"
)

// #7389: a rule's use-effort is the level the member is asked for, on an
// Anthropic request that has no thinking as well as on one that has it.
// output_config.effort (top-level or on a message) must not outrank it,
// and a budget cut down to max_tokens-1 must not be read back as a lower
// level. The same level is what a Chat, Responses or Gemini member is
// sent when the client spoke Anthropic.

func TestRuleUseEffortWithoutThinking(t *testing.T) {
	msg := quote(long(2000))
	for _, c := range []struct {
		name, model, use string
		levels           []string
		body             string // after max_tokens, before messages
		wantThinking     string // enabled, adaptive, disabled, or ""
		wantEffort       string // output_config.effort, "" if absent
		wantBudget       int    // 0 if no budget
		wantRoute        string
	}{
		{
			name: "adaptive, effort and no thinking", model: "claude-opus-5-5", use: "xhigh",
			levels:       []string{"low", "medium", "high", "xhigh", "max"},
			body:         `"output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"object"}}},`,
			wantThinking: "adaptive", wantEffort: "xhigh", wantRoute: "xhigh",
		},
		{
			name: "adaptive, no reasoning fields", model: "claude-opus-5-5", use: "xhigh",
			levels:       []string{"low", "medium", "high", "xhigh", "max"},
			wantThinking: "adaptive", wantEffort: "xhigh", wantRoute: "xhigh",
		},
		{
			name: "adaptive, message effort and no thinking", model: "claude-opus-5-5", use: "low",
			levels:       []string{"low", "medium", "high", "xhigh", "max"},
			body:         `"messages":[{"role":"user","content":` + msg + `},{"role":"system","content":"","output_config":{"effort":"high"}}]`,
			wantThinking: "adaptive", wantEffort: "low", wantRoute: "low",
		},
		{
			name: "adaptive, already thinking", model: "claude-opus-5-5", use: "low",
			levels:       []string{"low", "medium", "high", "xhigh", "max"},
			body:         `"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},`,
			wantThinking: "adaptive", wantEffort: "low", wantRoute: "low",
		},
		{
			name: "adaptive max", model: "claude-opus-5-5", use: "max",
			levels:       []string{"low", "medium", "high", "xhigh", "max"},
			wantThinking: "adaptive", wantEffort: "max", wantRoute: "max",
		},
		{
			name: "adaptive medium", model: "claude-opus-5-5", use: "medium",
			levels:       []string{"low", "medium", "high", "xhigh", "max"},
			wantThinking: "adaptive", wantEffort: "medium", wantRoute: "medium",
		},
		{
			name: "budget claude, effort and no thinking", model: "claude-sonnet-4-5", use: "low",
			levels:       []string{"low", "medium", "high"},
			body:         `"output_config":{"effort":"high"},`,
			wantThinking: "enabled", wantEffort: "low", wantBudget: budgetOf("low"), wantRoute: "low",
		},
		{
			name: "budget claude, no reasoning fields, small cap", model: "claude-sonnet-4-5", use: "high",
			levels:       []string{"low", "medium", "high"},
			wantThinking: "enabled", wantBudget: budgetOf("high"), wantRoute: "high",
		},
		{
			name: "glm reads the rule level, not the cut budget", model: "glm-5.3", use: "max",
			levels:       []string{"low", "high", "max"},
			wantThinking: "enabled", wantEffort: "max", wantRoute: "max",
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			s, up := anthropicRuleMember(t, c.model, c.levels, c.use)
			messages := `"messages":[{"role":"user","content":` + msg + `}]`
			if strings.Contains(c.body, `"messages"`) {
				messages = ""
			}
			body := `{"model":"group/r","max_tokens":8192,` + c.body + messages + `}`
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			sent := up.last()
			th, _ := sent["thinking"].(map[string]any)
			if c.wantThinking == "" {
				if th != nil {
					t.Fatalf("thinking %v", th)
				}
			} else if th["type"] != c.wantThinking {
				t.Fatalf("thinking %v, want type %s", th, c.wantThinking)
			}
			oc, _ := sent["output_config"].(map[string]any)
			if c.wantEffort == "" {
				if oc["effort"] != nil {
					t.Fatalf("output_config %v, want no effort", oc)
				}
			} else if oc["effort"] != c.wantEffort {
				t.Fatalf("output_config %v, want effort %s", oc, c.wantEffort)
			}
			if c.wantBudget != 0 && th["budget_tokens"] != float64(c.wantBudget) {
				t.Fatalf("budget %v, want %d", th["budget_tokens"], c.wantBudget)
			}
			if c.body != "" && strings.Contains(c.body, `"format"`) {
				format, _ := oc["format"].(map[string]any)
				if format["type"] != "json_schema" {
					t.Fatalf("format dropped: %v", oc)
				}
			}
			if strings.Contains(c.body, `"role":"system"`) {
				raw := gjson.GetBytes(up.raw(), "messages.#.output_config.effort").String()
				if !strings.Contains(raw, c.wantEffort) || strings.Contains(raw, "high") {
					t.Fatalf("message efforts %s, want %s", raw, c.wantEffort)
				}
			}
			r := lastRoute(s)
			if len(r.Tries) != 1 || r.Tries[0].Effort != c.wantRoute || r.Tries[0].RuleEffort != c.wantRoute {
				t.Fatalf("try %+v, want effort %q", r.Tries, c.wantRoute)
			}
		})
	}
}

// The level is the one a member of another protocol is asked for when the
// client spoke Anthropic and named no thinking.
func TestRuleUseEffortCrossProtocol(t *testing.T) {
	msg := quote(long(2000))
	levels := []string{"none", "low", "medium", "high", "xhigh", "max"}
	for _, c := range []struct {
		name, proto, model, use, extra, want string
	}{
		{"chat, request effort", "chat", "m", "low", `"output_config":{"effort":"high"},`, `"reasoning_effort":"low"`},
		{"chat, no reasoning fields", "chat", "m", "xhigh", "", `"reasoning_effort":"xhigh"`},
		{"chat, message effort", "chat", "m", "low", `"messages":[{"role":"user","content":` + msg + `},{"role":"system","content":"note","output_config":{"effort":"max"}}]`, `"reasoning_effort":"low"`},
		{"responses, request effort", "responses", "m", "low", `"output_config":{"effort":"high"},`, `"effort":"low"`},
		{"responses, no reasoning fields", "responses", "m", "high", "", `"effort":"high"`},
		{"gemini, request effort", "gemini", "gemini-3-flash", "low", `"output_config":{"effort":"high"},`, `"thinkingLevel":"low"`},
		{"gemini, no reasoning fields", "gemini", "gemini-3-flash", "high", "", `"thinkingLevel":"high"`},
	} {
		t.Run(c.name, func(t *testing.T) {
			s, up := otherRuleMember(t, c.proto, c.model, levels, c.use)
			messages := `"messages":[{"role":"user","content":` + msg + `}]`
			extra := c.extra
			if strings.Contains(c.extra, `"messages"`) {
				messages, extra = c.extra, ""
			}
			body := `{"model":"group/r","max_tokens":8192,` + extra + messages + `}`
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			if !strings.Contains(string(up.raw()), c.want) {
				t.Fatalf("sent %s, want %s", up.raw(), c.want)
			}
			r := lastRoute(s)
			wantRoute := strings.Trim(c.want[strings.LastIndex(c.want, `:`)+1:], `"`)
			if len(r.Tries) != 1 || r.Tries[0].Effort != wantRoute {
				t.Fatalf("try %+v, want %s", r.Tries, wantRoute)
			}
		})
	}
}

// capUp records the last request body and answers each protocol with a
// short completed reply.
type capUp struct {
	mu    sync.Mutex
	body  []byte
	proto string
}

func (u *capUp) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := io.ReadAll(r.Body)
	u.mu.Lock()
	u.body = append([]byte(nil), b...)
	u.mu.Unlock()
	switch u.proto {
	case "chat":
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	case "responses":
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, responsesHi)
	case "gemini":
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, vertexPong)
	default:
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}
}

func (u *capUp) raw() []byte {
	u.mu.Lock()
	defer u.mu.Unlock()
	return append([]byte(nil), u.body...)
}

func (u *capUp) last() map[string]any {
	var v map[string]any
	_ = decodeJSON(u.raw(), &v)
	return v
}

func anthropicRuleMember(t *testing.T, model string, levels []string, use string) (*Server, *capUp) {
	t.Helper()
	fresh(t)
	up := &capUp{proto: "anthropic"}
	srv := httptest.NewServer(up)
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{model}}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/"+model, levels); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "R", Members: []string{"relay/" + model}, Routing: provider.Ordered,
		Rules: []provider.Rule{{Use: "relay/" + model, Tokens: 1000, UseEffort: use}},
	}); err != nil {
		t.Fatal(err)
	}
	return New(), up
}

func otherRuleMember(t *testing.T, proto, model string, levels []string, use string) (*Server, *capUp) {
	t.Helper()
	fresh(t)
	up := &capUp{proto: proto}
	srv := httptest.NewServer(up)
	t.Cleanup(srv.Close)
	p := provider.Provider{ID: "relay", Name: "Relay", Key: "k", Models: []string{model}}
	switch proto {
	case "chat":
		p.Chat = srv.URL + "/v1"
	case "responses":
		p.Responses = srv.URL + "/v1"
	case "gemini":
		p.Gemini = srv.URL + "/v1beta"
	default:
		t.Fatalf("proto %s", proto)
	}
	if err := provider.Save(p); err != nil {
		t.Fatal(err)
	}
	id := "relay/" + model
	if err := provider.SetModelEfforts(id, levels); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "R", Members: []string{id}, Routing: provider.Ordered,
		Rules: []provider.Rule{{Use: id, Tokens: 1000, UseEffort: use}},
	}); err != nil {
		t.Fatal(err)
	}
	return New(), up
}

// A member fixed at none is asked with thinking off, and the request's
// own output_config.effort does not stay behind to turn it back on.
// use-effort itself is only low…max; none is a member's fixed level.
func TestFixedNoneDropsAnthropicEffort(t *testing.T) {
	fresh(t)
	up := &capUp{proto: "anthropic"}
	srv := httptest.NewServer(up)
	t.Cleanup(srv.Close)
	model := "claude-sonnet-4-5"
	if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{model}}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/"+model, []string{"none", "low", "medium", "high"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{Name: "R", Members: []string{"relay/" + model + ":none"}, Routing: provider.Ordered}); err != nil {
		t.Fatal(err)
	}
	s := New()
	body := `{"model":"group/r","max_tokens":8192,"output_config":{"effort":"high"},"messages":[{"role":"user","content":` + quote(long(2000)) + `},{"role":"system","content":"note","output_config":{"effort":"xhigh"}}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	sent := up.last()
	th, _ := sent["thinking"].(map[string]any)
	if th["type"] != "disabled" {
		t.Fatalf("thinking %v", th)
	}
	if oc, _ := sent["output_config"].(map[string]any); oc["effort"] != nil {
		t.Fatalf("output_config %v", oc)
	}
	if e := gjson.GetBytes(up.raw(), "messages.#.output_config.effort").String(); strings.Contains(e, "high") || strings.Contains(e, "xhigh") {
		t.Fatalf("message efforts %s", e)
	}
	if r := lastRoute(s); len(r.Tries) != 1 || r.Tries[0].Effort != "" && r.Tries[0].Effort != "none" {
		t.Fatalf("try %+v", r.Tries)
	}
}

// #7389 F1: raising max_tokens for a budget Claude's rule effort must not
// exceed the member's configured Max output or defeat context-overflow retry.
func TestRuleUseEffortRespectsMaxOutputAndRoomRetry(t *testing.T) {
	msg := quote(long(2000))
	t.Run("max output cap", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-sonnet-4-5", []string{"low", "medium", "high"}, "high")
		if err := provider.SetModelOutput("relay/claude-sonnet-4-5", 8192); err != nil {
			t.Fatal(err)
		}
		body := `{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		if got := gjson.GetBytes(up.raw(), "max_tokens").Int(); got != 8192 {
			t.Fatalf("max_tokens = %d, want 8192 (body=%s)", got, up.raw())
		}
		if got := gjson.GetBytes(up.raw(), "thinking.budget_tokens").Int(); got != 8191 {
			t.Fatalf("budget_tokens = %d, want 8191 (body=%s)", got, up.raw())
		}
	})

	t.Run("context overflow retry caps raised max_tokens", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var seenMax []int64
		var seenBudget []int64
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			maxTok := gjson.GetBytes(b, "max_tokens").Int()
			budget := gjson.GetBytes(b, "thinking.budget_tokens").Int()
			mu.Lock()
			seenMax = append(seenMax, maxTok)
			seenBudget = append(seenBudget, budget)
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if maxTok > 8192 {
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"input length and max_tokens exceed context limit: 191808 + 32192 > 200000"}}`)
				return
			}
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"}}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/claude-sonnet-4-5"}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "high"}},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		body := `{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(seenMax) != 2 || seenMax[0] != 32192 || seenMax[1] != 8192 || seenBudget[1] != 8191 {
			t.Fatalf("seen max=%v budget=%v, want [32192 8192] and retry budget 8191", seenMax, seenBudget)
		}
	})
}

// #7389 F2: a Chat-only Claude member preserves max (distinct from xhigh) and
// translated Chat members do not inflate max_tokens for Anthropic budgets.
func TestRuleUseEffortCrossProtocolPreservesMaxAndMaxTokens(t *testing.T) {
	msg := quote(long(2000))
	t.Run("chat-only claude-sonnet-4-5 keeps max and max_tokens", func(t *testing.T) {
		s, up := otherRuleMember(t, "chat", "claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}, "max")
		body := `{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "max" {
			t.Fatalf("reasoning_effort = %q, want max (body=%s)", got, raw)
		}
		if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 8192 {
			t.Fatalf("max_tokens = %d, want 8192 (body=%s)", got, raw)
		}
	})

	t.Run("chat member does not inflate small max_tokens", func(t *testing.T) {
		s, up := otherRuleMember(t, "chat", "m", []string{"low", "medium", "high"}, "low")
		body := `{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "low" {
			t.Fatalf("reasoning_effort = %q, want low (body=%s)", got, raw)
		}
		if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
			t.Fatalf("max_tokens = %d, want 100 (body=%s)", got, raw)
		}
	})
}

// #7389 F3: rewriting effort preserves exact large integers in output_config.format.schema.
func TestRuleUseEffortPreservesOutputConfigSchemaNumbers(t *testing.T) {
	msg := quote(long(2000))
	s, up := anthropicRuleMember(t, "glm-5.3", []string{"low", "medium", "high"}, "low")
	body := `{"model":"group/r","max_tokens":8192,"thinking":{"type":"enabled","budget_tokens":4096},"output_config":{"format":{"type":"json_schema","schema":{"type":"integer","const":9007199254740993}}},"messages":[{"role":"user","content":` + msg + `}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	raw := up.raw()
	if got := gjson.GetBytes(raw, "output_config.format.schema.const").Raw; got != "9007199254740993" {
		t.Fatalf("schema.const = %s, want 9007199254740993 (body=%s)", got, raw)
	}
	if got := gjson.GetBytes(raw, "output_config.effort").String(); got != "low" {
		t.Fatalf("output_config.effort = %q, want low (body=%s)", got, raw)
	}
}

// #7389 R2 F4: first fallback from Messages to Chat on a multi-protocol relay
// preserves use-effort=max and does not inflate max_tokens with an Anthropic budget.
func TestRuleUseEffortFirstMessagesToChatFallbackPreservesMaxAndMaxTokens(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	var chatBodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/v1/messages") {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"Model does not support this protocol."}}`)
			return
		}
		mu.Lock()
		chatBodies = append(chatBodies, string(b))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k",
		Anthropic: srv.URL, Chat: srv.URL + "/v1",
		Models: []string{"claude-sonnet-4-5"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "R", Members: []string{"relay/claude-sonnet-4-5"}, Routing: provider.Ordered,
		Rules: []provider.Rule{{Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "max"}},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	body := `{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
	for attempt := range 2 {
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("attempt %d: %d %s", attempt+1, rec.Code, rec.Body)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(chatBodies) != 2 {
		t.Fatalf("chat requests = %d, want 2", len(chatBodies))
	}
	for i, raw := range chatBodies {
		if got := gjson.Get(raw, "reasoning_effort").String(); got != "max" {
			t.Fatalf("chat #%d reasoning_effort = %q, want max (body=%s)", i+1, got, raw)
		}
		if got := gjson.Get(raw, "max_tokens").Int(); got != 100 {
			t.Fatalf("chat #%d max_tokens = %d, want 100 (body=%s)", i+1, got, raw)
		}
	}
}

// #7389 R2 F5: native thinking-needed retry must not bypass the configured Max output.
func TestRuleUseEffortThinkingNeededRetryRespectsMaxOutput(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	var seenMax []int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		seenMax = append(seenMax, gjson.GetBytes(b, "max_tokens").Int())
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"thinking.type must be enabled"}}`)
	}))
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"}}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelOutput("relay/claude-sonnet-4-5", 100); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "R", Members: []string{"relay/claude-sonnet-4-5"}, Routing: provider.Ordered,
		Rules: []provider.Rule{{Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "low"}},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	body := `{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	mu.Lock()
	defer mu.Unlock()
	for i, n := range seenMax {
		if n > 100 {
			t.Fatalf("attempt %d sent max_tokens=%d exceeding Max output 100 (seen=%v)", i+1, n, seenMax)
		}
	}
}

// #7389 R2 F6: native Anthropic request with forced tool_choice and no thinking
// is not given thinking by use-effort (which Anthropic rejects with 400).
func TestRuleUseEffortSkipsThinkingOnForcedToolChoice(t *testing.T) {
	for _, tc := range []string{`{"type":"tool","name":"lookup"}`, `{"type":"any"}`} {
		t.Run(tc, func(t *testing.T) {
			s, up := anthropicRuleMember(t, "claude-sonnet-4-5", []string{"low", "medium", "high"}, "high")
			body := `{"model":"group/r","max_tokens":100,"tool_choice":` + tc + `,"tools":[{"name":"lookup","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			raw := up.raw()
			if gjson.GetBytes(raw, "thinking").Exists() {
				t.Fatalf("unexpected thinking added with forced tool_choice: %s", raw)
			}
			if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
				t.Fatalf("max_tokens = %d, want 100 (body=%s)", got, raw)
			}
		})
	}
}

// #7389 R2 F7: Claude subscription bridge and WithSideEffort preserve max
// (instead of lossy budget -> xhigh) on claude-opus-4-5.
func TestRuleUseEffortAndSideEffortClaudeSubscriptionMax(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script stands in for Claude Code")
	}
	claudeMadeFirst(t, false)
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args.txt")
	script := `#!/bin/sh
printf '%s\n' "$@" > "` + argsFile + `"
while read -r line; do
  echo '{"type":"stream_event","event":{"type":"message_start","message":{"id":"m","model":"claude-opus-4-5","usage":{"input_tokens":1}}}}'
  echo '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}}'
  echo '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}}'
  echo '{"type":"stream_event","event":{"type":"message_stop"}}'
  echo '{"type":"result","subtype":"success","is_error":false,"result":""}'
done
`
	testenv.Program(t, filepath.Join(dir, "claude"), script)
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	runsEndWithTest(t, dir)

	if err := provider.SetModelEfforts("claude/claude-opus-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "R", Members: []string{"claude/claude-opus-4-5"}, Routing: provider.Ordered,
		Rules: []provider.Rule{{Use: "claude/claude-opus-4-5", Tokens: 1000, UseEffort: "max"}},
	}); err != nil {
		t.Fatal(err)
	}

	// Verify WithSideEffort preserves exact "max" when parsed back.
	sideBody := WithSideEffort([]byte(`{"model":"claude/claude-opus-4-5","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`), "claude/claude-opus-4-5", "max")
	parsed, err := parseAnthropic(sideBody)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Effort != "max" {
		t.Fatalf("WithSideEffort parsed effort = %q, want max (body=%s)", parsed.Effort, sideBody)
	}

	// Verify Claude subscription bridge receives --effort max via routing rule.
	s := New()
	t.Cleanup(s.subscription.abortAll)
	body := `{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	rawArgs, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.ReplaceAll(string(rawArgs), "\n", " ")
	if !strings.Contains(args, "--effort max ") {
		t.Fatalf("claude CLI args = %q, want --effort max", args)
	}
}

// #7389 R3 F1: fitting a native effort to the model's own word (low ->
// minimal) rewrites output_config and must keep large integers in the schema.
func TestRuleUseEffortFitPreservesSchemaIntegers(t *testing.T) {
	msg := quote(long(2000))
	s, up := anthropicRuleMember(t, "vendor-m", []string{"minimal", "high"}, "low")
	body := `{"model":"group/r","max_tokens":8192,"output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"integer","const":9007199254740993}}},"messages":[{"role":"user","content":` + msg + `}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	raw := up.raw()
	if got := gjson.GetBytes(raw, "output_config.format.schema.const").Raw; got != "9007199254740993" {
		t.Fatalf("schema.const = %s, want 9007199254740993 (body=%s)", got, raw)
	}
	if got := gjson.GetBytes(raw, "output_config.effort").String(); got != "minimal" {
		t.Fatalf("output_config.effort = %q, want minimal (body=%s)", got, raw)
	}
}

// #7389 R3 F2: the effort wire shape follows the upstream model name, not the
// local alias. review-sonnet sent as claude-sonnet-4-5 is a budget Claude.
func TestRuleUseEffortWireShapeUsesUpstreamName(t *testing.T) {
	fresh(t)
	up := &capUp{proto: "anthropic"}
	srv := httptest.NewServer(up)
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"review-sonnet"}}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("relay/review-sonnet", "claude-sonnet-4-5"); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/review-sonnet", []string{"low", "medium", "high"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "R", Members: []string{"relay/review-sonnet"}, Routing: provider.Ordered,
		Rules: []provider.Rule{{Use: "relay/review-sonnet", Tokens: 1000, UseEffort: "high"}},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	body := `{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	raw := up.raw()
	if got := gjson.GetBytes(raw, "model").String(); got != "claude-sonnet-4-5" {
		t.Fatalf("model = %q, want claude-sonnet-4-5 (body=%s)", got, raw)
	}
	if gjson.GetBytes(raw, "output_config.effort").Exists() {
		t.Fatalf("output_config.effort = %s, Sonnet 4.5 is asked by a budget (body=%s)", gjson.GetBytes(raw, "output_config.effort").Raw, raw)
	}
	if got := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); got != int64(budgetOf("high")) {
		t.Fatalf("budget_tokens = %d, want %d (body=%s)", got, budgetOf("high"), raw)
	}
	if got := gjson.GetBytes(raw, "max_tokens").Int(); got != int64(budgetOf("high")+8192) {
		t.Fatalf("max_tokens = %d, want %d (body=%s)", got, budgetOf("high")+8192, raw)
	}
}

// #7389 R3 F3: enabling thinking for a rule effort drops temperature, top_p
// and top_k, which Anthropic rejects beside thinking. A request that already
// thinks keeps the sampling it sent.
func TestRuleUseEffortDropsSamplingWhenEnablingThinking(t *testing.T) {
	msg := quote(long(2000))
	t.Run("omitted thinking", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-sonnet-4-5", []string{"low", "medium", "high"}, "high")
		body := `{"model":"group/r","max_tokens":100,"temperature":0,"top_p":0.9,"top_k":40,"messages":[{"role":"user","content":` + msg + `}]}`
		req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
		req.Header.Set("anthropic-version", "2023-06-01")
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("thinking %s, want enabled (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if got := gjson.GetBytes(raw, "max_tokens").Int(); got != int64(budgetOf("high")+100) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", got, budgetOf("high")+100, raw)
		}
		for _, key := range []string{"temperature", "top_p", "top_k"} {
			if gjson.GetBytes(raw, key).Exists() {
				t.Fatalf("%s kept beside enabled thinking: %s", key, raw)
			}
		}
	})
	t.Run("already thinking", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-sonnet-4-5", []string{"low", "medium", "high"}, "high")
		body := `{"model":"group/r","max_tokens":8192,"temperature":0,"thinking":{"type":"enabled","budget_tokens":4096},"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if got := gjson.GetBytes(raw, "temperature").Raw; got != "0" {
			t.Fatalf("temperature = %s, want 0 kept (body=%s)", got, raw)
		}
	})
}

// #7389 R3 F4: a JSON unicode escape of tool_choice is still a forced choice,
// so use-effort must not enable thinking beside it.
func TestRuleUseEffortSeesEscapedToolChoice(t *testing.T) {
	msg := quote(long(2000))
	for _, tc := range []string{`{"type":"any"}`, `{"type":"tool","name":"lookup"}`} {
		t.Run(tc, func(t *testing.T) {
			s, up := anthropicRuleMember(t, "claude-sonnet-4-5", []string{"low", "medium", "high"}, "high")
			body := `{"model":"group/r","max_tokens":100,"tool\u005fchoice":` + tc + `,"tools":[{"name":"lookup","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":` + msg + `}]}`
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			raw := up.raw()
			if gjson.GetBytes(raw, "thinking").Exists() {
				t.Fatalf("thinking added beside escaped tool_choice: %s", raw)
			}
			if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
				t.Fatalf("max_tokens = %d, want 100 (body=%s)", got, raw)
			}
			if got := gjson.GetBytes(raw, "tool_choice.type").String(); got != gjson.Get(tc, "type").String() {
				t.Fatalf("tool_choice.type = %q, want %s (body=%s)", got, gjson.Get(tc, "type").String(), raw)
			}
		})
	}
	t.Run("auto still thinks", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-sonnet-4-5", []string{"low", "medium", "high"}, "high")
		body := `{"model":"group/r","max_tokens":100,"tool\u005fchoice":{"type":"auto"},"tools":[{"name":"lookup","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("auto tool_choice left thinking off: %s", raw)
		}
	})
}

// #7389 R3 F5: a cached thinking-needed mark must not enable thinking on a
// later request whose tool_choice forces a call.
func TestThinkingNeededCacheSkipsForcedToolChoice(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	var bodies [][]byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, append([]byte(nil), b...))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		forced := gjson.GetBytes(b, "tool_choice.type").String()
		thinking := gjson.GetBytes(b, "thinking.type").String()
		if forced == "any" || forced == "tool" {
			if thinking == "enabled" || thinking == "adaptive" {
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"thinking is not supported when tool_choice forces a call"}}`)
				return
			}
		} else if thinking != "enabled" && thinking != "adaptive" {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"thinking.type must be enabled"}}`)
			return
		}
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"}}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "R", Members: []string{"relay/claude-sonnet-4-5"}, Routing: provider.Ordered,
		Rules: []provider.Rule{{Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "high"}},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	ordinary := `{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(ordinary)))
	if rec.Code != 200 {
		t.Fatalf("ordinary: %d %s", rec.Code, rec.Body)
	}
	ruled := `{"model":"group/r","max_tokens":100,"temperature":0,"tool_choice":{"type":"any"},"tools":[{"name":"lookup","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(ruled)))
	if rec.Code != 200 {
		t.Fatalf("ruled: %d %s", rec.Code, rec.Body)
	}
	mu.Lock()
	defer mu.Unlock()
	var last []byte
	for _, b := range bodies {
		if gjson.GetBytes(b, "tool_choice.type").String() == "any" {
			last = b
		}
	}
	if last == nil {
		t.Fatal("no request carried tool_choice any")
	}
	if gjson.GetBytes(last, "thinking").Exists() {
		t.Fatalf("cached thinking-needed enabled thinking beside forced tool_choice: %s", last)
	}
	if got := gjson.GetBytes(last, "max_tokens").Int(); got != 100 {
		t.Fatalf("max_tokens = %d, want 100 (body=%s)", got, last)
	}
	if got := gjson.GetBytes(last, "temperature").Raw; got != "0" {
		t.Fatalf("temperature = %s, want 0 kept when thinking is not added (body=%s)", got, last)
	}
}

// #7389 R3 F6: WithSideEffort follows a model pinned to Chat, so max stays
// max and max_tokens is not raised by an Anthropic budget.
func TestWithSideEffortRespectsChatAPIPin(t *testing.T) {
	fresh(t)
	up := &capUp{proto: "chat"}
	srv := httptest.NewServer(up)
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k",
		Anthropic: srv.URL, Chat: srv.URL + "/v1",
		Models: []string{"claude-sonnet-4-5"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelAPI("relay/claude-sonnet-4-5", "chat"); err != nil {
		t.Fatal(err)
	}
	body := WithSideEffort([]byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`), "relay/claude-sonnet-4-5", "max")
	s := New()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(string(body)))
	req.Header.Set("anthropic-version", "2023-06-01")
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	raw := up.raw()
	if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "max" {
		t.Fatalf("reasoning_effort = %q, want max (body=%s)", got, raw)
	}
	if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
		t.Fatalf("max_tokens = %d, want 100 (body=%s)", got, raw)
	}
}

// #7389 R4 F7: an upstream that only accepts thinking.type "enabled" is
// asked that way by a use-effort rule, both after the mark is cached and
// on the first refusal, when the rule has already written adaptive.
func TestRuleUseEffortThinkingNeededStaysEnabled(t *testing.T) {
	const model = "claude-opus-5-5"
	levels := []string{"low", "medium", "high", "xhigh", "max"}
	newRelay := func(t *testing.T) (*Server, *[][]byte, *sync.Mutex) {
		t.Helper()
		fresh(t)
		var mu sync.Mutex
		var bodies [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			bodies = append(bodies, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if gjson.GetBytes(b, "thinking.type").String() != "enabled" {
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"thinking.type must be \"enabled\""}}`)
				return
			}
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"`+model+`","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{model}}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/"+model, levels); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/" + model}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/" + model, Tokens: 1000, UseEffort: "xhigh"}},
		}); err != nil {
			t.Fatal(err)
		}
		return New(), &bodies, &mu
	}
	ruled := `{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
	send := func(t *testing.T, s *Server, body string) {
		t.Helper()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
	}

	t.Run("cached mark", func(t *testing.T) {
		s, bodies, mu := newRelay(t)
		send(t, s, `{"model":"relay/`+model+`","max_tokens":200,"messages":[{"role":"user","content":"hi"}]}`)
		mu.Lock()
		marked := len(*bodies)
		mu.Unlock()
		if marked < 2 {
			t.Fatalf("direct request made %d upstream calls, want a refusal then enabled", marked)
		}
		send(t, s, ruled)
		mu.Lock()
		defer mu.Unlock()
		if len(*bodies) != marked+1 {
			t.Fatalf("ruled request made %d upstream calls, want 1 already enabled (bodies=%d)", len(*bodies)-marked, len(*bodies))
		}
		got := (*bodies)[marked]
		if typ := gjson.GetBytes(got, "thinking.type").String(); typ != "enabled" {
			t.Fatalf("thinking.type = %q, want enabled (body=%s)", typ, got)
		}
		if n := gjson.GetBytes(got, "thinking.budget_tokens").Int(); n != int64(budgetOf("xhigh")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("xhigh"), got)
		}
		if n := gjson.GetBytes(got, "max_tokens").Int(); n <= int64(budgetOf("xhigh")) {
			t.Fatalf("max_tokens = %d, want room above the budget (body=%s)", n, got)
		}
		if e := gjson.GetBytes(got, "output_config.effort").String(); e != "xhigh" {
			t.Fatalf("output_config.effort = %q, want xhigh (body=%s)", e, got)
		}
	})

	t.Run("first refusal converts adaptive", func(t *testing.T) {
		s, bodies, mu := newRelay(t)
		send(t, s, ruled)
		mu.Lock()
		if len(*bodies) != 2 {
			mu.Unlock()
			t.Fatalf("upstream calls = %d, want adaptive then enabled", len(*bodies))
		}
		first, second := (*bodies)[0], (*bodies)[1]
		mu.Unlock()
		if typ := gjson.GetBytes(first, "thinking.type").String(); typ != "adaptive" {
			t.Fatalf("first thinking.type = %q, want adaptive (body=%s)", typ, first)
		}
		if gjson.GetBytes(first, "thinking.display").String() != "summarized" {
			t.Fatalf("first display = %s, want summarized when adaptive is newly asked", gjson.GetBytes(first, "thinking").Raw)
		}
		if typ := gjson.GetBytes(second, "thinking.type").String(); typ != "enabled" {
			t.Fatalf("retry thinking.type = %q, want enabled (body=%s)", typ, second)
		}
		if n := gjson.GetBytes(second, "thinking.budget_tokens").Int(); n != int64(budgetOf("xhigh")) {
			t.Fatalf("retry budget_tokens = %d, want %d (body=%s)", n, budgetOf("xhigh"), second)
		}
		send(t, s, ruled)
		mu.Lock()
		defer mu.Unlock()
		if len(*bodies) != 3 {
			t.Fatalf("second request made %d more calls, want 1", len(*bodies)-2)
		}
		if typ := gjson.GetBytes((*bodies)[2], "thinking.type").String(); typ != "enabled" {
			t.Fatalf("cached thinking.type = %q, want enabled", typ)
		}
	})
}

// #7389 R4 F8: a Max output too small for a thinking budget turns thinking
// back off, and the sampling the rule would have dropped for thinking stays.
func TestRuleUseEffortLowMaxOutputKeepsSampling(t *testing.T) {
	msg := quote(long(2000))
	t.Run("max output disables thinking", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-sonnet-4-5", []string{"low", "medium", "high"}, "high")
		if err := provider.SetModelOutput("relay/claude-sonnet-4-5", 100); err != nil {
			t.Fatal(err)
		}
		body := `{"model":"group/r","max_tokens":8192,"temperature":0.2,"top_p":0.9,"top_k":40,"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if typ := gjson.GetBytes(raw, "thinking.type").String(); typ == "enabled" || typ == "adaptive" {
			t.Fatalf("thinking.type = %q, want disabled under Max output 100 (body=%s)", typ, raw)
		}
		if got := gjson.GetBytes(raw, "temperature").Raw; got != "0.2" {
			t.Fatalf("temperature = %s, want 0.2 (body=%s)", got, raw)
		}
		if got := gjson.GetBytes(raw, "top_p").Raw; got != "0.9" {
			t.Fatalf("top_p = %s, want 0.9 (body=%s)", got, raw)
		}
		if got := gjson.GetBytes(raw, "top_k").Int(); got != 40 {
			t.Fatalf("top_k = %d, want 40 (body=%s)", got, raw)
		}
	})
	t.Run("room for thinking still drops sampling", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-sonnet-4-5", []string{"low", "medium", "high"}, "high")
		body := `{"model":"group/r","max_tokens":8192,"temperature":0.2,"top_p":0.9,"top_k":40,"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("thinking %s, want enabled (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		for _, key := range []string{"temperature", "top_p", "top_k"} {
			if gjson.GetBytes(raw, key).Exists() {
				t.Fatalf("%s kept beside enabled thinking: %s", key, raw)
			}
		}
	})
}

// #7389 R4 F9: a request that already asks adaptive thinking keeps that
// object, including Anthropic's default of no display. Display summarized
// is only the default when adaptive thinking is newly asked.
func TestRuleUseEffortPreservesAdaptiveThinking(t *testing.T) {
	msg := quote(long(2000))
	t.Run("already adaptive", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-opus-5-5", []string{"low", "medium", "high", "xhigh", "max"}, "low")
		body := `{"model":"group/r","max_tokens":8192,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" {
			t.Fatalf("thinking %s, want adaptive", gjson.GetBytes(raw, "thinking").Raw)
		}
		if gjson.GetBytes(raw, "thinking.display").Exists() {
			t.Fatalf("display injected into existing adaptive thinking: %s", gjson.GetBytes(raw, "thinking").Raw)
		}
		if got := gjson.GetBytes(raw, "output_config.effort").String(); got != "low" {
			t.Fatalf("output_config.effort = %q, want low", got)
		}
	})
	t.Run("enabled becomes adaptive with summarized", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-opus-5-5", []string{"low", "medium", "high", "xhigh", "max"}, "low")
		body := `{"model":"group/r","max_tokens":8192,"thinking":{"type":"enabled","budget_tokens":4096},"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" || gjson.GetBytes(raw, "thinking.display").String() != "summarized" {
			t.Fatalf("thinking %s, want adaptive summarized", gjson.GetBytes(raw, "thinking").Raw)
		}
		if gjson.GetBytes(raw, "thinking.budget_tokens").Exists() {
			t.Fatalf("budget kept on adaptive thinking: %s", gjson.GetBytes(raw, "thinking").Raw)
		}
	})
	t.Run("explicit display kept", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-opus-5-5", []string{"low", "medium", "high", "xhigh", "max"}, "high")
		body := `{"model":"group/r","max_tokens":8192,"thinking":{"type":"enabled","display":"omitted","budget_tokens":8000},"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" || gjson.GetBytes(raw, "thinking.display").String() != "omitted" {
			t.Fatalf("thinking %s, want adaptive display omitted", gjson.GetBytes(raw, "thinking").Raw)
		}
	})
}

// #7389 R4 F10: a budget Claude's effort is one decode and one encode, so a
// 2 MiB messages payload is not parsed into maps three times.
func TestWithAnthropicBudgetLargeBodyIsOnePass(t *testing.T) {
	payload := strings.Repeat("a", 2<<20)
	body := []byte(`{"model":"claude-sonnet-4-5","max_tokens":8192,"temperature":0.2,"top_p":0.9,"top_k":40,"output_config":{"effort":"low","format":{"type":"json_schema","schema":{"const":9007199254740993}}},"messages":[{"role":"user","content":"` + payload + `"}]}`)
	out := withAnthropicBudget(body, "xhigh")
	if got := gjson.GetBytes(out, "output_config.format.schema.const").Raw; got != "9007199254740993" {
		t.Fatalf("schema.const = %s, want 9007199254740993", got)
	}
	if gjson.GetBytes(out, "temperature").Exists() || gjson.GetBytes(out, "top_p").Exists() || gjson.GetBytes(out, "top_k").Exists() {
		t.Fatalf("sampling kept beside new thinking: %s", gjson.GetBytes(out, "temperature").Raw)
	}
	if got := gjson.GetBytes(out, "thinking.budget_tokens").Int(); got != int64(budgetOf("xhigh")) {
		t.Fatalf("budget_tokens = %d, want %d", got, budgetOf("xhigh"))
	}
	if got := gjson.GetBytes(out, "output_config.effort").String(); got != "xhigh" {
		t.Fatalf("output_config.effort = %q, want xhigh", got)
	}
	res := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		var sink []byte
		for i := 0; i < b.N; i++ {
			sink = withAnthropicBudget(body, "xhigh")
		}
		if len(sink) == 0 {
			b.Fatal("empty")
		}
	})
	t.Logf("withAnthropicBudget 2MiB: %d ns/op %d B/op %d allocs/op", res.NsPerOp(), res.AllocedBytesPerOp(), res.AllocsPerOp())
	if res.AllocedBytesPerOp() > 8<<20 {
		t.Fatalf("allocated %d bytes/op, want a single pass under 8MiB", res.AllocedBytesPerOp())
	}
}

func BenchmarkWithAnthropicBudgetLargeBody(b *testing.B) {
	payload := strings.Repeat("a", 2<<20)
	body := []byte(`{"model":"claude-sonnet-4-5","max_tokens":8192,"temperature":0.2,"top_p":0.9,"top_k":40,"messages":[{"role":"user","content":"` + payload + `"}]}`)
	b.ReportAllocs()
	b.ResetTimer()
	var sink []byte
	for i := 0; i < b.N; i++ {
		sink = withAnthropicBudget(body, "high")
	}
	if len(sink) == 0 {
		b.Fatal("empty")
	}
}

// #7389 R4 F11: WithSideEffort on an unpinned dual-protocol relay keeps a
// translated body for Chat, so a Messages refusal and the cached unfit
// both ask reasoning_effort max at the original max_tokens.
func TestWithSideEffortDualProtocolFallbackKeepsMax(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	var messages, chats [][]byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/v1/messages") {
			mu.Lock()
			messages = append(messages, append([]byte(nil), b...))
			mu.Unlock()
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"Model does not support this protocol."}}`)
			return
		}
		mu.Lock()
		chats = append(chats, append([]byte(nil), b...))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k",
		Anthropic: srv.URL, Chat: srv.URL + "/v1",
		Models: []string{"claude-sonnet-4-5"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	orig := []byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
	body := WithSideEffort(orig, "relay/claude-sonnet-4-5", "max")
	if !gjson.ValidBytes(body) {
		t.Fatalf("WithSideEffort returned invalid JSON: %s", body)
	}
	if gjson.GetBytes(body, "thinking.type").String() != "enabled" {
		t.Fatalf("native thinking %s, want enabled", gjson.GetBytes(body, "thinking").Raw)
	}
	if got := gjson.GetBytes(body, "max_tokens").Int(); got != 100+int64(budgetOf("max")) {
		t.Fatalf("native max_tokens = %d, want %d", got, 100+budgetOf("max"))
	}
	s := New()
	for attempt := range 2 {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(string(body)))
		req.Header.Set("anthropic-version", "2023-06-01")
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("attempt %d: %d %s", attempt+1, rec.Code, rec.Body)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(chats) != 2 {
		t.Fatalf("chat requests = %d, want 2", len(chats))
	}
	for i, raw := range chats {
		if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "max" {
			t.Fatalf("chat #%d reasoning_effort = %q, want max (body=%s)", i+1, got, raw)
		}
		if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
			t.Fatalf("chat #%d max_tokens = %d, want 100 (body=%s)", i+1, got, raw)
		}
	}
}

// Messages on the same unpinned relay still receives the native Anthropic
// body WithSideEffort built, budget and raised max_tokens included.
func TestWithSideEffortDualProtocolMessagesKeepsNative(t *testing.T) {
	fresh(t)
	up := &capUp{proto: "anthropic"}
	srv := httptest.NewServer(up)
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k",
		Anthropic: srv.URL, Chat: srv.URL + "/v1",
		Models: []string{"claude-sonnet-4-5"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	body := WithSideEffort([]byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`), "relay/claude-sonnet-4-5", "max")
	s := New()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(string(body)))
	req.Header.Set("anthropic-version", "2023-06-01")
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	raw := up.raw()
	if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
		t.Fatalf("thinking %s, want enabled", gjson.GetBytes(raw, "thinking").Raw)
	}
	if got := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); got != int64(budgetOf("max")) {
		t.Fatalf("budget_tokens = %d, want %d (body=%s)", got, budgetOf("max"), raw)
	}
	if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100+int64(budgetOf("max")) {
		t.Fatalf("max_tokens = %d, want %d (body=%s)", got, 100+budgetOf("max"), raw)
	}
	if gjson.GetBytes(raw, "reasoning_effort").Exists() {
		t.Fatalf("native body carried reasoning_effort: %s", raw)
	}
}

// #7389 R5 F12: max and xhigh share an Anthropic budget, so the native
// WithSideEffort bodies must still differ in bytes. Otherwise the second
// rememberSideEffort overwrites the first slot and a later send of the
// first body is asked at the second level.
func TestWithSideEffortMaxAndXhighDoNotShareACacheKey(t *testing.T) {
	fresh(t)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k",
		Anthropic: "http://127.0.0.1:1", Chat: "http://127.0.0.1:1/v1",
		Models: []string{"claude-sonnet-4-5"},
	}); err != nil {
		t.Fatal(err)
	}
	levels := []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}
	if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", levels); err != nil {
		t.Fatal(err)
	}
	model := "relay/claude-sonnet-4-5"
	for _, orig := range [][]byte{
		[]byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`),
		[]byte(`{"model":"relay/claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}]}`),
	} {
		seen := map[string][]byte{}
		for _, want := range levels {
			e := sideEffort(model, want)
			out := WithSideEffort(orig, model, want)
			if !gjson.ValidBytes(out) {
				t.Fatalf("want %s fitted %s: invalid JSON %q", want, e, out)
			}
			if prev, ok := seen[e]; ok {
				if !bytes.Equal(prev, out) {
					t.Fatalf("fitted %s encoded two ways", e)
				}
				continue
			}
			for other, b := range seen {
				if bytes.Equal(b, out) {
					t.Fatalf("fitted %q and %q share bytes for %s", e, other, orig)
				}
			}
			seen[e] = append([]byte(nil), out...)
		}
	}

	orig := []byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
	bMax := WithSideEffort(orig, model, "max")
	bXhigh := WithSideEffort(orig, model, "xhigh")
	if sha256.Sum256(bMax) == sha256.Sum256(bXhigh) {
		t.Fatalf("max and xhigh share a cache key: %s", bMax)
	}
	if canonJSON(t, bMax) != canonJSON(t, bXhigh) {
		t.Fatalf("parsed bodies differ:\nmax %s\nxhigh %s", bMax, bXhigh)
	}
	for _, body := range [][]byte{bMax, bXhigh} {
		if gjson.GetBytes(body, "max_tokens").Int() != 100+int64(budgetOf("max")) {
			t.Fatalf("max_tokens = %d, want %d (%s)", gjson.GetBytes(body, "max_tokens").Int(), 100+budgetOf("max"), body)
		}
		if gjson.GetBytes(body, "thinking.type").String() != "enabled" || gjson.GetBytes(body, "thinking.budget_tokens").Int() != int64(budgetOf("max")) {
			t.Fatalf("thinking %s, want enabled budget %d", gjson.GetBytes(body, "thinking").Raw, budgetOf("max"))
		}
		var m map[string]any
		if err := json.Unmarshal(body, &m); err != nil {
			t.Fatal(err)
		}
		for _, k := range []string{"model", "max_tokens", "messages", "thinking"} {
			if _, ok := m[k]; !ok {
				t.Fatalf("missing %s in %s", k, body)
			}
		}
		if len(m) != 4 {
			t.Fatalf("keys %v, want model, max_tokens, messages, thinking", keysOf(m))
		}
	}
	maxMeta, ok := sideEffortCached(bMax)
	if !ok || maxMeta.effort != "max" || !maxMeta.hadMaxTokens || maxMeta.origMaxTokens != 100 || !maxMeta.addedThinking {
		t.Fatalf("cached max meta = %+v ok=%v", maxMeta, ok)
	}
	xMeta, ok := sideEffortCached(bXhigh)
	if !ok || xMeta.effort != "xhigh" || !xMeta.hadMaxTokens || xMeta.origMaxTokens != 100 || !xMeta.addedThinking {
		t.Fatalf("cached xhigh meta = %+v ok=%v, the max slot was overwritten", xMeta, ok)
	}
}

func TestWithSideEffortMaxThenXhighKeepsEachChatFallback(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	var chats [][]byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/v1/messages") {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"Model does not support this protocol."}}`)
			return
		}
		mu.Lock()
		chats = append(chats, append([]byte(nil), b...))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k",
		Anthropic: srv.URL, Chat: srv.URL + "/v1",
		Models: []string{"claude-sonnet-4-5"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	orig := []byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
	bMax := WithSideEffort(orig, "relay/claude-sonnet-4-5", "max")
	bXhigh := WithSideEffort(orig, "relay/claude-sonnet-4-5", "xhigh")
	s := New()
	send := func(body []byte) {
		t.Helper()
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body))
		req.Header.Set("anthropic-version", "2023-06-01")
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
	}
	send(bMax)
	send(bMax)
	send(bXhigh)
	mu.Lock()
	defer mu.Unlock()
	if len(chats) != 3 {
		t.Fatalf("chat requests = %d, want 3", len(chats))
	}
	want := []string{"max", "max", "xhigh"}
	for i, raw := range chats {
		if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != want[i] {
			t.Fatalf("chat #%d reasoning_effort = %q, want %s (body=%s)", i+1, got, want[i], raw)
		}
		if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
			t.Fatalf("chat #%d max_tokens = %d, want 100 (body=%s)", i+1, got, raw)
		}
	}
}

// #7389 R5 F13: Max output and a context-window reply cap rewrite the
// attempt before the side-effort lookup. The twin is the untouched
// request, then the cap is applied to that twin.
func TestWithSideEffortFallbackSurvivesMaxOutputAndReplyCap(t *testing.T) {
	orig := []byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
	newRelay := func(t *testing.T, maxOut int) (*Server, *[][]byte, *sync.Mutex) {
		t.Helper()
		fresh(t)
		var mu sync.Mutex
		var chats [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			if strings.HasSuffix(r.URL.Path, "/v1/messages") {
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"Model does not support this protocol."}}`)
				return
			}
			mu.Lock()
			chats = append(chats, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, chatHi)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k",
			Anthropic: srv.URL, Chat: srv.URL + "/v1",
			Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		if maxOut > 0 {
			if err := provider.SetModelOutput("relay/claude-sonnet-4-5", maxOut); err != nil {
				t.Fatal(err)
			}
		}
		return New(), &chats, &mu
	}
	sendTwice := func(t *testing.T, s *Server, body []byte) {
		t.Helper()
		for attempt := range 2 {
			rec := httptest.NewRecorder()
			req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body))
			req.Header.Set("anthropic-version", "2023-06-01")
			s.Handler().ServeHTTP(rec, req)
			if rec.Code != 200 {
				t.Fatalf("attempt %d: %d %s", attempt+1, rec.Code, rec.Body)
			}
		}
	}
	t.Run("max output above the ask", func(t *testing.T) {
		s, chats, mu := newRelay(t, 8192)
		sendTwice(t, s, WithSideEffort(orig, "relay/claude-sonnet-4-5", "max"))
		mu.Lock()
		defer mu.Unlock()
		if len(*chats) != 2 {
			t.Fatalf("chat requests = %d, want 2", len(*chats))
		}
		for i, raw := range *chats {
			if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "max" {
				t.Fatalf("chat #%d reasoning_effort = %q, want max (body=%s)", i+1, got, raw)
			}
			if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
				t.Fatalf("chat #%d max_tokens = %d, want 100 (body=%s)", i+1, got, raw)
			}
		}
	})
	t.Run("max output below the ask", func(t *testing.T) {
		s, chats, mu := newRelay(t, 50)
		sendTwice(t, s, WithSideEffort(orig, "relay/claude-sonnet-4-5", "max"))
		mu.Lock()
		defer mu.Unlock()
		if len(*chats) != 2 {
			t.Fatalf("chat requests = %d, want 2", len(*chats))
		}
		for i, raw := range *chats {
			if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "max" {
				t.Fatalf("chat #%d reasoning_effort = %q, want max (body=%s)", i+1, got, raw)
			}
			if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 50 {
				t.Fatalf("chat #%d max_tokens = %d, want 50 (body=%s)", i+1, got, raw)
			}
		}
	})
	t.Run("reply cap", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var chats [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			if strings.HasSuffix(r.URL.Path, "/v1/messages") {
				if gjson.GetBytes(b, "max_tokens").Int() > 8192 {
					w.WriteHeader(http.StatusBadRequest)
					io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"input length and max_tokens exceed context limit: 191808 + 32192 > 200000"}}`)
					return
				}
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"Model does not support this protocol."}}`)
				return
			}
			mu.Lock()
			chats = append(chats, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, chatHi)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k",
			Anthropic: srv.URL, Chat: srv.URL + "/v1",
			Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		s := New()
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(WithSideEffort(orig, "relay/claude-sonnet-4-5", "max")))
		req.Header.Set("anthropic-version", "2023-06-01")
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(chats) != 1 {
			t.Fatalf("chat requests = %d, want 1", len(chats))
		}
		raw := chats[0]
		if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "max" {
			t.Fatalf("reasoning_effort = %q, want max (body=%s)", got, raw)
		}
		if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
			t.Fatalf("max_tokens = %d, want 100 (body=%s)", got, raw)
		}
	})
}

// #7389 R5 F14: a unicode escape of output_config is still that key.
// encoding/json unescapes a top-level map key; messages stay raw and the
// pre-filter has to notice the escape without parsing a 2 MiB array.
func TestEscapedOutputConfigStillRewritten(t *testing.T) {
	top := []byte(`{"model":"claude-sonnet-4-5","max\u005ftokens":100,"thinkin\u0067":{"type":"adaptive"},"output\u005fconfig":{"effort":"high"},"messages":[]}`)
	out := withAnthropicBudget(top, "xhigh")
	if got := gjson.GetBytes(out, "max_tokens").Int(); got != int64(budgetOf("xhigh")+100) {
		t.Fatalf("top-level escaped max_tokens = %d, want %d (body=%s)", got, budgetOf("xhigh")+100, out)
	}
	if gjson.GetBytes(out, "thinking.type").String() != "enabled" || gjson.GetBytes(out, "thinking.budget_tokens").Int() != int64(budgetOf("xhigh")) {
		t.Fatalf("top-level escaped thinking %s, want enabled at budget %d", gjson.GetBytes(out, "thinking").Raw, budgetOf("xhigh"))
	}
	if got := gjson.GetBytes(out, "output_config.effort").String(); got != "xhigh" {
		t.Fatalf("top-level escaped effort = %q, want xhigh (body=%s)", got, out)
	}
	for _, key := range []string{
		`output\u005fconfig`,
		`\u006futput_config`,
		`output_con\u0066ig`,
		`\u006f\u0075\u0074\u0070\u0075\u0074\u005f\u0063\u006f\u006e\u0066\u0069\u0067`,
	} {
		body := []byte(`{"model":"claude-sonnet-4-5","max_tokens":8192,"messages":[{"role":"user","content":"hi","` + key + `":{"effort":"high"}}]}`)
		got := withAnthropicBudget(body, "low")
		if e := gjson.GetBytes(got, "messages.0.output_config.effort").String(); e != "low" {
			t.Fatalf("key %s effort = %q, want low (body=%s)", key, e, got)
		}
	}
	untouched := []byte(`{"model":"claude-sonnet-4-5","max_tokens":8192,"messages":[{"role":"user","content":"a \u003c b"}]}`)
	if gjson.GetBytes(withAnthropicBudget(untouched, "low"), "messages.0.output_config").Exists() {
		t.Fatal("output_config invented on a message that only escaped <")
	}
}

// #7389 R5 F15: a cached thinking-needed mark forces the rule's budget on
// every model, including one asked in output_config and a budget Claude
// whose request already says adaptive.
func TestThinkingNeededCacheUsesRuleBudget(t *testing.T) {
	msg := quote(long(2000))
	newRelay := func(t *testing.T, model string, levels []string) (*Server, *[][]byte, *sync.Mutex) {
		t.Helper()
		fresh(t)
		var mu sync.Mutex
		var bodies [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			bodies = append(bodies, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if gjson.GetBytes(b, "thinking.type").String() != "enabled" {
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"thinking.type must be \"enabled\""}}`)
				return
			}
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"`+model+`","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{model}}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/"+model, levels); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/" + model}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/" + model, Tokens: 1000, UseEffort: "xhigh"}},
		}); err != nil {
			t.Fatal(err)
		}
		return New(), &bodies, &mu
	}
	mark := func(t *testing.T, s *Server, model string, bodies *[][]byte, mu *sync.Mutex) int {
		t.Helper()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"relay/`+model+`","max_tokens":200,"messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != 200 {
			t.Fatalf("mark: %d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(*bodies) < 2 {
			t.Fatalf("direct request made %d upstream calls, want a refusal then enabled", len(*bodies))
		}
		return len(*bodies)
	}
	t.Run("glm", func(t *testing.T) {
		const model = "glm-5.3"
		s, bodies, mu := newRelay(t, model, []string{"low", "medium", "high", "xhigh", "max"})
		marked := mark(t, s, model, bodies, mu)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":`+msg+`}]}`)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(*bodies) != marked+1 {
			t.Fatalf("ruled request made %d upstream calls, want 1", len(*bodies)-marked)
		}
		got := (*bodies)[marked]
		if typ := gjson.GetBytes(got, "thinking.type").String(); typ != "enabled" {
			t.Fatalf("thinking.type = %q, want enabled (body=%s)", typ, got)
		}
		if n := gjson.GetBytes(got, "thinking.budget_tokens").Int(); n != int64(budgetOf("xhigh")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("xhigh"), got)
		}
		if n := gjson.GetBytes(got, "max_tokens").Int(); n != int64(budgetOf("xhigh")+100) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, budgetOf("xhigh")+100, got)
		}
		if e := gjson.GetBytes(got, "output_config.effort").String(); e != "xhigh" {
			t.Fatalf("output_config.effort = %q, want xhigh (body=%s)", e, got)
		}
	})
	t.Run("budget claude already adaptive", func(t *testing.T) {
		const model = "claude-sonnet-4-5"
		s, bodies, mu := newRelay(t, model, []string{"low", "medium", "high", "xhigh", "max"})
		marked := mark(t, s, model, bodies, mu)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"group/r","max_tokens":100,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":`+msg+`}]}`)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(*bodies) != marked+1 {
			t.Fatalf("ruled request made %d upstream calls, want 1 already enabled", len(*bodies)-marked)
		}
		got := (*bodies)[marked]
		if typ := gjson.GetBytes(got, "thinking.type").String(); typ != "enabled" {
			t.Fatalf("thinking.type = %q, want enabled (body=%s)", typ, got)
		}
		if n := gjson.GetBytes(got, "thinking.budget_tokens").Int(); n != int64(budgetOf("xhigh")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("xhigh"), got)
		}
		if n := gjson.GetBytes(got, "max_tokens").Int(); n != int64(budgetOf("xhigh")+100) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, budgetOf("xhigh")+100, got)
		}
	})
}

func canonJSON(t *testing.T, b []byte) string {
	t.Helper()
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("unmarshal: %v (%s)", err, b)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func keysOf(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// sideEffortRelay refuses Messages and answers Chat, the dual-protocol
// fallback WithSideEffort keeps a level for.
func sideEffortRelay(t *testing.T) (*Server, *[][]byte, *sync.Mutex) {
	t.Helper()
	fresh(t)
	var mu sync.Mutex
	var chats [][]byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/v1/messages") {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"Model does not support this protocol."}}`)
			return
		}
		mu.Lock()
		chats = append(chats, append([]byte(nil), b...))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, chatHi)
	}))
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k",
		Anthropic: srv.URL, Chat: srv.URL + "/v1",
		Models: []string{"claude-sonnet-4-5"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	return New(), &chats, &mu
}

func postSideEffort(t *testing.T, s *Server, body []byte) {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body))
	req.Header.Set("anthropic-version", "2023-06-01")
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
}

// #7389 review: Privacy Redaction rewrites the body before the side-effort
// lookup. The Chat fallback must keep the secret masked and the exact
// level at the original max_tokens, not a budget read back as xhigh.
func TestWithSideEffortRedactionKeepsMaxAndOriginalMaxTokens(t *testing.T) {
	s, chats, mu := sideEffortRelay(t)
	const secret = "sk-proj-abcdEFGH1234ijklMNOP5678qrst"
	st := settings.Load()
	st.Redact = true
	if err := settings.Save(st); err != nil {
		t.Fatal(err)
	}
	orig := []byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"key ` + secret + `"}]}`)
	postSideEffort(t, s, WithSideEffort(orig, "relay/claude-sonnet-4-5", "max"))
	mu.Lock()
	defer mu.Unlock()
	if len(*chats) != 1 {
		t.Fatalf("chat requests = %d, want 1", len(*chats))
	}
	raw := (*chats)[0]
	if strings.Contains(string(raw), secret) {
		t.Fatalf("secret reached chat: %s", raw)
	}
	if !strings.Contains(string(raw), "{{API_KEY_") {
		t.Fatalf("placeholder missing: %s", raw)
	}
	if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "max" {
		t.Fatalf("reasoning_effort = %q, want max (body=%s)", got, raw)
	}
	if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
		t.Fatalf("max_tokens = %d, want 100 (body=%s)", got, raw)
	}
}

// #7389 review: a per-candidate rewrite must survive onto the translated
// fallback. The cached twin used to be the body from before that rewrite.
func TestWithSideEffortModelSystemReachesChatFallback(t *testing.T) {
	s, chats, mu := sideEffortRelay(t)
	const mark = "SIDE-SYSTEM-MARK"
	if err := provider.SetModelSystemPrompt("relay/claude-sonnet-4-5", mark); err != nil {
		t.Fatal(err)
	}
	orig := []byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
	postSideEffort(t, s, WithSideEffort(orig, "relay/claude-sonnet-4-5", "max"))
	mu.Lock()
	defer mu.Unlock()
	if len(*chats) != 1 {
		t.Fatalf("chat requests = %d, want 1", len(*chats))
	}
	raw := (*chats)[0]
	if !strings.Contains(string(raw), mark) {
		t.Fatalf("model system missing: %s", raw)
	}
	if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "max" {
		t.Fatalf("reasoning_effort = %q, want max (body=%s)", got, raw)
	}
	if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
		t.Fatalf("max_tokens = %d, want 100 (body=%s)", got, raw)
	}
}

// #7389 review: use-effort on a budget Claude that enters translate while
// the first API is still Anthropic (a stream-only account, a non-streaming
// client) must raise max_tokens around the full budget, and a later Chat
// fallback must keep the exact level at the client's max_tokens.
func TestRuleUseEffortTranslateAnthropicKeepsBudgetAndChatMax(t *testing.T) {
	msg := quote(long(2000))
	// web_search cannot be relayed, so the ask is translated while the
	// first API is still Anthropic (a stream-only account does the same).
	body := `{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":` + msg + `}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`

	t.Run("anthropic only", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var got [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			got = append(got, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, anthropicHi)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL,
			Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/claude-sonnet-4-5"}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "max"}},
		}); err != nil {
			t.Fatal(err)
		}
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(got) != 1 {
			t.Fatalf("upstream calls = %d, want 1", len(got))
		}
		raw := got[0]
		if !gjson.GetBytes(raw, "stream").Bool() {
			t.Fatalf("messages body was relayed, not translated: %s", raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100+int64(budgetOf("max")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 100+budgetOf("max"), raw)
		}
	})

	t.Run("chat fallback", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var messages, chats [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			if strings.HasSuffix(r.URL.Path, "/v1/messages") {
				mu.Lock()
				messages = append(messages, append([]byte(nil), b...))
				mu.Unlock()
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"Model does not support this protocol."}}`)
				return
			}
			mu.Lock()
			chats = append(chats, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, chatHi)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k",
			Anthropic: srv.URL, Chat: srv.URL + "/v1",
			Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/claude-sonnet-4-5"}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "max"}},
		}); err != nil {
			t.Fatal(err)
		}
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(messages) != 1 {
			t.Fatalf("messages calls = %d, want 1", len(messages))
		}
		anth := messages[0]
		if !gjson.GetBytes(anth, "stream").Bool() {
			t.Fatalf("messages body was relayed, not translated: %s", anth)
		}
		if n := gjson.GetBytes(anth, "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
			t.Fatalf("anthropic budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), anth)
		}
		if n := gjson.GetBytes(anth, "max_tokens").Int(); n != 100+int64(budgetOf("max")) {
			t.Fatalf("anthropic max_tokens = %d, want %d (body=%s)", n, 100+budgetOf("max"), anth)
		}
		if len(chats) != 1 {
			t.Fatalf("chat requests = %d, want 1", len(chats))
		}
		raw := chats[0]
		if got := gjson.GetBytes(raw, "reasoning_effort").String(); got != "max" {
			t.Fatalf("reasoning_effort = %q, want max (body=%s)", got, raw)
		}
		if got := gjson.GetBytes(raw, "max_tokens").Int(); got != 100 {
			t.Fatalf("chat max_tokens = %d, want 100 (body=%s)", got, raw)
		}
	})
}

// #7389 R2: s.translate rebuilds an Anthropic body from the upstream model,
// and an effort override keeps thinking and output_config.effort instead of
// dropping them under max_tokens 2048 or cutting the budget to a lower level.
func TestTranslateAnthropicKeepsEffortOnWireAndVendorModels(t *testing.T) {
	msg := quote(long(2000))
	web := `,"tools":[{"type":"web_search_20250305","name":"web_search"}]`
	levels := []string{"low", "medium", "high", "xhigh", "max"}
	newRelay := func(t *testing.T, model, wire, use string, maxTokens int) (*Server, *[][]byte, *sync.Mutex) {
		t.Helper()
		fresh(t)
		var mu sync.Mutex
		var got [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			got = append(got, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, anthropicHi)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{model},
		}); err != nil {
			t.Fatal(err)
		}
		if wire != "" && wire != model {
			if err := provider.SetUpstreamName("relay/"+model, wire); err != nil {
				t.Fatal(err)
			}
		}
		if err := provider.SetModelEfforts("relay/"+model, levels); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/" + model}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/" + model, Tokens: 1000, UseEffort: use}},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		body := `{"model":"group/r","max_tokens":` + strconv.Itoa(maxTokens) + `,"messages":[{"role":"user","content":` + msg + `}]` + web + `}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		return s, &got, &mu
	}
	t.Run("alias is a budget claude", func(t *testing.T) {
		_, got, mu := newRelay(t, "review-sonnet", "claude-sonnet-4-5", "max", 100)
		mu.Lock()
		defer mu.Unlock()
		if len(*got) != 1 {
			t.Fatalf("upstream calls = %d, want 1", len(*got))
		}
		raw := (*got)[0]
		if gjson.GetBytes(raw, "model").String() != "claude-sonnet-4-5" {
			t.Fatalf("model = %q, want claude-sonnet-4-5 (body=%s)", gjson.GetBytes(raw, "model").String(), raw)
		}
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("thinking %s, want enabled (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100+int64(budgetOf("max")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 100+budgetOf("max"), raw)
		}
		if gjson.GetBytes(raw, "output_config.effort").Exists() {
			t.Fatalf("output_config.effort = %s, Sonnet 4.5 is asked by a budget (body=%s)", gjson.GetBytes(raw, "output_config.effort").Raw, raw)
		}
	})
	t.Run("alias is an adaptive claude", func(t *testing.T) {
		_, got, mu := newRelay(t, "review-opus", "claude-opus-5-5", "max", 100)
		mu.Lock()
		defer mu.Unlock()
		raw := (*got)[0]
		if gjson.GetBytes(raw, "model").String() != "claude-opus-5-5" {
			t.Fatalf("model = %q (body=%s)", gjson.GetBytes(raw, "model").String(), raw)
		}
		if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" {
			t.Fatalf("thinking %s, want adaptive (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if gjson.GetBytes(raw, "output_config.effort").String() != "max" {
			t.Fatalf("effort = %q, want max (body=%s)", gjson.GetBytes(raw, "output_config.effort").String(), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
			t.Fatalf("max_tokens = %d, want 100 (body=%s)", n, raw)
		}
	})
	t.Run("glm keeps effort under 2048", func(t *testing.T) {
		_, got, mu := newRelay(t, "glm-5.3", "", "max", 100)
		mu.Lock()
		defer mu.Unlock()
		raw := (*got)[0]
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("thinking %s, want enabled (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if gjson.GetBytes(raw, "output_config.effort").String() != "max" {
			t.Fatalf("effort = %q, want max (body=%s)", gjson.GetBytes(raw, "output_config.effort").String(), raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100+int64(budgetOf("max")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 100+budgetOf("max"), raw)
		}
	})
	t.Run("vendor model is not cut to medium", func(t *testing.T) {
		_, got, mu := newRelay(t, "vendor-m", "", "max", 8192)
		mu.Lock()
		defer mu.Unlock()
		raw := (*got)[0]
		if gjson.GetBytes(raw, "output_config.effort").Exists() {
			t.Fatalf("output_config.effort = %s, a custom relay id is asked by a budget (body=%s)", gjson.GetBytes(raw, "output_config.effort").Raw, raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 8192+int64(budgetOf("max")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 8192+budgetOf("max"), raw)
		}
	})
}

// #7389 R2: a budget Claude asked at max is recorded as max. The budget
// 32000 also reads as xhigh, and that used to win in the route.
func TestBudgetClaudeMaxRecordedAsMax(t *testing.T) {
	msg := quote(long(2000))
	s, up := anthropicRuleMember(t, "claude-opus-4-5", []string{"low", "medium", "high", "xhigh", "max"}, "max")
	body := `{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":` + msg + `}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	raw := up.raw()
	if gjson.GetBytes(raw, "output_config.effort").Exists() {
		t.Fatalf("output_config.effort = %s, a budget Claude is asked by a budget (body=%s)", gjson.GetBytes(raw, "output_config.effort").Raw, raw)
	}
	if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
		t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), raw)
	}
	r := lastRoute(s)
	if len(r.Tries) != 1 || r.Tries[0].Effort != "max" || r.Tries[0].RuleEffort != "max" {
		t.Fatalf("try %+v, want effort max", r.Tries)
	}
}

// #7389 R2: WithSideEffort remembers a single-protocol body, and a rule or
// fixed member does not keep the side call's raised max_tokens.
func TestWithSideEffortSingleProtocolAndOverride(t *testing.T) {
	levels := []string{"low", "medium", "high", "xhigh", "max"}
	t.Run("anthropic only translate", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var got [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			got = append(got, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, anthropicHi)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", levels); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`)
		body := WithSideEffort(orig, "relay/claude-sonnet-4-5", "max")
		meta, ok := sideEffortCached(body)
		if !ok || meta.effort != "max" || !meta.hadMaxTokens || meta.origMaxTokens != 100 {
			t.Fatalf("side meta %+v ok=%v, want max at 100", meta, ok)
		}
		s := New()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(got) != 1 {
			t.Fatalf("upstream calls = %d, want 1", len(got))
		}
		raw := got[0]
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100+int64(budgetOf("max")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 100+budgetOf("max"), raw)
		}
		if r := lastRoute(s); len(r.Tries) != 1 || r.Tries[0].Effort != "max" {
			t.Fatalf("try %+v, want effort max", r.Tries)
		}
	})
	t.Run("rule replaces side budget", func(t *testing.T) {
		fresh(t)
		up := &capUp{proto: "anthropic"}
		srv := httptest.NewServer(up)
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", levels); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/claude-sonnet-4-5"}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "low"}},
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`)
		body := WithSideEffort(orig, "group/r", "max")
		if n := gjson.GetBytes(body, "max_tokens").Int(); n != 100+int64(budgetOf("max")) {
			t.Fatalf("side max_tokens = %d, want the raised budget", n)
		}
		s := New()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("low")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("low"), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100+int64(budgetOf("low")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 100+budgetOf("low"), raw)
		}
	})
	t.Run("fixed member replaces side budget", func(t *testing.T) {
		fresh(t)
		up := &capUp{proto: "anthropic"}
		srv := httptest.NewServer(up)
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", levels); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/claude-sonnet-4-5:low"}, Routing: provider.Ordered,
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
		body := WithSideEffort(orig, "group/r", "max")
		s := New()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100+int64(budgetOf("low")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 100+budgetOf("low"), raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("low")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("low"), raw)
		}
	})
}

// #7389 R2: a Chat-first WithSideEffort body failed over to a budget Claude
// is asked with a budget and a raised max_tokens, not the adaptive marker.
// The other way, a native side body failed over to Chat keeps max at the
// caller's max_tokens.
func TestWithSideEffortGroupFailover(t *testing.T) {
	levels := []string{"low", "medium", "high", "xhigh", "max"}
	t.Run("chat then budget claude", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var chats, anths [][]byte
		chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			chats = append(chats, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, `{"error":{"message":"upstream down","type":"api_error"}}`)
		}))
		t.Cleanup(chatSrv.Close)
		anthSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			anths = append(anths, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(anthSrv.Close)
		if err := provider.Save(provider.Provider{
			ID: "chatrel", Name: "Chat", Key: "k", Chat: chatSrv.URL + "/v1", Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.Save(provider.Provider{
			ID: "anthrel", Name: "Anth", Key: "k", Anthropic: anthSrv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("chatrel/claude-sonnet-4-5", levels); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("anthrel/claude-sonnet-4-5", levels); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members: []string{"chatrel/claude-sonnet-4-5", "anthrel/claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
		body := WithSideEffort(orig, "group/r", "max")
		if gjson.GetBytes(body, "thinking.type").String() != "adaptive" {
			t.Fatalf("chat-first side thinking %s, want adaptive", gjson.GetBytes(body, "thinking").Raw)
		}
		if n := gjson.GetBytes(body, "max_tokens").Int(); n != 100 {
			t.Fatalf("chat-first side max_tokens = %d, want 100", n)
		}
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(anths) != 1 {
			t.Fatalf("anthropic calls = %d, chats = %d", len(anths), len(chats))
		}
		raw := anths[0]
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("thinking %s, want enabled (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100+int64(budgetOf("max")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 100+budgetOf("max"), raw)
		}
	})
	t.Run("budget claude then chat", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var chats, anths [][]byte
		anthSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			anths = append(anths, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"upstream down"}}`)
		}))
		t.Cleanup(anthSrv.Close)
		chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			chats = append(chats, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, chatHi)
		}))
		t.Cleanup(chatSrv.Close)
		if err := provider.Save(provider.Provider{
			ID: "anthrel", Name: "Anth", Key: "k", Anthropic: anthSrv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.Save(provider.Provider{
			ID: "chatrel", Name: "Chat", Key: "k", Chat: chatSrv.URL + "/v1", Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("anthrel/claude-sonnet-4-5", levels); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("chatrel/claude-sonnet-4-5", levels); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members: []string{"anthrel/claude-sonnet-4-5", "chatrel/claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
		body := WithSideEffort(orig, "group/r", "max")
		if n := gjson.GetBytes(body, "max_tokens").Int(); n != 100+int64(budgetOf("max")) {
			t.Fatalf("native side max_tokens = %d, want raised", n)
		}
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(chats) != 1 {
			t.Fatalf("chat calls = %d, anthropic = %d", len(chats), len(anths))
		}
		raw := chats[0]
		if gjson.GetBytes(raw, "reasoning_effort").String() != "max" {
			t.Fatalf("reasoning_effort = %q, want max (body=%s)", gjson.GetBytes(raw, "reasoning_effort").String(), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
			t.Fatalf("max_tokens = %d, want 100 (body=%s)", n, raw)
		}
	})
}

// #7389 R2: a reasonless model fixed at an effort is not given that effort's
// budget. max_tokens and sampling stay as the client sent them.
func TestReasonlessFixedEffortKeepsSampling(t *testing.T) {
	fresh(t)
	up := &capUp{proto: "anthropic"}
	srv := httptest.NewServer(up)
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"glm"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelReasonless("relay/glm", true); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "R", Members: []string{"relay/glm:max"}, Routing: provider.Ordered,
	}); err != nil {
		t.Fatal(err)
	}
	body := `{"model":"group/r","max_tokens":100,"temperature":0,"top_p":0.9,"top_k":40,"output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"object"}}},"messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	raw := up.raw()
	if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
		t.Fatalf("max_tokens = %d, want 100 (body=%s)", n, raw)
	}
	for _, key := range []string{"temperature", "top_p", "top_k"} {
		if !gjson.GetBytes(raw, key).Exists() {
			t.Fatalf("%s dropped (body=%s)", key, raw)
		}
	}
	if gjson.GetBytes(raw, "thinking").Exists() {
		t.Fatalf("thinking sent to a reasonless model: %s", raw)
	}
	if gjson.GetBytes(raw, "output_config.effort").Exists() {
		t.Fatalf("effort sent to a reasonless model: %s", raw)
	}
	if gjson.GetBytes(raw, "output_config.format.type").String() != "json_schema" {
		t.Fatalf("format dropped: %s", raw)
	}
}

// #7389 R2: a cached thinking-needed mark, on the local name or the upstream
// name, converts an adaptive body to enabled before the request is sent.
func TestThinkingNeededCacheConvertsAdaptive(t *testing.T) {
	t.Run("marked model", func(t *testing.T) {
		fresh(t)
		up := &capUp{proto: "anthropic"}
		srv := httptest.NewServer(up)
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-opus-5-5"},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		s.markUnfit("relay", thinkingNeeded("claude-opus-5-5"), provider.Anthropic)
		body := `{"model":"relay/claude-opus-5-5","max_tokens":8192,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("thinking %s, want enabled without a 400 (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("high"), raw)
		}
	})
	t.Run("marked upstream name", func(t *testing.T) {
		fresh(t)
		up := &capUp{proto: "anthropic"}
		srv := httptest.NewServer(up)
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"review-opus"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetUpstreamName("relay/review-opus", "claude-opus-5-5"); err != nil {
			t.Fatal(err)
		}
		s := New()
		s.markUnfit("relay", thinkingNeeded("claude-opus-5-5"), provider.Anthropic)
		body := `{"model":"relay/review-opus","max_tokens":8192,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "model").String() != "claude-opus-5-5" {
			t.Fatalf("model = %q (body=%s)", gjson.GetBytes(raw, "model").String(), raw)
		}
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("thinking %s, want enabled from the upstream mark (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("high"), raw)
		}
	})
}

// #7389 R3: native WithSideEffort keeps replyRoom above the thinking budget
// (for non-Claude budget models and adaptive Claudes alike), and routing-group
// failover fits the requested level to each member's known efforts (leaving
// unknown-effort members untouched without leaking the first member's ask).
func TestSideEffortNativeReplyRoomAndFailoverFit(t *testing.T) {
	t.Run("native budget model keeps replyRoom", func(t *testing.T) {
		fresh(t)
		up := &capUp{proto: "anthropic"}
		srv := httptest.NewServer(up)
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"glm-5.3"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/glm-5.3", []string{"low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"relay/glm-5.3","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`)
		body := WithSideEffort(orig, "relay/glm-5.3", "high")
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("thinking %s, want enabled (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("high"), raw)
		}
		wantMax := int64(replyRoom(4096, "high"))
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != wantMax {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, wantMax, raw)
		}
		if e := gjson.GetBytes(raw, "output_config.effort").String(); e != "high" {
			t.Fatalf("output_config.effort = %q, want high (body=%s)", e, raw)
		}
	})
	t.Run("native adaptive claude keeps replyRoom", func(t *testing.T) {
		fresh(t)
		up := &capUp{proto: "anthropic"}
		srv := httptest.NewServer(up)
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-opus-5-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-opus-5-5", []string{"low", "medium", "high", "max"}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"relay/claude-opus-5-5","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`)
		body := WithSideEffort(orig, "relay/claude-opus-5-5", "max")
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" {
			t.Fatalf("thinking %s, want adaptive (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		wantMax := int64(replyRoom(4096, "max"))
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != wantMax {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, wantMax, raw)
		}
		if e := gjson.GetBytes(raw, "output_config.effort").String(); e != "max" {
			t.Fatalf("output_config.effort = %q, want max (body=%s)", e, raw)
		}
	})
	t.Run("failover fits member efforts and skips unknown efforts", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var m2Body, m3Body []byte
		failSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"down"}}`)
		}))
		t.Cleanup(failSrv.Close)
		m2Srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			m2Body = append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"down"}}`)
		}))
		t.Cleanup(m2Srv.Close)
		m3Srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			m3Body = append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"plain-model","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(m3Srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "p1", Name: "P1", Key: "k", Anthropic: failSrv.URL, Models: []string{"claude-opus-5-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.Save(provider.Provider{
			ID: "p2", Name: "P2", Key: "k", Anthropic: m2Srv.URL, Models: []string{"glm-5.3"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.Save(provider.Provider{
			ID: "p3", Name: "P3", Key: "k", Anthropic: m3Srv.URL, Models: []string{"plain-model"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("p1/claude-opus-5-5", []string{"low", "medium", "high", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("p2/glm-5.3", []string{"low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members: []string{"p1/claude-opus-5-5", "p2/glm-5.3", "p3/plain-model"},
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`)
		body := WithSideEffort(orig, "group/r", "max")
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(m2Body) == 0 || len(m3Body) == 0 {
			t.Fatalf("expected failover through m2 (%d) and m3 (%d)", len(m2Body), len(m3Body))
		}
		// m2 has efforts up to high, so max fits to high with high's replyRoom
		if gjson.GetBytes(m2Body, "output_config.effort").String() != "high" {
			t.Fatalf("m2 effort = %q, want high (body=%s)", gjson.GetBytes(m2Body, "output_config.effort").String(), m2Body)
		}
		if n := gjson.GetBytes(m2Body, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
			t.Fatalf("m2 budget = %d, want %d (body=%s)", n, budgetOf("high"), m2Body)
		}
		if n := gjson.GetBytes(m2Body, "max_tokens").Int(); n != int64(replyRoom(4096, "high")) {
			t.Fatalf("m2 max_tokens = %d, want %d (body=%s)", n, replyRoom(4096, "high"), m2Body)
		}
		// m3 has unknown efforts, so WithSideEffort does not inject reasoning or raise max_tokens
		if n := gjson.GetBytes(m3Body, "max_tokens").Int(); n != 4096 {
			t.Fatalf("m3 max_tokens = %d, want 4096 (body=%s)", n, m3Body)
		}
		if gjson.GetBytes(m3Body, "thinking.type").String() == "enabled" || gjson.GetBytes(m3Body, "thinking.type").String() == "adaptive" {
			t.Fatalf("m3 got leaked thinking %s (body=%s)", gjson.GetBytes(m3Body, "thinking").Raw, m3Body)
		}
		if gjson.GetBytes(m3Body, "output_config.effort").Exists() {
			t.Fatalf("m3 got leaked output_config.effort (body=%s)", m3Body)
		}
	})
}

// #7389 R3: translated Anthropic requests preserve sampling params on
// cross-protocol targets (Chat/Responses) and on translated Anthropic targets
// whose OutputCap prevents emitting thinking.
func TestTranslatedEffortPreservesSamplingAndLowCap(t *testing.T) {
	t.Run("anthropic client to chat keeps temperature and top_p", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var chatBody []byte
		chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			chatBody = append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, chatHi)
		}))
		t.Cleanup(chatSrv.Close)
		if err := provider.Save(provider.Provider{
			ID: "chatrel", Name: "Chat", Key: "k", Chat: chatSrv.URL + "/v1", Models: []string{"gpt-5.4"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("chatrel/gpt-5.4", []string{"low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members: []string{"chatrel/gpt-5.4"},
			Rules: []provider.Rule{{
				Use: "chatrel/gpt-5.4", Tokens: 1000, UseEffort: "high",
			}},
		}); err != nil {
			t.Fatal(err)
		}
		body := `{"model":"group/r","max_tokens":100,"temperature":0.3,"top_p":0.85,"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if gjson.GetBytes(chatBody, "reasoning_effort").String() != "high" {
			t.Fatalf("reasoning_effort = %q, want high (body=%s)", gjson.GetBytes(chatBody, "reasoning_effort").String(), chatBody)
		}
		if got := gjson.GetBytes(chatBody, "temperature"); !got.Exists() || got.Float() != 0.3 {
			t.Fatalf("temperature = %v, want 0.3 (body=%s)", got.Raw, chatBody)
		}
		if got := gjson.GetBytes(chatBody, "top_p"); !got.Exists() || got.Float() != 0.85 {
			t.Fatalf("top_p = %v, want 0.85 (body=%s)", got.Raw, chatBody)
		}
	})
	t.Run("translated anthropic keeps temperature when OutputCap disables thinking", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var raw []byte
		const anthSSE = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			raw = append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, anthSSE)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelOutput("relay/claude-sonnet-4-5", 100); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members: []string{"relay/claude-sonnet-4-5"},
			Rules: []provider.Rule{{
				Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "high",
			}},
		}); err != nil {
			t.Fatal(err)
		}
		// web_search_20250305 enters translate -> forwardTranslated -> anthropicBody
		// with from==Anthropic, EffortOverride==true, OutputCap==100, and Temp==0.3.
		body := `{"model":"group/r","max_tokens":4096,"temperature":0.3,"messages":[{"role":"user","content":` + quote(long(2000)) + `}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if gjson.GetBytes(raw, "thinking").Exists() {
			t.Fatalf("thinking emitted under OutputCap=100: %s", raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
			t.Fatalf("max_tokens = %d, want 100 (body=%s)", n, raw)
		}
		if got := gjson.GetBytes(raw, "temperature"); !got.Exists() || got.Float() != 0.3 {
			t.Fatalf("temperature = %v, want 0.3 (body=%s)", got.Raw, raw)
		}
	})
	t.Run("group levels clamp on client effort does not inflate max_tokens in translate", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var raw []byte
		const anthSSE = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			raw = append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, anthSSE)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members: []string{"relay/claude-sonnet-4-5"},
			Levels:  []string{"low", "medium", "high"},
		}); err != nil {
			t.Fatal(err)
		}
		body := `{"model":"group/r","max_tokens":4096,"thinking":{"type":"adaptive"},"output_config":{"effort":"max"},"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n > 4096 {
			t.Fatalf("client max_tokens inflated to %d > 4096 without effort override (body=%s)", n, raw)
		}
	})
}

// #7389 R4: WithSideEffort edge cases:
// - "none" records addedThinking so failover to an unknown-effort model does not leak thinking:disabled
// - group whose first member has unknown efforts still carries want onto a later member
// - EffortMaps evaluate against the candidate's fitted WithSideEffort level (not budget 32000 -> xhigh)
func TestSideEffortGroupNoneUnknownFirstAndEffortMap(t *testing.T) {
	t.Run("none does not leak thinking disabled on failover to unknown-effort member", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var m2Body []byte
		m1Srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"down"}}`)
		}))
		t.Cleanup(m1Srv.Close)
		m2Srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			m2Body = append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(m2Srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "p1", Name: "P1", Key: "k", Anthropic: m1Srv.URL, Models: []string{"m-off"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.Save(provider.Provider{
			ID: "p2", Name: "P2", Key: "k", Anthropic: m2Srv.URL, Models: []string{"claude-opus-5-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("p1/m-off", []string{"none", "low", "high"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members: []string{"p1/m-off", "p2/claude-opus-5-5"},
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`)
		body := WithSideEffort(orig, "group/r", "none")
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if gjson.GetBytes(m2Body, "thinking").Exists() {
			t.Fatalf("m2 got leaked thinking %s (body=%s)", gjson.GetBytes(m2Body, "thinking").Raw, m2Body)
		}
	})
	t.Run("unknown-effort first member still carries want to second member", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var m2Body []byte
		m1Srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"down"}}`)
		}))
		t.Cleanup(m1Srv.Close)
		m2Srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			m2Body = append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"glm-5.3","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(m2Srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "p1", Name: "P1", Key: "k", Anthropic: m1Srv.URL, Models: []string{"plain-model"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.Save(provider.Provider{
			ID: "p2", Name: "P2", Key: "k", Anthropic: m2Srv.URL, Models: []string{"glm-5.3"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("p2/glm-5.3", []string{"low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members: []string{"p1/plain-model", "p2/glm-5.3"},
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`)
		bodyLow := WithSideEffort(orig, "group/r", "low")
		bodyHigh := WithSideEffort(orig, "group/r", "high")
		if bytes.Equal(bodyLow, bodyHigh) {
			t.Fatalf("distinct group wants produced identical cache keys")
		}
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(bodyHigh)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if gjson.GetBytes(m2Body, "output_config.effort").String() != "high" {
			t.Fatalf("m2 effort = %q, want high (body=%s)", gjson.GetBytes(m2Body, "output_config.effort").String(), m2Body)
		}
		if n := gjson.GetBytes(m2Body, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
			t.Fatalf("m2 budget = %d, want %d (body=%s)", n, budgetOf("high"), m2Body)
		}
		if n := gjson.GetBytes(m2Body, "max_tokens").Int(); n != int64(replyRoom(4096, "high")) {
			t.Fatalf("m2 max_tokens = %d, want %d (body=%s)", n, replyRoom(4096, "high"), m2Body)
		}
	})
	t.Run("EffortMaps sees exact WithSideEffort max on budget claude", func(t *testing.T) {
		fresh(t)
		up := &capUp{proto: "anthropic"}
		srv := httptest.NewServer(up)
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members:    []string{"relay/claude-sonnet-4-5"},
			EffortMaps: map[string]map[string]string{"relay/claude-sonnet-4-5": {"max": "medium"}},
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`)
		body := WithSideEffort(orig, "group/r", "max")
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("medium")) {
			t.Fatalf("budget_tokens = %d, want %d from EffortMap max->medium (body=%s)", n, budgetOf("medium"), raw)
		}
	})
}

// #7389 R4: forwardTranslated does not burn a duplicate 400 retry when
// attemptMaxOut <= neededBudget prevents enabling thinking.
func TestForwardTranslatedThinkingNeededNoDuplicateRetryWhenCapped(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"thinking.type must be enabled"}}`)
	}))
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-opus-5-5"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelOutput("relay/claude-opus-5-5", 100); err != nil {
		t.Fatal(err)
	}
	s := New()
	body := `{"model":"relay/claude-opus-5-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400 (%s)", rec.Code, rec.Body)
	}
	mu.Lock()
	defer mu.Unlock()
	if calls != 1 {
		t.Fatalf("upstream calls = %d, want 1 without redundant retry when capped at 100", calls)
	}
	if s.fits("relay", thinkingNeeded("claude-opus-5-5"), provider.Anthropic) {
		t.Fatalf("expected thinkingNeeded to be remembered for future uncapped requests")
	}
}

// #7389 R5:
//  1. Translated Anthropic routes (web_search/stream-only) do not emit synthetic
//     thinking alongside forced tool_choice (preserving sampling and max_tokens).
//  2. EffortMaps on WithSideEffort matches raw want before fitting to member efforts.
//  3. WithSideEffort keeps replyRoom on sideOverride (native) and adaptive Claude translate.
//  4. WithSideEffort cache keys disambiguate want != e across group members.
//  5. 200-SSE thinkingNeededRefusal does not retry when capped at <= neededBudget.
func TestR5TranslatedForcedToolEffortMapReplyRoomAndSSECap(t *testing.T) {
	const anthSSE = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	t.Run("translated anthropic skips synthetic thinking on forced tool_choice", func(t *testing.T) {
		for _, tc := range []struct {
			model    string
			keepTemp bool
		}{
			{model: "claude-opus-5-5", keepTemp: false},
			{model: "claude-sonnet-4-5", keepTemp: true},
		} {
			t.Run(tc.model, func(t *testing.T) {
				fresh(t)
				var mu sync.Mutex
				var raw []byte
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					b, _ := io.ReadAll(r.Body)
					mu.Lock()
					raw = append([]byte(nil), b...)
					mu.Unlock()
					w.Header().Set("Content-Type", "text/event-stream")
					io.WriteString(w, anthSSE)
				}))
				t.Cleanup(srv.Close)
				if err := provider.Save(provider.Provider{
					ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{tc.model},
				}); err != nil {
					t.Fatal(err)
				}
				if err := provider.SetModelEfforts("relay/"+tc.model, []string{"low", "medium", "high"}); err != nil {
					t.Fatal(err)
				}
				if err := provider.SaveGroup(provider.Group{
					Name: "R", Routing: provider.Ordered,
					Members: []string{"relay/" + tc.model},
					Rules:   []provider.Rule{{Use: "relay/" + tc.model, Tokens: 1000, UseEffort: "high"}},
				}); err != nil {
					t.Fatal(err)
				}
				body := `{"model":"group/r","max_tokens":100,"temperature":0.3,"tool_choice":{"type":"any"},"tools":[{"name":"lookup","description":"lookup","input_schema":{"type":"object"}},{"type":"web_search_20250305","name":"web_search"}],"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
				rec := httptest.NewRecorder()
				New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
				if rec.Code != 200 {
					t.Fatalf("%d %s", rec.Code, rec.Body)
				}
				mu.Lock()
				defer mu.Unlock()
				if gjson.GetBytes(raw, "thinking").Exists() {
					t.Fatalf("synthetic thinking emitted alongside forced tool_choice: %s", raw)
				}
				if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
					t.Fatalf("max_tokens = %d, want 100 (body=%s)", n, raw)
				}
				if tc.keepTemp {
					if got := gjson.GetBytes(raw, "temperature"); !got.Exists() || got.Float() != 0.3 {
						t.Fatalf("temperature = %v, want 0.3 (body=%s)", got.Raw, raw)
					}
				}
			})
		}
	})

	t.Run("EffortMaps evaluates raw WithSideEffort want before fitting and keeps replyRoom", func(t *testing.T) {
		fresh(t)
		up := &capUp{proto: "anthropic"}
		srv := httptest.NewServer(up)
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"glm-5.3"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/glm-5.3", []string{"low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members:    []string{"relay/glm-5.3"},
			EffortMaps: map[string]map[string]string{"relay/glm-5.3": {"high": "low", "max": "high", "none": "low", "low": "medium"}},
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`)
		// want="max" must hit EffortMap["max"]="high" (not pre-fit to "high" -> "low"),
		// AND keep replyRoom(4096, "high") = 28096 rather than 1 output token.
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(WithSideEffort(orig, "group/r", "max"))))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "output_config.effort").String() != "high" {
			t.Fatalf("effort = %q, want high (body=%s)", gjson.GetBytes(raw, "output_config.effort").String(), raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
			t.Fatalf("budget = %d, want %d (body=%s)", n, budgetOf("high"), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != int64(replyRoom(4096, "high")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, replyRoom(4096, "high"), raw)
		}
		// want="none" must hit EffortMap["none"]="low" (not pre-fit to "low" -> "medium").
		rec2 := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec2, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(WithSideEffort(orig, "group/r", "none"))))
		if rec2.Code != 200 {
			t.Fatalf("%d %s", rec2.Code, rec2.Body)
		}
		raw2 := up.raw()
		if gjson.GetBytes(raw2, "output_config.effort").String() != "low" {
			t.Fatalf("none->low effort = %q, want low (body=%s)", gjson.GetBytes(raw2, "output_config.effort").String(), raw2)
		}
	})

	t.Run("WithSideEffort adaptive claude keeps replyRoom through translate", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var raw []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			raw = append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, anthSSE)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-opus-5-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-opus-5-5", []string{"low", "medium", "high", "max"}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"relay/claude-opus-5-5","max_tokens":4096,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`)
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(WithSideEffort(orig, "relay/claude-opus-5-5", "max"))))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" {
			t.Fatalf("thinking = %s, want adaptive (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != int64(replyRoom(4096, "max")) {
			t.Fatalf("max_tokens = %d, want %d (body=%s)", n, replyRoom(4096, "max"), raw)
		}
	})

	t.Run("WithSideEffort disambiguates want != e in cache across group members", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var m2Body []byte
		m1Srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"down"}}`)
		}))
		t.Cleanup(m1Srv.Close)
		m2Srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			m2Body = append([]byte(nil), b...)
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(m2Srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "p1", Name: "P1", Key: "k", Anthropic: m1Srv.URL, Models: []string{"glm-5.3"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.Save(provider.Provider{
			ID: "p2", Name: "P2", Key: "k", Anthropic: m2Srv.URL, Models: []string{"claude-opus-5-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("p1/glm-5.3", []string{"low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("p2/claude-opus-5-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Routing: provider.Ordered,
			Members: []string{"p1/glm-5.3", "p2/claude-opus-5-5"},
		}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"group/r","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`)
		bMax := WithSideEffort(orig, "group/r", "max")
		bHigh := WithSideEffort(orig, "group/r", "high")
		if bytes.Equal(bMax, bHigh) {
			t.Fatalf("want=max (fitted high) collided with want=high in sideTranslatedCache")
		}
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(bMax)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if e := gjson.GetBytes(m2Body, "output_config.effort").String(); e != "max" {
			t.Fatalf("m2 effort = %q, want max (body=%s)", e, m2Body)
		}
	})

	t.Run("200-SSE thinkingNeededRefusal does not retry when capped at <= neededBudget", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		calls := 0
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			calls++
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"thinking.type must be enabled\"}}\n\n")
		}))
		t.Cleanup(srv.Close)
		p := provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-opus-5-5"},
		}
		if err := provider.Save(p); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelOutput("relay/claude-opus-5-5", 100); err != nil {
			t.Fatal(err)
		}
		s := New()
		body := []byte(`{"model":"relay/claude-opus-5-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
		req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
		rec := httptest.NewRecorder()
		s.translate(rec, req, p, provider.Chat, provider.Anthropic, "claude-opus-5-5", body, &Usage{})
		mu.Lock()
		defer mu.Unlock()
		if calls != 1 {
			t.Fatalf("upstream calls = %d, want 1 without redundant SSE retry when capped at 100", calls)
		}
		if s.fits("relay", thinkingNeeded("claude-opus-5-5"), provider.Anthropic) {
			t.Fatalf("expected thinkingNeeded to be remembered for future uncapped requests")
		}
	})
}

// #7389 R3: forwardTranslated checks thinkingNeeded on both local model and
// upstream alias (SetUpstreamName) and marks both so forwardOnce never turns
// "enabled" back into "adaptive".
func TestForwardTranslatedThinkingNeededUpstreamAlias(t *testing.T) {
	const anthSSE = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	for _, tc := range []struct {
		name string
		mark string
	}{
		{name: "marked upstream name", mark: "claude-opus-5-5"},
		{name: "marked local alias only", mark: "review-opus"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fresh(t)
			var mu sync.Mutex
			var seen [][]byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				mu.Lock()
				seen = append(seen, append([]byte(nil), b...))
				mu.Unlock()
				if gjson.GetBytes(b, "thinking.type").String() != "enabled" {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusBadRequest)
					io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"thinking.type must be enabled"}}`)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, anthSSE)
			}))
			t.Cleanup(srv.Close)
			if err := provider.Save(provider.Provider{
				ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"review-opus"},
			}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetUpstreamName("relay/review-opus", "claude-opus-5-5"); err != nil {
				t.Fatal(err)
			}
			s := New()
			s.markUnfit("relay", thinkingNeeded(tc.mark), provider.Anthropic)
			body := `{"model":"relay/review-opus","reasoning_effort":"high","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(seen) != 1 {
				t.Fatalf("upstream calls = %d, want 1 without 400 retry (bodies=%q)", len(seen), seen)
			}
			raw := seen[0]
			if gjson.GetBytes(raw, "model").String() != "claude-opus-5-5" {
				t.Fatalf("model = %q, want claude-opus-5-5 (body=%s)", gjson.GetBytes(raw, "model").String(), raw)
			}
			if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
				t.Fatalf("thinking = %s, want enabled (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
			}
			if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
				t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("high"), raw)
			}
		})
	}
}

// #7389 R6: per-attempt Anthropic translation state and max_output caps must
// not leak into nested ownAsk serve calls (web search, advisor), passthrough
// must inspect the upstream wire ID for Claude / output_config checks, and
// anthropicBody must not inflate max_tokens above the model's ceiling.
func TestR6Fixes(t *testing.T) {
	t.Run("nested web_search does not inherit outer translated body or caps", func(t *testing.T) {
		fresh(t)
		searchSandbox(t)
		var mu sync.Mutex
		var searchBodies [][]byte
		var mainCalls int
		const anthToolUse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"web_search\",\"input\":{}}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\":\\\"NESTED-QUERY-R6\\\"}\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
		mainSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			mainCalls++
			call := mainCalls
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			if call == 1 {
				io.WriteString(w, anthToolUse)
				return
			}
			io.WriteString(w, anthropicHi)
		}))
		t.Cleanup(mainSrv.Close)
		searchSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			searchBodies = append(searchBodies, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, chatHi)
		}))
		t.Cleanup(searchSrv.Close)
		searchesOn(t, provider.Chat, searchSrv.URL)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: mainSrv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.Save(provider.Provider{
			ID: "srch", Name: "Searcher", Key: "k", Chat: searchSrv.URL + "/v1", Models: []string{"search-model"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelOutput("relay/claude-sonnet-4-5", 200); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/claude-sonnet-4-5"}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "max"}},
		}); err != nil {
			t.Fatal(err)
		}
		st := settings.Load()
		st.Searcher = "srch/search-model"
		if err := settings.Save(st); err != nil {
			t.Fatal(err)
		}
		msg := quote("OUTER-USER-PROMPT " + long(2000))
		body := `{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":` + msg + `}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(searchBodies) != 1 {
			t.Fatalf("search calls = %d, want 1", len(searchBodies))
		}
		sb := string(searchBodies[0])
		if !strings.Contains(sb, "NESTED-QUERY-R6") {
			t.Fatalf("nested search body lost its query (hijacked by outer translatedFallback): %s", sb)
		}
		if strings.Contains(sb, "OUTER-USER-PROMPT") {
			t.Fatalf("nested search body inherited outer request prompt: %s", sb)
		}
		if gjson.GetBytes(searchBodies[0], "reasoning_effort").String() == "max" {
			t.Fatalf("nested search inherited outer effortOverride: %s", sb)
		}
	})

	t.Run("passthrough checks upstream wire model for aliased claude and glm", func(t *testing.T) {
		t.Run("aliased claude with forced tool_choice stays untouched and within max output", func(t *testing.T) {
			fresh(t)
			up := &capUp{proto: "anthropic"}
			srv := httptest.NewServer(up)
			t.Cleanup(srv.Close)
			if err := provider.Save(provider.Provider{
				ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"review-opus"},
			}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetUpstreamName("relay/review-opus", "claude-opus-5-5"); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetModelEfforts("relay/review-opus", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetModelOutput("relay/review-opus", 100); err != nil {
				t.Fatal(err)
			}
			if err := provider.SaveGroup(provider.Group{
				Name: "R", Members: []string{"relay/review-opus"}, Routing: provider.Ordered,
				Rules: []provider.Rule{{Use: "relay/review-opus", Tokens: 1000, UseEffort: "high"}},
			}); err != nil {
				t.Fatal(err)
			}
			msg := quote(long(2000))
			body := `{"model":"group/r","max_tokens":100,"tool_choice":{"type":"any"},"tools":[{"name":"pick","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":` + msg + `}]}`
			rec := httptest.NewRecorder()
			New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			raw := up.raw()
			if gjson.GetBytes(raw, "thinking").Exists() {
				t.Fatalf("thinking injected on aliased claude with forced tool_choice: %s", raw)
			}
			if gjson.GetBytes(raw, "output_config.effort").Exists() {
				t.Fatalf("output_config.effort injected on aliased claude with forced tool_choice: %s", raw)
			}
			if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
				t.Fatalf("max_tokens = %d, want 100 (body=%s)", n, raw)
			}
		})

		t.Run("aliased glm receives withOutputEffort in passthrough", func(t *testing.T) {
			fresh(t)
			up := &capUp{proto: "anthropic"}
			srv := httptest.NewServer(up)
			t.Cleanup(srv.Close)
			if err := provider.Save(provider.Provider{
				ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"review-glm"},
			}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetUpstreamName("relay/review-glm", "glm-5.3"); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetModelEfforts("relay/review-glm", []string{"low", "medium", "high"}); err != nil {
				t.Fatal(err)
			}
			body := `{"model":"relay/review-glm","max_tokens":32768,"thinking":{"type":"enabled","budget_tokens":16000},"messages":[{"role":"user","content":"hi"}]}`
			rec := httptest.NewRecorder()
			New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			raw := up.raw()
			if e := gjson.GetBytes(raw, "output_config.effort").String(); e != "high" {
				t.Fatalf("output_config.effort = %q, want high for aliased glm-5.3 (body=%s)", e, raw)
			}
		})
	})

	t.Run("translate anthropicBody does not inflate max_tokens past ceiling", func(t *testing.T) {
		t.Run("non-side override with max_tokens above budget keeps 64000", func(t *testing.T) {
			fresh(t)
			var mu sync.Mutex
			var got [][]byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				mu.Lock()
				got = append(got, append([]byte(nil), b...))
				mu.Unlock()
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, anthropicHi)
			}))
			t.Cleanup(srv.Close)
			if err := provider.Save(provider.Provider{
				ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
			}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
				t.Fatal(err)
			}
			body := `{"model":"relay/claude-sonnet-4-5:max","max_tokens":64000,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
			rec := httptest.NewRecorder()
			New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(got) != 1 {
				t.Fatalf("upstream calls = %d, want 1", len(got))
			}
			if n := gjson.GetBytes(got[0], "max_tokens").Int(); n != 64000 {
				t.Fatalf("max_tokens = %d, want 64000 (body=%s)", n, got[0])
			}
			if n := gjson.GetBytes(got[0], "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
				t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), got[0])
			}
		})

		t.Run("WithSideEffort without max_tokens does not inflate fillsOutput ceiling", func(t *testing.T) {
			fresh(t)
			var mu sync.Mutex
			var got [][]byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				mu.Lock()
				got = append(got, append([]byte(nil), b...))
				mu.Unlock()
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, anthropicHi)
			}))
			t.Cleanup(srv.Close)
			if err := provider.Save(provider.Provider{
				ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-opus-5-5"},
			}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetModelEfforts("relay/claude-opus-5-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
				t.Fatal(err)
			}
			orig := []byte(`{"model":"relay/claude-opus-5-5","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`)
			body := WithSideEffort(orig, "relay/claude-opus-5-5", "max")
			rec := httptest.NewRecorder()
			New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(got) != 1 {
				t.Fatalf("upstream calls = %d, want 1", len(got))
			}
			wantMax := int64(askedOutput(provider.Provider{ID: "relay"}, "claude-opus-5-5"))
			if wantMax == 0 {
				wantMax = defaultMaxTokens
			}
			if n := gjson.GetBytes(got[0], "max_tokens").Int(); n != wantMax {
				t.Fatalf("max_tokens = %d, want %d without replyRoom inflation (body=%s)", n, wantMax, got[0])
			}
		})
	})

	t.Run("forwardOnce WithExtraBody keeps enabled thinking when marked thinkingNeeded", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var bodies [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			bodies = append(bodies, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if gjson.GetBytes(b, "thinking.type").String() != "enabled" {
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"thinking.type must be \"enabled\""}}`)
				return
			}
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL,
			Models:    []string{"claude-opus-5-5"},
			ExtraBody: provider.ExtraBody{"custom_tag": json.RawMessage(`"r7"`)},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		body := `{"model":"relay/claude-opus-5-5","max_tokens":4096,"thinking":{"type":"adaptive"},"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(bodies) != 2 {
			t.Fatalf("upstream calls = %d, want 2 (initial refusal + enabled retry)", len(bodies))
		}
		retry := bodies[1]
		if gjson.GetBytes(retry, "thinking.type").String() != "enabled" {
			t.Fatalf("retry thinking = %s, want enabled (body=%s)", gjson.GetBytes(retry, "thinking").Raw, retry)
		}
		if gjson.GetBytes(retry, "custom_tag").String() != "r7" {
			t.Fatalf("ExtraBody field missing: %s", retry)
		}
	})

	t.Run("R8 F1 client enabled converted to adaptive retries as enabled on refusal", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var bodies [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			bodies = append(bodies, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if gjson.GetBytes(b, "thinking.type").String() != "enabled" {
				w.WriteHeader(http.StatusBadRequest)
				io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"thinking.type must be \"enabled\""}}`)
				return
			}
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL,
			Models: []string{"claude-opus-5-5"},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		body := `{"model":"relay/claude-opus-5-5","max_tokens":32768,"thinking":{"type":"enabled","budget_tokens":16000},"messages":[{"role":"user","content":"hi"}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(bodies) != 2 {
			t.Fatalf("upstream calls = %d, want 2 (initial adaptive refusal + enabled retry)", len(bodies))
		}
		if gjson.GetBytes(bodies[0], "thinking.type").String() != "adaptive" {
			t.Fatalf("first call thinking = %s, want adaptive", gjson.GetBytes(bodies[0], "thinking").Raw)
		}
		if gjson.GetBytes(bodies[1], "thinking.type").String() != "enabled" || gjson.GetBytes(bodies[1], "thinking.budget_tokens").Int() != 16000 {
			t.Fatalf("retry thinking = %s, want enabled 16000", gjson.GetBytes(bodies[1], "thinking").Raw)
		}
	})

	t.Run("R8 F2 rule on budget Claude converts client adaptive to enabled budget", func(t *testing.T) {
		msg := quote(long(2000))
		s, up := anthropicRuleMember(t, "claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}, "high")
		body := `{"model":"group/r","max_tokens":8192,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
			t.Fatalf("thinking = %s, want enabled on budget Claude sonnet-4-5 (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
			t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("high"), raw)
		}
	})

	t.Run("R8 F3 fillsOutput cap overrides larger SetModelOutput when max_tokens omitted", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var got [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			got = append(got, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, anthropicHi)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelOutput("relay/claude-sonnet-4-5", 64000); err != nil {
			t.Fatal(err)
		}
		ctxWin := 8192
		if err := provider.SetModelSpec("relay/claude-sonnet-4-5", provider.SpecChange{Context: &ctxWin}); err != nil {
			t.Fatal(err)
		}
		body := `{"model":"relay/claude-sonnet-4-5:max","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(got) != 1 {
			t.Fatalf("upstream calls = %d, want 1", len(got))
		}
		if n := gjson.GetBytes(got[0], "max_tokens").Int(); n > 8192 {
			t.Fatalf("max_tokens = %d, exceeds fillsOutput ceiling 8192 (body=%s)", n, got[0])
		}
	})

	t.Run("R8 F4 translated Anthropic omits effort when output cap prevents thinking", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var got [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			got = append(got, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, anthropicHi)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelOutput("relay/claude-sonnet-4-5", 100); err != nil {
			t.Fatal(err)
		}
		s := New()
		body := `{"model":"relay/claude-sonnet-4-5:max","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		raw := got[0]
		mu.Unlock()
		if gjson.GetBytes(raw, "thinking").Exists() || gjson.GetBytes(raw, "output_config.effort").Exists() {
			t.Fatalf("expected thinking and output_config.effort omitted at cap 100: %s", raw)
		}
		if r := lastRoute(s); len(r.Tries) != 1 || r.Tries[0].Effort != "" {
			t.Fatalf("try %+v, want empty effort when thinking was omitted", r.Tries)
		}
	})
}

// #7389: a native budget, on a budget Claude or a vendor that does not read
// output_config.effort, is the level's full budget with max_tokens raised
// past it. A model that does read the field keeps it, and is raised the
// same way. A client's own effort is rewritten; one it did not send is not
// added.
func TestNativeBudgetRaisesMaxTokensWithoutSyntheticEffort(t *testing.T) {
	msg := quote(long(2000))
	levels := []string{"low", "medium", "high", "xhigh", "max"}
	post := func(t *testing.T, s *Server, body string) {
		t.Helper()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
	}
	for _, model := range []string{"claude-sonnet-4-5", "kimi-k3", "MiniMax-M3", "GLM-5-Turbo", "qwen3.5-plus", "vendor-m"} {
		t.Run(model+" full budget", func(t *testing.T) {
			s, up := anthropicRuleMember(t, model, levels, "xhigh")
			post(t, s, `{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":`+msg+`}]}`)
			raw := up.raw()
			if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("xhigh")) {
				t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("xhigh"), raw)
			}
			if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 8192+int64(budgetOf("xhigh")) {
				t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 8192+budgetOf("xhigh"), raw)
			}
			if gjson.GetBytes(raw, "output_config.effort").Exists() {
				t.Fatalf("synthetic output_config.effort %s (body=%s)", gjson.GetBytes(raw, "output_config.effort").Raw, raw)
			}
		})
		t.Run(model+" client budget 1024 is raised", func(t *testing.T) {
			s, up := anthropicRuleMember(t, model, levels, "high")
			post(t, s, `{"model":"group/r","max_tokens":1024,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":`+msg+`}]}`)
			raw := up.raw()
			if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
				t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("high"), raw)
			}
			if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 1024+int64(budgetOf("high")) {
				t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 1024+budgetOf("high"), raw)
			}
		})
	}
	for _, model := range []string{"glm-5.3", "MiniMax-M3.1-Flash"} {
		t.Run(model+" effort and full budget", func(t *testing.T) {
			s, up := anthropicRuleMember(t, model, []string{"low", "high", "max"}, "max")
			post(t, s, `{"model":"group/r","max_tokens":100,"messages":[{"role":"user","content":`+msg+`}]}`)
			raw := up.raw()
			if gjson.GetBytes(raw, "thinking.type").String() != "enabled" {
				t.Fatalf("thinking %s, want enabled (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
			}
			if n := gjson.GetBytes(raw, "thinking.budget_tokens").Int(); n != int64(budgetOf("max")) {
				t.Fatalf("budget_tokens = %d, want %d (body=%s)", n, budgetOf("max"), raw)
			}
			if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100+int64(budgetOf("max")) {
				t.Fatalf("max_tokens = %d, want %d (body=%s)", n, 100+budgetOf("max"), raw)
			}
			if gjson.GetBytes(raw, "output_config.effort").String() != "max" {
				t.Fatalf("effort = %q, want max (body=%s)", gjson.GetBytes(raw, "output_config.effort").String(), raw)
			}
		})
	}
	t.Run("existing effort on kimi is rewritten", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "kimi-k3", levels, "high")
		post(t, s, `{"model":"group/r","max_tokens":8192,"output_config":{"effort":"low"},"messages":[{"role":"user","content":`+msg+`},{"role":"system","content":"n","output_config":{"effort":"max"}}]}`)
		raw := up.raw()
		if gjson.GetBytes(raw, "output_config.effort").String() != "high" {
			t.Fatalf("effort = %q, want high (body=%s)", gjson.GetBytes(raw, "output_config.effort").String(), raw)
		}
		if gjson.GetBytes(raw, "messages.1.output_config.effort").String() != "high" {
			t.Fatalf("message effort = %q, want high (body=%s)", gjson.GetBytes(raw, "messages.1.output_config.effort").String(), raw)
		}
	})
}

// #7389: a Max output that cannot hold the least thinking budget does not
// synthesize thinking (withMaxOutput would write disabled, and
// thinkingOffAsTaken would then grow max_tokens) or a synthetic
// output_config.effort. The same tight flag is what a context-window reply
// cap at or under that budget passes. Adaptive thinking is not a budget, so
// a cap leaves it unless the upstream takes only enabled.
func TestTightOutputCapOmitsSyntheticThinking(t *testing.T) {
	msg := quote(long(2000))
	levels := []string{"low", "medium", "high", "xhigh", "max"}
	for _, model := range []string{"claude-opus-5-5", "claude-fable-5", "claude-mythos-5"} {
		t.Run(model+" thinkingNeeded stays inside the cap", func(t *testing.T) {
			s, up := anthropicRuleMember(t, model, levels, "max")
			if err := provider.SetModelOutput("relay/"+model, 100); err != nil {
				t.Fatal(err)
			}
			s.markUnfit("relay", thinkingNeeded(model), provider.Anthropic)
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
				`{"model":"group/r","max_tokens":8192,"temperature":0.3,"messages":[{"role":"user","content":`+msg+`}]}`)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			raw := up.raw()
			if gjson.GetBytes(raw, "thinking").Exists() {
				t.Fatalf("thinking %s, want none under Max output 100 (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
			}
			if gjson.GetBytes(raw, "output_config.effort").Exists() {
				t.Fatalf("synthetic effort %s (body=%s)", gjson.GetBytes(raw, "output_config.effort").Raw, raw)
			}
			if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
				t.Fatalf("max_tokens = %d, want 100 (body=%s)", n, raw)
			}
			if r := lastRoute(s); len(r.Tries) != 1 || r.Tries[0].Effort != "" {
				t.Fatalf("try %+v, want empty effort", r.Tries)
			}
		})
	}
	t.Run("adaptive without thinkingNeeded keeps its level under the cap", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "claude-opus-5-5", levels, "max")
		if err := provider.SetModelOutput("relay/claude-opus-5-5", 100); err != nil {
			t.Fatal(err)
		}
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
			`{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":`+msg+`}]}`)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" {
			t.Fatalf("thinking %s, want adaptive (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
		}
		if gjson.GetBytes(raw, "output_config.effort").String() != "max" {
			t.Fatalf("effort = %q, want max (body=%s)", gjson.GetBytes(raw, "output_config.effort").String(), raw)
		}
		if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
			t.Fatalf("max_tokens = %d, want 100 (body=%s)", n, raw)
		}
	})
	for _, model := range []string{"glm-5.3", "MiniMax-M3.1-Flash", "k3.1"} {
		t.Run(model+" cap omits disabled thinking and effort", func(t *testing.T) {
			s, up := anthropicRuleMember(t, model, levels, "max")
			if err := provider.SetModelOutput("relay/"+model, 100); err != nil {
				t.Fatal(err)
			}
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
				`{"model":"group/r","max_tokens":8192,"temperature":0.3,"messages":[{"role":"user","content":`+msg+`}]}`)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			raw := up.raw()
			// thinkingOffUnlessAsked may still say disabled for a vendor.
			// The cap must not have turned a synthetic enabled budget into
			// that, and must not have invented an effort beside it.
			if typ := gjson.GetBytes(raw, "thinking.type").String(); typ == "enabled" || typ == "adaptive" {
				t.Fatalf("thinking.type = %q, want none enabled (body=%s)", typ, raw)
			}
			if gjson.GetBytes(raw, "output_config.effort").Exists() {
				t.Fatalf("synthetic effort %s (body=%s)", gjson.GetBytes(raw, "output_config.effort").Raw, raw)
			}
			if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
				t.Fatalf("max_tokens = %d, want 100 (body=%s)", n, raw)
			}
			if gjson.GetBytes(raw, "temperature").Raw != "0.3" {
				t.Fatalf("temperature = %s, want 0.3 (body=%s)", gjson.GetBytes(raw, "temperature").Raw, raw)
			}
			if r := lastRoute(s); len(r.Tries) != 1 || r.Tries[0].Effort != "" {
				t.Fatalf("try %+v, want empty effort", r.Tries)
			}
		})
	}
	t.Run("room cap flag omits the same way", func(t *testing.T) {
		body := []byte(`{"max_tokens":8192,"temperature":0.2,"output_config":{"effort":"low"},"messages":[{"role":"user","content":"hi","output_config":{"effort":"high"}}]}`)
		out := withAnthropicFixedMode(body, "", "glm-5.3", "max", false, true, false, true)
		if gjson.GetBytes(out, "thinking").Exists() {
			t.Fatalf("thinking %s, want none when the cap cannot hold a budget", gjson.GetBytes(out, "thinking").Raw)
		}
		if gjson.GetBytes(out, "output_config.effort").String() != "max" || gjson.GetBytes(out, "messages.0.output_config.effort").String() != "max" {
			t.Fatalf("efforts not rewritten: %s", out)
		}
		if gjson.GetBytes(out, "temperature").Raw != "0.2" || gjson.GetBytes(out, "max_tokens").Int() != 8192 {
			t.Fatalf("cap flag changed the rest of the body: %s", out)
		}
	})
}

// #7389: a forced tool_choice adds no thinking, and still rewrites an
// output_config.effort the client already sent so it cannot outrank the rule.
func TestRuleUseEffortRewritesEffortOnForcedToolChoice(t *testing.T) {
	msg := quote(long(2000))
	for _, model := range []string{"claude-sonnet-4-5", "glm-5.3", "k3.1"} {
		t.Run(model, func(t *testing.T) {
			s, up := anthropicRuleMember(t, model, []string{"low", "medium", "high", "max"}, "low")
			body := `{"model":"group/r","max_tokens":100,"tool_choice":{"type":"any"},"tools":[{"name":"lookup","input_schema":{"type":"object"}}],"output_config":{"effort":"high"},"messages":[{"role":"user","content":` + msg + `},{"role":"system","content":"n","output_config":{"effort":"max"}}]}`
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body)
			}
			raw := up.raw()
			if typ := gjson.GetBytes(raw, "thinking.type").String(); typ == "enabled" || typ == "adaptive" {
				t.Fatalf("thinking %s added beside a forced tool_choice (body=%s)", typ, raw)
			}
			if model == "claude-sonnet-4-5" && gjson.GetBytes(raw, "thinking").Exists() {
				t.Fatalf("thinking added beside a forced tool_choice: %s", raw)
			}
			if gjson.GetBytes(raw, "output_config.effort").String() != "low" {
				t.Fatalf("effort = %q, want low (body=%s)", gjson.GetBytes(raw, "output_config.effort").String(), raw)
			}
			if gjson.GetBytes(raw, "messages.1.output_config.effort").String() != "low" {
				t.Fatalf("message effort = %q, want low (body=%s)", gjson.GetBytes(raw, "messages.1.output_config.effort").String(), raw)
			}
			if gjson.GetBytes(raw, "max_tokens").Int() != 100 {
				t.Fatalf("max_tokens = %d, want 100 (body=%s)", gjson.GetBytes(raw, "max_tokens").Int(), raw)
			}
		})
	}
	t.Run("no effort is not synthesized", func(t *testing.T) {
		s, up := anthropicRuleMember(t, "glm-5.3", []string{"low", "high", "max"}, "max")
		body := `{"model":"group/r","max_tokens":100,"tool_choice":{"type":"tool","name":"lookup"},"tools":[{"name":"lookup","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":` + msg + `}]}`
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		raw := up.raw()
		if gjson.GetBytes(raw, "output_config.effort").Exists() {
			t.Fatalf("synthetic effort: %s", raw)
		}
		if typ := gjson.GetBytes(raw, "thinking.type").String(); typ == "enabled" || typ == "adaptive" {
			t.Fatalf("thinking %s synthesized beside a forced tool_choice (body=%s)", typ, raw)
		}
	})
}

// #7389: withAdaptiveAsEnabled keeps the level outputConfigEffort reads, a
// message's effort and a \u-escaped output_config key, instead of the least
// budget a top-level path misses.
func TestAdaptiveAsEnabledReadsMessageAndEscapedEffort(t *testing.T) {
	msg, ok := withAdaptiveAsEnabled([]byte(`{"thinking":{"type":"adaptive"},"max_tokens":2048,"messages":[{"role":"user","content":"hi","output_config":{"effort":"high"}}]}`))
	if !ok {
		t.Fatal("message effort not applied")
	}
	if n := gjson.GetBytes(msg, "thinking.budget_tokens").Int(); n != int64(budgetOf("high")) {
		t.Fatalf("message effort budget = %d, want %d (%s)", n, budgetOf("high"), msg)
	}
	esc, ok := withAdaptiveAsEnabled([]byte(`{"thinking":{"type":"adaptive"},"max_tokens":2048,"output\u005fconfig":{"effort":"xhigh"}}`))
	if !ok {
		t.Fatal("escaped key not applied")
	}
	if n := gjson.GetBytes(esc, "thinking.budget_tokens").Int(); n != int64(budgetOf("xhigh")) {
		t.Fatalf("escaped effort budget = %d, want %d (%s)", n, budgetOf("xhigh"), esc)
	}
	plain, ok := withAdaptiveAsEnabled([]byte(`{"thinking":{"type":"adaptive"},"max_tokens":2048,"messages":[{"role":"user","content":"hi"}]}`))
	if !ok {
		t.Fatal("least budget not applied")
	}
	if n := gjson.GetBytes(plain, "thinking.budget_tokens").Int(); n != neededBudget {
		t.Fatalf("no effort budget = %d, want %d (%s)", n, neededBudget, plain)
	}
}

// #7389: a settings.Save while a turn is upstream does not split the effort
// shape from the model that turn already went out under. The next turn is
// the one the new name is for, shape and model together.
func TestUseEffortWireShapeStaysWithTheTurnsModel(t *testing.T) {
	fresh(t)
	var mu sync.Mutex
	var seen [][]byte
	named := false
	relay := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		seen = append(seen, append([]byte(nil), b...))
		save := !named
		named = true
		mu.Unlock()
		if save {
			st := settings.Load()
			st.ModelWires = map[string]string{"relay/review-sonnet": "claude-opus-5-5"}
			if err := settings.Save(st); err != nil {
				t.Error(err)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(relay.Close)
	if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: relay.URL, Models: []string{"review-sonnet"}}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetUpstreamName("relay/review-sonnet", "claude-sonnet-4-5"); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("relay/review-sonnet", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "R", Members: []string{"relay/review-sonnet"}, Routing: provider.Ordered,
		Rules: []provider.Rule{{Use: "relay/review-sonnet", Tokens: 1000, UseEffort: "high"}},
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	body := `{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":` + quote(long(2000)) + `}]}`
	for range 2 {
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 2 {
		t.Fatalf("upstream calls = %d, want 2", len(seen))
	}
	first := seen[0]
	if gjson.GetBytes(first, "model").String() != "claude-sonnet-4-5" {
		t.Fatalf("first model = %q, want the name the turn began with (body=%s)", gjson.GetBytes(first, "model").String(), first)
	}
	if gjson.GetBytes(first, "thinking.budget_tokens").Int() != int64(budgetOf("high")) || gjson.GetBytes(first, "output_config.effort").Exists() {
		t.Fatalf("first effort shape is not the budget Claude the model names: %s", first)
	}
	second := seen[1]
	if gjson.GetBytes(second, "model").String() != "claude-opus-5-5" {
		t.Fatalf("second model = %q, want the name saved mid-flight (body=%s)", gjson.GetBytes(second, "model").String(), second)
	}
	if gjson.GetBytes(second, "thinking.type").String() != "adaptive" || gjson.GetBytes(second, "output_config.effort").String() != "high" {
		t.Fatalf("second effort shape is not the adaptive Claude the model names: %s", second)
	}
}

// #7389 R10 regressions: clipped translated budgets log effortOfBudget(b),
// DeepSeek host writes output_config.effort natively, tight caps preserve
// client adaptive thinking and omit translated synthetic thinking under
// thinkingNeeded, aliased Gemini 4 sends adaptive without budget_tokens,
// and WithSideEffort minimal raises max_tokens once.
func TestRuleUseEffortR10Regressions(t *testing.T) {
	msg := quote(long(2000))

	t.Run("F1 clipped translated Anthropic budget logs effortOfBudget", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var got [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			got = append(got, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, anthropicHi)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelOutput("relay/claude-sonnet-4-5", 8192); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/claude-sonnet-4-5"}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/claude-sonnet-4-5", Tokens: 1000, UseEffort: "high"}},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		// Translated /v1/messages via web_search_20250305
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
			`{"model":"group/r","max_tokens":8192,"messages":[{"role":"user","content":`+msg+`}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`)))
		if rec.Code != 200 {
			t.Fatalf("translated messages: %d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		raw := got[0]
		mu.Unlock()
		b := int(gjson.GetBytes(raw, "thinking.budget_tokens").Int())
		if b != 8191 {
			t.Fatalf("budget_tokens = %d, want 8191 (body=%s)", b, raw)
		}
		want := effortOfBudget(b)
		if r := lastRoute(s); len(r.Tries) != 1 || r.Tries[0].Effort != want || want == "" {
			t.Fatalf("try %+v, want Effort=%q", r.Tries, want)
		}
		// Chat -> Anthropic with max_tokens 8192 and reasoning_effort high (budget clipped to 7168)
		rec = httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(
			`{"model":"relay/claude-sonnet-4-5","stream":true,"max_tokens":8192,"reasoning_effort":"high","messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != 200 {
			t.Fatalf("chat->anthropic: %d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		raw2 := got[1]
		mu.Unlock()
		b2 := int(gjson.GetBytes(raw2, "thinking.budget_tokens").Int())
		want2 := effortOfBudget(b2)
		if r := lastRoute(s); len(r.Tries) != 1 || r.Tries[0].Effort != want2 || want2 == "" {
			t.Fatalf("chat->anthropic try %+v, want Effort=%q (budget=%d)", r.Tries, want2, b2)
		}
	})

	t.Run("F2 DeepSeek host writes output_config.effort natively and in WithSideEffort", func(t *testing.T) {
		fresh(t)
		body := []byte(`{"model":"deepseek-reasoner","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`)
		out := withAnthropicFixedMode(body, deepseekHost, "deepseek-reasoner", "max", false, false, false, false)
		out = withMaxOutput(provider.Anthropic, out, 4096)
		if ef := gjson.GetBytes(out, "output_config.effort").String(); ef != "max" {
			t.Fatalf("deepseek output_config.effort = %q, want max (body=%s)", ef, out)
		}
		if b := gjson.GetBytes(out, "thinking.budget_tokens").Int(); b != 4095 {
			t.Fatalf("deepseek budget_tokens = %d, want 4095 (body=%s)", b, out)
		}
		// WithSideEffort on DeepSeek's host keeps output_config.effort = max
		if err := provider.Save(provider.Provider{
			ID: "ds", Name: "DeepSeek", Key: "k",
			Anthropic: "https://api.deepseek.com/anthropic",
			Models:    []string{"deepseek-reasoner"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("ds/deepseek-reasoner", []string{"low", "medium", "high", "xhigh", "max"}); err != nil {
			t.Fatal(err)
		}
		side := WithSideEffort([]byte(`{"model":"ds/deepseek-reasoner","max_tokens":4096,"messages":[{"role":"user","content":"hi"}]}`), "ds/deepseek-reasoner", "max")
		if ef := gjson.GetBytes(side, "output_config.effort").String(); ef != "max" {
			t.Fatalf("WithSideEffort deepseek effort = %q, want max (body=%s)", ef, side)
		}
	})

	t.Run("F3 tight cap preserves client adaptive thinking and omits translated synthetic thinking", func(t *testing.T) {
		levels := []string{"low", "medium", "high", "xhigh", "max"}
		// 1. offNever Claude with thinkingNeeded + SetModelOutput 100 + client adaptive -> max_tokens stays 100
		for _, model := range []string{"claude-opus-5-5", "claude-fable-5", "claude-mythos-5"} {
			s, up := anthropicRuleMember(t, model, levels, "max")
			if err := provider.SetModelOutput("relay/"+model, 100); err != nil {
				t.Fatal(err)
			}
			s.markUnfit("relay", thinkingNeeded(model), provider.Anthropic)
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
				`{"model":"group/r","max_tokens":8192,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":`+msg+`}]}`)))
			if rec.Code != 200 {
				t.Fatalf("%s: %d %s", model, rec.Code, rec.Body)
			}
			raw := up.raw()
			if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" {
				t.Fatalf("%s thinking = %s, want adaptive (body=%s)", model, gjson.GetBytes(raw, "thinking").Raw, raw)
			}
			if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
				t.Fatalf("%s max_tokens = %d, want 100 (body=%s)", model, n, raw)
			}
		}
		// 2. glm-5.3 with SetModelOutput 100 + client adaptive + UseEffort max -> keeps adaptive + max at 100
		{
			s, up := anthropicRuleMember(t, "glm-5.3", levels, "max")
			if err := provider.SetModelOutput("relay/glm-5.3", 100); err != nil {
				t.Fatal(err)
			}
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
				`{"model":"group/r","max_tokens":8192,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":`+msg+`}]}`)))
			if rec.Code != 200 {
				t.Fatalf("glm-5.3: %d %s", rec.Code, rec.Body)
			}
			raw := up.raw()
			if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" {
				t.Fatalf("glm-5.3 thinking = %s, want adaptive (body=%s)", gjson.GetBytes(raw, "thinking").Raw, raw)
			}
			if ef := gjson.GetBytes(raw, "output_config.effort").String(); ef != "max" {
				t.Fatalf("glm-5.3 effort = %q, want max (body=%s)", ef, raw)
			}
			if n := gjson.GetBytes(raw, "max_tokens").Int(); n != 100 {
				t.Fatalf("glm-5.3 max_tokens = %d, want 100 (body=%s)", n, raw)
			}
		}
		// 3. Translated /v1/messages (web_search) with no client thinking, thinkingNeeded, SetModelOutput 100 on adaptive-only Claudes
		for _, model := range []string{"claude-opus-5-5", "claude-opus-4-6"} {
			fresh(t)
			var mu sync.Mutex
			var got [][]byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				mu.Lock()
				got = append(got, append([]byte(nil), b...))
				mu.Unlock()
				w.Header().Set("Content-Type", "text/event-stream")
				io.WriteString(w, anthropicHi)
			}))
			t.Cleanup(srv.Close)
			if err := provider.Save(provider.Provider{
				ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{model},
			}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetModelEfforts("relay/"+model, levels); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetModelOutput("relay/"+model, 100); err != nil {
				t.Fatal(err)
			}
			if err := provider.SaveGroup(provider.Group{
				Name: "R", Members: []string{"relay/" + model}, Routing: provider.Ordered,
				Rules: []provider.Rule{{Use: "relay/" + model, Tokens: 1000, UseEffort: "max"}},
			}); err != nil {
				t.Fatal(err)
			}
			s := New()
			s.markUnfit("relay", thinkingNeeded(model), provider.Anthropic)
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
				`{"model":"group/r","max_tokens":100,"temperature":0.4,"messages":[{"role":"user","content":`+msg+`}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`)))
			if rec.Code != 200 {
				t.Fatalf("translated %s: %d %s", model, rec.Code, rec.Body)
			}
			mu.Lock()
			raw := got[0]
			mu.Unlock()
			if gjson.GetBytes(raw, "thinking").Exists() || gjson.GetBytes(raw, "output_config.effort").Exists() {
				t.Fatalf("translated %s leaked synthetic thinking/effort: %s", model, raw)
			}
			if gjson.GetBytes(raw, "max_tokens").Int() != 100 {
				t.Fatalf("translated %s max_tokens = %d, want 100: %s", model, gjson.GetBytes(raw, "max_tokens").Int(), raw)
			}
			if model == "claude-opus-4-6" && gjson.GetBytes(raw, "temperature").Raw != "0.4" {
				t.Fatalf("translated %s dropped temperature: %s", model, raw)
			}
			if r := lastRoute(s); len(r.Tries) != 1 || r.Tries[0].Effort != "" {
				t.Fatalf("translated %s try %+v, want empty effort", model, r.Tries)
			}
		}
	})

	t.Run("F4 aliased Gemini 4 Anthropic relay sends adaptive without budget_tokens", func(t *testing.T) {
		fresh(t)
		var mu sync.Mutex
		var seen [][]byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			mu.Lock()
			seen = append(seen, append([]byte(nil), b...))
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"gemini-4-hydrogen","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(srv.Close)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL, Models: []string{"my-gemini"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetUpstreamName("relay/my-gemini", "gemini-4-hydrogen"); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/my-gemini", []string{"low", "medium", "high", "xhigh"}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "R", Members: []string{"relay/my-gemini"}, Routing: provider.Ordered,
			Rules: []provider.Rule{{Use: "relay/my-gemini", Tokens: 1000, UseEffort: "xhigh"}},
		}); err != nil {
			t.Fatal(err)
		}
		s := New()
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
			`{"model":"group/r","max_tokens":4096,"messages":[{"role":"user","content":`+msg+`}]}`)))
		if rec.Code != 200 {
			t.Fatalf("rule aliased gemini: %d %s", rec.Code, rec.Body)
		}
		// Direct client budget request to aliased Gemini 4 also converts via geminiLevelBody
		rec = httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(
			`{"model":"relay/my-gemini","max_tokens":12000,"thinking":{"type":"enabled","budget_tokens":8000},"messages":[{"role":"user","content":"hi"}]}`)))
		if rec.Code != 200 {
			t.Fatalf("direct aliased gemini: %d %s", rec.Code, rec.Body)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(seen) != 2 {
			t.Fatalf("calls = %d, want 2", len(seen))
		}
		for i, raw := range seen {
			if gjson.GetBytes(raw, "thinking.type").String() != "adaptive" || gjson.GetBytes(raw, "thinking.budget_tokens").Exists() {
				t.Fatalf("call[%d] leaked budget or non-adaptive thinking: %s", i, raw)
			}
			if gjson.GetBytes(raw, "model").String() != "gemini-4-hydrogen" {
				t.Fatalf("call[%d] model = %q, want gemini-4-hydrogen", i, gjson.GetBytes(raw, "model").String())
			}
		}
		if n := gjson.GetBytes(seen[0], "max_tokens").Int(); n != 4096 {
			t.Fatalf("aliased gemini rule inflated max_tokens to %d, want 4096 (body=%s)", n, seen[0])
		}
	})

	t.Run("F6 WithSideEffort minimal raises max_tokens once on budget model", func(t *testing.T) {
		fresh(t)
		if err := provider.Save(provider.Provider{
			ID: "relay", Name: "Relay", Key: "k", Anthropic: "http://127.0.0.1:9", Models: []string{"claude-sonnet-4-5"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SetModelEfforts("relay/claude-sonnet-4-5", []string{"minimal", "low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
		orig := []byte(`{"model":"relay/claude-sonnet-4-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
		out := WithSideEffort(orig, "relay/claude-sonnet-4-5", "minimal")
		wantMax := int64(100 + budgetOf("low"))
		if n := gjson.GetBytes(out, "max_tokens").Int(); n != wantMax {
			t.Fatalf("WithSideEffort minimal max_tokens = %d, want %d (body=%s)", n, wantMax, out)
		}
		if b := gjson.GetBytes(out, "thinking.budget_tokens").Int(); b != int64(budgetOf("low")) {
			t.Fatalf("WithSideEffort minimal budget_tokens = %d, want %d (body=%s)", b, budgetOf("low"), out)
		}
	})
}
