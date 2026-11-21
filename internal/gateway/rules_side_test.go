package gateway

import (
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/access"
	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
)

// A session-title call shares the turn's key — the session and the user's
// first words — but it is its own short conversation. The rule it matches
// routes that call, and the turn stays where it was decided, before the
// call and on the tool rounds after it. A later turn of the conversation
// is decided again (#7455).
func TestRuleSideCallDoesNotRewriteTheTurn(t *testing.T) {
	const (
		user  = "write a haiku about rivers"
		agent = "You are a coding agent."
	)
	titleRule := provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "system", Op: "starts", Value: "You are an expert conversation summarizer"}}}
	turnRule := provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "message", Op: "contains", Value: "ship it"}}}
	s, _, _ := ruled(t, titleRule, turnRule)
	sys := func(system, body string) string {
		return strings.Replace(body, `"messages":[`, `"messages":[{"role":"system","content":`+quote(system)+`},`, 1)
	}
	post := func(session, body string) (string, RuleHit) {
		t.Helper()
		code, out := postAs(t, s, session, body)
		r := lastRoute(s)
		if code != 200 || r.Rule == nil {
			t.Fatalf("%s: %d %s %+v", session, code, out, r.Rule)
		}
		return out, *r.Rule
	}
	main := sys(agent, chat(user, nil, 0, ""))
	title := sys("You are an expert conversation summarizer. Your goal is to create a concise, 2-5 word title that captures the main intent of the user's message.", chat(user, nil, 0, `,"max_tokens":64`))
	out, hit := post("side", main)
	if !strings.Contains(out, "from ka") || hit.N != 0 || hit.Held || hit.Aside {
		t.Fatalf("turn: %s %+v", out, hit)
	}
	out, hit = post("side", title)
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Held || !hit.Aside {
		t.Fatalf("title: %s %+v", out, hit)
	}
	out, hit = post("side", sys(agent, chat(user, nil, 1, "")))
	if !strings.Contains(out, "from ka") || hit.N != 0 || !hit.Held || hit.Aside {
		t.Fatalf("tool round: %s %+v", out, hit)
	}
	// the next thing the user says is decided on its own, and that holds
	out, hit = post("side", sys(agent, chat(user, []string{"ship it"}, 0, "")))
	if !strings.Contains(out, "from kb") || hit.N != 2 || hit.Held || hit.Aside {
		t.Fatalf("next turn: %s %+v", out, hit)
	}
	out, hit = post("side", sys(agent, chat(user, []string{"ship it"}, 1, "")))
	if !strings.Contains(out, "from kb") || hit.N != 2 || !hit.Held {
		t.Fatalf("next turn's tool round: %s %+v", out, hit)
	}
	// a side call whose prompt isn't a known title, after the turn is
	// stored, routes itself and leaves the turn
	suggest := provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "system", Op: "starts", Value: "Suggest one short follow-up"}}}
	s, _, _ = ruled(t, suggest)
	out, hit = post("suggest", sys(agent, chat(user, nil, 0, "")))
	if !strings.Contains(out, "from ka") || hit.N != 0 || hit.Held {
		t.Fatalf("before the suggestion: %s %+v", out, hit)
	}
	out, hit = post("suggest", sys("Suggest one short follow-up the user might ask.", chat(user, nil, 0, `,"max_tokens":64`)))
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Held || !hit.Aside {
		t.Fatalf("suggestion: %s %+v", out, hit)
	}
	out, hit = post("suggest", sys(agent, chat(user, nil, 1, "")))
	if !strings.Contains(out, "from ka") || hit.N != 0 || !hit.Held || hit.Aside {
		t.Fatalf("after the suggestion: %s %+v", out, hit)
	}
}

// A known title prompt never writes the turn, even when it is the first
// request on the key. The turn that follows is decided as if the title
// had not been there. The same for each agent's title prompt (#7455).
func TestRuleTitleBeforeTheTurnLeavesIt(t *testing.T) {
	user := "write a haiku about rivers"
	prompts := []string{
		"You are an expert conversation summarizer. Your goal is to create a concise, 2-5 word title that captures the main intent of the user's message.",
		"You write short session titles.",
		"Generate a concise, sentence-case title (3-7 words) that captures the main topic or goal of this coding session.",
		"Create a concise title for an AI coding-assistant session from the supplied human messages.",
		"You are a title generator. Reply with the title only.",
		"You name AI coding-assistant sessions. Reply with the name only.",
	}
	for i, p := range prompts {
		s, _, _ := ruled(t, provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "system", Op: "starts", Value: p}}})
		title := `{"model":"group/r","max_tokens":64,"messages":[{"role":"system","content":` + quote(p) + `},{"role":"user","content":` + quote(user) + `}]}`
		main := `{"model":"group/r","messages":[{"role":"system","content":"You are a coding agent."},{"role":"user","content":` + quote(user) + `}]}`
		sess := "title-first-" + string(rune('a'+i))
		code, out := postAs(t, s, sess, title)
		hit := lastRoute(s).Rule
		if code != 200 || !strings.Contains(out, "from kb") || hit == nil || hit.N != 1 || hit.Held || !hit.Aside {
			t.Fatalf("title %q: %d %s %+v", p, code, out, hit)
		}
		code, out = postAs(t, s, sess, main)
		hit = lastRoute(s).Rule
		if code != 200 || !strings.Contains(out, "from ka") || hit == nil || hit.N != 0 || hit.Held || hit.Aside {
			t.Fatalf("turn after %q: %d %s %+v", p, code, out, hit)
		}
		code, out = postAs(t, s, sess, strings.Replace(chat(user, nil, 1, ""), `"messages":[`, `"messages":[{"role":"system","content":"You are a coding agent."},`, 1))
		hit = lastRoute(s).Rule
		if code != 200 || !strings.Contains(out, "from ka") || hit == nil || hit.N != 0 || !hit.Held {
			t.Fatalf("tool round after %q: %d %s %+v", p, code, out, hit)
		}
	}
}

const agentPrompt = "You are a coding agent."

// billed is an agent prompt with Claude Code's billing line and a reminder,
// both of which change between a turn and its retry.
func billed(cch, reminder string) string {
	return "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli; cch=" + cch + ";\n" +
		agentPrompt + "\n<system-reminder>\n" + reminder + "\n</system-reminder>"
}

func withSystem(system, body string) string {
	return strings.Replace(body, `"messages":[`, `"messages":[{"role":"system","content":`+quote(system)+`},`, 1)
}

func withTools(body string) string {
	i := strings.Index(body, `"messages":`)
	if i < 0 {
		return body
	}
	return body[:i] + `"tools":[{"type":"function","function":{"name":"Bash","parameters":{"type":"object"}}}],` + body[i:]
}

func mustRoute(t *testing.T, s *Server, session, body string) (string, RuleHit) {
	t.Helper()
	code, out := postAs(t, s, session, body)
	r := lastRoute(s)
	if code != 200 || r.Rule == nil {
		t.Fatalf("%s: %d %s %+v", session, code, out, r.Rule)
	}
	return out, *r.Rule
}

func plantTurnState(input int) {
	turnRules.Lock()
	defer turnRules.Unlock()
	for k, tr := range turnRules.m {
		tr.input = input
		tr.hour = hourCache{tokens: input, at: tr.at}
		turnRules.m[k] = tr
	}
}

func turnStateKept(input int) bool {
	turnRules.Lock()
	defer turnRules.Unlock()
	if len(turnRules.m) == 0 {
		return false
	}
	for _, tr := range turnRules.m {
		if tr.input != input || tr.hour.tokens != input {
			return false
		}
	}
	return true
}

func rewriteTurnUse(use string) {
	turnRules.Lock()
	defer turnRules.Unlock()
	for k, tr := range turnRules.m {
		tr.use, tr.n = use, 7
		turnRules.m[k] = tr
	}
}

func turnUses() []string {
	turnRules.Lock()
	defer turnRules.Unlock()
	var out []string
	for _, tr := range turnRules.m {
		out = append(out, tr.use)
	}
	return out
}

func stickModels() map[string]string {
	sticks.Lock()
	defer sticks.Unlock()
	out := map[string]string{}
	for k, st := range sticks.m {
		out[k] = st.model
	}
	return out
}

// Editing turn 2, and rewinding turn 3 back to turn 2, are the turn. The
// tool rounds after them hold the rule that edit or rewind decided.
func TestRuleEditAndRewindHoldTheNewRule(t *testing.T) {
	ship := provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "message", Op: "contains", Value: "ship it"}}}
	s, _, _ := ruled(t, ship)
	main := func(turns []string, tools int) string {
		return withSystem(agentPrompt, withTools(chat("hello", turns, tools, "")))
	}
	out, hit := mustRoute(t, s, "edit", main(nil, 0))
	if !strings.Contains(out, "from ka") || hit.N != 0 || hit.Aside {
		t.Fatalf("turn 1: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "edit", main([]string{"draft the note"}, 0))
	if !strings.Contains(out, "from ka") || hit.N != 0 || hit.Held || hit.Aside {
		t.Fatalf("turn 2: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "edit", main([]string{"ship it"}, 0))
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Held || hit.Aside {
		t.Fatalf("edited turn 2: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "edit", main([]string{"ship it"}, 1))
	if !strings.Contains(out, "from kb") || hit.N != 1 || !hit.Held || hit.Aside {
		t.Fatalf("edited turn's tool round: %s %+v", out, hit)
	}

	s, _, _ = ruled(t, ship)
	out, hit = mustRoute(t, s, "rewind", main(nil, 0))
	if !strings.Contains(out, "from ka") || hit.Aside {
		t.Fatalf("rewind turn 1: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "rewind", main([]string{"ship it"}, 0))
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Aside {
		t.Fatalf("rewind turn 2: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "rewind", main([]string{"ship it", "later"}, 0))
	if !strings.Contains(out, "from ka") || hit.N != 0 || hit.Aside {
		t.Fatalf("turn 3: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "rewind", main([]string{"ship it"}, 0))
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Held || hit.Aside {
		t.Fatalf("rewound to turn 2: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "rewind", main([]string{"ship it"}, 1))
	if !strings.Contains(out, "from kb") || hit.N != 1 || !hit.Held || hit.Aside {
		t.Fatalf("rewound turn's tool round: %s %+v", out, hit)
	}
}

// A retry of a failed turn 1 keeps the turn: a turn_aborted line, a new
// reminder and a new billing hash are not another conversation.
func TestRuleRetryOfAFailedTurnUpdatesIt(t *testing.T) {
	if systemID(&Request{System: billed("aaa", "plan A")}) != systemID(&Request{System: billed("bbb", "plan B")}) {
		t.Fatal("cch and a reminder changed the prompt's identity")
	}
	s, _, _ := ruled(t, provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "message", Op: "contains", Value: "ship it"}}})
	turn := withSystem(billed("aaa", "plan A"), withTools(chat("ship it", nil, 0, "")))
	out, hit := mustRoute(t, s, "retry", turn)
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Aside {
		t.Fatalf("turn 1: %s %+v", out, hit)
	}
	rewriteTurnUse("nope")
	retry := strings.Replace(turn, quote(billed("aaa", "plan A")), quote(billed("bbb", "plan B")), 1)
	retry = strings.Replace(retry, quote("ship it"), quote("<turn_aborted>\nThe user interrupted.\n</turn_aborted>")+`},{"role":"user","content":`+quote("ship it"), 1)
	out, hit = mustRoute(t, s, "retry", retry)
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Held || hit.Aside {
		t.Fatalf("retry: %s %+v", out, hit)
	}
	if uses := turnUses(); len(uses) != 1 || uses[0] != "b/big" {
		t.Fatalf("retry left the turn at %v", uses)
	}
	out, hit = mustRoute(t, s, "retry", withSystem(billed("ccc", "plan C"), withTools(chat("ship it", nil, 1, ""))))
	if !strings.Contains(out, "from kb") || hit.N != 1 || !hit.Held || hit.Aside {
		t.Fatalf("retry's tool round: %s %+v", out, hit)
	}
}

// A summary or a prompt suggestion replays the transcript and adds one
// question, so it counts as a later turn. It is still a side call: the
// turn stays at 1, and the real turn 2 is decided on its own.
func TestRuleTranscriptSideCallLeavesTheTurn(t *testing.T) {
	ask := provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "last", Op: "contains", Value: "3-5 words"}}}
	suggest := provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "last", Op: "contains", Value: "SUGGESTION MODE"}}}
	s, _, _ := ruled(t, ask, suggest)
	main := withSystem(agentPrompt, withTools(chat("write a haiku about rivers", nil, 0, "")))
	out, hit := mustRoute(t, s, "summary", main)
	if !strings.Contains(out, "from ka") || hit.N != 0 || hit.Aside || hit.Turn != 1 {
		t.Fatalf("turn 1: %s %+v", out, hit)
	}
	summary := withSystem(agentPrompt, withTools(chat("write a haiku about rivers", []string{"Describe your most recent action in 3-5 words."}, 0, "")))
	out, hit = mustRoute(t, s, "summary", summary)
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Held || !hit.Aside {
		t.Fatalf("summary: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "summary", withSystem(agentPrompt, withTools(chat("write a haiku about rivers", nil, 1, ""))))
	if !strings.Contains(out, "from ka") || hit.N != 0 || !hit.Held || hit.Aside || hit.Turn != 1 {
		t.Fatalf("tool round after the summary: %s %+v", out, hit)
	}
	suggestion := withSystem(agentPrompt, withTools(chat("write a haiku about rivers", []string{"[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code.]"}, 0, "")))
	out, hit = mustRoute(t, s, "summary", suggestion)
	if !strings.Contains(out, "from kb") || hit.N != 2 || !hit.Aside {
		t.Fatalf("suggestion: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "summary", withSystem(agentPrompt, withTools(chat("write a haiku about rivers", []string{"ship the haiku"}, 0, ""))))
	if !strings.Contains(out, "from ka") || hit.Held || hit.Aside || hit.Turn != 2 {
		t.Fatalf("real turn 2: %s %+v", out, hit)
	}
}

// A short title after a long turn is matched at its own length. The turn's
// token floor, the intent it was given and the effort kept for the cache
// stay the turn's.
func TestRuleSideCallKeepsItsOwnSize(t *testing.T) {
	title := "You are an expert conversation summarizer. Your goal is to create a concise, 2-5 word title of the chat."
	s, _, _ := ruled(t,
		provider.Rule{Use: "b/big", Tokens: 20000},
		provider.Rule{Use: "a/small", Where: []provider.Cond{{Field: "system", Op: "starts", Value: "You are an expert conversation summarizer"}}},
	)
	out, hit := mustRoute(t, s, "floor", withSystem(agentPrompt, withTools(chat(long(30000), nil, 0, ""))))
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Aside || hit.Tokens < 20000 {
		t.Fatalf("long turn: %s %+v", out, hit)
	}
	plantTurnState(30000)
	titleBody := withSystem(title, chat("write a haiku about rivers", nil, 0, `,"max_tokens":64`))
	out, hit = mustRoute(t, s, "floor", titleBody)
	if !strings.Contains(out, "from ka") || hit.N != 2 || hit.Held || !hit.Aside || hit.Tokens >= 20000 {
		t.Fatalf("title took the turn's length: %s %+v", out, hit)
	}
	if !turnStateKept(30000) {
		t.Fatal("the title wrote the turn's counted size")
	}

	g := provider.Group{
		ID: "r", Effort: provider.EffortAuto, Classifier: "c/m",
		Rules: []provider.Rule{{Use: "b/big", Intent: "ship it"}},
	}
	req := &Request{
		System:   title,
		Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "write a haiku about rivers"}}}},
		Effort:   "low", Thinking: true,
	}
	key := "group/r|effort|x"
	turnRules.Lock()
	turnRules.m[key] = turnRule{turn: 1, use: "a/small", at: time.Now(), input: 30000, intent: "ship it", effort: "xhigh", sys: "main"}
	turnRules.Unlock()
	t.Cleanup(func() {
		turnRules.Lock()
		delete(turnRules.m, key)
		turnRules.Unlock()
	})
	got := ruleFor(key, g, nil, req, "claude", nil, "")
	if got == nil || !got.Aside || got.Use != "" || got.Pick != "" || got.Tokens >= 20000 {
		t.Fatalf("effort: %+v", got)
	}
	if got.Classified != nil && (got.Classified.After != "" || got.Classified.Fallback != "" || got.Classified.Intent != "") {
		t.Fatalf("inherited the turn: %+v", got.Classified)
	}
	turnRules.Lock()
	tr := turnRules.m[key]
	turnRules.Unlock()
	if tr.use != "a/small" || tr.effort != "xhigh" || tr.intent != "ship it" || tr.input != 30000 {
		t.Fatalf("the title wrote the turn: %+v", tr)
	}
}

// A side call inside a group — the outer ruled or not — does not move the
// stick or the counted size of either group's turn.
func TestRuleNestedSideCallLeavesTheTurn(t *testing.T) {
	for _, outerRuled := range []bool{false, true} {
		t.Run(map[bool]string{false: "unruled outer", true: "ruled outer"}[outerRuled], func(t *testing.T) {
			fresh(t)
			a, b := &ruleUp{key: "ka"}, &ruleUp{key: "kb"}
			no, yes := false, true
			for _, x := range []struct {
				id, model string
				up        *ruleUp
				m         catalog.Model
			}{
				{"a", "small", a, catalog.Model{ID: "small", Context: 64000, ImageInput: &no}},
				{"b", "big", b, catalog.Model{ID: "big", Context: 1000000, Images: true, ImageInput: &yes}},
			} {
				srv := httptest.NewServer(x.up)
				t.Cleanup(srv.Close)
				if err := provider.Save(provider.Provider{ID: x.id, Name: strings.ToUpper(x.id), Key: "k" + x.id, Models: []string{x.model}, Chat: srv.URL + "/v1"}); err != nil {
					t.Fatal(err)
				}
				if err := catalog.SaveLive(x.id, srv.URL+"/v1", []catalog.Model{x.m}); err != nil {
					t.Fatal(err)
				}
			}
			inner := provider.Group{Name: "Inner", Members: []string{"a/small", "b/big"}, Routing: provider.Ordered,
				Rules: []provider.Rule{{Use: "b/big", Where: []provider.Cond{{Field: "system", Op: "starts", Value: "You are an expert conversation summarizer"}}}}}
			if err := provider.SaveGroup(inner); err != nil {
				t.Fatal(err)
			}
			outer := provider.Group{Name: "Outer", Members: []string{"group/inner"}, Routing: provider.Ordered}
			if outerRuled {
				outer.Rules = []provider.Rule{{Use: "group/inner", Where: []provider.Cond{{Field: "system", Op: "starts", Value: agentPrompt}}}}
			}
			if err := provider.SaveGroup(outer); err != nil {
				t.Fatal(err)
			}
			s := New()
			asOuter := func(body string) string {
				return strings.Replace(body, `"model":"group/r"`, `"model":"group/outer"`, 1)
			}
			post := func(body string) (string, *RuleHit) {
				t.Helper()
				code, out := postAs(t, s, "nest", body)
				if code != 200 {
					t.Fatalf("%d %s", code, out)
				}
				return out, lastRoute(s).Rule
			}
			main := asOuter(withSystem(agentPrompt, withTools(chat("write a haiku about rivers", nil, 0, ""))))
			out, hit := post(main)
			if !strings.Contains(out, "from ka") {
				t.Fatalf("turn: %s", out)
			}
			r := lastRoute(s)
			if outerRuled && (hit == nil || hit.Aside || hit.Use != "group/inner") {
				t.Fatalf("outer rule: %+v", hit)
			}
			if !outerRuled && hit != nil {
				t.Fatalf("unruled outer produced a rule: %+v", hit)
			}
			if len(r.Nested) != 1 || r.Nested[0].Rule == nil || r.Nested[0].Rule.Aside || r.Nested[0].Rule.N != 0 {
				t.Fatalf("inner rule: %+v", r.Nested)
			}
			plantTurnState(30000)
			stuck := stickModels()
			title := `{"model":"group/outer","max_tokens":64,"messages":[{"role":"system","content":` + quote("You are an expert conversation summarizer. Your goal is to create a concise, 2-5 word title") + `},{"role":"user","content":` + quote("write a haiku about rivers") + `}]}`
			out, hit = post(title)
			if !strings.Contains(out, "from kb") {
				t.Fatalf("title: %s %+v", out, hit)
			}
			nr := lastRoute(s)
			if len(nr.Nested) != 1 || nr.Nested[0].Rule == nil || !nr.Nested[0].Rule.Aside || nr.Nested[0].Rule.Use != "b/big" {
				t.Fatalf("nested title: %+v", nr.Nested)
			}
			if outerRuled && (hit == nil || !hit.Aside) {
				t.Fatalf("outer title: %+v", hit)
			}
			if !outerRuled && hit != nil {
				t.Fatalf("unruled outer title: %+v", hit)
			}
			if !turnStateKept(30000) {
				t.Fatal("the title wrote a nested turn")
			}
			if got := stickModels(); len(got) != len(stuck) {
				t.Fatalf("sticks %v, were %v", got, stuck)
			} else {
				for k, model := range stuck {
					if got[k] != model {
						t.Fatalf("stick %s moved from %s to %s", k, model, got[k])
					}
				}
			}
			tool := asOuter(withSystem(agentPrompt, withTools(chat("write a haiku about rivers", nil, 1, ""))))
			out, hit = post(tool)
			if !strings.Contains(out, "from ka") {
				t.Fatalf("tool round: %s %+v nested %+v", out, hit, lastRoute(s).Nested)
			}
			if nr := lastRoute(s); len(nr.Nested) != 1 || nr.Nested[0].Rule == nil || nr.Nested[0].Rule.Aside || !nr.Nested[0].Rule.Held {
				t.Fatalf("inner tool round: %+v", nr.Nested)
			}
		})
	}

	t.Run("ruled outer routes unrelated helper to unvisited ruled inner", func(t *testing.T) {
		fresh(t)
		a, b := &ruleUp{key: "ka"}, &ruleUp{key: "kb"}
		no, yes := false, true
		for _, x := range []struct {
			id, model string
			up        *ruleUp
			m         catalog.Model
		}{
			{"a", "small", a, catalog.Model{ID: "small", Context: 64000, ImageInput: &no}},
			{"b", "big", b, catalog.Model{ID: "big", Context: 1000000, Images: true, ImageInput: &yes}},
		} {
			srv := httptest.NewServer(x.up)
			t.Cleanup(srv.Close)
			if err := provider.Save(provider.Provider{ID: x.id, Name: strings.ToUpper(x.id), Key: "k" + x.id, Models: []string{x.model}, Chat: srv.URL + "/v1"}); err != nil {
				t.Fatal(err)
			}
			if err := catalog.SaveLive(x.id, srv.URL+"/v1", []catalog.Model{x.m}); err != nil {
				t.Fatal(err)
			}
		}
		helpers := provider.Group{
			Name: "Helpers", Members: []string{"a/small", "b/big"}, Routing: provider.Ordered,
			Effort: provider.EffortAuto, Classifier: "b/big",
			Rules: []provider.Rule{
				{Use: "a/small", Intent: "coding"},
				{Use: "b/big", Where: []provider.Cond{{Field: "system", Op: "starts", Value: "Suggest one short follow-up"}}},
			},
		}
		if err := provider.SaveGroup(helpers); err != nil {
			t.Fatal(err)
		}
		outer := provider.Group{
			Name: "Outer", Members: []string{"a/small", "group/helpers"}, Routing: provider.Ordered,
			Rules: []provider.Rule{
				{Use: "group/helpers", Where: []provider.Cond{{Field: "system", Op: "starts", Value: "Suggest one short follow-up"}}},
			},
		}
		if err := provider.SaveGroup(outer); err != nil {
			t.Fatal(err)
		}
		s := New()
		asOuter := func(body string) string {
			return strings.Replace(body, `"model":"group/r"`, `"model":"group/outer"`, 1)
		}
		post := func(body string) (string, Route) {
			t.Helper()
			code, out := postAs(t, s, "nest-helper", body)
			if code != 200 {
				t.Fatalf("%d %s", code, out)
			}
			return out, lastRoute(s)
		}
		// Turn 1 goes to a/small; group/helpers has never been visited on this key.
		out, r := post(asOuter(withSystem(agentPrompt, withTools(chat("write a haiku about rivers", nil, 0, "")))))
		if !strings.Contains(out, "from ka") || r.Rule == nil || r.Rule.Aside || len(r.Nested) != 0 {
			t.Fatalf("turn 1: %s rule=%+v nested=%+v", out, r.Rule, r.Nested)
		}
		plantTurnState(30000)
		stuck := stickModels()
		helper := `{"model":"group/outer","max_tokens":64,"messages":[{"role":"system","content":` + quote("Suggest one short follow-up the user might ask.") + `},{"role":"user","content":` + quote("write a haiku about rivers") + `}]}`
		out, r = post(helper)
		if !strings.Contains(out, "from kb") {
			t.Fatalf("helper: %s", out)
		}
		if r.Rule == nil || !r.Rule.Aside || r.Rule.Use != "group/helpers" {
			t.Fatalf("outer helper rule: %+v", r.Rule)
		}
		if len(r.Nested) != 1 || r.Nested[0].Rule == nil {
			t.Fatalf("expected nested rule hit: %+v", r.Nested)
		}
		nh := r.Nested[0].Rule
		if !nh.Aside || nh.Use != "b/big" || nh.N != 2 || nh.Classified != nil || nh.Pick != "" {
			t.Fatalf("nested helper should propagate Aside without classifying: %+v", nh)
		}
		turnRules.Lock()
		ruleCount := len(turnRules.m)
		turnRules.Unlock()
		if ruleCount != 1 {
			t.Fatalf("unvisited nested group wrote turnRules: count=%d", ruleCount)
		}
		if !turnStateKept(30000) {
			t.Fatal("nested helper overwrote outer turn state")
		}
		if got := stickModels(); len(got) != len(stuck) {
			t.Fatalf("sticks %v, were %v", got, stuck)
		} else {
			for k, model := range stuck {
				if got[k] != model {
					t.Fatalf("stick %s moved from %s to %s", k, model, got[k])
				}
			}
		}
		// Tool round of Turn 1 still holds a/small.
		out, r = post(asOuter(withSystem(agentPrompt, withTools(chat("write a haiku about rivers", nil, 1, "")))))
		if !strings.Contains(out, "from ka") || r.Rule == nil || !r.Rule.Held || r.Rule.Aside || r.Rule.N != 0 {
			t.Fatalf("tool round: %s %+v", out, r.Rule)
		}
	})
}

// An unrecognized tool-less helper racing Turn 1 simultaneously through
// ruleFor must not overwrite the main turn's decision in turnRules when
// the main turn writes first (#7455).
func TestRuleSideCallRaceWithMainTurn(t *testing.T) {
	fresh(t)
	g := provider.Group{
		ID: "r", Effort: provider.EffortAuto, Classifier: "c/m",
		Rules: []provider.Rule{
			{Use: "a/small", Intent: "coding"},
			{Use: "b/big", Where: []provider.Cond{{Field: "system", Op: "starts", Value: "Suggest one short follow-up"}}},
		},
	}
	userMsg := Message{Role: "user", Parts: []Part{{Kind: Text, Text: "write a haiku about rivers"}}}
	mainReq := &Request{
		System:      agentPrompt,
		systemParts: []string{agentPrompt},
		Tools:       []Tool{{Name: "Bash"}},
		Messages:    []Message{userMsg},
		Thinking:    true,
		Effort:      "low",
	}
	helperPrompt := "Suggest one short follow-up the user might ask."
	helperReq := &Request{
		System:      helperPrompt,
		systemParts: []string{helperPrompt},
		Messages:    []Message{userMsg},
	}
	key := "group/r|race-turn1|x"

	helperEntered := make(chan struct{})
	mainDone := make(chan struct{})
	ask := func(_ access.Identity, _ string, _ []string, _ before, effort bool, _ string) (verdict, error) {
		if !effort {
			close(helperEntered)
			<-mainDone
			return verdict{Intent: "coding"}, nil
		}
		return verdict{Intent: "coding", Effort: "high"}, nil
	}

	var helperHit, mainHit *RuleHit
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		helperHit = ruleFor(key, g, nil, helperReq, "claude", ask, "")
	}()
	// Wait until the helper has read had == false and entered classify.
	<-helperEntered
	mainHit = ruleFor(key, g, nil, mainReq, "claude", ask, "")
	close(mainDone)
	wg.Wait()

	if mainHit == nil || mainHit.Aside || mainHit.Use != "a/small" || mainHit.N != 1 || mainHit.Pick != "high" {
		t.Fatalf("main turn: %+v", mainHit)
	}
	if helperHit == nil || !helperHit.Aside || helperHit.Use != "b/big" || helperHit.N != 2 || helperHit.Classified != nil || helperHit.Pick != "" || helperHit.Wanted != "" || helperHit.Ended != 0 {
		t.Fatalf("racing helper: %+v", helperHit)
	}
	turnRules.Lock()
	tr := turnRules.m[key]
	turnRules.Unlock()
	if tr.use != "a/small" || tr.n != 1 || tr.intent != "coding" || tr.effort != "high" || tr.sys != systemID(mainReq) {
		t.Fatalf("turnRules overwritten by racing helper: %+v", tr)
	}
	toolReq := &Request{
		System:      agentPrompt,
		systemParts: []string{agentPrompt},
		Tools:       []Tool{{Name: "Bash"}},
		Messages: []Message{
			userMsg,
			{Role: "assistant", Parts: []Part{{Kind: ToolCall, CallID: "c1", Name: "Bash"}}},
			{Role: "user", Parts: []Part{{Kind: ToolResult, CallID: "c1", Text: "ok"}}},
		},
		Thinking: true,
		Effort:   "low",
	}
	toolHit := ruleFor(key, g, nil, toolReq, "claude", nil, "")
	if toolHit == nil || !toolHit.Held || toolHit.Aside || toolHit.Use != "a/small" || toolHit.N != 1 || toolHit.Pick != "high" {
		t.Fatalf("tool round after race: %+v", toolHit)
	}
}

// A title block after a long preamble is still a title, on each agent's
// prompt, before the turn, during it and after it. A tool-less call that
// arrives first does not keep a tool-bearing turn out.
func TestRuleTitlesAroundTheTurn(t *testing.T) {
	prompts := []string{
		"You are an expert conversation summarizer. Your goal is to create a concise, 2-5 word title",
		"You write short session titles.",
		"Generate a concise, sentence-case title (3-7 words) that captures the main topic or goal of this coding session.",
		"Create a concise title for an AI coding-assistant session",
		"You are a title generator",
		"You name AI coding-assistant sessions",
	}
	user := "write a haiku about rivers"
	for i, p := range prompts {
		s, _, _ := ruled(t, provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "system", Op: "starts", Value: p}}})
		sess := "around-" + string(rune('a'+i))
		main := withSystem(agentPrompt, withTools(chat(user, nil, 0, "")))
		out, hit := mustRoute(t, s, sess, main)
		if !strings.Contains(out, "from ka") || hit.Aside {
			t.Fatalf("turn before %q: %s %+v", p, out, hit)
		}
		title := `{"model":"group/r","max_tokens":64,"messages":[{"role":"system","content":` + quote(p+". Reply with the title only.") + `},{"role":"user","content":` + quote(user) + `}]}`
		out, hit = mustRoute(t, s, sess, title)
		if !strings.Contains(out, "from kb") || hit.N != 1 || !hit.Aside {
			t.Fatalf("title during %q: %s %+v", p, out, hit)
		}
		out, hit = mustRoute(t, s, sess, withSystem(agentPrompt, withTools(chat(user, nil, 1, ""))))
		if !strings.Contains(out, "from ka") || !hit.Held || hit.Aside {
			t.Fatalf("tool round during %q: %s %+v", p, out, hit)
		}
		out, hit = mustRoute(t, s, sess, withSystem(agentPrompt, withTools(chat(user, []string{"and a second verse"}, 0, ""))))
		if !strings.Contains(out, "from ka") || hit.Aside || hit.Turn != 2 {
			t.Fatalf("turn 2 before a late title %q: %s %+v", p, out, hit)
		}
		out, hit = mustRoute(t, s, sess, title)
		if !strings.Contains(out, "from kb") || !hit.Aside {
			t.Fatalf("title after %q: %s %+v", p, out, hit)
		}
		out, hit = mustRoute(t, s, sess, withSystem(agentPrompt, withTools(chat(user, []string{"and a second verse"}, 1, ""))))
		if !strings.Contains(out, "from ka") || !hit.Held || hit.Aside {
			t.Fatalf("turn 2's tool round after %q: %s %+v", p, out, hit)
		}
	}

	// a preamble longer than compactLead, then the title in its own block
	p := prompts[0]
	s, _, _ := ruled(t, provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "system", Op: "starts", Value: p}}})
	preamble := strings.Repeat("context ", 400)
	title := `{"model":"group/r","max_tokens":64,"system":[{"type":"text","text":` + quote(preamble) + `},{"type":"text","text":` + quote(p+". Reply with the title only.") + `}],"messages":[{"role":"user","content":` + quote(user) + `}]}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(title))
	req.Header.Set("x-session-id", "preamble")
	s.Handler().ServeHTTP(rec, req)
	if pre := lastRoute(s).Rule; rec.Code != 200 || !strings.Contains(rec.Body.String(), "from kb") || pre == nil || !pre.Aside || pre.N != 1 {
		t.Fatalf("preamble: %d %s %+v", rec.Code, rec.Body.String(), lastRoute(s).Rule)
	}
	out, hit := mustRoute(t, s, "preamble", withSystem(agentPrompt, withTools(chat(user, nil, 0, ""))))
	if !strings.Contains(out, "from ka") || hit.Held || hit.Aside {
		t.Fatalf("turn after the preamble title: %s %+v", out, hit)
	}

	// an unrecognized tool-less call before the turn does not keep the
	// tool-bearing turn out of its own rule
	custom := "You title chats for a custom agent in a few words."
	s, _, _ = ruled(t, provider.Rule{Use: "b/big", Where: []provider.Cond{{Field: "system", Op: "starts", Value: "You title chats"}}})
	early := `{"model":"group/r","max_tokens":64,"messages":[{"role":"system","content":` + quote(custom) + `},{"role":"user","content":` + quote(user) + `}]}`
	out, hit = mustRoute(t, s, "early", early)
	if !strings.Contains(out, "from kb") || hit.N != 1 || hit.Aside {
		t.Fatalf("early helper, nothing stored yet: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "early", withSystem(agentPrompt, withTools(chat(user, nil, 0, ""))))
	if !strings.Contains(out, "from ka") || hit.N != 0 || hit.Held || hit.Aside {
		t.Fatalf("tool-bearing turn after the helper: %s %+v", out, hit)
	}
	out, hit = mustRoute(t, s, "early", withSystem(agentPrompt, withTools(chat(user, nil, 1, ""))))
	if !strings.Contains(out, "from ka") || !hit.Held || hit.Aside {
		t.Fatalf("its tool round: %s %+v", out, hit)
	}
}
