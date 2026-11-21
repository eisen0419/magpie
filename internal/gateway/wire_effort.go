package gateway

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"hash/fnv"
	"net/http"
	"strings"

	"github.com/yetone/magpie/internal/provider"
)

// thinkingBackOn is req with the thinking off it asked for taken back, for
// a model that refused it: at the least the model has, as a request asking
// none was sent before thinking could be turned off (#1636).
func thinkingBackOn(p provider.Provider, model string, req *Request) *Request {
	r := *req
	r.ThinkOff = false
	if r.Effort == "" || offEffort(r.Effort) {
		r.Effort = onEffort(p, model)
	}
	return &r
}

// wireAt is where a translated request leaves the reasoning it was built
// at, for the log (wireEffortOf); set says it was.
type wireAt struct {
	effort string
	set    bool
	// tier: the service tier the try's request went out at, whichever
	// path sent it (noteWireTier)
	tier string
	// model: the model id the plugin the try went to said it sent its
	// vendor (noteWireModel)
	model string
	// pluginEffort says a Gemini plugin was told effort in a header, and
	// builtLevel is the thinking level the body had then. noteWireBody
	// keeps the header's effort unless the body that leaves names another.
	pluginEffort bool
	builtLevel   string
	// askedOff says the client asked for thinking off, as against
	// thinkingOffUnlessAsked appending disabled on a request that named
	// no effort. extraDisabled says WithExtraBody set thinking off.
	// noteSum is the body noteWireEffort read; 0 means it has not.
	// fromBody says the effort was read out of a body, so a later body
	// that no longer names one clears it instead of keeping it.
	askedOff      bool
	extraDisabled bool
	geminiCompat  bool
	sentLevel     string
	sentLevelSet  bool
	noteSum       uint64
	fromBody      bool
}

type wireAtKey struct{}

// withWireAt is ctx carrying w, which forwardTranslated fills.
func withWireAt(ctx context.Context, w *wireAt) context.Context {
	return context.WithValue(ctx, wireAtKey{}, w)
}

// noteWireEffort leaves, where the request's context has a place for it,
// the reasoning req goes to to at. body is the request as it was built,
// which Gemini and Code Assist read for the thinkingLevel they sent.
func noteWireEffort(ctx context.Context, to provider.Protocol, req *Request, body []byte) {
	w, _ := ctx.Value(wireAtKey{}).(*wireAt)
	if w == nil {
		return
	}
	// each attempt starts clean. A retry that cleared the effort
	// (ownEffortRefused), or a fallback onto another endpoint, must not
	// keep the failed attempt's none (#7400). The tier and the model the
	// reply names are filled as that reply comes back.
	w.effort, w.set = "", false
	w.pluginEffort = false
	w.builtLevel = ""
	w.askedOff = false
	w.extraDisabled = false
	w.geminiCompat = req != nil && req.GeminiCompat
	w.sentLevel, w.sentLevelSet = "", false
	w.noteSum = 0
	w.fromBody = false
	if req != nil && (req.ThinkOff || req.Effort == "none") {
		w.askedOff = true
	}
	w.noteSum = effortBodySum(body)
	if _, ok := bodyWireEffortAt(to, body, w.geminiCompat); ok {
		w.fromBody = true
	}
	if e, ok := wireEffort(to, req, body); ok {
		w.effort, w.set = e, true
	}
}

// noteGeminiCompat records that a relayed Chat request targets Gemini's
// OpenAI-compatible API, so noteWireBody reads extra_body.google.thinking_config
// even when the model id is an opaque alias (#7400).
func noteGeminiCompat(ctx context.Context) {
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w != nil {
		w.geminiCompat = true
	}
}

// noteWireSentLevel records the explicit thinking level carried on the final
// outgoing body (after WithExtraBody), so level-refusal retries and unfit
// marks match only the level actually sent (#7400).
func noteWireSentLevel(ctx context.Context, to provider.Protocol, body []byte) {
	w, _ := ctx.Value(wireAtKey{}).(*wireAt)
	if w == nil {
		return
	}
	w.sentLevel, w.sentLevelSet = "", true
	switch to {
	case provider.Gemini, provider.CodeAssist:
		if l, ok := geminiWireLevel(body); ok {
			if e, ok := normWireLevel(l); ok {
				w.sentLevel = e
			}
		}
	case provider.Chat:
		if w.geminiCompat {
			if e, ok := geminiCompatWireLevel(body); ok {
				w.sentLevel = e
			}
		}
	}
}

// sentGeminiLevel returns the explicit thinking level actually sent on the
// wire for this attempt (including any WithExtraBody override). Budgets without
// a thinkingLevel return "" so a budget request is never misread as xhigh.
func sentGeminiLevel(ctx context.Context, to provider.Protocol, body []byte, geminiCompat bool) string {
	if w, _ := ctx.Value(wireAtKey{}).(*wireAt); w != nil && w.sentLevelSet {
		return w.sentLevel
	}
	switch to {
	case provider.Gemini, provider.CodeAssist:
		if l, ok := geminiWireLevel(body); ok {
			if e, ok := normWireLevel(l); ok {
				return e
			}
		}
	case provider.Chat:
		if geminiCompat {
			if e, ok := geminiCompatWireLevel(body); ok {
				return e
			}
		}
	}
	return ""
}

// wireEffort is the reasoning a request built from req asks the model for,
// in the words the log uses: what the client asked can go as another
// level once translated — effort none and minimal read as low where the
// model's levels don't say it can stop (effortOf), thinking turned off
// goes as off to Anthropic's API — and the log said the one asked while
// another was sent (#1636). Gemini and Code Assist say the thinkingLevel
// the body names, when it names one (#7400). Gemini's OpenAI-compatible
// API says the thinking_level it was sent, or reasoning_effort when that
// is all it got. A Chat or Anthropic relay that geminiLevelBody rewrote
// says the effort in the body, not the one asked before the rewrite. A
// budget, or no thinkingConfig, is left as asked.
func wireEffort(to provider.Protocol, req *Request, body []byte) (string, bool) {
	switch {
	case req != nil && req.GeminiCompat:
		if e, ok := geminiCompatWireLevel(body); ok {
			return e, true
		}
		return "", false
	case to == provider.Gemini || to == provider.CodeAssist:
		if level, ok := geminiWireLevel(body); ok {
			if e, ok := normWireLevel(level); ok {
				return e, true
			}
		}
		return "", false
	case to == provider.Anthropic:
		// the body is what left: thinking disabled or between_tools is
		// none even when output_config still names an effort, and a body
		// WithExtraBody turned back on is that effort, not the none noted
		// from ThinkOff (#7400). budgetOf("max") is 32000, which
		// effortOfBudget reads as xhigh: a translated max stays max.
		if e, ok := anthropicWireEffort(body); ok {
			if req != nil && req.Effort == "max" && e == "xhigh" && preservesTranslatedMax(body) {
				return "max", true
			}
			return e, true
		}
		if req != nil && req.ThinkOff {
			return "none", true
		}
		if req == nil {
			return "", false
		}
		return req.Effort, true
	case to == provider.Chat:
		if e, ok := chatWireEffort(body, req != nil && req.GeminiCompat); ok {
			return e, true
		}
		if req == nil {
			return "", false
		}
		return req.Effort, true
	case to == provider.Responses:
		if req == nil {
			return "", false
		}
		return req.Effort, true
	}
	return "", false
}

// noteWireBody is noteWireEffort for the body as it leaves, after the
// provider's own fields (WithExtraBody) have been written in. It replaces
// the effort noted at build when the final body names one.
func noteWireBody(ctx context.Context, to provider.Protocol, body []byte) {
	w, _ := ctx.Value(wireAtKey{}).(*wireAt)
	if w == nil {
		return
	}
	// a Gemini plugin is told the effort in a header. The body's
	// thinkingLevel is only what Gemini's config can say, so it does not
	// replace that effort unless the final body names a different level
	// (WithExtraBody).
	if w.pluginEffort && (to == provider.Gemini || to == provider.CodeAssist) {
		if level, ok := geminiWireLevel(body); ok {
			if e, ok := normWireLevel(level); ok && e != w.builtLevel {
				w.effort, w.set = e, true
			}
		}
		return
	}
	e, ok := bodyWireEffortAt(to, body, w.geminiCompat)
	// thinking turned off was noted as none; a leftover output_config
	// effort is not what went out. A final body that turned thinking
	// back on (thinkingOffAsTaken, WithExtraBody) is. A body that no
	// longer names thinking or an effort cleared that none (#7400).
	if to == provider.Anthropic && w.set && w.effort == "none" && ok && e == "none" {
		return
	}
	if !ok {
		// the field the last attempt noted is gone. A none left from
		// that attempt, or any effort read out of the body, must not
		// stick on the retry (#7400). It stays set and empty, as
		// wireEffort does when the request named no effort, so the log
		// does not fall back to the level that was asked. An effort
		// noted only from the request, which this body never carried, stays.
		if w.fromBody || (w.set && w.effort == "none") {
			w.effort, w.set, w.fromBody = "", true, false
		}
		return
	}
	// thinkingOffUnlessAsked appends disabled on a non-Claude request
	// that asked for no effort. That is not a none the client asked,
	// and it is not what the log keeps (#7400).
	if to == provider.Anthropic && e == "none" && !w.askedOff && !w.extraDisabled {
		return
	}
	// a translated max was noted from the request. effortOfBudget reads
	// its 32000-token budget as xhigh, and that must not replace it.
	if w.set && w.effort == "max" && e == "xhigh" && preservesTranslatedMax(body) {
		return
	}
	w.effort, w.set, w.fromBody = e, true, true
}

// bodyWireEffort is the effort the final body names, for the protocols
// whose wire effort is read back out of it.
func bodyWireEffort(to provider.Protocol, body []byte) (string, bool) {
	return bodyWireEffortAt(to, body, false)
}

func bodyWireEffortAt(to provider.Protocol, body []byte, geminiChat bool) (string, bool) {
	switch to {
	case provider.Gemini, provider.CodeAssist:
		level, ok := geminiWireLevel(body)
		if !ok {
			return "", false
		}
		return normWireLevel(level)
	case provider.Chat:
		return chatWireEffort(body, geminiChat)
	case provider.Anthropic:
		return anthropicWireEffort(body)
	case provider.Responses:
		// a mid-thread change is the effort that left, not the stale
		// top-level reasoning.effort (#7400)
		if e := effortUpdated(body); e != "" {
			return normWireLevel(e)
		}
		if !bodyHas(body, `"reasoning"`) {
			return "", false
		}
		var v struct {
			Reasoning *struct {
				Effort string `json:"effort"`
			} `json:"reasoning"`
		}
		if json.Unmarshal(body, &v) != nil || v.Reasoning == nil {
			return "", false
		}
		return normWireLevel(v.Reasoning.Effort)
	}
	return "", false
}

// noteGeminiPluginEffort keeps the effort a Gemini plugin is told in
// X-Magpie-Effort. The thinkingLevel the body was built with is remembered
// so a later WithExtraBody that changes it is what the log keeps instead.
func noteGeminiPluginEffort(ctx context.Context, effort string, body []byte) {
	w, _ := ctx.Value(wireAtKey{}).(*wireAt)
	if w == nil || effort == "" {
		return
	}
	w.pluginEffort = true
	if level, ok := geminiWireLevel(body); ok {
		if e, ok := normWireLevel(level); ok {
			w.builtLevel = e
		}
	}
	w.effort, w.set = effort, true
}

// normWireLevel is a thinking level in the log's words. minimal stays
// minimal: effortOf would read it as low.
func normWireLevel(s string) (string, bool) {
	switch l := strings.ToLower(strings.TrimSpace(s)); l {
	case "minimal", "none", "low", "medium", "high", "xhigh", "max":
		return l, true
	default:
		return "", false
	}
}

// geminiCompatWireLevel is the level AI Studio's OpenAI-compatible body
// asks for: extra_body.google.thinking_config.thinking_level, else
// reasoning_effort.
func geminiCompatWireLevel(body []byte) (string, bool) {
	if !bodyHas(body, "reasoning_effort") && !bodyHas(body, "thinking_level") && !bodyHas(body, "thinkingLevel") {
		return "", false
	}
	var top struct {
		ReasoningEffort json.RawMessage `json:"reasoning_effort"`
		Extra           json.RawMessage `json:"extra_body"`
	}
	if json.Unmarshal(body, &top) != nil {
		return "", false
	}
	if e, ok := geminiCompatExtraLevel(top.Extra); ok {
		return e, true
	}
	return rawWireLevel(top.ReasoningEffort)
}

// geminiCompatExtraLevel is extra_body.google.thinking_config's level, in
// either spelling. Each nested field is its own RawMessage, so a
// non-object thinking_config does not hide a valid thinkingConfig, and a
// non-string level does not hide its sibling (#7400).
func geminiCompatExtraLevel(extra json.RawMessage) (string, bool) {
	extra = jsonObject(extra)
	if extra == nil {
		return "", false
	}
	var wrap struct {
		Google json.RawMessage `json:"google"`
	}
	if json.Unmarshal(extra, &wrap) != nil {
		return "", false
	}
	google := jsonObject(wrap.Google)
	if google == nil {
		return "", false
	}
	var g struct {
		Snake json.RawMessage `json:"thinking_config"`
		Camel json.RawMessage `json:"thinkingConfig"`
	}
	if json.Unmarshal(google, &g) != nil {
		return "", false
	}
	for _, raw := range []json.RawMessage{g.Snake, g.Camel} {
		if e, ok := geminiCompatThinkLevel(raw); ok {
			return e, true
		}
	}
	return "", false
}

// geminiCompatThinkLevel is thinking_level, else thinkingLevel, when the
// value is a string naming a level.
func geminiCompatThinkLevel(raw json.RawMessage) (string, bool) {
	raw = jsonObject(raw)
	if raw == nil {
		return "", false
	}
	var tc struct {
		Level      json.RawMessage `json:"thinking_level"`
		LevelCamel json.RawMessage `json:"thinkingLevel"`
	}
	if json.Unmarshal(raw, &tc) != nil {
		return "", false
	}
	for _, lv := range []json.RawMessage{tc.Level, tc.LevelCamel} {
		if e, ok := rawWireLevel(lv); ok {
			return e, true
		}
	}
	return "", false
}

// rawWireLevel is a JSON string that names a thinking level. A number or
// object is not one, and does not fail the object it sits beside.
func rawWireLevel(raw json.RawMessage) (string, bool) {
	s, ok := rawJSONString(raw)
	if !ok {
		return "", false
	}
	return normWireLevel(s)
}

func rawJSONString(raw json.RawMessage) (string, bool) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '"' {
		return "", false
	}
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return "", false
	}
	return s, true
}

func jsonObject(raw json.RawMessage) []byte {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '{' {
		return nil
	}
	return raw
}

// chatGeminiLevel is the level a Chat body names for Gemini: the thinking
// level in extra_body, else reasoning_effort.
func chatGeminiLevel(body []byte) (string, bool) {
	return geminiCompatWireLevel(body)
}

// chatWireEffort is a Chat body's wire effort: for Gemini models or Gemini's
// OpenAI-compatible API, extra_body.google.thinking_config else
// reasoning_effort; for non-Gemini models, top-level reasoning_effort alone so
// nested extra_body fields (even with non-object types) cannot mask or break
// decoding of the fitted effort in logs (#7400).
func chatWireEffort(body []byte, geminiChat bool) (string, bool) {
	if !bodyHas(body, "reasoning_effort") && !bodyHas(body, "thinking_level") && !bodyHas(body, "thinkingLevel") {
		return "", false
	}
	var top struct {
		Model           string          `json:"model"`
		ReasoningEffort json.RawMessage `json:"reasoning_effort"`
		Extra           json.RawMessage `json:"extra_body"`
	}
	if json.Unmarshal(body, &top) != nil {
		return "", false
	}
	low := strings.ToLower(top.Model)
	if geminiChat || geminiGeneration(low) > 0 || strings.Contains(low, "gemini") {
		if e, ok := geminiCompatExtraLevel(top.Extra); ok {
			return e, true
		}
	}
	return rawWireLevel(top.ReasoningEffort)
}

// anthropicWireEffort is the effort an Anthropic body went out at.
// thinking disabled, or between_tools, is none — requestEffort reads
// those as no ask — even when output_config still names a level.
// Otherwise it is output_config.effort, which geminiLevelBody writes
// for a Gemini relay.
func anthropicWireEffort(body []byte) (string, bool) {
	if !bodyHas(body, `"thinking"`) && !bodyHas(body, `"output_config"`) {
		return "", false
	}
	var v struct {
		Model    string `json:"model"`
		Thinking *struct {
			Type   string `json:"type"`
			Budget int    `json:"budget_tokens"`
		} `json:"thinking"`
		OutputConfig struct {
			Effort string `json:"effort"`
		} `json:"output_config"`
	}
	if json.Unmarshal(body, &v) != nil {
		return "", false
	}
	if v.Thinking != nil && (v.Thinking.Type == "disabled" || v.Thinking.Type == "between_tools") {
		return "none", true
	}
	// Claude Code and Claude Desktop keep the turn's level on a message,
	// beside a budget that only approximates it (#5379). A budget with
	// no effort named is the level that went out: withThinkingEnabled
	// and WithExtraBody turn a disabled request into one (#7400).
	// none and minimal stay themselves (turnWireEffort): effortOf would
	// read both as low.
	turn, turnOK := turnWireEffort(body)
	if v.Thinking != nil && v.Thinking.Type == "enabled" {
		if e, ok := normWireLevel(v.OutputConfig.Effort); ok {
			return e, true
		}
		if turnOK {
			return turn, true
		}
		if e := effortOfBudget(v.Thinking.Budget); e != "" {
			return e, true
		}
		return "", false
	}
	if v.Thinking == nil || v.Thinking.Type == "adaptive" {
		// a session title carries output_config.effort and no thinking.
		// That is not an effort the route asked, except on a model that
		// cannot turn thinking off: the effort is how it thinks.
		// Adaptive thinking names the level that went out, on the
		// request or only on the turn (#5379, #7400).
		e, ok := normWireLevel(v.OutputConfig.Effort)
		if !ok {
			e, ok = turn, turnOK
		}
		if v.Thinking != nil {
			return e, ok
		}
		if !ok || thinkingOffOf(v.Model) != offNever {
			return "", false
		}
		return e, true
	}
	return normWireLevel(v.OutputConfig.Effort)
}

// preservesTranslatedMax reports an enabled thinking budget that
// budgetOf("max") wrote and that names no effort of its own.
// effortOfBudget reads 32000 as xhigh, so a translated max must not be
// logged as that.
func preservesTranslatedMax(body []byte) bool {
	var v struct {
		MaxTokens int `json:"max_tokens"`
		Thinking  *struct {
			Type   string `json:"type"`
			Budget int    `json:"budget_tokens"`
		} `json:"thinking"`
		OutputConfig struct {
			Effort string `json:"effort"`
		} `json:"output_config"`
	}
	if json.Unmarshal(body, &v) != nil || v.Thinking == nil || v.Thinking.Type != "enabled" {
		return false
	}
	if strings.TrimSpace(v.OutputConfig.Effort) != "" {
		return false
	}
	if _, ok := turnWireEffort(body); ok {
		return false
	}
	// anthropicBody clamps the budget under max_tokens, leaving room for
	// the answer. 32000 tokens of room leaves 30976, which effortOfBudget
	// reads as xhigh; that clamp is still a translated max (#7400).
	budget := v.Thinking.Budget
	return budget == budgetOf("max") ||
		v.MaxTokens > 0 && budget == min(budgetOf("max"), v.MaxTokens-1024) && effortOfBudget(budget) == "xhigh"
}

// geminiWireLevel is the thinking level a Gemini body names:
// generationConfig.thinkingConfig on Gemini's own request, or the same
// under Code Assist's request, in camelCase or snake_case and mixed.
// "" and false when the body has no thinking level, or only a budget,
// or the level is not a string.
func geminiWireLevel(body []byte) (string, bool) {
	if len(body) == 0 || (!bodyHas(body, "thinkingLevel") && !bodyHas(body, "thinking_level")) {
		return "", false
	}
	var top struct {
		GenerationConfig json.RawMessage `json:"generationConfig"`
		GenerationSnake  json.RawMessage `json:"generation_config"`
		Request          json.RawMessage `json:"request"`
	}
	if json.Unmarshal(body, &top) != nil {
		return "", false
	}
	if l, ok := geminiGenLevel(top.GenerationConfig); ok {
		return l, true
	}
	if l, ok := geminiGenLevel(top.GenerationSnake); ok {
		return l, true
	}
	// Code Assist's envelope, one level down. A non-object request (a
	// string id beside a valid generationConfig) does not abort that
	// sibling (#7400).
	if req := jsonObject(top.Request); req != nil {
		var inner struct {
			GenerationConfig json.RawMessage `json:"generationConfig"`
			GenerationSnake  json.RawMessage `json:"generation_config"`
		}
		if json.Unmarshal(req, &inner) == nil {
			if l, ok := geminiGenLevel(inner.GenerationConfig); ok {
				return l, true
			}
			if l, ok := geminiGenLevel(inner.GenerationSnake); ok {
				return l, true
			}
		}
	}
	return "", false
}

// geminiGenLevel is generationConfig's thinking level. thinking_config and
// thinkingConfig are decoded on their own, so a non-object sibling does
// not hide the valid spelling.
func geminiGenLevel(raw json.RawMessage) (string, bool) {
	raw = jsonObject(raw)
	if raw == nil {
		return "", false
	}
	var g struct {
		ThinkingConfig json.RawMessage `json:"thinkingConfig"`
		ThinkingSnake  json.RawMessage `json:"thinking_config"`
	}
	if json.Unmarshal(raw, &g) != nil {
		return "", false
	}
	if l, ok := geminiThinkLevel(g.ThinkingConfig); ok {
		return l, true
	}
	return geminiThinkLevel(g.ThinkingSnake)
}

// geminiThinkLevel is thinkingLevel, else thinking_level, when it is a
// non-empty string. A number is not a level.
func geminiThinkLevel(raw json.RawMessage) (string, bool) {
	raw = jsonObject(raw)
	if raw == nil {
		return "", false
	}
	var t struct {
		Level      json.RawMessage `json:"thinkingLevel"`
		LevelSnake json.RawMessage `json:"thinking_level"`
	}
	if json.Unmarshal(raw, &t) != nil {
		return "", false
	}
	for _, lv := range []json.RawMessage{t.Level, t.LevelSnake} {
		s, ok := rawJSONString(lv)
		if !ok {
			continue
		}
		if s = strings.TrimSpace(s); s != "" {
			return s, true
		}
	}
	return "", false
}

// noteRelayEffort marks that an inbound relayed body carried a wire effort,
// before passthrough or forwardOnce mutates it. If a learned refused shape
// (withoutTurnEffort, withSystemOnTop) or WithExtraBody then clears that
// effort before noteWireBody runs, noteWireBody clears the effort to empty
// instead of letting serveRoute fall back to the effort asked (#7400).
// noteSum stays 0 so wireBodyStale stays true.
func noteRelayEffort(ctx context.Context, proto provider.Protocol, body []byte) {
	w, _ := ctx.Value(wireAtKey{}).(*wireAt)
	if w == nil || w.fromBody {
		return
	}
	if _, ok := bodyWireEffortAt(proto, body, w.geminiCompat); ok {
		w.fromBody = true
	}
}

// noteAnthropicAsked remembers that the client asked for thinking off,
// before thinkingOffUnlessAsked appends a disabled block of its own.
func noteAnthropicAsked(ctx context.Context, body []byte) {
	w, _ := ctx.Value(wireAtKey{}).(*wireAt)
	if w == nil || w.askedOff {
		return
	}
	if !bodyHas(body, `"thinking"`) && !bodyHas(body, `"output_config"`) {
		return
	}
	var v struct {
		Thinking *struct {
			Type string `json:"type"`
		} `json:"thinking"`
		OutputConfig struct {
			Effort string `json:"effort"`
		} `json:"output_config"`
	}
	if json.Unmarshal(body, &v) != nil {
		return
	}
	if v.Thinking != nil && (v.Thinking.Type == "disabled" || v.Thinking.Type == "between_tools") {
		w.askedOff = true
		return
	}
	// a later top-level effort is the one asked. An earlier turn's none
	// does not override it (#7400).
	asked := strings.TrimSpace(v.OutputConfig.Effort)
	if strings.EqualFold(asked, "none") {
		w.askedOff = true
		return
	}
	if asked != "" {
		return
	}
	if e, ok := turnWireEffort(body); ok && e == "none" {
		w.askedOff = true
	}
}

// extraDisablesThinking reports that the provider's extra body sets
// thinking to disabled or between_tools, forced or not. thinkingOffUnlessAsked
// may already have appended the same type, so the body no longer changes
// when those fields are applied (#7400). An account other than Codex is
// not sent extraBody (WithExtraBody): an unused one does not count.
func extraDisablesThinking(p provider.Provider) bool {
	if p.Account != nil && p.Account.Agent != "codex" {
		return false
	}
	for k, raw := range p.ExtraBody {
		n, _ := strings.CutPrefix(strings.TrimSpace(k), "!")
		if strings.TrimSpace(n) != "thinking" {
			continue
		}
		var v struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(raw, &v) != nil {
			continue
		}
		if v.Type == "disabled" || v.Type == "between_tools" {
			return true
		}
	}
	return false
}

// noteExtraDisabled remembers that the provider's own fields turned
// thinking off. The log then keeps none.
func noteExtraDisabled(ctx context.Context) {
	w, _ := ctx.Value(wireAtKey{}).(*wireAt)
	if w != nil {
		w.extraDisabled = true
	}
}

// wireBodyStale says body is not the one noteWireEffort read, so the
// effort has to be read again. A relay never noted one.
func wireBodyStale(ctx context.Context, body []byte) bool {
	w, _ := ctx.Value(wireAtKey{}).(*wireAt)
	if w == nil || w.noteSum == 0 {
		return true
	}
	return effortBodySum(body) != w.noteSum
}

func effortBodySum(b []byte) uint64 {
	h := fnv.New64a()
	_, _ = h.Write(b)
	return h.Sum64()
}

// noteWireModel leaves, where the request's context has a place for it, the
// model id a plugin's answer res says it sent its vendor
// (provider.SentModelHeader). Only a plugin is taken at its word: a vendor
// or a relay naming one is not what magpie sent.
func noteWireModel(ctx context.Context, p provider.Provider, res *http.Response) {
	if res == nil || !p.IsPlugin() {
		return
	}
	m := strings.TrimSpace(res.Header.Get(provider.SentModelHeader))
	res.Header.Del(provider.SentModelHeader)
	w, _ := ctx.Value(wireAtKey{}).(*wireAt)
	if w == nil || m == "" || len(m) > 200 {
		return
	}
	w.model = m
}

// wireName is the model id a try went out under, when it is another than
// model, the id magpie asked it for: what the plugin said it sent, else the
// name the user gave the model upstream (sentAs, provider.SentNameOnIn).
func wireName(model, plugin, sentAs string) string {
	if w := cmp.Or(plugin, sentAs); w != model {
		return w
	}
	return ""
}
