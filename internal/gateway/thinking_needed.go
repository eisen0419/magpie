package gateway

import (
	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/lazyre"
	"github.com/yetone/magpie/internal/provider"
)

// thinkingNeededRefusal is an upstream turning away a Messages request
// whose thinking isn't "enabled": thinking left out, or turned off as
// magpie does for a vendor's model asked without it
// (thinkingOffUnlessAsked). A relay in front of such a model answered
// Claude Desktop's connection check ({"max_tokens":1,"messages":[…"."]},
// no thinking) with `thinking.type must be "enabled"` (cybercaius on
// Discord, v0.1.1862, his own 天翼云 provider). Read off the reply, not
// the provider: any upstream that says it is asked again thinking.
var thinkingNeededRefusal = lazyre.New(`(?i)thinking[._ ]type\W+(?:must|should|has to) be\W+enabled\b`)

// thinkingNeeded is how unfit remembers an upstream that takes model only
// with thinking enabled (thinkingNeededRefusal).
func thinkingNeeded(model string) string { return "thinking needed\x00" + model }

// neededBudget is the thinking budget a request that asked for none is
// sent at, for a model that needs thinking: Anthropic's least, which the
// reply limit is raised past (withThinkingEnabled).
const neededBudget = 1024

// withThinkingEnabled is a Messages body that asked for no thinking, or
// turned it off, sent thinking at the least budget instead, for a model
// that takes no other: max_tokens is raised past the budget by what was
// asked (a budget must be below it), and the sampling a thinking request
// is refused with (temperature, top_p, top_k) left out. false for a body
// that asks to think already, or isn't JSON.
func withThinkingEnabled(body []byte) ([]byte, bool) {
	if !gjson.ValidBytes(body) {
		return nil, false
	}
	switch gjson.GetBytes(body, "thinking.type").String() {
	case "enabled", "adaptive":
		return nil, false
	}
	// a forced tool_choice cannot take thinking: the retry and the
	// cached mark both go through here, so neither adds it (#7389)
	if forcedAnthropicToolChoice(body) {
		return nil, false
	}
	maxTokens := gjson.GetBytes(body, "max_tokens").Int()
	if maxTokens <= neededBudget {
		maxTokens = neededBudget + max(maxTokens, 1)
	}
	out := withoutFields(body, "temperature", "top_p", "top_k")
	out = withFields(out, map[string]any{
		"thinking":   map[string]any{"type": "enabled", "budget_tokens": neededBudget},
		"max_tokens": maxTokens,
	})
	return out, true
}

// withThinkingEnabledCapped is withThinkingEnabled bounded by maxOut: when
// maxOut > 0 cannot fit the minimum thinking budget plus at least one output
// token (maxOut <= neededBudget), it refuses to raise max_tokens past the
// configured limit and returns false (#7389).
func withThinkingEnabledCapped(body []byte, maxOut int) ([]byte, bool) {
	if maxOut > 0 && maxOut <= neededBudget {
		return nil, false
	}
	out, ok := withThinkingEnabled(body)
	if !ok {
		return nil, false
	}
	if maxOut > 0 {
		out = withMaxOutput(provider.Anthropic, out, maxOut)
	}
	return out, true
}

// withAdaptiveAsEnabled turns thinking.type adaptive into enabled at the
// effort output_config names, or the least budget when it names none.
// withThinkingEnabled leaves an adaptive body alone, and its tests stay
// that way: this is only the retry after thinking.type must be "enabled"
// (#7389). false when the body isn't adaptive, forces a tool call, or
// isn't JSON.
func withAdaptiveAsEnabled(body []byte) ([]byte, bool) {
	if !gjson.ValidBytes(body) || gjson.GetBytes(body, "thinking.type").String() != "adaptive" {
		return nil, false
	}
	if forcedAnthropicToolChoice(body) {
		return nil, false
	}
	budget := neededBudget
	// per-message output_config.effort and a \u-escaped output_config key
	// name the level too; a raw path drops both and the budget falls to
	// the least (#7389)
	if level := outputConfigEffort(body); level != "" {
		budget = budgetOf(level)
	}
	maxTokens := gjson.GetBytes(body, "max_tokens").Int()
	if maxTokens > 0 && maxTokens <= int64(budget) {
		maxTokens = int64(budget) + max(maxTokens, 1)
	} else if maxTokens <= 0 {
		maxTokens = int64(budget) + 1
	}
	out := withoutFields(body, "temperature", "top_p", "top_k")
	out = withFields(out, map[string]any{
		"thinking":   map[string]any{"type": "enabled", "budget_tokens": budget},
		"max_tokens": maxTokens,
	})
	return out, true
}

// withAdaptiveAsEnabledCapped is withAdaptiveAsEnabled bounded by maxOut.
// A limit that cannot hold the least thinking budget is refused, so the
// retry does not raise max_tokens past the configured Max output (#7389).
func withAdaptiveAsEnabledCapped(body []byte, maxOut int) ([]byte, bool) {
	if maxOut > 0 && maxOut <= neededBudget {
		return nil, false
	}
	out, ok := withAdaptiveAsEnabled(body)
	if !ok {
		return nil, false
	}
	if maxOut > 0 {
		out = withMaxOutput(provider.Anthropic, out, maxOut)
		if gjson.GetBytes(out, "thinking.type").String() != "enabled" {
			return nil, false
		}
	}
	return out, true
}

// markThinkingNeeded remembers that p takes model only thinking, by the
// name magpie knows it and the vendor's (wire), which forwardOnce reads
// off the body: thinking enabled isn't made adaptive for it there, as
// "enabled" is the one type it said it takes.
func (s *Server) markThinkingNeeded(p provider.Provider, to provider.Protocol, model, wire string) {
	s.markUnfit(p.ID, thinkingNeeded(model), to)
	if wire != "" && wire != model {
		s.markUnfit(p.ID, thinkingNeeded(wire), to)
	}
}

// thinkingNeededMarked reports whether either magpie's name or the
// vendor's wire name is marked as requiring thinking.type "enabled".
func (s *Server) thinkingNeededMarked(pid, model, wire string, to provider.Protocol) bool {
	return !s.fits(pid, thinkingNeeded(model), to) || (wire != "" && wire != model && !s.fits(pid, thinkingNeeded(wire), to))
}

// thinkingNeededBothMarked reports whether every applicable name (model
// and distinct wire) is already marked in unfit.
func (s *Server) thinkingNeededBothMarked(pid, model, wire string, to provider.Protocol) bool {
	if s.fits(pid, thinkingNeeded(model), to) {
		return false
	}
	return wire == "" || wire == model || !s.fits(pid, thinkingNeeded(wire), to)
}
