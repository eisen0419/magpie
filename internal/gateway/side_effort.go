package gateway

import (
	"bytes"
	"context"
	"crypto/sha256"
	"slices"
	"sync"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/settings"
)

// The reasoning level picked in Settings for a model magpie asks on its
// own, beside the model in use: Image recognition's and the searcher's
// (MOMO on Discord, "外置模型能不能提供思考深度选项"). Only those calls
// are asked so; every other request goes as it did.

// ModelEfforts are the reasoning levels model ("<provider>/<model>") has,
// weakest first, for Settings to offer beside it; none for one whose
// levels aren't known, or a routing group.
func ModelEfforts(model string) []string {
	p, m, ok := provider.Resolve(model)
	if !ok {
		return nil
	}
	return ByStrength(p.Efforts(m))
}

// sideEffort is want fitted to model's own levels: "" when none is
// wanted, or the model's levels aren't known, so nothing is sent that it
// might refuse.
func sideEffort(model, want string) string {
	levels := ModelEfforts(model)
	if want == "" || len(levels) == 0 {
		return ""
	}
	return fitEffort(want, levels)
}

// visionEffort is the level the Image recognition model is asked to read
// an image at: the one picked in Settings, else magpie's own, the least
// it has (leastEffort), as a description is wanted quickly.
func visionEffort(model string) string {
	if e := sideEffort(model, provider.HeldSettings().VisionEffort); e != "" {
		return e
	}
	return leastEffort(model)
}

// leastEffort is the least reasoning model takes: none when it can go
// without, else its lowest level; "" when its levels aren't known, which
// leaves it to the vendor.
func leastEffort(model string) string {
	p, m, ok := provider.Resolve(model)
	if !ok {
		return ""
	}
	levels := p.Efforts(m)
	if len(levels) == 0 {
		return ""
	}
	return fitEffort("none", levels)
}

// VisionEffortNow is the level the Image recognition model in use is
// asked at when one is picked in Settings ("" when none is): the picked
// one fitted to the model's levels, for the row to show.
func VisionEffortNow() string {
	m, ok := seer()
	if !ok {
		return ""
	}
	return sideEffort(m, provider.HeldSettings().VisionEffort)
}

// VisionEfforts are the levels of the model that describes images now.
func VisionEfforts() []string {
	if m, ok := seer(); ok {
		return ModelEfforts(m)
	}
	return nil
}

// searchEffort is the level the searcher's model, provider p's model, is
// asked to search at: the one picked in Settings fitted to its levels,
// or "" to send none.
func searchEffort(p provider.Provider, model string) string {
	if model == "" || model == codexSearchModel {
		return ""
	}
	return sideEffort(p.ID+"/"+model, settings.Shared().SearcherEffort)
}

// SearchEffortNow is the level the searcher in use is asked at, when one
// is picked in Settings.
func SearchEffortNow() string {
	if p, m, ok := searcher(); ok {
		return searchEffort(p, m)
	}
	return ""
}

// SearchEfforts are the levels of the model that searches now: none for
// a Kimi Code plan's search service, which has no model.
func SearchEfforts() []string {
	if p, m, ok := searcher(); ok && m != "" && m != codexSearchModel {
		return ModelEfforts(p.ID + "/" + m)
	}
	return nil
}

// replyRoom is the reply limit a side call asked to think at effort is
// sent with: its usual limit with room for the thinking above it, so a
// level picked higher doesn't spend the reply on thinking and leave no
// answer. The model's own limit still caps it (withMaxOutput).
func replyRoom(usual int, effort string) int {
	switch effort {
	case "", "none":
		return usual
	case "minimal":
		return usual + 1024
	}
	return usual + budgetOf(effortOf(effort))
}

// SideEffort is the level a call magpie makes on its own with model is
// asked at when want is picked for it: want fitted to the model's own
// levels, "" when none is picked or its levels aren't known.
func SideEffort(model, want string) string { return sideEffort(model, want) }

// WithSideEffort is an Anthropic Messages request magpie makes on its own
// with model (memory's reading of conversations and its nightly tidy)
// asked to reason at want, fitted to the model's levels as SideEffort
// fits it, as a group's member fixed at a level is asked (withModelEffort):
// none turns thinking off, any other level asks for it, with room above
// the reply's usual limit to think in (replyRoom). A model pinned to
// another API (SetModelAPI) is asked as a translated request, so the
// level stays exact and max_tokens is not raised by an Anthropic budget.
// Nothing picked, or a model whose levels aren't known: the body as it was.
func WithSideEffort(body []byte, model, want string) []byte {
	e := sideEffort(model, want)
	if e == "" {
		// A routing group's first member may have unknown efforts while a
		// later member (failover, QuotaFirst, or a routing rule) does have
		// known efforts: remember want under a distinct RFC 8259 whitespace
		// trail without mutating the JSON fields (#7389).
		if want == "" || !provider.IsGroupRef(model) {
			return body
		}
		facts := sideEffortFactsOf(body)
		out := bytes.TrimRight(body, " \t\r\n")
		out = append(out, sideEffortAliasTrail("group:"+want)...)
		rememberSideEffort(out, sideEffortMeta{
			want:          want,
			origMaxTokens: facts.maxTokens,
			hadMaxTokens:  facts.hadMaxTokens,
		})
		return out
	}
	p, m, ok := provider.Resolve(model)
	translated := ok && ((p.Account != nil && p.Account.Agent == "claude") ||
		!sideSpeaksAnthropic(p, m))
	orig := body
	facts := sideEffortFactsOf(orig)
	shaped := model
	if ok {
		shaped = effortShapeModelAt(settings.Shared().Wires(), p, m)
	}
	if !translated {
		var v struct {
			MaxTokens int `json:"max_tokens"`
		}
		if decodeJSON(body, &v) == nil && v.MaxTokens > 0 {
			roomLevel := e
			if roomLevel == "minimal" && !anthropicAdaptive(shaped) {
				roomLevel = effortOf(roomLevel)
			}
			body = withFields(body, map[string]any{"max_tokens": replyRoom(v.MaxTokens, roomLevel)})
		}
	}
	var out []byte
	if ok {
		out = withModelEffortAt(provider.Anthropic, body, p.VendorHost(), shaped, e, p.Efforts(m), translated)
	} else {
		out = withModelEffort(provider.Anthropic, body, model, e, nil)
	}
	// The level and the caller's max_tokens are remembered beside every
	// body this rewrites, including a single-protocol relay and one that
	// will be translated (#7389). A stream-only account, a search and a
	// routing group's other member all read it back. max and xhigh share
	// a budget, and minimal and low do when max_tokens is omitted, so the
	// native bytes would be identical and the second remember would
	// replace the first. Trailing whitespace is not a JSON value (RFC
	// 8259), and it is different for every fitted level that can encode
	// alike. Only that metadata is kept: the transcript itself is not.
	out = bytes.TrimRight(out, " \t\r\n")
	if want != "" && want != e {
		out = append(out, sideEffortAliasTrail(e+":"+want)...)
	} else {
		out = append(out, sideEffortTrail(e)...)
	}
	rememberSideEffort(out, sideEffortMeta{
		want:          want,
		effort:        e,
		origMaxTokens: facts.maxTokens,
		hadMaxTokens:  facts.hadMaxTokens,
		addedThinking: !facts.hadThinking && hasThinking(out),
		addedOutput:   !facts.hadOutput && outputConfigEffort(out) != "",
	})
	return out
}

// sideTranslatedSlots is how many WithSideEffort bodies keep a level.
// A read does not move or drop a slot, so the same bytes still hit.
const sideTranslatedSlots = 64

// sideEffortMeta is what WithSideEffort changed, and nothing of the
// transcript: the caller's requested level, the first-member fitted level,
// the max_tokens the caller named, and whether thinking or
// output_config.effort was added here. serve stamps that onto the live body,
// after redaction and each candidate's own rewrites.
type sideEffortMeta struct {
	want          string
	effort        string
	origMaxTokens int
	hadMaxTokens  bool
	addedThinking bool
	addedOutput   bool
}

func (m sideEffortMeta) active() bool { return m.want != "" || m.effort != "" }

type sideEffortFacts struct {
	maxTokens    int
	hadMaxTokens bool
	hadThinking  bool
	hadOutput    bool
}

func hasThinking(body []byte) bool {
	r := gjson.GetBytes(body, "thinking")
	return r.Exists() && r.Type != gjson.Null
}

type sideTranslatedSlot struct {
	key  [32]byte
	meta sideEffortMeta
	used bool
}

var (
	sideTranslatedMu    sync.Mutex
	sideTranslatedCache [sideTranslatedSlots]sideTranslatedSlot
	sideTranslatedNext  int
)

// sideEffortFactsOf reads the max_tokens and thinking the caller sent,
// before WithSideEffort raises either.
func sideEffortFactsOf(body []byte) sideEffortFacts {
	var v struct {
		MaxTokens *int `json:"max_tokens"`
	}
	_ = decodeJSON(body, &v)
	f := sideEffortFacts{hadThinking: hasThinking(body), hadOutput: outputConfigEffort(body) != ""}
	if v.MaxTokens != nil {
		f.hadMaxTokens = true
		f.maxTokens = *v.MaxTokens
	}
	return f
}

// rememberSideEffort records sha256(native) -> meta. native is the
// Anthropic body WithSideEffort returned. The transcript is not kept.
func rememberSideEffort(native []byte, meta sideEffortMeta) {
	if len(native) == 0 || !meta.active() {
		return
	}
	key := sha256.Sum256(native)
	sideTranslatedMu.Lock()
	defer sideTranslatedMu.Unlock()
	for i := range sideTranslatedCache {
		if sideTranslatedCache[i].used && sideTranslatedCache[i].key == key {
			sideTranslatedCache[i].meta = meta
			return
		}
	}
	sideTranslatedCache[sideTranslatedNext] = sideTranslatedSlot{key: key, meta: meta, used: true}
	sideTranslatedNext = (sideTranslatedNext + 1) % sideTranslatedSlots
}

// sideEffortCached is the metadata WithSideEffort stored for body, when
// body is a native dual-protocol ask. The second result is false when
// body isn't one of those.
func sideEffortCached(body []byte) (sideEffortMeta, bool) {
	if len(body) == 0 {
		return sideEffortMeta{}, false
	}
	key := sha256.Sum256(body)
	sideTranslatedMu.Lock()
	defer sideTranslatedMu.Unlock()
	for i := range sideTranslatedCache {
		if sideTranslatedCache[i].used && sideTranslatedCache[i].key == key && sideTranslatedCache[i].meta.active() {
			return sideTranslatedCache[i].meta, true
		}
	}
	return sideEffortMeta{}, false
}

// sideEffortBox is one request's lookup of sideEffortCached, taken by the
// serve that request enters. A later serve on the same context (a search,
// a title) looks its own body up instead of reusing this one.
type sideEffortBox struct {
	mu   sync.Mutex
	meta sideEffortMeta
	ok   bool
	used bool
}

type sideEffortMetaKey struct{}

// stashSideEffort looks body up once, before onRequest, requestModel or
// redaction rewrite it. A second stash on the same context keeps the first.
func stashSideEffort(ctx context.Context, body []byte) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if _, already := ctx.Value(sideEffortMetaKey{}).(*sideEffortBox); already {
		return ctx
	}
	meta, ok := sideEffortCached(body)
	return context.WithValue(ctx, sideEffortMetaKey{}, &sideEffortBox{meta: meta, ok: ok})
}

// takeSideEffort is the metadata stashed for this request, or a lookup of
// body when nothing was stashed or a nested serve already took it.
func takeSideEffort(ctx context.Context, body []byte) (sideEffortMeta, bool) {
	if ctx != nil {
		if box, _ := ctx.Value(sideEffortMetaKey{}).(*sideEffortBox); box != nil {
			box.mu.Lock()
			if !box.used {
				box.used = true
				meta, ok := box.meta, box.ok
				box.mu.Unlock()
				return meta, ok
			}
			box.mu.Unlock()
		}
	}
	return sideEffortCached(body)
}

// restoreSideEffortBase undoes what WithSideEffort added: the caller's
// max_tokens (or none, when they named none), thinking cleared when it
// was added here, and output_config.effort cleared when it was added
// here. A thinking block or effort the caller sent is left as it is.
// A rule, a fixed member or an effort map then asks from that base, so
// the side call's raised budget never leaks into the override (#7389).
func restoreSideEffortBase(body []byte, meta sideEffortMeta) []byte {
	if meta.effort == "" {
		return body
	}
	if meta.hadMaxTokens {
		body = withFields(body, map[string]any{"max_tokens": meta.origMaxTokens})
	} else {
		body = withoutFields(body, "max_tokens")
	}
	if meta.addedThinking {
		body = withoutFields(body, "thinking")
	}
	if meta.addedOutput {
		body = editJSON(body, clearOutputEfforts)
	}
	return body
}

// sideEffortLevelFor fits WithSideEffort's requested level to a candidate's
// known efforts, returning "" when the candidate's efforts are unknown (so
// failover never injects reasoning onto a model WithSideEffort leaves alone).
func sideEffortLevelFor(meta sideEffortMeta, efforts []string) string {
	if !meta.active() || len(efforts) == 0 {
		return ""
	}
	want := meta.want
	if want == "" {
		want = meta.effort
	}
	return fitEffort(want, efforts)
}

// applySideEffortMeta stamps level onto body: the caller's max_tokens (or
// none, when they named none), thinking cleared when WithSideEffort added
// it, and output_config.effort set to level. A thinking block the caller
// sent is left as it is. The translated shape is what parse reads back as
// that exact level, without the native budget's raise.
func applySideEffortMeta(body []byte, meta sideEffortMeta, level string) []byte {
	if !meta.active() || level == "" {
		return body
	}
	return withModelEffortMode(provider.Anthropic, restoreSideEffortBase(body, meta), "", level, nil, true)
}

// sideEffortTrail is RFC 8259 whitespace after the native body, so two
// fitted levels that encode to the same Anthropic fields (max and xhigh
// share budgetOf 32000; minimal and low do when replyRoom is not applied)
// still hash to different sideTranslatedCache keys. low, medium, high and
// xhigh already differ by budget. Anything else gets a trail no named
// level uses, unique to that word.
func sideEffortTrail(effort string) []byte {
	switch effort {
	case "max":
		return []byte("\n")
	case "minimal":
		return []byte(" ")
	case "low", "medium", "high", "xhigh":
		return nil
	case "ultra":
		return []byte("\t")
	case "none":
		return []byte("\r")
	default:
		return sideEffortAliasTrail(effort)
	}
}

// sideEffortAliasTrail encodes effort as runs of spaces split by tabs,
// after a CR LF no named level uses, so two vendor words never share bytes.
func sideEffortAliasTrail(effort string) []byte {
	b := make([]byte, 0, 2+len(effort)*2)
	b = append(b, '\r', '\n')
	for i := 0; i < len(effort); i++ {
		n := int(effort[i]) + 1
		for range n {
			b = append(b, ' ')
		}
		b = append(b, '\t')
	}
	return b
}

// sideSpeaksAnthropic reports whether a side call on model is asked on
// Anthropic's API. A model pinned to another API (SetModelAPI) is not,
// even when the provider also has a Messages URL (#7389).
func sideSpeaksAnthropic(p provider.Provider, model string) bool {
	if proto, pinned := p.ModelAPI(model); pinned {
		return proto == provider.Anthropic
	}
	return slices.Contains(p.Speaks(), provider.Anthropic)
}
