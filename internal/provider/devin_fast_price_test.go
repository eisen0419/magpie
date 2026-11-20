package provider

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/plugin"
	"github.com/yetone/magpie/internal/settings"
)

// Devin's GetCliModelConfigs lists every model with cache write 0. A base
// Claude id is priced from models.dev, which has the 5-minute and 1-hour
// writes. A -fast id is not on models.dev, so the plugin's 0 was the
// price and a long cached session was undercounted (#7360, kaimo455).
// The plugin's input, output and cache read are already the fast rate it
// charges — 2×, 6× or something of its own — and stay. Only a missing
// write is filled, from the base model's writes times that input ratio.
// A write the plugin set, a -fast id that is not Claude's, and Factory's
// own Fast Mode price stay as they were.
func TestDevinClaudeFastCacheWrite(t *testing.T) {
	claudeHome(t)
	t.Setenv("MAGPIE_BUN", "")
	os.MkdirAll(filepath.Dir(catalog.CachePath()), 0o755)
	// models.dev as it lists them: fast mode is not its own id, and a
	// 1-hour write is filled at load (OneHourFor), 2× input.
	os.WriteFile(catalog.CachePath(), []byte(`{
	  "anthropic":{"id":"anthropic","models":{
	    "claude-opus-5-5":{"id":"claude-opus-5-5","cost":{"input":4,"output":20,"cache_read":0.2,"cache_write":5}},
	    "claude-opus-5":{"id":"claude-opus-5","cost":{"input":5,"output":25,"cache_read":0.5,"cache_write":6.25}},
	    "claude-opus-4-8":{"id":"claude-opus-4-8","cost":{"input":5,"output":25,"cache_read":0.5,"cache_write":6.25}},
	    "claude-opus-4-6":{"id":"claude-opus-4-6","cost":{"input":5,"output":25,"cache_read":0.5,"cache_write":6.25}},
	    "opus-canon-zero1h":{"id":"opus-canon-zero1h","canonical_model_id":"anthropic/claude-opus-zero1h","cost":{"input":1e308,"output":1e308,"cache_read":0.5,"cache_write":1}},
	    "claude-fable-5-2":{"id":"claude-fable-5-2","cost":{"input":8,"output":40,"cache_read":0.2,"cache_write":10}},
	    "claude-sonnet-5-5":{"id":"claude-sonnet-5-5","cost":{"input":2,"output":10,"cache_read":0.1,"cache_write":2.5}}}},
	  "openai":{"id":"openai","models":{
	    "gpt-6-sol":{"id":"gpt-6-sol","cost":{"input":2,"output":12,"cache_read":0.2,"cache_write":2.5}},
	    "gpt-5.5":{"id":"gpt-5.5","cost":{"input":5,"output":30,"cache_read":0.5,"cache_write":0.625,
	      "tiers":[{"tier":{"type":"context","size":272000},"input":10,"output":45,"cache_read":1,"cache_write":1.25}]}},
	    "gpt-5.3-codex":{"id":"gpt-5.3-codex","cost":{"input":1.75,"output":14,"cache_read":0.175,"cache_write":0.175}}}}}`), 0o644)
	catalog.Reset()
	t.Cleanup(catalog.Reset)

	devinSpec := "@magpie-community/opencode-devin-auth"
	acmeSpec := "opencode-acme-auth"
	os.MkdirAll(settings.Dir(), 0o700)
	write := func(name string, v any) {
		t.Helper()
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(settings.Dir(), name), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("plugins.json", map[string]any{"plugins": []map[string]any{{"spec": devinSpec}, {"spec": acmeSpec}}})
	write("plugin-auth.json", map[string]map[string]any{
		"devin": {"type": "api", "key": "devin-key-1234"},
		"acme":  {"type": "api", "key": "acme-key-1234"},
	})
	write("migrations.json", map[string]any{"devin": map[string]any{"state": "plugin", "package": devinSpec}})
	plugin.UseCached([]plugin.Provider{
		{ID: "devin", Spec: devinSpec, Name: "Devin",
			Methods:  []plugin.Method{{Type: "api", Label: "API key"}},
			Accounts: []plugin.Account{{Key: "devin", Type: "api"}},
			Models: listedModels(t, `[
				{"id":"claude-opus-5-5","cost":{"input":5,"output":25,"cache":{"read":0.5,"write":0}}},
				{"id":"claude-opus-5-5-fast","cost":{"input":8,"output":40,"cache":{"read":0.4,"write":0}}},
				{"id":"claude-opus-5-5-high-fast","cost":{"input":8,"output":40,"cache":{"read":0.4,"write":0}}},
				{"id":"claude-opus-5-5-max-fast","cost":{"input":8,"output":40,"cache":{"read":0.4,"write":0}}},
				{"id":"claude-opus-5-5-fast@1m","cost":{"input":8,"output":40,"cache":{"read":0.4,"write":0}}},
				{"id":"claude-opus-5-5-fast[1m]","cost":{"input":8,"output":40,"cache":{"read":0.4,"write":0}}},
				{"id":"claude-fable-5-2-fast","cost":{"input":16,"output":80,"cache":{"read":0.4,"write":0}}},
				{"id":"claude-opus-5-fast","cost":{"input":10,"output":50,"cache":{"read":1,"write":0}}},
				{"id":"claude-opus-5-thinking-high-fast","cost":{"input":10,"output":50,"cache":{"read":1,"write":0}}},
				{"id":"claude-opus-4.8-fast","cost":{"input":10,"output":50,"cache":{"read":1,"write":0}}},
				{"id":"claude-opus-4-8-max-fast","cost":{"input":10,"output":50,"cache":{"read":1,"write":0}}},
				{"id":"claude-opus-4-6-20260101-fast","cost":{"input":10,"output":50,"cache":{"read":1,"write":0}}},
				{"id":"claude-opus-4-6-fast[1.5m]","cost":{"input":10,"output":50,"cache":{"read":1,"write":0}}},
				{"id":"claude-sonnet-5-5-fast","cost":{"input":4,"output":20,"cache":{"read":0.2,"write":3}}},
				{"id":"swe-2-fast","cost":{"input":3,"output":15,"cache":{"read":0.5,"write":0}}},
				{"id":"gpt-5.5-fast","cost":{"input":12.5,"output":75,"cache":{"read":1.25,"write":0}}},
				{"id":"gpt-5.3-codex-fast","cost":{"input":3.5,"output":28,"cache":{"read":0.35,"write":0}}}
			]`)},
		{ID: "acme", Spec: acmeSpec, Name: "Acme",
			Methods:  []plugin.Method{{Type: "api", Label: "API key"}},
			Accounts: []plugin.Account{{Key: "acme", Type: "api"}},
			Models: listedModels(t, `[
				{"id":"claude-opus-5-5-fast","cost":{"input":24,"output":120,"cache":{"read":1.2,"write":0}}},
				{"id":"claude-fable-5-2-fast","cost":{"input":12,"output":100,"cache":{"read":0.5,"write":0}}},
				{"id":"claude-gpt-6-sol-fast","cost":{"input":4,"output":24,"cache":{"read":0.4,"write":0}}},
				{"id":"claude-gpt-5.5-fast","cost":{"input":12.5,"output":75,"cache":{"read":1.25,"write":0}}}
			]`)},
	})
	t.Cleanup(func() { plugin.UseCached(nil) })

	devin := Provider{ID: "devin"}
	// the rates the issue's workaround sets, and the same 2× for the
	// other Claude -fast ids Devin lists
	wantList(t, devin, "claude-opus-5-5-fast", 8, 40, 0.4, 10, 16)
	wantList(t, devin, "claude-opus-5-5-high-fast", 8, 40, 0.4, 10, 16)
	wantList(t, devin, "claude-opus-5-5-max-fast", 8, 40, 0.4, 10, 16)
	wantList(t, devin, "claude-opus-5-5-fast@1m", 8, 40, 0.4, 10, 16)
	wantList(t, devin, "claude-opus-5-5-fast[1m]", 8, 40, 0.4, 10, 16)
	wantList(t, devin, "claude-fable-5-2-fast", 16, 80, 0.4, 20, 32)
	wantList(t, devin, "claude-opus-5-fast", 10, 50, 1, 12.5, 20)
	wantList(t, devin, "claude-opus-5-thinking-high-fast", 10, 50, 1, 12.5, 20)
	wantList(t, devin, "claude-opus-4.8-fast", 10, 50, 1, 12.5, 20)
	wantList(t, devin, "claude-opus-4-8-max-fast", 10, 50, 1, 12.5, 20)
	wantList(t, devin, "claude-opus-4-6-20260101-fast", 10, 50, 1, 12.5, 20)
	wantList(t, devin, "claude-opus-4-6-fast[1.5m]", 10, 50, 1, 12.5, 20)
	if pr, ok := EffectivePrice("devin", "claude-opus-5-5-fast"); !ok || pr.Input != 8 || pr.CacheRead != 0.4 || pr.CacheWrite != 10 || pr.CacheWrite1h != 16 {
		t.Errorf("EffectivePrice(devin/claude-opus-5-5-fast) = %+v, %v; want $8/$40, read $0.4, write $10 and $16", pr, ok)
	}
	// raw plugin catalog metadata keeps what the plugin reported ($0 write)
	if pp, ok := PluginOf("devin"); !ok {
		t.Fatal("PluginOf(devin) missing")
	} else {
		var raw *catalog.Price
		for _, cm := range pluginCatalog(pp) {
			if cm.ID == "claude-opus-5-5-fast" {
				raw = cm.Price
				break
			}
		}
		if raw == nil || raw.CacheWrite != 0 || raw.CacheWrite1h != 0 {
			t.Errorf("pluginCatalog(devin) claude-opus-5-5-fast price = %+v; want raw plugin write 0/0", raw)
		}
	}
	// the base id keeps models.dev's price over the plugin's write of 0
	wantList(t, devin, "claude-opus-5-5", 4, 20, 0.2, 5, 8)

	// a cache write the plugin did set is kept, and the 1-hour price is
	// not invented beside it
	wantList(t, devin, "claude-sonnet-5-5-fast", 4, 20, 0.2, 3, 0)
	// not a Claude -fast model: the plugin's 0 write stays 0
	wantList(t, devin, "swe-2-fast", 3, 15, 0.5, 0, 0)
	wantList(t, devin, "gpt-5.5-fast", 12.5, 75, 1.25, 0, 0)
	wantList(t, devin, "gpt-5.3-codex-fast", 3.5, 28, 0.35, 0, 0)

	// another plugin's input is the ratio, not a fixed 2×: 6× Opus, and
	// a Fable whose output and cache read are not the input's multiple
	acme := Provider{ID: "acme"}
	wantList(t, acme, "claude-opus-5-5-fast", 24, 120, 1.2, 30, 48)
	wantList(t, acme, "claude-fable-5-2-fast", 12, 100, 0.5, 15, 24)
	// a relay's "claude-" prefix on a non-Claude -fast model stays at its
	// plugin price and never picks up OpenAI writes or Anthropic 1-hour writes
	wantList(t, acme, "claude-gpt-6-sol-fast", 4, 24, 0.4, 0, 0)
	wantList(t, acme, "claude-gpt-5.5-fast", 12.5, 75, 1.25, 0, 0)

	// non-finite or negative direct-caller inputs stay untouched
	for _, bad := range []catalog.Price{
		{Input: math.NaN(), Output: 40},
		{Input: math.Inf(1), Output: 40},
		{Input: 1e308, Output: 40},
		{Input: 8, Output: 40, CacheWrite: -1},
	} {
		got := claudeFastWrites("claude-opus-5-5-fast", bad)
		if got.CacheWrite != bad.CacheWrite || got.CacheWrite1h != bad.CacheWrite1h {
			t.Errorf("claudeFastWrites(claude-opus-5-5-fast, %+v) = %+v; want writes unchanged", bad, got)
		}
	}
	// OneHourFor fallback (2*Input) overflowing to +Inf must leave writes unchanged
	if std, ok := anthropicListPrice("claude-opus-zero1h"); !ok || std.CacheWrite1h != 0 {
		t.Fatalf("anthropicListPrice(claude-opus-zero1h) = %+v, %v; want matched base with CacheWrite1h=0", std, ok)
	}
	if got := claudeFastWrites("claude-opus-zero1h-fast", catalog.Price{Input: 1e308, Output: 40}); got.CacheWrite != 0 || got.CacheWrite1h != 0 {
		t.Errorf("claudeFastWrites(claude-opus-zero1h-fast, 1e308) = %+v; want 0/0 without +Inf", got)
	}

	// Factory's Fast Mode is its own multiple of the base, tiers included
	factory := Provider{ID: "factory", Account: &Account{Agent: "factory"}}
	wantList(t, factory, "claude-opus-5-5-fast", 8, 40, 0.4, 10, 16)
	pr, ok := factory.ListPrice("gpt-5.5-fast")
	if !ok || pr.Input != 12.5 || pr.Output != 75 || pr.CacheRead != 1.25 || pr.CacheWrite != 1.5625 || pr.CacheWrite1h != 0 {
		t.Errorf("factory ListPrice(gpt-5.5-fast) = %+v, %v; want 2.5× gpt-5.5", pr, ok)
	} else if long := pr.At(300000); long.Input != 25 || long.Output != 112.5 || long.CacheWrite != 3.125 {
		t.Errorf("factory gpt-5.5-fast past 272k = %+v; want 2.5× its tier", long)
	}
	// an Anthropic account's -fast id is not this price
	claude := Provider{ID: "claude", Account: &Account{Agent: "claude"}}
	if pr, ok := claude.ListPrice("claude-opus-5-5-fast"); ok {
		t.Errorf("claude ListPrice(claude-opus-5-5-fast) = %+v; want unpriced", pr)
	}
}

func wantList(t *testing.T, p Provider, model string, in, out, read, write, write1h float64) {
	t.Helper()
	pr, ok := p.ListPrice(model)
	if !ok || pr.Input != in || pr.Output != out || pr.CacheRead != read || pr.CacheWrite != write || pr.CacheWrite1h != write1h {
		t.Errorf("%s ListPrice(%s) = %+v, %v; want in %v out %v read %v write %v 1h %v", p.ID, model, pr, ok, in, out, read, write, write1h)
	}
}
