package agent

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/yetone/magpie/internal/edit"
	"github.com/yetone/magpie/internal/gateway"
	"github.com/yetone/magpie/internal/provider"
	"gopkg.in/yaml.v3"
)

// omp 19's /fast ultra looks at the runtime model's serviceTiers and refuses
// ("does not offer the Ultrafast") before it sends anything (#7374). omp's
// models.yml model entries have no serviceTiers field (ModelDefinitionSchema
// in models-config-schema-bundle.ts, 18.15.0 and current): the tiers arrive
// only when discovery is openai-models-list, which copies a /v1/models row's
// service_tiers onto the model (oh-my-pi#17577, omp 18.15.0). The static
// entry still says the API, thinking, window and the rest, and an older omp,
// or one whose version isn't known, is left without discovery, which its
// schema may not name.
func TestOmpDiscoversServiceTiers(t *testing.T) {
	home := syncHome(t)
	was := ompVersion
	ompVersion = func() string { return "19.0.1" }
	t.Cleanup(func() { ompVersion = was })
	oa := provider.Provider{ID: "oa", Name: "OpenAI", Key: "k", Responses: "https://api.openai.com/v1", Models: []string{"gpt-6.2-sol"}}
	if err := provider.Save(oa); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetContext(oa, "gpt-6.2-sol", 272000); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("oa/gpt-6.2-sol", []string{"low", "high", "xhigh"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.Save(provider.Provider{ID: "anth", Name: "Anth", Key: "k", Anthropic: "http://127.0.0.1:1", Models: []string{"claude-sonnet-5"}}); err != nil {
		t.Fatal(err)
	}
	models := filepath.Join(home, ".omp", "agent", "models.yml")
	read := func() map[string]any {
		t.Helper()
		b, err := os.ReadFile(models)
		if err != nil {
			t.Fatal(err)
		}
		var file map[string]any
		if err := yaml.Unmarshal(b, &file); err != nil {
			t.Fatal(err)
		}
		pm, _ := file["providers"].(map[string]any)["magpie"].(map[string]any)
		if pm == nil {
			t.Fatalf("no magpie provider:\n%s", b)
		}
		return pm
	}
	model := func(pm map[string]any, id string) map[string]any {
		t.Helper()
		for _, raw := range pm["models"].([]any) {
			m := raw.(map[string]any)
			if m["id"] == id {
				return m
			}
		}
		t.Fatalf("%s not listed", id)
		return nil
	}
	if err := omp(home).Field("model").Set("magpie/oa/gpt-6.2-sol"); err != nil {
		t.Fatal(err)
	}
	if d := omp(home).Check(); d != "" {
		t.Fatalf("wired omp 19.0.1 drifted: %q", d)
	}
	pm := read()
	if want := gateway.AgentURL(gateway.URL(), "omp") + "/v1"; pm["baseUrl"] != want {
		t.Fatalf("baseUrl %v, want %q so keyless discovery identifies as omp", pm["baseUrl"], want)
	}
	disc, _ := pm["discovery"].(map[string]any)
	if disc["type"] != "openai-models-list" || disc["advertisedReasoning"] != false {
		t.Fatalf("discovery: %#v", pm["discovery"])
	}
	if wsl := ompProviderAt("http://172.23.80.1:3425", "19.0.1"); wsl.BaseURL != "http://172.23.80.1:3425/agent/omp/v1" {
		t.Fatalf("WSL omp 19.0.1 baseUrl %q", wsl.BaseURL)
	}
	sol := model(pm, "oa/gpt-6.2-sol")
	if _, ok := sol["serviceTiers"]; ok || sol["api"] != "openai-responses" || sol["contextWindow"] != 272000 {
		t.Fatalf("codex model: %#v", sol)
	}
	th, _ := sol["thinking"].(map[string]any)
	if th["mode"] != "effort" {
		t.Fatalf("thinking: %#v", sol["thinking"])
	}
	chat := model(pm, "relay/glm-4.6")
	if _, ok := chat["serviceTiers"]; ok || chat["api"] != nil {
		t.Fatalf("chat model: %#v", chat)
	}
	anth := model(pm, "anth/claude-sonnet-5")
	if anth["api"] != "anthropic-messages" || anth["baseUrl"] != gateway.URL() {
		t.Fatalf("anthropic model: %#v", anth)
	}
	if pm["api"] != "openai-completions" || pm["compat"].(map[string]any)["promptCacheSessionHeader"] != "x-session-affinity" {
		t.Fatalf("provider: api %v compat %v", pm["api"], pm["compat"])
	}

	// a discovery the user wrote stays, including one that already asks the list
	if err := edit.SetYAML(models, edit.KV{Path: "providers.magpie.discovery", Value: map[string]any{"type": "openai-models-list", "timeoutMs": 1500}}); err != nil {
		t.Fatal(err)
	}
	if err := edit.SetYAML(models, edit.KV{Path: "providers.magpie.modelOverrides", Value: map[string]any{"oa/gpt-6.2-sol": map[string]any{"contextWindow": 64000}}}); err != nil {
		t.Fatal(err)
	}
	if err := omp(home).Sync(); err != nil {
		t.Fatal(err)
	}
	pm = read()
	disc, _ = pm["discovery"].(map[string]any)
	if disc["type"] != "openai-models-list" || disc["timeoutMs"] != 1500 || disc["advertisedReasoning"] != nil {
		t.Fatalf("user discovery: %#v", pm["discovery"])
	}
	over, _ := pm["modelOverrides"].(map[string]any)["oa/gpt-6.2-sol"].(map[string]any)
	if over["contextWindow"] != 64000 || model(pm, "oa/gpt-6.2-sol")["api"] != "openai-responses" {
		t.Fatalf("override or api lost: %#v", pm)
	}

	// 18.14 has the discovery key but not the service_tiers copy, and a
	// discovery magpie wrote is taken back so an older omp is not pointed
	// at a list it would take over the static models for no gain
	if err := edit.DelYAML(models, "providers.magpie.discovery"); err != nil {
		t.Fatal(err)
	}
	ompVersion = func() string { return "19.0.1" }
	if err := omp(home).Sync(); err != nil {
		t.Fatal(err)
	}
	if read()["discovery"] == nil {
		t.Fatal("19.0.1 lost discovery after the user's was taken out")
	}
	ompVersion = func() string { return "18.14.2" }
	if err := omp(home).Sync(); err != nil {
		t.Fatal(err)
	}
	if pm := read(); pm["discovery"] != nil || pm["baseUrl"] != gateway.URL()+"/v1" {
		t.Fatalf("omp 18.14.2 discovery=%#v baseUrl=%v", pm["discovery"], pm["baseUrl"])
	}
	if d := omp(home).Check(); d != "" {
		t.Fatalf("wired omp 18.14.2 drifted: %q", d)
	}
	ompVersion = func() string { return "" }
	if err := omp(home).Sync(); err != nil {
		t.Fatal(err)
	}
	if pm := read(); pm["discovery"] != nil || pm["baseUrl"] != gateway.URL()+"/v1" {
		t.Fatalf("unknown omp version discovery=%#v baseUrl=%v", pm["discovery"], pm["baseUrl"])
	}
	// and a discovery that isn't magpie's is left on the older omp too
	if err := edit.SetYAML(models, edit.KV{Path: "providers.magpie.discovery", Value: map[string]any{"type": "ollama"}}); err != nil {
		t.Fatal(err)
	}
	if err := omp(home).Sync(); err != nil {
		t.Fatal(err)
	}
	if d, _ := read()["discovery"].(map[string]any); d["type"] != "ollama" {
		t.Fatalf("user discovery on an older omp: %#v", read()["discovery"])
	}
}

// Moving the gateway port must still flag drift for omp 18.15+ even before a
// background magpieAt probe has answered, so Rewire re-points omp off the old
// /agent/omp/v1 port; reverse proxies and bili chains stay drift-free.
func TestOmpCheckPortMoveAndProxy(t *testing.T) {
	home := syncHome(t)
	wasVer := ompVersion
	ompVersion = func() string { return "19.0.1" }
	t.Cleanup(func() { ompVersion = wasVer })
	wasAt := magpieAt
	magpieAt = func(string) (bool, bool) { return false, false }
	t.Cleanup(func() { magpieAt = wasAt })

	if err := omp(home).Field("model").Set("magpie/relay/glm-4.6"); err != nil {
		t.Fatal(err)
	}
	models := filepath.Join(home, ".omp", "agent", "models.yml")
	check := func(ver, base string) string {
		t.Helper()
		ompVersion = func() string { return ver }
		if err := edit.SetYAML(models, edit.KV{Path: "providers.magpie.baseUrl", Value: base}); err != nil {
			t.Fatal(err)
		}
		return omp(home).Check()
	}

	// Old loopback ports must report drift so Rewire updates them
	for _, tc := range []struct {
		name, ver, url string
	}{
		{"old-port-agent-path-19", "19.0.1", "http://127.0.0.1:3999/agent/omp/v1"},
		{"old-port-agent-path-unknown", "", "http://127.0.0.1:3999/agent/omp/v1"},
		{"wrong-port-v1", "19.0.1", "http://127.0.0.1:1/v1"},
		{"wrong-port-agent-path", "19.0.1", "http://127.0.0.1:1/agent/omp/v1"},
	} {
		if d := check(tc.ver, tc.url); d == "" {
			t.Errorf("%s (%s): expected drift, got none", tc.name, tc.url)
		}
	}

	// Wired address, reverse proxies, and bili chains stay drift-free
	gw := gateway.URL()
	for _, tc := range []struct {
		name, url string
	}{
		{"wired", gateway.AgentURL(gw, "omp") + "/v1"},
		{"reverse-proxy-v1", "https://ai.corp.example/magpie/v1"},
		{"reverse-proxy-agent-path", "https://ai.corp.example/magpie/agent/omp/v1"},
		{"local-proxy-v1", "http://127.0.0.1:8788/proxy/v1"},
		{"local-proxy-agent-path", "http://127.0.0.1:8788/proxy/agent/omp/v1"},
		{"bili-v1", "http://127.0.0.1:18787/bili/" + gw + "/v1"},
		{"bili-agent-path", "http://127.0.0.1:18787/bili/" + gw + "/agent/omp/v1"},
	} {
		if d := check("19.0.1", tc.url); d != "" {
			t.Errorf("%s (%s): unexpected drift %q", tc.name, tc.url, d)
		}
	}
}
