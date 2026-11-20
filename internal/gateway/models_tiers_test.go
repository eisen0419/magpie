package gateway

import (
	"encoding/json"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/usage"
)

// The model list every client reads names the tiers Codex here is offered
// on each model, not only the one another magpie asks for: Cindy shows its
// Fast switch on a model whose service_tiers hold priority, and its link
// asks /v1/models with no magpie header (tsuixl, #1914). A model offered
// no tier says nothing.
func TestModelsListTiersForEveryClient(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Models: []string{"gpt-6.1-sol", "glm-5"}, Responses: "http://relay.invalid/v1"}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	New().Handler().ServeHTTP(rec, httptest.NewRequest("GET", "/v1/models", nil))
	var list struct {
		Data []struct {
			ID    string `json:"id"`
			Tiers []struct {
				ID string `json:"id"`
			} `json:"service_tiers"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("%v: %s", err, rec.Body)
	}
	got := map[string]string{}
	for _, m := range list.Data {
		s := ""
		for _, x := range m.Tiers {
			s += x.ID + " "
		}
		got[m.ID] = s
	}
	// Ultrafast too: OpenAI serves it on GPT-6.1 Sol, and a provider added
	// by its address offers it as its editor does (OoopsHh on X)
	for id, want := range map[string]string{"relay/gpt-6.1-sol": "priority ultrafast ", "relay/glm-5": ""} {
		if s, ok := got[id]; !ok || s != want {
			t.Errorf("%s tiers %q (listed %v), want %q", id, s, ok, want)
		}
	}
}

// omp's openai-models-list discovery runs through bare fetch (Bun's User-Agent,
// no Authorization on loopback), so magpie points omp 18.15.0+ at
// /agent/omp/v1 (#7374). That path identifies the request as omp, so omp's
// hidden models stay out of the discovered list, BareModelIDs / BareOnly do
// not append bare duplicates or replace the provider/model ids omp matches
// service_tiers against, and Codex models still carry service_tiers.
func TestOmpDiscoveryModelsList(t *testing.T) {
	fresh(t)
	// package agent registers built-in agents in usage.Agents, and isn't linked here
	was := usage.PluginAgents
	usage.PluginAgents = func() []usage.Known {
		return []usage.Known{{ID: "omp", Names: []string{"omp", "oh-my-pi"}, UA: []string{"oh-my-pi"}}}
	}
	t.Cleanup(func() { usage.PluginAgents = was })

	if err := provider.Save(provider.Provider{ID: "oa", Name: "OpenAI", Key: "k", Models: []string{"gpt-6.2-sol", "hidden-chat"}, Responses: "https://api.openai.com/v1"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetHiddenModels("omp", []string{"oa/hidden-chat"}); err != nil {
		t.Fatal(err)
	}

	h := lanGuard(New().Handler())
	fetch := func(path string) ([]string, map[string][]string) {
		t.Helper()
		req := httptest.NewRequest("GET", path, nil)
		req.RemoteAddr = "127.0.0.1:50000"
		req.Header.Set("User-Agent", "Bun/1.4.2")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("GET %s: %d %s", path, rec.Code, rec.Body.String())
		}
		var list struct {
			Data []struct {
				ID    string `json:"id"`
				Tiers []struct {
					ID string `json:"id"`
				} `json:"service_tiers"`
			} `json:"data"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
			t.Fatalf("GET %s: %v: %s", path, err, rec.Body.String())
		}
		var ids []string
		tiers := map[string][]string{}
		for _, m := range list.Data {
			ids = append(ids, m.ID)
			for _, tier := range m.Tiers {
				tiers[m.ID] = append(tiers[m.ID], tier.ID)
			}
		}
		return ids, tiers
	}

	// Control: without /agent/omp, Bun's User-Agent gets the unfiltered catalog
	unfiltered, _ := fetch("/v1/models")
	if !slices.Contains(unfiltered, "oa/hidden-chat") {
		t.Fatalf("control /v1/models omitted hidden-chat: %v", unfiltered)
	}

	for _, mode := range []struct {
		name           string
		bare, bareOnly bool
	}{
		{"default", false, false},
		{"bare-ids", true, false},
		{"bare-only", true, true},
	} {
		t.Run(mode.name, func(t *testing.T) {
			if err := provider.SetBareIDs(mode.bare, mode.bareOnly); err != nil {
				t.Fatal(err)
			}
			ids, tiers := fetch("/agent/omp/v1/models")
			if slices.Contains(ids, "oa/hidden-chat") || slices.Contains(ids, "hidden-chat") {
				t.Fatalf("hidden model leaked into omp discovery: %v", ids)
			}
			if slices.Contains(ids, "gpt-6.2-sol") {
				t.Fatalf("bare duplicate leaked into omp discovery: %v", ids)
			}
			if !slices.Contains(ids, "oa/gpt-6.2-sol") {
				t.Fatalf("oa/gpt-6.2-sol missing from omp discovery: %v", ids)
			}
			if got := tiers["oa/gpt-6.2-sol"]; !slices.Contains(got, "priority") || !slices.Contains(got, "ultrafast") {
				t.Fatalf("oa/gpt-6.2-sol service_tiers = %v, want priority & ultrafast", got)
			}
		})
	}
}
