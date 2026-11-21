package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/edit"
)

// SideModel is the model of magpie's Codex is on beside its ChatGPT
// sign-in, which the gateway answers the Codex app's titles, descriptions
// and summaries on (atom_Ray on X): none on one of Codex's own models, and
// none with magpie Codex's provider, where those requests reach magpie's
// own API on the model Codex asks for.
func TestCodexSideModel(t *testing.T) {
	home, _ := codexHome(t, `{"tokens":{"access_token":"x","id_token":"x.e30.x"}}`, "model = \"gpt-a\"\n")
	os.WriteFile(filepath.Join(home, ".codex", "models_cache.json"),
		[]byte(`{"models":[{"slug":"gpt-a","display_name":"A","priority":1}]}`), 0o644)
	cx := codex(home)
	if got := cx.SideModel(); got != "" {
		t.Fatalf("not connected: %q", got)
	}
	if ok, err := cx.Join(); err != nil || !ok {
		t.Fatalf("join: %v %v", ok, err)
	}
	if got := cx.SideModel(); got != "" {
		t.Fatalf("joined on Codex's own model: %q", got)
	}
	if err := cx.Fields[0].Set("fake/m1"); err != nil {
		t.Fatal(err)
	}
	if got := cx.SideModel(); got != "fake/m1" {
		t.Fatalf("a magpie model beside the sign-in: %q", got)
	}
	if got := CodexSideModel(); got != "fake/m1" {
		t.Fatalf("the gateway's lookup: %q", got)
	}
	if err := cx.Field("login").Set("api"); err != nil {
		t.Fatal(err)
	}
	if got := cx.SideModel(); got != "" {
		t.Fatalf("magpie Codex's provider: %q", got)
	}
}

// CodexSideSet says whether Codex's config names the model for its
// memories itself, the gateway then leaving those calls as Codex sent
// them; a config it can't read counts as naming one. Its subagents' model
// isn't asked: the gateway leaves a subagent on the model it was named.
func TestCodexSideSet(t *testing.T) {
	home, _ := codexHome(t, `{}`, "model = \"gpt-a\"\n")
	if CodexSideSet("memories") {
		t.Fatal("nothing set")
	}
	config := filepath.Join(home, ".codex", "config.toml")
	os.WriteFile(config, []byte("[agents]\ndefault_subagent_model = \"gpt-b\"\n[memories]\nextract_model = \"gpt-c\"\n"), 0o644)
	if !CodexSideSet("memories") {
		t.Fatal("set")
	}
	os.WriteFile(config, []byte("[agents]\ndefault_subagent_model = \"gpt-b\"\n"), 0o644)
	if CodexSideSet("memories") {
		t.Fatal("the subagents' alone")
	}
	os.WriteFile(config, []byte("[agents\n"), 0o644)
	if !CodexSideSet("memories") {
		t.Fatal("an unreadable config counts as set")
	}
}

// A Codex wired by its profile alone (codex -p magpie) has no beside-sign-in
// model and no failover hook. The gateway asks both on a side turn and on
// the models list; calling the missing hook panicked the process (#7452).
// Those calls answer as if the hook said no, unconnected and connected.
// Quit's EndCodexFailover does nothing. Uninstall's TakeCodexLeftovers
// removes a leftover magpie profile whose model no longer resolves, and
// leftover magpie wiring in config.toml, and leaves a provider and base
// URL the user owns.
func TestCodexProfileSideAndFailoverHooksDoNotPanic(t *testing.T) {
	const ownModel = "model = \"gpt-a\"\n"
	home, read := codexHome(t, `{"tokens":{"access_token":"x","id_token":"x.e30.x"}}`, ownModel)
	dir := filepath.Join(home, ".codex")
	if err := codex(home).Field("scope").Set("profile"); err != nil {
		t.Fatal(err)
	}
	profPath := filepath.Join(dir, codexProfileFile)
	if _, err := os.Stat(profPath); !os.IsNotExist(err) {
		t.Fatalf("profile file before a model: %v", err)
	}
	assertProfileHooksQuiet(t)
	took, err := TakeCodexLeftovers()
	if err != nil || took {
		t.Fatalf("nothing of magpie's yet: took %v, %v", took, err)
	}
	if read() != ownModel {
		t.Fatalf("unconnected profile changed config.toml:\n%s", read())
	}

	if err := codex(home).Apply("model", "fake/m1"); err != nil {
		t.Fatal(err)
	}
	prof, err := os.ReadFile(profPath)
	if err != nil || !strings.Contains(string(prof), `model = "fake/m1"`) || !strings.Contains(string(prof), `model_provider = "magpie"`) {
		t.Fatalf("connected profile:\n%s (%v)", prof, err)
	}
	if !codex(home).Wired() {
		t.Fatal("profile not connected")
	}
	assertProfileHooksQuiet(t)
	if read() != ownModel {
		t.Fatalf("connecting the profile changed config.toml:\n%s", read())
	}

	// The model no longer resolves, so uninstall's Disconnect never sees
	// the profile, and config.toml still holds magpie's wiring beside the
	// user's own keys (the #7452 reporter's stash).
	if err := edit.SetTOMLTop(profPath, edit.KV{Path: "model", Value: "gone/missing"}); err != nil {
		t.Fatal(err)
	}
	if codex(home).Wired() {
		t.Fatal("an unresolvable model still reads as connected")
	}
	gw := codexGatewayURL()
	cfg := "model = \"gpt-a\"\nmodel_provider = \"magpie\"\nopenai_base_url = \"" + gw + "\"\nmodel_catalog_json = \"magpie-models.json\"\n\n" +
		"[model_providers.magpie]\nname = \"magpie\"\nbase_url = \"" + gw + "\"\nwire_api = \"responses\"\n\n" +
		"[projects.\"/x\"]\ntrust_level = \"trusted\"\n"
	if err := os.WriteFile(filepath.Join(dir, "config.toml"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	took, err = TakeCodexLeftovers()
	if err != nil || !took {
		t.Fatalf("leftovers: took %v, %v", took, err)
	}
	if _, err := os.Stat(profPath); !os.IsNotExist(err) {
		t.Fatalf("leftover profile: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "magpie-models.json")); !os.IsNotExist(err) {
		t.Fatalf("leftover catalog: %v", err)
	}
	got := read()
	if strings.Contains(got, "openai_base_url") || strings.Contains(got, `model_provider = "magpie"`) || strings.Contains(got, "model_catalog_json") {
		t.Fatalf("magpie's config.toml entries left:\n%s", got)
	}
	if !strings.Contains(got, `model = "gpt-a"`) || !strings.Contains(got, `trust_level = "trusted"`) || !strings.Contains(got, "[model_providers.magpie]") {
		t.Fatalf("the user's keys or magpie's provider table disturbed:\n%s", got)
	}

	const userOwned = "model = \"gpt-a\"\nmodel_provider = \"mine\"\nopenai_base_url = \"https://api.openai.com/v1\"\n\n[projects.\"/x\"]\ntrust_level = \"trusted\"\n"
	if err := os.WriteFile(filepath.Join(dir, "config.toml"), []byte(userOwned), 0o644); err != nil {
		t.Fatal(err)
	}
	if took, err = TakeCodexLeftovers(); err != nil || took {
		t.Fatalf("the user's own provider and base URL: took %v, %v", took, err)
	}
	if read() != userOwned {
		t.Fatalf("user-owned provider and base URL changed:\n%s", read())
	}
}

func assertProfileHooksQuiet(t *testing.T) {
	t.Helper()
	if got := CodexSideModel(); got != "" {
		t.Fatalf("profile wiring has no beside-sign-in model, got %q", got)
	}
	if CodexOwnOnly() {
		t.Fatal("profile wiring is not account failover")
	}
	ended, err := EndCodexFailover()
	if err != nil || ended {
		t.Fatalf("end failover: ended %v, %v", ended, err)
	}
}
