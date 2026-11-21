package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"testing"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/gateway"
	"github.com/yetone/magpie/internal/provider"
)

// openBotUser is openbot-custom-providers-v1.json as OpenBot saves it
// (CustomProviderStore, v0.58.0): an LM Studio endpoint with no key, and
// one with a key sealed in secret that runs on Pi.
const openBotUser = `{
  "version": 1,
  "providers": [
    {
      "id": "lm-studio",
      "name": "LM Studio",
      "baseUrl": "http://127.0.0.1:1234/v1",
      "models": [{"id": "qwen3-coder-30b", "name": "Qwen3 Coder 30B", "contextWindow": 262144}],
      "secret": null
    },
    {
      "id": "work",
      "name": "Work proxy",
      "baseUrl": "https://llm.example.com/v1",
      "models": [{"id": "gpt-5", "name": "gpt-5"}],
      "secret": "djEwAAAAsealedbyelectronsafestorage==",
      "harness": "pi"
    }
  ]
}
`

// openBotModelDoc is one model as OpenBot's modelSchema reads it. A nil
// pointer is a field left out. For inputs and reasoningEfforts that is
// not the same as []: an empty list is text-only and no reasoning effort,
// and a missing one is OpenCode's default.
type openBotModelDoc struct {
	ID               *string   `json:"id"`
	Name             *string   `json:"name"`
	ContextWindow    *int      `json:"contextWindow"`
	MaxOutput        *int      `json:"maxOutput"`
	ReasoningEfforts *[]string `json:"reasoningEfforts"`
	Inputs           *[]string `json:"inputs"`
}

// openBotSpecEfforts and openBotSpecInputs are the only orders OpenBot's
// isCustomProviderModel accepts (on disk, modelSchema drops an invalid
// field via .catch(undefined), and IPC validation rejects it). They are
// the check's own copy of that schema, not the lists the writer uses.
var openBotSpecEfforts = []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}
var openBotSpecInputs = []string{"image", "pdf"}

// openBotEntryDoc is a providers entry with OpenBot's fields alone
// (entrySchema), so a field of magpie's own fails to decode.
type openBotEntryDoc struct {
	ID      *string            `json:"id"`
	Name    *string            `json:"name"`
	BaseURL *string            `json:"baseUrl"`
	Models  *[]openBotModelDoc `json:"models"`
	Secret  *string            `json:"secret"`
	Harness *string            `json:"harness"`
}

// openBotOrdered reports whether got is a strictly ordered subset of
// order: each value one of order's, once, and only in order's order.
// nil is left out, which is allowed.
func openBotOrdered(got *[]string, order []string) bool {
	if got == nil {
		return true
	}
	at := -1
	for _, g := range *got {
		i := slices.Index(order, g)
		if i <= at {
			return false
		}
		at = i
	}
	return true
}

// openBotAgentModel is isAgentModel (ipc-agent-identity.ts): what a
// composed provider/model id must be, or OpenBot drops the whole list.
var openBotAgentModel = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/\[\],=-]*$`)

// openBotRead is the file's magpie entry, checked as OpenBot reads one
// (fileSchema, then isCustomProviderSummary, which fails closed on the
// whole list): version 1, the required fields there and nothing else, at
// most 64 models with ids and names it takes, windows in its range, a
// maxOutput only from 256 to 1,000,000, and inputs and reasoningEfforts
// each a strictly ordered subset of the kinds OpenBot names (or left
// out); and the user's entries, as written.
func openBotRead(t *testing.T, path string) (magpie openBotEntryDoc, others []json.RawMessage) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Version   *int              `json:"version"`
		Providers []json.RawMessage `json:"providers"`
	}
	if err := json.Unmarshal(b, &doc); err != nil || doc.Version == nil || *doc.Version != 1 || doc.Providers == nil {
		t.Fatalf("not a file OpenBot reads (version 1 and providers required): %v\n%s", err, b)
	}
	found := false
	for _, raw := range doc.Providers {
		var e openBotEntryDoc
		d := json.NewDecoder(bytes.NewReader(raw))
		d.DisallowUnknownFields()
		if err := d.Decode(&e); err != nil {
			t.Fatalf("not OpenBot's entry: %v\n%s", err, raw)
		}
		if e.ID == nil || e.Name == nil || e.BaseURL == nil || e.Models == nil {
			t.Fatalf("entry without a required field: %s", raw)
		}
		if *e.ID != magpieID {
			others = append(others, raw)
			continue
		}
		if found {
			t.Fatalf("two entries of magpie's: %s", b)
		}
		found, magpie = true, e
	}
	if !found {
		t.Fatalf("no entry of magpie's: %s", b)
	}
	if len(*magpie.Name) > 80 {
		t.Errorf("name %q", *magpie.Name)
	}
	if n := len(*magpie.Models); n > 64 {
		t.Errorf("%d models, OpenBot takes 64", n)
	}
	for _, m := range *magpie.Models {
		if m.ID == nil || m.Name == nil {
			t.Fatalf("model without id or name: %s", b)
		}
		if id := magpieID + "/" + *m.ID; len(*m.ID) > 128 || len(id) > 160 || !openBotAgentModel.MatchString(id) {
			t.Errorf("model id %q OpenBot refuses", *m.ID)
		}
		if len([]rune(*m.Name)) > 160 {
			t.Errorf("model name %q", *m.Name)
		}
		if w := m.ContextWindow; w != nil && (*w < 1024 || *w > 100_000_000) {
			t.Errorf("%s: window %d", *m.ID, *w)
		}
		if o := m.MaxOutput; o != nil && (*o < 256 || *o > 1_000_000) {
			t.Errorf("%s: maxOutput %d", *m.ID, *o)
		}
		if !openBotOrdered(m.ReasoningEfforts, openBotSpecEfforts) {
			t.Errorf("%s: reasoningEfforts %v", *m.ID, *m.ReasoningEfforts)
		}
		if !openBotOrdered(m.Inputs, openBotSpecInputs) {
			t.Errorf("%s: inputs %v", *m.ID, *m.Inputs)
		}
	}
	return magpie, others
}

// openBotListed is the model id in e.
func openBotListed(t *testing.T, e openBotEntryDoc, id string) openBotModelDoc {
	t.Helper()
	for _, m := range *e.Models {
		if m.ID != nil && *m.ID == id {
			return m
		}
	}
	t.Fatalf("model %s not listed", id)
	return openBotModelDoc{}
}

// openBotRawModel is the model object as written, so a field left out
// (no key) can be told from null or [].
func openBotRawModel(t *testing.T, path, id string) map[string]json.RawMessage {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Providers []json.RawMessage `json:"providers"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	for _, raw := range doc.Providers {
		if g := jsonID(raw); g != magpieID {
			continue
		}
		var e struct {
			Models []json.RawMessage `json:"models"`
		}
		if err := json.Unmarshal(raw, &e); err != nil {
			t.Fatal(err)
		}
		for _, m := range e.Models {
			if jsonID(m) != id {
				continue
			}
			var obj map[string]json.RawMessage
			if err := json.Unmarshal(m, &obj); err != nil {
				t.Fatal(err)
			}
			return obj
		}
	}
	t.Fatalf("model %s not in %s", id, b)
	return nil
}

func jsonID(raw json.RawMessage) string {
	var o struct {
		ID string `json:"id"`
	}
	json.Unmarshal(raw, &o)
	return o.ID
}

// OpenBot (nightly-labs/openbot) lists the user's own endpoints in its
// model picker, from openbot-custom-providers-v1.json in its userData
// folder. magpie adds itself as the endpoint "magpie" on the gateway's
// Chat Completions, keyless, keeps the user's entries as they were and the
// name, harness and key they give its entry, writes nothing when nothing
// changed, and off takes only its own entry out.
func TestOpenBotProvider(t *testing.T) {
	home := syncHome(t)
	cfg := filepath.Join(home, ".config")
	if err := provider.Save(provider.Provider{ID: "deepseek", Name: "DeepSeek", Chat: "https://api.deepseek.com/v1", Key: "k", Models: []string{"pro", "flash"}}); err != nil {
		t.Fatal(err)
	}
	a := openBot(home, cfg)
	if !slices.ContainsFunc(All(), func(x *Agent) bool { return x.ID == "openbot" && x.Name == "OpenBot" }) {
		t.Fatal("OpenBot is not among the agents")
	}
	if a.Detected() {
		t.Fatal("found with nothing of it here")
	}
	os.MkdirAll(a.Dir, 0o755)
	if !a.Detected() {
		t.Fatal("not found with its userData folder")
	}
	path := a.Path
	if filepath.Base(path) != "openbot-custom-providers-v1.json" || filepath.Dir(path) != a.Dir {
		t.Fatalf("path %s", path)
	}
	os.WriteFile(path, []byte(openBotUser), 0o600)
	f := a.Field("provider")
	if f.Get() != "" {
		t.Fatalf("get: %q", f.Get())
	}
	if err := f.Set(magpieID); err != nil {
		t.Fatal(err)
	}
	if f.Get() != magpieID {
		t.Fatalf("get after set: %q", f.Get())
	}
	// the restart it needs, said while it runs
	running := Running
	t.Cleanup(func() { Running = running })
	Running = func(...string) bool { return true }
	if a.Notice() != noticeOpenBot.String() {
		t.Errorf("notice while it runs: %q", a.Notice())
	}
	Running = func(...string) bool { return false }
	if a.Notice() != "" {
		t.Errorf("notice while it doesn't: %q", a.Notice())
	}

	gw := gateway.URL()
	e, others := openBotRead(t, path)
	if *e.BaseURL != gw+"/agent/openbot/v1" || *e.Name != "magpie" || e.Secret != nil || e.Harness != nil {
		t.Errorf("entry: base %q name %q secret %v harness %v", *e.BaseURL, *e.Name, e.Secret, e.Harness)
	}
	var got, want []string
	for _, m := range *e.Models {
		got = append(got, *m.ID)
	}
	for _, m := range magpieModels("openbot") {
		want = append(want, m.ID)
	}
	if len(want) < 2 || !slices.Equal(got, want) {
		t.Errorf("models %v, want %v", got, want)
	}
	var user struct {
		Providers []json.RawMessage `json:"providers"`
	}
	json.Unmarshal([]byte(openBotUser), &user)
	if len(others) != 2 || !sameJSON(string(others[0]), user.Providers[0]) || !sameJSON(string(others[1]), user.Providers[1]) {
		t.Errorf("the user's entries: %s", others)
	}

	// nothing changed: nothing written
	b, _ := os.ReadFile(path)
	st, _ := os.Stat(path)
	if err := a.Sync(); err != nil {
		t.Fatal(err)
	}
	if err := f.Set(magpieID); err != nil {
		t.Fatal(err)
	}
	if b2, _ := os.ReadFile(path); !bytes.Equal(b, b2) {
		t.Errorf("rewritten with nothing changed:\n%s\n%s", b, b2)
	}
	if st2, _ := os.Stat(path); !st2.ModTime().Equal(st.ModTime()) {
		t.Error("written with nothing changed")
	}

	// renamed, given a key and put on Pi in OpenBot, and a model gone: a
	// sync keeps the three and puts the model back
	var doc map[string]any
	json.Unmarshal(b, &doc)
	for _, x := range doc["providers"].([]any) {
		if m := x.(map[string]any); m["id"] == magpieID {
			m["name"], m["secret"], m["harness"] = "My magpie", "c2VhbGVk", "pi"
			m["models"] = m["models"].([]any)[:1]
		}
	}
	edited, _ := json.MarshalIndent(doc, "", "  ")
	os.WriteFile(path, edited, 0o600)
	if err := a.Sync(); err != nil {
		t.Fatal(err)
	}
	e, _ = openBotRead(t, path)
	if *e.Name != "My magpie" || e.Secret == nil || *e.Secret != "c2VhbGVk" || e.Harness == nil || *e.Harness != "pi" {
		t.Errorf("the user's name, key and harness not kept: %+v", e)
	}
	if len(*e.Models) != len(want) {
		t.Errorf("models after sync: %d, want %d", len(*e.Models), len(want))
	}

	// off: only magpie's entry goes, the file stays
	if err := f.Set(""); err != nil {
		t.Fatal(err)
	}
	if f.Get() != "" {
		t.Fatalf("get after off: %q", f.Get())
	}
	b, _ = os.ReadFile(path)
	var after struct {
		Version   int               `json:"version"`
		Providers []json.RawMessage `json:"providers"`
	}
	json.Unmarshal(b, &after)
	if after.Version != 1 || len(after.Providers) != 2 || !sameJSON(string(after.Providers[0]), user.Providers[0]) || !sameJSON(string(after.Providers[1]), user.Providers[1]) {
		t.Errorf("after off: %s", b)
	}
	// a sync of an entry that isn't there writes none
	if err := a.Sync(); err != nil {
		t.Fatal(err)
	}
	if b2, _ := os.ReadFile(path); !bytes.Equal(b, b2) {
		t.Errorf("a sync off wrote:\n%s", b2)
	}
}

// OpenBot not run yet: magpie makes the file with the version 1 OpenBot
// needs, and off takes the file away again.
func TestOpenBotMakesItsFile(t *testing.T) {
	home := syncHome(t)
	a := openBot(home, filepath.Join(home, ".config"))
	if err := a.Field("provider").Set(magpieID); err != nil {
		t.Fatal(err)
	}
	_, others := openBotRead(t, a.Path)
	if len(others) != 0 {
		t.Errorf("others: %s", others)
	}
	b, _ := os.ReadFile(a.Path)
	var top map[string]json.RawMessage
	json.Unmarshal(b, &top)
	if len(top) != 2 || string(top["version"]) != "1" {
		t.Errorf("made: %s", b)
	}
	if err := a.Field("provider").Set(""); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(a.Path); !os.IsNotExist(err) {
		t.Errorf("magpie's own file left behind: %v", err)
	}
}

// A file OpenBot can't read leaves it with no endpoints and every save
// refused, so it may be a newer OpenBot's: magpie leaves it as it is.
func TestOpenBotLeavesAlone(t *testing.T) {
	home := syncHome(t)
	a := openBot(home, filepath.Join(home, ".config"))
	os.MkdirAll(a.Dir, 0o755)
	for _, s := range []string{
		`{"version": 2, "providers": []}`,
		`{"providers": []}`,
		`{"version": "1", "providers": []}`,
		`{"version": 1}`,
		`{"version": 1, "providers": {}}`,
		`[]`,
		`{"version": 1,`,
	} {
		os.WriteFile(a.Path, []byte(s), 0o600)
		if err := a.Field("provider").Set(magpieID); err == nil {
			t.Errorf("%s: set", s)
		}
		if b, _ := os.ReadFile(a.Path); string(b) != s {
			t.Errorf("%s: written: %s", s, b)
		}
	}
}

// OpenBot drops every model of an endpoint when one would not do: more than
// 64, an id it won't compose as magpie/<id>, a window out of its range. So
// magpie writes the first 64 it takes, and a window only inside the range.
func TestOpenBotModelsItTakes(t *testing.T) {
	syncHome(t)
	ms := []string{"has space", "ünï"}
	for i := range 70 {
		ms = append(ms, fmt.Sprintf("m%02d", i))
	}
	if err := provider.Save(provider.Provider{ID: "many", Name: "Many", Chat: "https://api.example/v1", Key: "k", Models: ms}); err != nil {
		t.Fatal(err)
	}
	all := magpieModels("openbot")
	if len(all) <= openBotMaxModels || !slices.ContainsFunc(all, func(m catalog.Model) bool { return m.ID == "has space" || m.ID == "many/has space" }) {
		t.Fatalf("fixture: %d models", len(all))
	}
	e := openBotEntry(gateway.URL())
	if len(e.Models) != openBotMaxModels {
		t.Errorf("%d models", len(e.Models))
	}
	for _, m := range e.Models {
		if !openBotAgentModel.MatchString(magpieID + "/" + m.ID) {
			t.Errorf("model %q OpenBot refuses", m.ID)
		}
		if m.ContextWindow != 0 && (m.ContextWindow < 1024 || m.ContextWindow > 100_000_000) {
			t.Errorf("%s: window %d", m.ID, m.ContextWindow)
		}
	}
}

// OpenBot shows an upload and a thinking control from inputs and
// reasoningEfforts. The fields are the ones its own check takes, in its
// order, or the whole list is empty: inputs an ordered subset of image
// then pdf, or []; reasoningEfforts an ordered subset of none, minimal,
// low, medium, high, xhigh, max, or []; maxOutput an integer from 256 to
// 1,000,000 and strictly below the context window. A known text-only
// model writes inputs [], and a model that does not think, or whose
// levels OpenBot does not name, writes reasoningEfforts []. A model that
// thinks and has no levels omits reasoningEfforts, and nothing said of
// the input omits inputs. An output outside the range, or not below the
// window, is omitted. A sync writes nothing while they are unchanged,
// and writes again when they change.
func TestOpenBotModelsCarryCapabilities(t *testing.T) {
	home := syncHome(t)
	was := provider.Described
	provider.Described = func() bool { return false }
	t.Cleanup(func() { provider.Described = was })
	const base = "https://ob.example/v1"
	ids := []string{"both", "lo", "hi", "under", "over", "plain", "odd", "switch", "unsaid", "noimg"}
	if err := provider.Save(provider.Provider{ID: "ob", Name: "OB", Chat: base, Key: "k", Models: ids}); err != nil {
		t.Fatal(err)
	}
	no := false
	if err := provider.SetModelImage("ob/noimg", &no); err != nil {
		t.Fatal(err)
	}
	ms := []catalog.Model{
		{ID: "both", Name: "Both", Context: 200000, Output: 128000, Images: true,
			InputModalities: []string{"pdf", "text", "image"},
			Efforts:         []string{"max", "ultra", "LOW", "low", "none", "persistent", "High", "minimal", "xhigh", "medium", "extra_high"}},
		{ID: "lo", Name: "Lo", Context: 8192, Output: 256, Images: true, InputModalities: []string{"text", "image"}},
		// 1,000,000 is the top of the range, and only while it is below the window
		{ID: "hi", Name: "Hi", Context: 2_000_000, Output: 1_000_000, InputModalities: []string{"text", "pdf"}},
		{ID: "under", Name: "Under", Context: 512, Output: 255, Efforts: []string{"high"}, InputModalities: []string{"text"}},
		{ID: "over", Name: "Over", Context: 8192, Output: 1_000_001, Images: true, InputModalities: []string{"text"}},
		{ID: "plain", Name: "Plain", InputModalities: []string{"text"}},
		{ID: "odd", Name: "Odd", Output: 100, Efforts: []string{"light", "ULTRA", "persistent", "LIGHT"}, InputModalities: []string{"text"}},
		// thinks, and has no levels: reasoningEfforts left out, inputs known text
		{ID: "switch", Name: "Switch", Reasoning: true, InputModalities: []string{"text"}},
		// nothing said of input or reasoning
		{ID: "unsaid", Name: "Unsaid"},
		// bare model marked text-only via provider.SetModelImage(..., &false)
		{ID: "noimg", Name: "NoImg"},
	}
	if err := catalog.SaveLive("ob", base, ms); err != nil {
		t.Fatal(err)
	}
	a := openBot(home, filepath.Join(home, ".config"))
	if err := a.Field("provider").Set(magpieID); err != nil {
		t.Fatal(err)
	}
	e, _ := openBotRead(t, a.Path)
	want := map[string]struct {
		in, eff       []string
		inSet, effSet bool
		out           int
		outSet        bool
	}{
		"ob/both":   {[]string{"image", "pdf"}, []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}, true, true, 128000, true},
		"ob/lo":     {[]string{"image"}, nil, true, true, 256, true},
		"ob/hi":     {[]string{"pdf"}, nil, true, true, 1_000_000, true},
		"ob/under":  {nil, []string{"high"}, true, true, 0, false},
		"ob/over":   {[]string{"image"}, nil, true, true, 0, false},
		"ob/plain":  {nil, nil, true, true, 0, false},
		"ob/odd":    {nil, nil, true, true, 0, false},
		"ob/switch": {nil, nil, true, false, 0, false},
		"ob/unsaid": {nil, nil, false, true, 0, false},
		"ob/noimg":  {nil, nil, true, true, 0, false},
	}
	// syncHome also lists relay/glm-4.6; the models under test are ob's
	for id, w := range want {
		m := openBotListed(t, e, id)
		if w.inSet != (m.Inputs != nil) || !slices.Equal(ptrStrings(m.Inputs), w.in) {
			t.Errorf("%s inputs %v (set %v), want %v (set %v)", id, ptrStrings(m.Inputs), m.Inputs != nil, w.in, w.inSet)
		}
		if w.effSet != (m.ReasoningEfforts != nil) || !slices.Equal(ptrStrings(m.ReasoningEfforts), w.eff) {
			t.Errorf("%s reasoningEfforts %v (set %v), want %v (set %v)", id, ptrStrings(m.ReasoningEfforts), m.ReasoningEfforts != nil, w.eff, w.effSet)
		}
		switch {
		case w.outSet && (m.MaxOutput == nil || *m.MaxOutput != w.out):
			t.Errorf("%s maxOutput %v, want %d", id, m.MaxOutput, w.out)
		case !w.outSet && m.MaxOutput != nil:
			t.Errorf("%s maxOutput %d, want it left out", id, *m.MaxOutput)
		}
	}
	// [] is the two-character list, not left out and not null. A missing
	// inputs is image on, and a missing reasoningEfforts is low, medium
	// and high.
	for id, w := range want {
		raw := openBotRawModel(t, a.Path, id)
		if w.inSet && len(w.in) == 0 && !rawEmptyList(raw["inputs"]) {
			t.Errorf("%s inputs %s, want []", id, raw["inputs"])
		}
		if !w.inSet && raw["inputs"] != nil {
			t.Errorf("%s inputs %s, want it left out", id, raw["inputs"])
		}
		if w.effSet && len(w.eff) == 0 && !rawEmptyList(raw["reasoningEfforts"]) {
			t.Errorf("%s reasoningEfforts %s, want []", id, raw["reasoningEfforts"])
		}
		if !w.effSet && raw["reasoningEfforts"] != nil {
			t.Errorf("%s reasoningEfforts %s, want it left out", id, raw["reasoningEfforts"])
		}
		if !w.outSet && raw["maxOutput"] != nil {
			t.Errorf("%s maxOutput %s, want it left out", id, raw["maxOutput"])
		}
	}
	if raw := openBotRawModel(t, a.Path, "ob/under"); raw["contextWindow"] != nil {
		t.Errorf("under window %s", raw["contextWindow"])
	}

	b, _ := os.ReadFile(a.Path)
	st, _ := os.Stat(a.Path)
	if err := a.Sync(); err != nil {
		t.Fatal(err)
	}
	if b2, _ := os.ReadFile(a.Path); !bytes.Equal(b, b2) {
		t.Errorf("sync wrote with nothing changed:\n%s\n%s", b, b2)
	}
	if st2, _ := os.Stat(a.Path); !st2.ModTime().Equal(st.ModTime()) {
		t.Error("sync touched the file with nothing changed")
	}

	// the plain model learns images, a level, a window and an output inside
	// it: the next sync writes those, and only then. The output needs the
	// window, or maxOutput stays omitted.
	for i := range ms {
		if ms[i].ID == "plain" {
			ms[i].Images = true
			ms[i].InputModalities = []string{"text", "image", "pdf"}
			ms[i].Efforts = []string{"high", "low", "high"}
			ms[i].Context = 128000
			ms[i].Output = 4096
		}
	}
	if err := catalog.SaveLive("ob", base, ms); err != nil {
		t.Fatal(err)
	}
	if err := a.Sync(); err != nil {
		t.Fatal(err)
	}
	e, _ = openBotRead(t, a.Path)
	m := openBotListed(t, e, "ob/plain")
	if !slices.Equal(ptrStrings(m.Inputs), []string{"image", "pdf"}) || !slices.Equal(ptrStrings(m.ReasoningEfforts), []string{"low", "high"}) || m.MaxOutput == nil || *m.MaxOutput != 4096 {
		t.Errorf("plain after the change: inputs %v efforts %v maxOutput %v", ptrStrings(m.Inputs), ptrStrings(m.ReasoningEfforts), m.MaxOutput)
	}
	if !slices.Equal(ptrStrings(openBotListed(t, e, "ob/both").Inputs), []string{"image", "pdf"}) {
		t.Error("both lost its inputs when plain changed")
	}
}

// A model that can't see is still told it takes images when Vision
// describes them (provider.Described): told it doesn't, OpenBot turns the
// picture away before the gateway is asked. A pdf its source lists stays
// after image.
func TestOpenBotImageWhenVisionDescribes(t *testing.T) {
	home := syncHome(t)
	was := provider.Described
	provider.Described = func() bool { return true }
	t.Cleanup(func() { provider.Described = was })
	const base = "https://ob-see.example/v1"
	if err := provider.Save(provider.Provider{ID: "obsee", Name: "See", Chat: base, Key: "k", Models: []string{"words", "paper"}}); err != nil {
		t.Fatal(err)
	}
	if err := catalog.SaveLive("obsee", base, []catalog.Model{
		{ID: "words", Name: "Words", InputModalities: []string{"text"}},
		{ID: "paper", Name: "Paper", InputModalities: []string{"text", "pdf"}},
	}); err != nil {
		t.Fatal(err)
	}
	a := openBot(home, filepath.Join(home, ".config"))
	if err := a.Field("provider").Set(magpieID); err != nil {
		t.Fatal(err)
	}
	e, _ := openBotRead(t, a.Path)
	if got := ptrStrings(openBotListed(t, e, "obsee/words").Inputs); !slices.Equal(got, []string{"image"}) {
		t.Errorf("words inputs %v, want [image]", got)
	}
	if got := ptrStrings(openBotListed(t, e, "obsee/paper").Inputs); !slices.Equal(got, []string{"image", "pdf"}) {
		t.Errorf("paper inputs %v, want [image pdf]", got)
	}
}

func ptrStrings(p *[]string) []string {
	if p == nil {
		return nil
	}
	return *p
}

// rawEmptyList reports whether v is the JSON list [], not null and not
// a missing key.
func rawEmptyList(v json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(v), []byte("[]"))
}

func sameStringList(got, want *[]string) bool {
	if (got == nil) != (want == nil) {
		return false
	}
	return slices.Equal(ptrStrings(got), ptrStrings(want))
}

// image follows Images, including on a raw catalog.Model whose source
// listed image. A known text-only model is [], and nothing said of the
// input is left out.
func TestOpenBotInputsFollowImages(t *testing.T) {
	no, yes := false, true
	for _, c := range []struct {
		name string
		m    catalog.Model
		want *[]string
	}{
		{"source listed image, Images false", catalog.Model{InputModalities: []string{"text", "image", "pdf"}}, &[]string{"pdf"}},
		{"image then pdf", catalog.Model{Images: true, InputModalities: []string{"pdf", "image"}}, &[]string{"image", "pdf"}},
		{"known text", catalog.Model{InputModalities: []string{"text"}}, openBotEmpty()},
		{"explicit image no", catalog.Model{ImageInput: &no}, openBotEmpty()},
		{"explicit image yes, Images false", catalog.Model{ImageInput: &yes}, openBotEmpty()},
		{"images, no list", catalog.Model{Images: true}, &[]string{"image"}},
		{"unsaid", catalog.Model{}, nil},
	} {
		if got := openBotInputs(c.m); !sameStringList(got, c.want) {
			t.Errorf("%s: %v, want %v", c.name, ptrStrings(got), ptrStrings(c.want))
		}
	}
}

// reasoningEfforts [] is no effort. A model that thinks and has no levels
// omits the field, so OpenCode offers low, medium and high. Levels OpenBot
// cannot name are [], not omitted: omitting would offer levels the model
// has not.
func TestOpenBotEffortField(t *testing.T) {
	for _, c := range []struct {
		name string
		m    catalog.Model
		want *[]string
	}{
		{"plain", catalog.Model{}, openBotEmpty()},
		{"thinks, no levels", catalog.Model{Reasoning: true}, nil},
		{"named levels", catalog.Model{Reasoning: true, Efforts: []string{"HIGH", "low", "low"}}, &[]string{"low", "high"}},
		{"only levels OpenBot refuses", catalog.Model{Reasoning: true, Efforts: []string{"ultra", "light"}}, openBotEmpty()},
		{"levels, reasoning flag off", catalog.Model{Efforts: []string{"none", "max"}}, &[]string{"none", "max"}},
	} {
		if got := openBotEffortField(c.m); !sameStringList(got, c.want) {
			t.Errorf("%s: %v, want %v", c.name, ptrStrings(got), ptrStrings(c.want))
		}
	}
}

// SetModelReasonless clears the levels and the thinking flag. The file
// then says reasoningEfforts [], so OpenCode sends no reasoning_effort.
func TestOpenBotReasonlessWritesNoEfforts(t *testing.T) {
	home := syncHome(t)
	const base = "https://ob-off.example/v1"
	if err := provider.Save(provider.Provider{ID: "oboff", Name: "Off", Chat: base, Key: "k", Models: []string{"thinks"}}); err != nil {
		t.Fatal(err)
	}
	if err := catalog.SaveLive("oboff", base, []catalog.Model{
		{ID: "thinks", Name: "Thinks", Efforts: []string{"high", "low"}, Reasoning: true, InputModalities: []string{"text"}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelReasonless("oboff/thinks", true); err != nil {
		t.Fatal(err)
	}
	a := openBot(home, filepath.Join(home, ".config"))
	if err := a.Field("provider").Set(magpieID); err != nil {
		t.Fatal(err)
	}
	raw := openBotRawModel(t, a.Path, "oboff/thinks")
	if !rawEmptyList(raw["reasoningEfforts"]) || !rawEmptyList(raw["inputs"]) {
		t.Errorf("reasonless: %s", raw)
	}
}

// maxOutput is written only inside 256 to 1,000,000 and strictly below a
// context window OpenBot accepts. A wider output, an equal one, and one
// beside a window of 0 or under 1,024 are left out.
func TestOpenBotMaxOutputStaysInsideTheWindow(t *testing.T) {
	home := syncHome(t)
	const base = "https://ob-out.example/v1"
	ids := []string{"roomless", "nowindow", "wider", "equal", "fits"}
	if err := provider.Save(provider.Provider{ID: "obout", Name: "Out", Chat: base, Key: "k", Models: ids}); err != nil {
		t.Fatal(err)
	}
	if err := catalog.SaveLive("obout", base, []catalog.Model{
		{ID: "roomless", Name: "Roomless", Context: 512, Output: 4096},
		{ID: "nowindow", Name: "No window", Context: 0, Output: 4096},
		{ID: "wider", Name: "Wider", Context: 128000, Output: 384000},
		{ID: "equal", Name: "Equal", Context: 4096, Output: 4096},
		{ID: "fits", Name: "Fits", Context: 128000, Output: 16384},
	}); err != nil {
		t.Fatal(err)
	}
	a := openBot(home, filepath.Join(home, ".config"))
	if err := a.Field("provider").Set(magpieID); err != nil {
		t.Fatal(err)
	}
	e, _ := openBotRead(t, a.Path)
	for _, id := range []string{"obout/roomless", "obout/nowindow", "obout/wider", "obout/equal"} {
		m := openBotListed(t, e, id)
		if m.MaxOutput != nil {
			t.Errorf("%s maxOutput %d, want it left out", id, *m.MaxOutput)
		}
		if raw := openBotRawModel(t, a.Path, id); raw["maxOutput"] != nil {
			t.Errorf("%s wrote maxOutput %s", id, raw["maxOutput"])
		}
	}
	if m := openBotListed(t, e, "obout/roomless"); m.ContextWindow != nil {
		t.Errorf("roomless window %d, want it left out", *m.ContextWindow)
	}
	if m := openBotListed(t, e, "obout/nowindow"); m.ContextWindow != nil {
		t.Errorf("nowindow window %d", *m.ContextWindow)
	}
	for _, id := range []string{"obout/wider", "obout/equal"} {
		if m := openBotListed(t, e, id); m.ContextWindow == nil {
			t.Errorf("%s context window left out", id)
		}
	}
	m := openBotListed(t, e, "obout/fits")
	if m.ContextWindow == nil || *m.ContextWindow != 128000 || m.MaxOutput == nil || *m.MaxOutput != 16384 {
		t.Errorf("fits window %v maxOutput %v", m.ContextWindow, m.MaxOutput)
	}
}

// The file is where Electron puts OpenBot's userData on each system: app
// data joined with its product name.
func TestOpenBotDirPerOS(t *testing.T) {
	home := filepath.Join("h", "me")
	cfg := filepath.Join(home, ".cfg")
	for _, c := range []struct{ goos, appData, want string }{
		{"darwin", "", filepath.Join(home, "Library", "Application Support", "OpenBot")},
		{"linux", "", filepath.Join(cfg, "OpenBot")},
		{"windows", filepath.Join("R", "Roaming"), filepath.Join("R", "Roaming", "OpenBot")},
		{"windows", "", filepath.Join(home, "AppData", "Roaming", "OpenBot")},
	} {
		if got := openBotDirOn(c.goos, home, cfg, c.appData); got != c.want {
			t.Errorf("%s (APPDATA %q): %s, want %s", c.goos, c.appData, got, c.want)
		}
	}
	if got := openBotDir(home, cfg); got != openBotDirOn(runtime.GOOS, home, cfg, os.Getenv("APPDATA")) {
		t.Errorf("here: %s", got)
	}
}
