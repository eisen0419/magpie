package agent

// OpenBot (nightly-labs/openbot, openbot.run; jiezhengj on Discord) is an
// Electron workspace that runs Codex, Claude, OpenCode and other agents,
// and lists the user's own OpenAI-compatible endpoints in its model picker.
// It keeps them in openbot-custom-providers-v1.json in Electron's userData
// folder, app data joined with its product name: ~/Library/Application
// Support/OpenBot on a Mac, %APPDATA%\OpenBot on Windows, ~/.config/OpenBot
// on Linux (src/main/index.ts, application-services.ts; v0.58.0, model
// capability fields from v0.74.0, where reasoningEfforts: [] is accepted
// too). The file (custom-provider-store.ts, its zod fileSchema):
//
//	{"version":1,"providers":[{"id":"magpie","name":"magpie",
//	  "baseUrl":"http://127.0.0.1:3425/agent/openbot/v1",
//	  "models":[{"id":"<model>","name":"<name>","contextWindow":<n>,
//	    "maxOutput":<n>,"reasoningEfforts":["low","high"],
//	    "inputs":["image","pdf"]},…],
//	  "secret":null}]}
//
// version must be 1: a file OpenBot can't parse leaves it with no endpoints
// and refuses every save, so magpie leaves alone a file it can't read the
// same way. An entry's key and headers are one ciphertext, secret, sealed
// with Electron's safeStorage (the Keychain, DPAPI), which magpie doesn't
// open, so magpie's entry has none: OpenBot hands it to OpenCode keyless
// (opencode-config.ts, providerEntry), and the gateway takes a request with
// no key from this computer. OpenCode sends it, with OpenCode's
// User-Agent, so the address names OpenBot instead: the gateway reads
// /agent/openbot off it (gateway.AgentPath) and counts the request as
// OpenBot's. A secret the user gave magpie's entry in OpenBot stays.
//
// OpenBot's list decoders fail closed: one entry or model it wouldn't accept
// empties the whole list. So the models are the ones picked for it, at most
// 64 (CUSTOM_PROVIDER_LIMITS.models), each with an id OpenBot takes as
// magpie/<id> (isAgentModel: ASCII letters, digits and ._:/[],=-, at most
// 160), and a context window only inside its 1,024 to 100,000,000. A reply
// limit is maxOutput only inside 256 to 1,000,000 and strictly below the
// context window written beside it. OpenBot ignores a max output when the
// window is absent (opencode-config.ts) and refuses that edit
// (maxOutputNeedsWindow); its probe keeps one only when it is below the
// window, so the prompt still has room. reasoningEfforts is a subset of
// none, minimal, low, medium, high, xhigh, max, in that order (ultra and
// any other word left out: modelSchema drops an out-of-order or unknown
// field via .catch(undefined), and IPC validation rejects it). inputs is
// a subset of image, pdf, image first: image when Images is set, Vision's
// stand-in included, and pdf when its source lists pdf. A Chat
// Completions file part is carried on to Responses, Messages and Gemini;
// a Chat upstream that can't hold a file is told of it in words. A known
// text-only model writes "inputs": [], and a model that does not think,
// or whose levels OpenBot does not name, writes "reasoningEfforts": [].
// An empty list is not the field left out. inputs [] is modalities text
// in and text out, so OpenCode sends no image (a text-only model answers
// 400) and the dialog says Text only; left out, modalities stay unset,
// OpenCode sends images, and the dialog defaults an opencode row to
// ["image"]. reasoningEfforts [] is variants {}, so OpenCode sends no
// reasoning_effort and the dialog says No reasoning; left out, OpenCode
// invents low, medium and high. A model that thinks and has no levels
// omits reasoningEfforts, so those three are offered and the gateway fits
// them (#7439). When maxOutput is omitted, OpenBot fills its default output
// limit if contextWindow is set, and omits limit altogether when there is
// no contextWindow (opencode-config.ts modelEntry).
//
// OpenBot reads the file once, when it starts, and writes it whole from
// what it holds only when an endpoint is saved, edited or removed there.
// So a change shows after a restart, and an endpoint saved in OpenBot
// before that writes magpie's entry out again: the notice says both. The
// model is picked in OpenBot's own picker, so what magpie sets is whether
// its models are in it. The user's other entries stay as written, and so
// do the name and harness they give magpie's.
//
// MCP servers live in OpenBot's SQLite database (McpServerStore), which
// magpie doesn't write; magpie's own (/mcp/magpie/…) are added there by
// hand as HTTP servers.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/tidwall/gjson"

	"github.com/yetone/magpie/internal/appdir"
	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/edit"
	"github.com/yetone/magpie/internal/gateway"
	"github.com/yetone/magpie/internal/lazyre"
)

// openBotFile is the file of OpenBot's custom endpoints in its userData.
const openBotFile = "openbot-custom-providers-v1.json"

// openBotApp is OpenBot's macOS bundle (electron-builder.yml: appId
// app.openbot.desktop, productName OpenBot).
var openBotApp = desktopApp{id: "app.openbot.desktop", names: []string{"OpenBot.app"}}

// openBotMaxModels is how many models OpenBot takes for one endpoint
// (CUSTOM_PROVIDER_LIMITS.models); a list longer than that hides them all.
const openBotMaxModels = 64

// openBotModelID is a model id OpenBot lists as magpie/<id>
// (isAgentModel, ipc-agent-identity.ts).
var openBotModelID = lazyre.New(`^[A-Za-z0-9._:/\[\],=-]+$`)

// openBotProc is OpenBot's process: the .app's and the .exe's OpenBot, the
// AppImage's openbot.
var openBotProc = []string{`(^|/)(OpenBot|openbot)( |$)`}

var noticeOpenBot = newNotice("OpenBot reads its custom endpoints at start-up — restart OpenBot to see magpie's models in its model picker, before adding or editing an endpoint there.")

func openBot(home, cfg string) *Agent {
	dir := openBotDir(home, cfg)
	path := filepath.Join(dir, openBotFile)
	write := func(on bool) error {
		if !on {
			return openBotUnwire(path)
		}
		return openBotWire(path, openBotEntry(gateway.URL()))
	}
	return atomic(&Agent{
		ID: "openbot", Name: "OpenBot", Icon: "openbot", Aliases: []string{"open-bot"},
		Dir: dir, Path: path,
		// its userData folder once it has run, else its app
		detect: func() bool {
			return isDir(dir) || macBundle(openBotApp) || openBotWindowsApp()
		},
		Notice: func() string {
			if openBotWired(path) && Running(openBotProc...) {
				return noticeOpenBot.String()
			}
			return ""
		},
		Sync: func() error {
			if !openBotWired(path) {
				return nil
			}
			return write(true)
		},
		// its picker lists the models picked for it (#1508)
		ListsModels: true,
		Fields: []Field{{
			Key: "provider", Label: "provider",
			Get: func() string {
				if openBotWired(path) {
					return magpieID
				}
				return ""
			},
			Set: func(v string) error { return write(v != "") },
			Options: func(map[string]string) []Option {
				return []Option{{Value: magpieID, Label: "magpie", Icon: "magpie", Note: "every magpie model in OpenBot's model picker"}}
			},
		}},
	}, path)
}

// openBotDir is OpenBot's userData folder, as Electron finds it.
func openBotDir(home, cfg string) string {
	return openBotDirOn(runtime.GOOS, home, cfg, appdir.Getenv("APPDATA"))
}

// openBotDirOn is openBotDir on the system goos, with the XDG_CONFIG_HOME
// folder cfg and %APPDATA% appData.
func openBotDirOn(goos, home, cfg, appData string) string {
	switch goos {
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", "OpenBot")
	case "windows":
		d := appData
		if d == "" {
			d = filepath.Join(home, "AppData", "Roaming")
		}
		return filepath.Join(d, "OpenBot")
	}
	return filepath.Join(cfg, "OpenBot")
}

// openBotWindowsApp says OpenBot's installer has put it where it installs
// it for one user, %LOCALAPPDATA%\Programs\<its name>, or for all of them.
func openBotWindowsApp() bool {
	if runtime.GOOS != "windows" {
		return false
	}
	var dirs []string
	if d := appdir.Getenv("LOCALAPPDATA"); d != "" {
		dirs = append(dirs, filepath.Join(d, "Programs"))
	}
	for _, v := range []string{"ProgramW6432", "ProgramFiles"} {
		if d := appdir.Getenv(v); d != "" {
			dirs = append(dirs, d)
		}
	}
	for _, d := range dirs {
		for _, name := range []string{"OpenBot", "openbot"} {
			if isFile(filepath.Join(d, name, "OpenBot.exe")) {
				return true
			}
		}
	}
	return false
}

// openBotEffortOrder is the reasoning levels OpenBot takes, weakest first
// (isCustomProviderModelEfforts). A list in any other order is refused.
var openBotEffortOrder = []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}

// openBotInputOrder is the input kinds OpenBot takes, in the only order
// it accepts (isCustomProviderInputs).
var openBotInputOrder = []string{"image", "pdf"}

// openBotModel is a model of an entry, as OpenBot writes one.
// ReasoningEfforts and Inputs are pointers so an empty list is written
// as [] (OpenBot's "none") and a nil pointer is left out (OpenBot's
// default). omitempty on a []string drops both, and a nil slice inside
// the pointer would be null.
type openBotModel struct {
	ID               string    `json:"id"`
	Name             string    `json:"name"`
	ContextWindow    int       `json:"contextWindow,omitempty"`
	MaxOutput        int       `json:"maxOutput,omitempty"`
	ReasoningEfforts *[]string `json:"reasoningEfforts,omitempty"`
	Inputs           *[]string `json:"inputs,omitempty"`
}

// openBotProvider is an entry of providers in the order OpenBot writes one.
type openBotProvider struct {
	ID      string          `json:"id"`
	Name    string          `json:"name"`
	BaseURL string          `json:"baseUrl"`
	Models  []openBotModel  `json:"models"`
	Secret  json.RawMessage `json:"secret"`
	Harness string          `json:"harness,omitempty"`
}

// openBotEntry is magpie's entry for a gateway at gw: its Chat Completions,
// no key, and the models picked for it that OpenBot would list, at most 64.
func openBotEntry(gw string) openBotProvider {
	ms := []openBotModel{}
	seen := map[string]bool{}
	for _, m := range sharedModels("openbot") {
		if m.ID == "" || seen[m.ID] || len(m.ID) > 128 || len(magpieID)+1+len(m.ID) > 160 || !openBotModelID.MatchString(m.ID) {
			continue
		}
		seen[m.ID] = true
		name := m.Name
		if name == "" || len([]rune(name)) > 160 {
			name = m.ID
		}
		e := openBotModel{ID: m.ID, Name: name}
		if m.Context >= 1024 && m.Context <= 100_000_000 {
			e.ContextWindow = m.Context
		}
		// below the window: OpenBot ignores maxOutput with no window, its
		// form refuses saving one without a window (maxOutputNeedsWindow),
		// and model-server-probe keeps one only while maxOutput < contextWindow
		if e.ContextWindow > 0 && m.Output >= 256 && m.Output <= 1_000_000 && m.Output < e.ContextWindow {
			e.MaxOutput = m.Output
		}
		e.ReasoningEfforts = openBotEffortField(m)
		e.Inputs = openBotInputs(m)
		ms = append(ms, e)
		if len(ms) == openBotMaxModels {
			break
		}
	}
	return openBotProvider{ID: magpieID, Name: "magpie", BaseURL: gateway.AgentURL(gw, "openbot") + "/v1", Models: ms, Secret: json.RawMessage("null")}
}

// openBotEfforts are the levels of in that OpenBot offers, in its order.
// A word it doesn't name (ultra, a vendor's light) is left out, and the
// rest are read lower case, so HIGH is high and a repeated low is one.
// None is nil. openBotEffortField decides whether that is [] or left out.
func openBotEfforts(in []string) []string {
	have := map[string]bool{}
	for _, e := range in {
		l := strings.ToLower(strings.TrimSpace(e))
		if slices.Contains(openBotEffortOrder, l) {
			have[l] = true
		}
	}
	if len(have) == 0 {
		return nil
	}
	out := make([]string, 0, len(have))
	for _, l := range openBotEffortOrder {
		if have[l] {
			out = append(out, l)
		}
	}
	return out
}

// openBotEmpty is an empty list written as [], not left out and not null.
// A nil []string marshals as null; omitempty on the pointer leaves a nil
// pointer out.
func openBotEmpty() *[]string {
	empty := []string{}
	return &empty
}

// openBotEffortField is the reasoningEfforts to write for m.
//
// A list OpenBot names is that list. Otherwise the field is [], except a
// model that thinks and whose Efforts are empty: that one is left out.
//
// [] is variants: {} (opencode-config.ts modelEntry). OpenCode then offers
// no effort and sends no reasoning_effort, the dialog says No reasoning,
// Pi treats it as no reasoning, and OpenBot's Ollama probe skips the model
// (the probe calls /api/version once per provider when any model leaves
// reasoningEfforts unset, then /api/show per unset model; when every model
// sets reasoningEfforts it skips /api/version too). Left out, OpenCode
// invents low, medium and high until the async probe finishes (or whenever
// magpie is not running yet), and on a model whose levels OpenBot cannot
// name (ultra, light) /api/show still reports "thinking" because the model
// thinks.
//
// The model that thinks with no levels is the exception, as Command Code
// marks it reasoning and gives no list: OpenCode then offers low, medium
// and high, and the gateway fits them to at most high (fitFor,
// openCodeReasons). [] would say No reasoning and send nothing, so the
// switch is never asked. A model that has levels, none of which OpenBot
// names (ultra, light), is [] rather than left out: the field cannot carry
// those words, and leaving it out would have OpenCode offer low, medium
// and high, which that model has not. OpenCode's own config writes an
// empty variants set when a model has no levels to offer (openCodeVariants).
func openBotEffortField(m catalog.Model) *[]string {
	if got := openBotEfforts(m.Efforts); len(got) > 0 {
		return &got
	}
	if m.Reasoning && len(m.Efforts) == 0 {
		return nil
	}
	return openBotEmpty()
}

// openBotInputs are the kinds OpenBot's picker offers for m, image then
// pdf, in the only order it accepts (openBotInputOrder). image when
// m.Images is set, which Vision's stand-in sets and a raw model's false
// keeps even when the source listed image (openCodeInputs decides image
// the same way). pdf when the source listed pdf.
//
// No such kind, when the input is known, is []. Known is a source list
// (InputModalities non-nil) or an explicit image answer (ImageInput
// non-nil): the pair InputsOf treats as the source having spoken, and the
// same split as model-server-probe.ts, which keeps an empty modalities
// array and leaves a model with no array out. [] is modalities text in
// and text out, so OpenCode shows Text only and does not send an attached
// image, which magpie refuses with 400 on a text-only model. Left out,
// modalities stay unset, OpenCode sends images, and the dialog defaults
// an opencode row to ["image"]. Nothing said of the input stays left out.
func openBotInputs(m catalog.Model) *[]string {
	var out []string
	for _, k := range openBotInputOrder {
		switch {
		case k == "image" && m.Images:
			out = append(out, k)
		case k != "image" && slices.Contains(m.InputModalities, k):
			out = append(out, k)
		}
	}
	if len(out) > 0 {
		return &out
	}
	if m.InputModalities != nil || m.ImageInput != nil {
		return openBotEmpty()
	}
	return nil
}

// openBotWired says the file has magpie's entry.
func openBotWired(path string) bool {
	_, i, _ := openBotProviders(path)
	return i >= 0
}

// openBotProviders are the entries of the file's providers as written, and
// the place of magpie's among them (-1 for none). A file that isn't there or
// holds nothing has none; one OpenBot wouldn't read is an error, so magpie
// leaves it alone.
func openBotProviders(path string) (list []json.RawMessage, i int, err error) {
	i = -1
	b, err := edit.Read(path)
	if err != nil {
		return nil, i, err
	}
	if len(bytes.TrimSpace(b)) == 0 {
		return nil, i, nil
	}
	if !json.Valid(b) || !gjson.ParseBytes(b).IsObject() {
		return nil, i, fmt.Errorf("%s isn't JSON OpenBot reads", path)
	}
	if v := gjson.GetBytes(b, "version"); v.Type != gjson.Number || v.Raw != "1" {
		return nil, i, fmt.Errorf("%s says version %s, not the 1 OpenBot reads, so magpie leaves it alone", path, v.Raw)
	}
	raw := gjson.GetBytes(b, "providers")
	if !raw.IsArray() || json.Unmarshal([]byte(raw.Raw), &list) != nil {
		return nil, i, fmt.Errorf("%s: providers is not a list", path)
	}
	for j, e := range list {
		if gjson.GetBytes(e, "id").String() == magpieID {
			return list, j, nil
		}
	}
	return list, i, nil
}

// openBotWire puts magpie's entry into the file, made when there is none;
// an entry there already keeps the name, harness and secret the user gave
// it in OpenBot. The file is written only when the entry says something
// else.
func openBotWire(path string, want openBotProvider) error {
	list, i, err := openBotProviders(path)
	if err != nil {
		return err
	}
	if i >= 0 {
		if n := gjson.GetBytes(list[i], "name"); n.Type == gjson.String && n.String() != "" {
			want.Name = n.String()
		}
		if h := gjson.GetBytes(list[i], "harness"); h.Type == gjson.String {
			want.Harness = h.String()
		}
		if s := gjson.GetBytes(list[i], "secret"); s.Type == gjson.String {
			want.Secret = json.RawMessage(s.Raw)
		}
	}
	w, err := json.Marshal(want)
	if err != nil {
		return err
	}
	if i >= 0 && sameJSON(string(list[i]), want) {
		return nil
	}
	if i >= 0 {
		list[i] = w
	} else {
		list = append(list, w)
	}
	if _, has := edit.GetJSON(path, "version"); !has {
		return edit.SetJSON(path, edit.KV{Path: "version", Value: 1}, edit.KV{Path: "providers", Value: list})
	}
	return edit.SetJSON(path, edit.KV{Path: "providers", Value: list})
}

// openBotUnwire takes magpie's entry out of the file, and the file when
// nothing else is left in it: the version and an empty list are what OpenBot
// starts without a file on too.
func openBotUnwire(path string) error {
	list, i, err := openBotProviders(path)
	if err != nil || i < 0 {
		return err
	}
	list = append(list[:i], list[i+1:]...)
	if len(list) == 0 {
		var rest map[string]json.RawMessage
		if b, err := edit.Read(path); err == nil && json.Unmarshal(b, &rest) == nil {
			delete(rest, "version")
			delete(rest, "providers")
			if len(rest) == 0 {
				return os.Remove(path)
			}
		}
	}
	return edit.SetJSON(path, edit.KV{Path: "providers", Value: list})
}
