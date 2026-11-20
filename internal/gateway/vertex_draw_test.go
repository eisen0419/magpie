package gateway

import (
	"encoding/base64"
	"encoding/json"
	"maps"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
)

// vertexImageAnswer is a generateContent reply that drew one PNG, with a
// thought image ahead of it that a drawing must not return.
func vertexImageAnswer() string {
	return `{"candidates":[{"content":{"role":"model","parts":[{"text":"here"},{"thought":true,"inlineData":{"mimeType":"image/png","data":"dGhvdWdodA=="}},{"inlineData":{"mimeType":"image/png","data":"` +
		base64.StdEncoding.EncodeToString(pngBytes) + `"}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1290}}`
}

func (v *vertexUp) paths() []string {
	v.mu.Lock()
	defer v.mu.Unlock()
	out := make([]string, len(v.asked))
	for i := range v.asked {
		out[i] = v.asked[i].path
	}
	return out
}

// A Gemini image model on google-vertex draws on the project's
// generateContent, signed with the Google token, including one a relay
// names google/gemini-…. The OpenAI chat completions a partner model uses
// answer no image, and that was a 502 ("drew nothing").
func TestVertexGeminiImageDrawsOnGenerateContent(t *testing.T) {
	v := &vertexUp{reply: func(int, map[string]any) (int, string) { return 200, vertexImageAnswer() }}
	s := vertexServe(t, v, `{}`)
	for _, c := range []struct {
		model, bare, size string
		imageSize         bool
	}{
		{"gemini-3.1-flash-image", "gemini-3.1-flash-image", "1536x1024", true},
		{"google/gemini-3.1-flash-image", "gemini-3.1-flash-image", "1536x1024", true},
		{"google/models/gemini-3.1-flash-image", "gemini-3.1-flash-image", "1536x1024", true},
		{"gemini-nano-banana-2.1", "gemini-nano-banana-2.1", "1536x1024", true},
		{"gemini-3-pro-image", "gemini-3-pro-image", "1536x1024", true},
		{"gemini-2.5-flash-image", "gemini-2.5-flash-image", "1536x1024", false},
	} {
		before := v.count()
		body := `{"model":"google-vertex/` + c.model + `","prompt":"a magpie","size":"` + c.size + `"}`
		code, a, raw := postImages(t, s, "/v1/images/generations", "application/json", body)
		got := v.paths()[before:]
		want := "/v1/projects/proj-1/locations/global/publishers/google/models/" + c.bare + ":generateContent"
		if len(got) != 1 || got[0] != want {
			t.Fatalf("%s: asked %v, want %s; HTTP %d %s", c.model, got, want, code, raw)
		}
		if code != 200 || len(a.Data) != 1 || a.Text != "here" || a.Usage.Input != 5 || a.Usage.Output != 1290 {
			t.Fatalf("%s: %d %s", c.model, code, raw)
		}
		if b, _ := base64.StdEncoding.DecodeString(a.Data[0].B64); string(b) != string(pngBytes) {
			t.Fatalf("%s: image %q", c.model, b)
		}
		asked := v.last(t)
		if asked.auth != "Bearer ya29.vertex-test" || asked.key != "" || asked.query != "" {
			t.Fatalf("%s: auth %q key %q query %q", c.model, asked.auth, asked.key, asked.query)
		}
		b, _ := json.Marshal(asked.body)
		if g := gjson.GetBytes(b, "generationConfig.responseModalities").Raw; g != `["TEXT","IMAGE"]` {
			t.Fatalf("%s: responseModalities %s", c.model, g)
		}
		if g := gjson.GetBytes(b, "generationConfig.imageConfig.aspectRatio").String(); g != "3:2" {
			t.Fatalf("%s: aspectRatio %q: %s", c.model, g, b)
		}
		if has := gjson.GetBytes(b, "generationConfig.imageConfig.imageSize").String(); (has == "2K") != c.imageSize {
			t.Fatalf("%s: imageSize %q, want set %v: %s", c.model, has, c.imageSize, b)
		}
		if g := gjson.GetBytes(b, "contents.0.parts.0.text").String(); g != "a magpie\n\nAspect ratio: 3:2." || gjson.GetBytes(b, "contents.0.role").String() != "user" {
			t.Fatalf("%s: contents %s", c.model, b)
		}
		if gjson.GetBytes(b, "model").Exists() || bytesContainsModalities(b) {
			t.Fatalf("%s: a chat field went to generateContent: %s", c.model, b)
		}
	}
}

func bytesContainsModalities(b []byte) bool {
	return strings.Contains(string(b), "modalities") || strings.Contains(string(b), "image_config")
}

// Express mode (an API key, no project) draws the same model at the global
// publishers path, the key in x-goog-api-key, and mints no Google token.
func TestVertexGeminiImageDrawsInExpressMode(t *testing.T) {
	v := &vertexUp{reply: func(int, map[string]any) (int, string) { return 200, vertexImageAnswer() }}
	vertexServe(t, v, `{}`)
	if err := provider.Save(provider.Provider{ID: "google-vertex", Name: "Google Vertex AI", Preset: provider.VertexPreset, Key: "AQ.Ab8RN6-gw"}); err != nil {
		t.Fatal(err)
	}
	s := New()
	for _, c := range []struct{ model, bare string }{
		{"gemini-3.1-flash-image", "gemini-3.1-flash-image"},
		{"google/gemini-3.1-flash-image", "gemini-3.1-flash-image"},
		{"gemini-nano-banana-2.1", "gemini-nano-banana-2.1"},
	} {
		before := v.count()
		code, a, raw := postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/`+c.model+`","prompt":"a magpie","size":"1024x1024"}`)
		got := v.paths()[before:]
		want := "/v1/publishers/google/models/" + c.bare + ":generateContent"
		if len(got) != 1 || got[0] != want {
			t.Fatalf("%s: asked %v, want %s; HTTP %d %s", c.model, got, want, code, raw)
		}
		if code != 200 || len(a.Data) != 1 {
			t.Fatalf("%s: %d %s", c.model, code, raw)
		}
		asked := v.last(t)
		if asked.key != "AQ.Ab8RN6-gw" || asked.auth != "" || strings.Contains(asked.query, "AQ.") {
			t.Fatalf("%s: auth %q key %q query %q", c.model, asked.auth, asked.key, asked.query)
		}
	}
	if n := v.minted(); n != 0 {
		t.Fatalf("%d Google tokens minted for a key", n)
	}
	before := v.count()
	code, _, raw := postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/imagen-4.0-generate-001","prompt":"a magpie"}`)
	for _, p := range v.paths()[before:] {
		if strings.Contains(p, ":generateContent") {
			t.Fatalf("imagen reached generateContent at %s; HTTP %d %s", p, code, raw)
		}
	}
	if code == 200 {
		t.Fatalf("imagen drew in express mode: %s", raw)
	}
}

// A partner model, a Gemini that only chats, and an Imagen stay off
// generateContent: partners on chat completions, Imagen on the images API.
func TestVertexGeminiImageDrawSkipsPartnersAndImagen(t *testing.T) {
	v := &vertexUp{reply: func(int, map[string]any) (int, string) { return 200, vertexImageAnswer() }}
	s := vertexServe(t, v, `{}`)
	for _, c := range []struct{ model, want string }{
		{"deepseek-ai/deepseek-v4.1-flash-maas", "/endpoints/openapi/chat/completions"},
		{"meta/llama-4-maverick-maas", "/endpoints/openapi/chat/completions"},
		{"qwen/qwen-image", "/images/generations"},
		{"imagen-4.0-generate-001", "/images/generations"},
		{"google/imagen-4.0-generate-001", "/images/generations"},
		{"google/gemini-3.8-flash", "/endpoints/openapi/chat/completions"},
		{"google/gemma-4", "/endpoints/openapi/chat/completions"},
	} {
		before := v.count()
		code, _, raw := postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/`+c.model+`","prompt":"a magpie"}`)
		got := v.paths()[before:]
		for _, p := range got {
			if strings.Contains(p, ":generateContent") {
				t.Fatalf("%s reached generateContent at %s; HTTP %d %s", c.model, p, code, raw)
			}
		}
		if len(got) != 1 || !strings.HasSuffix(got[0], c.want) {
			t.Fatalf("%s: asked %v, want a path ending %s; HTTP %d %s", c.model, got, c.want, code, raw)
		}
	}
}

// Express mode has no chat base, and still lists a Gemini image model the
// user added. A partner model is not a drawer. A project lists the same
// image models, and an Imagen the user added.
func TestVertexGeminiImageModelsAreDrawers(t *testing.T) {
	fresh(t)
	catalog.Reset()
	t.Cleanup(catalog.Reset)
	models := []string{"gemini-3.1-flash-image", "google/gemini-3.1-flash-image", "gemini-nano-banana-2.1", "deepseek-ai/deepseek-v4.1-flash-maas", "imagen-4.0-generate-001"}
	if err := provider.Save(provider.Provider{ID: "google-vertex", Name: "Google Vertex AI", Preset: provider.VertexPreset, Key: "AQ.Ab8RN6-gw", Models: models}); err != nil {
		t.Fatal(err)
	}
	p, err := provider.Find("google-vertex")
	if err != nil {
		t.Fatal(err)
	}
	express := drawerIDs(*p)
	for _, id := range []string{"gemini-3.1-flash-image", "google/gemini-3.1-flash-image", "gemini-nano-banana-2.1"} {
		if !containsID(express, id) {
			t.Errorf("express mode omits %s: %v", id, express)
		}
	}
	if containsID(express, "deepseek-ai/deepseek-v4.1-flash-maas") {
		t.Errorf("express mode lists a partner model: %v", express)
	}
	if err := provider.Save(provider.Provider{ID: "google-vertex", Name: "Google Vertex AI", Preset: provider.VertexPreset,
		Vertex: &provider.Vertex{Project: "proj-1", Location: "global"}, Models: models}); err != nil {
		t.Fatal(err)
	}
	p, err = provider.Find("google-vertex")
	if err != nil {
		t.Fatal(err)
	}
	proj := drawerIDs(*p)
	for _, id := range []string{"gemini-3.1-flash-image", "google/gemini-3.1-flash-image", "gemini-nano-banana-2.1", "imagen-4.0-generate-001"} {
		if !containsID(proj, id) {
			t.Errorf("project omits %s: %v", id, proj)
		}
	}
	if containsID(proj, "deepseek-ai/deepseek-v4.1-flash-maas") {
		t.Errorf("project lists a partner model among its drawers: %v", proj)
	}
}

// Chat completions and a Gemini client ask a relay-named Gemini image
// model on the project's generateContent for its bare id. google/ and
// google/models/ are not a second publisher segment, streamed or not.
// A Gemini that only chats (google/gemini-3.8-flash) stays on chat
// completions, its full id in the body.
func TestVertexGeminiImageChatAndPassthroughUseBareModel(t *testing.T) {
	v := &vertexUp{}
	s := vertexServe(t, v, vertexLevels)
	const bare = "/v1/projects/proj-1/locations/global/publishers/google/models/gemini-3.1-flash-image:"
	for _, id := range []string{"google/gemini-3.1-flash-image", "google/models/gemini-3.1-flash-image"} {
		for _, stream := range []string{"true", "false"} {
			vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/`+id+`","stream":`+stream+`,"messages":[{"role":"user","content":"hi"}]}`)
			if p := v.last(t).path; p != bare+"streamGenerateContent" {
				t.Errorf("chat %s stream %s asked at %s", id, stream, p)
			}
		}
		vertexAsk(t, s, "/v1beta/models/google-vertex/"+id+":streamGenerateContent?alt=sse", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
		if p := v.last(t).path; p != bare+"streamGenerateContent" {
			t.Errorf("gemini stream %s asked at %s", id, p)
		}
		vertexAsk(t, s, "/v1beta/models/google-vertex/"+id+":generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
		if p := v.last(t).path; p != bare+"generateContent" {
			t.Errorf("gemini whole %s asked at %s", id, p)
		}
	}
	vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/google/gemini-3.8-flash","messages":[{"role":"user","content":"hi"}]}`)
	if a := v.last(t); a.path != "/v1/projects/proj-1/locations/global/endpoints/openapi/chat/completions" || a.body["model"] != "google/gemini-3.8-flash" {
		t.Errorf("chat-only google/gemini-3.8-flash asked at %s as %v", a.path, a.body["model"])
	}
}

// A relay-named Gemini 2.5 image model is asked as that model: its effort
// is fitted (vertexEffort) and it is sent a thinking budget, which 2.5
// takes, not a thinking level. xhigh, which 2.5 has not, is its high.
func TestVertexGeminiImageChatUsesThinkingBudget(t *testing.T) {
	v := &vertexUp{}
	s := vertexServe(t, v, vertexLevels)
	want := map[string]any{"thinkingBudget": 24000.0, "includeThoughts": true}
	for _, id := range []string{"google/gemini-2.5-flash-image", "google/models/gemini-2.5-flash-image"} {
		vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/`+id+`","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
		a := v.last(t)
		if p := a.path; p != "/v1/projects/proj-1/locations/global/publishers/google/models/gemini-2.5-flash-image:streamGenerateContent" {
			t.Errorf("%s asked at %s", id, p)
		}
		if got := thinkingOf(a); !maps.Equal(got, want) {
			t.Errorf("%s thinkingConfig %v, want %v", id, got, want)
		}
	}
}

// Saved per-model effort levels for a relay-named Gemini image model
// (google-vertex/google/gemini-…) stay in force during vertexEffort; if
// none are saved on the prefixed id, levels on the bare id still apply.
func TestVertexGeminiImageChatKeepsSavedEfforts(t *testing.T) {
	v := &vertexUp{}
	s := vertexServe(t, v, vertexLevels)
	if err := provider.SetModelEfforts("google-vertex/google/gemini-3.1-flash-image", []string{"minimal", "low", "high"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("google-vertex/google/gemini-2.5-flash-image", []string{"low", "high", "xhigh"}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SetModelEfforts("google-vertex/gemini-3-pro-image", []string{"minimal", "low", "high"}); err != nil {
		t.Fatal(err)
	}

	vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/google/gemini-3.1-flash-image","stream":true,"reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]}`)
	wantLevel := map[string]any{"thinkingLevel": "minimal", "includeThoughts": false}
	if got := thinkingOf(v.last(t)); !maps.Equal(got, wantLevel) {
		t.Errorf("saved prefixed 3.1 effort: thinkingConfig %v, want %v", got, wantLevel)
	}

	vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/google/gemini-2.5-flash-image","stream":true,"reasoning_effort":"xhigh","messages":[{"role":"user","content":"hi"}]}`)
	wantBudget := map[string]any{"thinkingBudget": 24576.0, "includeThoughts": true}
	if got := thinkingOf(v.last(t)); !maps.Equal(got, wantBudget) {
		t.Errorf("saved prefixed 2.5 effort: thinkingConfig %v, want %v", got, wantBudget)
	}

	vertexAsk(t, s, "/v1/chat/completions", `{"model":"google-vertex/google/models/gemini-3-pro-image","stream":true,"reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]}`)
	if got := thinkingOf(v.last(t)); !maps.Equal(got, wantLevel) {
		t.Errorf("bare fallback effort: thinkingConfig %v, want %v", got, wantLevel)
	}
}

func drawerIDs(p provider.Provider) []string {
	var ids []string
	for _, m := range Drawers(p) {
		ids = append(ids, m.ID)
	}
	return ids
}

func containsID(ids []string, id string) bool {
	for _, s := range ids {
		if s == id {
			return true
		}
	}
	return false
}
