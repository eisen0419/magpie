package provider

import (
	"context"
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// A Gemini image model is named on Vertex AI's publishers/google/models
// path by its bare id. A relay's google/ and google/models/ are not
// another publisher segment. A chat-only Gemini written google/…, a
// partner publisher, a namespaced id and an Imagen stay spelled as given.
func TestVertexGeminiImagePaths(t *testing.T) {
	const (
		stream = "/publishers/google/models/gemini-3.1-flash-image:streamGenerateContent?alt=sse"
		whole  = "/publishers/google/models/gemini-3.1-flash-image:generateContent"
	)
	for _, id := range []string{
		"gemini-3.1-flash-image",
		"google/gemini-3.1-flash-image",
		"google/models/gemini-3.1-flash-image",
		"models/gemini-3.1-flash-image",
	} {
		if got := VertexPath(id); got != stream {
			t.Errorf("VertexPath(%q) = %q, want %q", id, got, stream)
		}
		if got := VertexWholePath(id); got != whole {
			t.Errorf("VertexWholePath(%q) = %q, want %q", id, got, whole)
		}
	}
	for _, c := range []struct{ id, stream, whole string }{
		{"google/gemini-3.8-flash", "/publishers/google/models/google/gemini-3.8-flash:streamGenerateContent?alt=sse", "/publishers/google/models/google/gemini-3.8-flash:generateContent"},
		{"deepseek-ai/deepseek-v4.1-flash-maas", "/publishers/google/models/deepseek-ai/deepseek-v4.1-flash-maas:streamGenerateContent?alt=sse", "/publishers/google/models/deepseek-ai/deepseek-v4.1-flash-maas:generateContent"},
		{"qwen/qwen-image", "/publishers/google/models/qwen/qwen-image:streamGenerateContent?alt=sse", "/publishers/google/models/qwen/qwen-image:generateContent"},
		{"example/model-demo", "/publishers/google/models/example/model-demo:streamGenerateContent?alt=sse", "/publishers/google/models/example/model-demo:generateContent"},
		{"imagen-4.0-generate-001", "/publishers/google/models/imagen-4.0-generate-001:streamGenerateContent?alt=sse", "/publishers/google/models/imagen-4.0-generate-001:generateContent"},
		{"google/imagen-4.0-generate-001", "/publishers/google/models/google/imagen-4.0-generate-001:streamGenerateContent?alt=sse", "/publishers/google/models/google/imagen-4.0-generate-001:generateContent"},
		{"anthropic/claude-x", "/publishers/google/models/anthropic/claude-x:streamGenerateContent?alt=sse", "/publishers/google/models/anthropic/claude-x:generateContent"},
	} {
		if got := VertexPath(c.id); got != c.stream {
			t.Errorf("VertexPath(%q) = %q, want %q", c.id, got, c.stream)
		}
		if got := VertexWholePath(c.id); got != c.whole {
			t.Errorf("VertexWholePath(%q) = %q, want %q", c.id, got, c.whole)
		}
	}
}

func TestVertexGeminiImage(t *testing.T) {
	for _, c := range []struct {
		model, bare string
		ok          bool
	}{
		{"gemini-3.1-flash-image", "gemini-3.1-flash-image", true},
		{" google/gemini-3.1-flash-image ", "gemini-3.1-flash-image", true},
		{"google/models/gemini-3.1-flash-image", "gemini-3.1-flash-image", true},
		{"models/gemini-nano-banana-2.1", "gemini-nano-banana-2.1", true},
		{"gemini-2.5-flash-image", "gemini-2.5-flash-image", true},
		{"gemini-3-pro-image", "gemini-3-pro-image", true},
		{"nano-banana-pro", "nano-banana-pro", true},
		{"gemini-3.8-flash", "", false},
		{"google/gemini-3.8-flash", "", false},
		{"google/gemma-4", "", false},
		{"deepseek-ai/deepseek-v4.1-flash-maas", "", false},
		{"meta/llama-4-maverick-maas", "", false},
		{"qwen/qwen-image", "", false},
		{"imagen-4.0-generate-001", "", false},
		{"google/imagen-4.0-generate-001", "", false},
		{"claude-sonnet-4-5@20250929", "", false},
		{"", "", false},
	} {
		bare, ok := VertexGeminiImage(c.model)
		if bare != c.bare || ok != c.ok {
			t.Errorf("%q: %q %v, want %q %v", c.model, bare, ok, c.bare, c.ok)
		}
	}
}

// A Gemini image model is served on generateContent even when a relay
// names it google/gemini-…. That prefix is not a partner publisher.
// A Gemini that only chats keeps the partner treatment the slash gives
// it, and so does every other publisher, including an Imagen written
// google/imagen-….
func TestVertexGeminiImageModelAPIs(t *testing.T) {
	p := Provider{ID: "google-vertex", Preset: VertexPreset, Vertex: &Vertex{Project: "acme-vertex", Location: "global"}}
	gemini := []Protocol{Gemini}
	chat := []Protocol{Chat}
	for _, c := range []struct {
		model string
		want  []Protocol
	}{
		{"gemini-3.1-flash-image", gemini},
		{"google/gemini-3.1-flash-image", gemini},
		{"google/models/gemini-3.1-flash-image", gemini},
		{"models/gemini-3.1-flash-image", gemini},
		{"gemini-nano-banana-2.1", gemini},
		{"gemini-3-pro-image", gemini},
		{"gemini-2.5-flash-image", gemini},
		{"nano-banana-pro", gemini},
		{"gemini-3.8-flash", gemini},
		{"google/gemini-3.8-flash", chat},
		{"google/gemma-4-31b", chat},
		{"deepseek-ai/deepseek-v4.1-flash-maas", chat},
		{"meta/llama-4-maverick-maas", chat},
		{"qwen/qwen-image", chat},
		{"anthropic/claude-x", chat},
		{"imagen-4.0-generate-001", gemini},
		{"google/imagen-4.0-generate-001", chat},
		{"claude-sonnet-4-5@20250929", []Protocol{Anthropic}},
	} {
		if got := p.APIs(c.model); !slices.Equal(got, c.want) {
			t.Errorf("%s: %v, want %v", c.model, got, c.want)
		}
	}
}

// Testing a relay-named Gemini image model asks generateContent for the
// bare id, with the project's token. A partner model tested beside it is
// still asked on chat completions, its full id in the body.
func TestVertexGeminiImageModelIsTestedOnGenerateContent(t *testing.T) {
	vertexHome(t)
	f := newVertexGoogle(t)
	file := filepath.Join(t.TempDir(), "application_default_credentials.json")
	writeFile(t, file, vertexUser("1//0g-test-refresh"))
	if err := Save(vertexAt("global", file)); err != nil {
		t.Fatal(err)
	}
	p, err := Find("vertex-prod")
	if err != nil {
		t.Fatal(err)
	}
	rs := p.TestModels(context.Background(), []string{"google/gemini-3.1-flash-image", "deepseek-ai/deepseek-v4.1-flash-maas"})
	if len(rs) != 2 || !rs[0].OK || rs[0].Protocol != Gemini || !rs[1].OK || rs[1].Protocol != Chat {
		t.Fatalf("models' tests: %+v", rs)
	}
	var image, chats int
	for _, c := range f.vertexCalls() {
		switch {
		case strings.HasSuffix(c.path, ":generateContent"):
			image++
			if c.path != "/vertex/v1/projects/acme-vertex/locations/global/publishers/google/models/gemini-3.1-flash-image:generateContent" ||
				c.header.Get("Authorization") != "Bearer ya29.minted-1" || strings.Contains(c.path, "google/gemini") {
				t.Errorf("image model asked at %s with %q", c.path, c.header.Get("Authorization"))
			}
		case strings.HasSuffix(c.path, "/endpoints/openapi/chat/completions"):
			chats++
			var body struct {
				Model string `json:"model"`
			}
			json.Unmarshal([]byte(c.body), &body)
			if c.path != "/vertex/v1/projects/acme-vertex/locations/global/endpoints/openapi/chat/completions" ||
				c.header.Get("Authorization") != "Bearer ya29.minted-1" || body.Model != "deepseek-ai/deepseek-v4.1-flash-maas" {
				t.Errorf("partner model asked at %s with %q: %s", c.path, c.header.Get("Authorization"), c.body)
			}
		default:
			t.Errorf("asked at %s", c.path)
		}
	}
	if image != 1 || chats != 1 {
		t.Errorf("generateContent %d times, chat completions %d, want 1 and 1", image, chats)
	}
}
