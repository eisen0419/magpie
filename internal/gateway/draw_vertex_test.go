package gateway

import (
	"encoding/base64"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// vertexDrawn is a Gemini image model's generateContent answer on Vertex
// AI, in the shape Google documents for it (candidates[].content.parts[],
// the picture as inlineData beside a line of text): #7263 gave no answer
// of its own, since it never reached this endpoint.
func vertexDrawn() string {
	return `{"candidates":[{"content":{"role":"model","parts":[{"text":"Here is a magpie."},{"inlineData":{"mimeType":"image/png","data":"` +
		base64.StdEncoding.EncodeToString(pngBytes) + `"}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":6,"candidatesTokenCount":1290,"totalTokenCount":1296,"trafficType":"ON_DEMAND"},"modelVersion":"gemini-3.1-flash-image"}`
}

// A Gemini image model on Google Vertex AI draws on Vertex AI's own
// generateContent at the project, signed with the Google token, not on
// the OpenAI-compatible chat completions its partner models are served
// on, which answered #7263's gemini-3.1-flash-image with no picture (502
// "drew nothing"). The id as a relay writes it (google/…) is asked
// without its google/. An express-mode key draws on the global endpoint,
// the key in x-goog-api-key, and lists its Gemini image models.
func TestVertexGeminiImageDrawsOnGenerateContent(t *testing.T) {
	v := &vertexUp{reply: func(int, map[string]any) (int, string) { return http.StatusOK, vertexDrawn() }}
	s := vertexServe(t, v, `{}`)
	for _, model := range []string{"gemini-3.1-flash-image", "google/gemini-3.1-flash-image", "gemini-nano-banana-2.1"} {
		code, a, raw := postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/`+model+`","prompt":"a magpie","size":"1536x1024"}`)
		if code != 200 || len(a.Data) != 1 {
			t.Fatalf("%s: %d %s", model, code, raw)
		}
		if b, _ := base64.StdEncoding.DecodeString(a.Data[0].B64); string(b) != string(pngBytes) {
			t.Fatalf("%s: image %q", model, b)
		}
		got := v.last(t)
		bare := strings.TrimPrefix(model, "google/")
		if want := "/v1/projects/proj-1/locations/global/publishers/google/models/" + bare + ":generateContent"; got.path != want || got.query != "" {
			t.Fatalf("%s was asked at %s?%s, want %s", model, got.path, got.query, want)
		}
		if got.auth != "Bearer ya29.vertex-test" || got.key != "" {
			t.Fatalf("%s was signed %q, key %q", model, got.auth, got.key)
		}
		gen, _ := got.body["generationConfig"].(map[string]any)
		if mods, _ := gen["responseModalities"].([]any); !slices.Equal(mods, []any{"TEXT", "IMAGE"}) {
			t.Fatalf("%s was sent %v", model, got.body)
		}
		if ic, _ := gen["imageConfig"].(map[string]any); ic["aspectRatio"] != "3:2" {
			t.Fatalf("%s: imageConfig %v", model, gen["imageConfig"])
		}
	}

	// express mode: a key and no project
	if err := provider.Save(provider.Provider{ID: "google-vertex", Name: "Google Vertex AI", Preset: provider.VertexPreset, Key: "AQ.Ab8RN6-gw",
		Models: []string{"gemini-3.1-flash-image", "gemini-3.8-flash"}}); err != nil {
		t.Fatal(err)
	}
	p, err := provider.Find("google-vertex")
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, m := range Drawers(*p) {
		ids = append(ids, m.ID)
	}
	if !slices.Contains(ids, "gemini-3.1-flash-image") || slices.ContainsFunc(ids, func(id string) bool { return !vertexDrawsGemini(*p, id) }) {
		t.Fatalf("express drawers %v", ids)
	}
	s = New()
	code, a, raw := postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/gemini-3.1-flash-image","prompt":"a magpie"}`)
	if code != 200 || len(a.Data) != 1 {
		t.Fatalf("express: %d %s", code, raw)
	}
	got := v.last(t)
	if got.path != "/v1/publishers/google/models/gemini-3.1-flash-image:generateContent" || got.key != "AQ.Ab8RN6-gw" || got.auth != "" || strings.Contains(got.query, "AQ.") {
		t.Fatalf("express was asked %+v", got)
	}
	for _, a := range v.asked {
		if strings.Contains(a.path, "/endpoints/openapi/") {
			t.Fatalf("a drawing went to chat completions: %s", a.path)
		}
	}
}

// Only Vertex AI's Gemini image models are drawn on generateContent: AI
// Studio's key keeps its own, and its Imagen stays on the images API. A
// relay's models (named with google/ or not) and OpenRouter's stay where
// they were. At Vertex AI a partner model and Claude are neither, an
// Imagen on a project draws on :predict, and express mode's Imagen stays
// off that route (express mode serves Gemini alone).
func TestVertexGeminiDrawOnlyForVertex(t *testing.T) {
	studio := provider.Provider{ID: "gemini", Chat: "https://generativelanguage.googleapis.com/v1beta/openai", Key: "k"}
	relay := provider.Provider{ID: "relay", Chat: "https://relay.example/v1", Key: "k"}
	openrouter := provider.Provider{ID: "openrouter", Chat: "https://openrouter.ai/api/v1", Key: "k"}
	vertex := provider.Provider{ID: "google-vertex", Preset: provider.VertexPreset, Vertex: &provider.Vertex{Project: "proj-1", Location: "global"}}
	// express mode: a key and no project, served Gemini alone
	express := provider.Provider{ID: "google-vertex", Preset: provider.VertexPreset, Key: "AQ.Ab8RN6-gw"}
	for _, c := range []struct {
		p     provider.Provider
		model string
		via   drawVia
		vtx   bool
	}{
		{studio, "gemini-3.1-flash-image", viaGemini, false},
		{studio, "imagen-4.0-generate-001", viaImages, false},
		{relay, "gemini-3.1-flash-image", viaChat, false},
		{relay, "google/gemini-3.1-flash-image", viaChat, false},
		{relay, "imagen-4.0-generate-001", viaImages, false},
		{relay, "google/imagen-4.0-generate-001", viaImages, false},
		{openrouter, "google/gemini-3.1-flash-image", viaChat, false},
		{openrouter, "google/imagen-4.0-generate-001", viaImages, false},
		{vertex, "gemini-3.1-flash-image", viaGemini, true},
		{vertex, "google/gemini-3.1-flash-image", viaGemini, true},
		{vertex, "models/gemini-2.5-flash-image", viaGemini, true},
		{vertex, "deepseek-ai/deepseek-v4.1-flash-maas", viaChat, false},
		{vertex, "claude-opus-5-5", viaChat, false},
		{vertex, "openai/gpt-image-1", viaImages, false},
		{vertex, "imagen-4.0-generate-001", viaPredict, false},
		{vertex, "google/imagen-4.0-generate-001", viaPredict, false},
		{vertex, "models/imagen-4.0-generate-001", viaPredict, false},
		{vertex, "imagen-4.0-fast-generate-001", viaPredict, false},
		{express, "gemini-3.1-flash-image", viaGemini, true},
		{express, "imagen-4.0-generate-001", viaImages, false},
		{express, "claude-opus-5-5", viaChat, false},
		{express, "deepseek-ai/deepseek-v4.1-flash-maas", viaChat, false},
	} {
		if got := viaFor(c.p, c.model); got != c.via {
			t.Errorf("%s %s: drawn on %s, want %s", c.p.ID, c.model, got, c.via)
		}
		if got := vertexDrawsGemini(c.p, c.model); got != c.vtx {
			t.Errorf("%s %s: Vertex Gemini drawer %v, want %v", c.p.ID, c.model, got, c.vtx)
		}
	}
}

// vertexImagenDrawn is an Imagen :predict answer as Vertex AI documents it
// (predictions[].bytesBase64Encoded): #7365 never reached this endpoint,
// so the fixture is that shape, not a capture.
func vertexImagenDrawn() string {
	b64 := base64.StdEncoding.EncodeToString(pngBytes)
	return `{"predictions":[{"mimeType":"image/png","bytesBase64Encoded":"` + b64 + `"},{"mimeType":"image/png","bytesBase64Encoded":"` + b64 + `"}]}`
}

// An Imagen model on a Google Vertex AI project draws on :predict, signed
// with the Google token. Vertex serves it only at
// …/publishers/google/models/<model>:predict; the OpenAI-compatible
// /endpoints/openapi/images/generations, where viaFor sent it, is not that
// API (#7365). A google/ or models/ before the id is dropped. The body is
// Imagen's: instances[0].prompt, parameters.sampleCount and aspectRatio,
// and sampleImageSize for the Imagen 4 models that take one. An edit is
// refused here: Imagen 4 doesn't take one. Express mode, which serves
// Gemini alone, lists no Imagen and is not asked.
func TestVertexImagenDrawsOnPredict(t *testing.T) {
	v := &vertexUp{reply: func(_ int, body map[string]any) (int, string) {
		inst, _ := body["instances"].([]any)
		if len(inst) > 0 {
			if one, _ := inst[0].(map[string]any); one["prompt"] == "a blocked magpie" {
				return http.StatusOK, `{"predictions":[{"raiFilteredReason":"blocked by safety"}]}`
			}
		}
		return http.StatusOK, vertexImagenDrawn()
	}}
	s := vertexServe(t, v, `{}`)
	for _, model := range []string{"imagen-4.0-generate-001", "google/imagen-4.0-generate-001", "models/imagen-4.0-generate-001"} {
		code, a, raw := postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/`+model+`","prompt":"a magpie","size":"16:9","n":2}`)
		got := v.last(t)
		bare := strings.TrimPrefix(strings.TrimPrefix(model, "google/"), "models/")
		want := "/v1/projects/proj-1/locations/global/publishers/google/models/" + bare + ":predict"
		if got.path != want || got.query != "" {
			t.Fatalf("%s was asked at %s?%s, want %s", model, got.path, got.query, want)
		}
		if code != 200 || len(a.Data) != 2 {
			t.Fatalf("%s: %d %s", model, code, raw)
		}
		if b, _ := base64.StdEncoding.DecodeString(a.Data[0].B64); string(b) != string(pngBytes) {
			t.Fatalf("%s: image %q", model, b)
		}
		if got.auth != "Bearer ya29.vertex-test" || got.key != "" {
			t.Fatalf("%s was signed %q, key %q", model, got.auth, got.key)
		}
		inst, _ := got.body["instances"].([]any)
		one, _ := inst[0].(map[string]any)
		params, _ := got.body["parameters"].(map[string]any)
		if len(inst) != 1 || one["prompt"] != "a magpie" || params["sampleCount"] != float64(2) || params["aspectRatio"] != "16:9" || params["sampleImageSize"] != nil {
			t.Fatalf("%s was sent %v", model, got.body)
		}
		if _, ok := got.body["model"]; ok {
			t.Fatalf("%s named the model in the body: %v", model, got.body["model"])
		}
	}

	// 2048px is 2K, which imagen-4.0-generate-001 takes as sampleImageSize;
	// the fast model does not, and is asked the ratio and count alone
	code, a, raw := postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/imagen-4.0-generate-001","prompt":"a magpie","size":"2048x2048"}`)
	if code != 200 || len(a.Data) != 2 {
		t.Fatalf("2K: %d %s", code, raw)
	}
	got := v.last(t)
	params, _ := got.body["parameters"].(map[string]any)
	if params["aspectRatio"] != "1:1" || params["sampleCount"] != float64(1) || params["sampleImageSize"] != "2K" {
		t.Fatalf("2K was sent %v", got.body)
	}
	code, _, raw = postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/imagen-4.0-fast-generate-001","prompt":"a magpie","size":"2048x2048"}`)
	if code != 200 {
		t.Fatalf("fast: %d %s", code, raw)
	}
	got = v.last(t)
	if want := "/v1/projects/proj-1/locations/global/publishers/google/models/imagen-4.0-fast-generate-001:predict"; got.path != want {
		t.Fatalf("fast was asked at %s, want %s", got.path, want)
	}
	params, _ = got.body["parameters"].(map[string]any)
	if params["sampleImageSize"] != nil || params["aspectRatio"] != "1:1" {
		t.Fatalf("fast was sent %v", got.body)
	}

	// 4K is not a sampleImageSize; ultra is asked at 2K, the largest it takes
	code, _, raw = postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/imagen-4.0-ultra-generate-001","prompt":"a magpie","size":"16:9","resolution":"4K"}`)
	if code != 200 {
		t.Fatalf("ultra: %d %s", code, raw)
	}
	got = v.last(t)
	if want := "/v1/projects/proj-1/locations/global/publishers/google/models/imagen-4.0-ultra-generate-001:predict"; got.path != want {
		t.Fatalf("ultra was asked at %s, want %s", got.path, want)
	}
	params, _ = got.body["parameters"].(map[string]any)
	if params["aspectRatio"] != "16:9" || params["sampleImageSize"] != "2K" {
		t.Fatalf("ultra was sent %v", got.body)
	}

	// a ratio Imagen doesn't take is refused before anything is asked
	before := v.count()
	code, _, raw = postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/imagen-4.0-generate-001","prompt":"a magpie","size":"3:2"}`)
	if code != 400 || !strings.Contains(raw, "3:2") || v.count() != before {
		t.Fatalf("ratio: %d %s, asked %d more", code, raw, v.count()-before)
	}
	code, _, raw = postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/imagen-4.0-generate-001","prompt":"a blocked magpie","size":"1:1"}`)
	if code != 502 || !strings.Contains(raw, "blocked by safety") {
		t.Fatalf("filtered: %d %s", code, raw)
	}

	// an edit is not the images API's multipart, and not :predict either
	img := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)
	before = v.count()
	code, _, raw = postImages(t, s, "/v1/images/edits", "application/json", `{"model":"google-vertex/imagen-4.0-generate-001","prompt":"bluer","image":"`+img+`"}`)
	if code != 400 || !strings.Contains(raw, "doesn't take an image") || v.count() != before {
		t.Fatalf("edit: %d %s, asked %d more", code, raw, v.count()-before)
	}

	// a project lists an Imagen the user named; express mode does not, and
	// asking it draws nothing at Vertex
	p, err := provider.Find("google-vertex")
	if err != nil {
		t.Fatal(err)
	}
	p.Models = append(p.Models, "imagen-4.0-generate-001", "gemini-3.1-flash-image")
	if err := provider.Save(*p); err != nil {
		t.Fatal(err)
	}
	p, err = provider.Find("google-vertex")
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, m := range Drawers(*p) {
		ids = append(ids, m.ID)
	}
	if !slices.Contains(ids, "imagen-4.0-generate-001") {
		t.Fatalf("project drawers %v", ids)
	}
	if err := provider.Save(provider.Provider{ID: "google-vertex", Name: "Google Vertex AI", Preset: provider.VertexPreset, Key: "AQ.Ab8RN6-gw",
		Models: []string{"imagen-4.0-generate-001", "gemini-3.1-flash-image"}}); err != nil {
		t.Fatal(err)
	}
	p, err = provider.Find("google-vertex")
	if err != nil {
		t.Fatal(err)
	}
	ids = nil
	for _, m := range Drawers(*p) {
		ids = append(ids, m.ID)
	}
	if slices.Contains(ids, "imagen-4.0-generate-001") || !slices.Contains(ids, "gemini-3.1-flash-image") {
		t.Fatalf("express drawers %v", ids)
	}
	before = v.count()
	code, _, raw = postImages(t, s, "/v1/images/generations", "application/json", `{"model":"google-vertex/imagen-4.0-generate-001","prompt":"a magpie"}`)
	if code != 400 || v.count() != before {
		asked := ""
		if v.count() != before {
			asked = v.last(t).path
		}
		t.Fatalf("express: %d %s, asked %q", code, raw, asked)
	}
	for _, a := range v.asked {
		if strings.Contains(a.path, "/endpoints/openapi/images/") {
			t.Fatalf("a drawing went to the images API: %s", a.path)
		}
	}
}
