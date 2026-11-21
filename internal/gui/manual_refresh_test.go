package gui

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
)

// A list the user asked for from the editor or the add form waits out the
// handler's deadline, not listBound. listBound is shrunk and the vendor
// answers past it: the request fails when WithManualRefresh is left off
// the handler, and the model comes back when it is on (#7409).

func manualHome(t *testing.T) {
	t.Helper()
	h := t.TempDir()
	t.Setenv("HOME", h)
	t.Setenv("USERPROFILE", os.Getenv("HOME"))
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	t.Setenv("DSH_HOME", "")
	t.Cleanup(catalog.SetListBoundForTest(50 * time.Millisecond))
}

// lateModels answers one OpenAI-shaped model after delay, or sooner when
// the request is cancelled.
func lateModels(t *testing.T, delay time.Duration) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		timer := time.NewTimer(delay)
		defer timer.Stop()
		select {
		case <-r.Context().Done():
			return
		case <-timer.C:
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"gpt-4.1","name":"GPT-4.1"}]}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// The editor's Fetch models, including a custom OpenAI Responses provider
// whose list is at an explicit Models URL (the report in #7409).
func TestProviderModelsManualRefreshWaitsPastListBound(t *testing.T) {
	manualHome(t)
	srv := lateModels(t, 150*time.Millisecond)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "sk",
		Responses: srv.URL + "/v1",
		ModelsURL: srv.URL + "/custom/models",
	}); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	providerRoutes(mux, nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/api/provider/models", strings.NewReader(`{"id":"relay"}`)))
	if w.Code != 200 {
		t.Fatalf("Fetch models: %d %s", w.Code, w.Body)
	}
	var got struct {
		Count int `json:"count"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Count < 1 {
		t.Fatalf("count %d, body %s", got.Count, w.Body)
	}
	ms, _, ok := catalog.Live("relay")
	if !ok || len(ms) != 1 || ms[0].ID != "gpt-4.1" {
		t.Fatalf("saved list %v ok=%v", ms, ok)
	}
}

// The add form's Fetch models, for the same custom Responses provider
// before it is saved.
func TestProviderListManualRefreshWaitsPastListBound(t *testing.T) {
	manualHome(t)
	srv := lateModels(t, 150*time.Millisecond)
	mux := http.NewServeMux()
	providerRoutes(mux, nil)
	body := `{"typed":true,"name":"Relay","key":"sk","responses":"` + srv.URL + `/v1","modelsURL":"` + srv.URL + `/custom/models"}`
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/api/provider/list", strings.NewReader(body)))
	if w.Code != 200 {
		t.Fatalf("Fetch models on the add form: %d %s", w.Code, w.Body)
	}
	var got struct {
		Models []struct {
			ID string `json:"id"`
		} `json:"models"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Models) != 1 || got.Models[0].ID != "gpt-4.1" {
		t.Fatalf("listed %+v", got.Models)
	}
	if ps := provider.All(); len(ps) != 0 {
		t.Fatalf("the add form saved %v", ps)
	}
}
