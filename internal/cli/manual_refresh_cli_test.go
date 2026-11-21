package cli

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
)

// magpie provider refresh and magpie provider add wait out the command's
// deadline, not listBound. The cap is shrunk and the vendor answers past
// it, so either command fails to keep the list when WithManualRefresh is
// left off (#7409).

func manualHome(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("USERPROFILE", os.Getenv("HOME"))
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	t.Setenv("DSH_HOME", "")
	t.Cleanup(catalog.SetListBoundForTest(50 * time.Millisecond))
}

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

func TestProviderRefreshWaitsPastListBound(t *testing.T) {
	manualHome(t)
	srv := lateModels(t, 150*time.Millisecond)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "sk",
		Responses: srv.URL + "/v1",
		ModelsURL: srv.URL + "/custom/models",
	}); err != nil {
		t.Fatal(err)
	}
	if err := providerCmd([]string{"provider", "refresh", "relay"}); err != nil {
		t.Fatal(err)
	}
	if _, ok := mustFind(t, "relay").Fetched(); !ok {
		t.Fatal("refresh fetched no list")
	}
	ms, _, ok := catalog.Live("relay")
	if !ok || len(ms) != 1 || ms[0].ID != "gpt-4.1" {
		t.Fatalf("saved list %v ok=%v", ms, ok)
	}
}

func TestProviderAddCustomResponsesWaitsPastListBound(t *testing.T) {
	manualHome(t)
	srv := lateModels(t, 150*time.Millisecond)
	err := providerCmd([]string{"provider", "add", "Relay",
		"key=sk",
		"responses=" + srv.URL + "/v1",
		"models.url=" + srv.URL + "/custom/models",
	})
	if err != nil {
		t.Fatal(err)
	}
	p := mustFind(t, "relay")
	if p.Responses == "" || p.ModelsURL == "" {
		t.Fatalf("saved %+v", p)
	}
	// add returns nil even when the fetch failed, and says so on stdout:
	// the list kept is what shows the fetch waited past listBound
	if _, ok := p.Fetched(); !ok {
		t.Fatal("add fetched no list")
	}
	ms, _, ok := catalog.Live("relay")
	if !ok || len(ms) != 1 || ms[0].ID != "gpt-4.1" {
		t.Fatalf("saved list %v ok=%v", ms, ok)
	}
}
