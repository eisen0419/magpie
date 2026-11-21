package tui

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
)

// The TUI's refetch and add wait out their own deadlines, not listBound.
// The cap is shrunk and the vendor answers past it: the status line says
// the list failed when WithManualRefresh is left off the command (#7409).

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

func TestRefetchCmdWaitsPastListBound(t *testing.T) {
	manualHome(t)
	srv := lateModels(t, 150*time.Millisecond)
	if err := provider.Save(provider.Provider{
		ID: "relay", Name: "Relay", Key: "sk",
		Responses: srv.URL + "/v1",
		ModelsURL: srv.URL + "/custom/models",
	}); err != nil {
		t.Fatal(err)
	}
	msg := refetchCmd("relay")()
	flash, ok := msg.(flashMsg)
	if !ok {
		t.Fatalf("msg %T", msg)
	}
	if !flash.ok || !strings.Contains(flash.text, "1 models") {
		t.Fatalf("refetch: %+v", flash)
	}
}

func TestAddCmdWaitsPastListBound(t *testing.T) {
	manualHome(t)
	srv := lateModels(t, 150*time.Millisecond)
	msg := addCmd(provider.Provider{
		Name: "Relay", Key: "sk",
		Responses: srv.URL + "/v1",
		ModelsURL: srv.URL + "/custom/models",
	})()
	flash, ok := msg.(flashMsg)
	if !ok {
		t.Fatalf("msg %T", msg)
	}
	if !flash.ok || !strings.Contains(flash.text, "1 models") {
		t.Fatalf("add: %+v", flash)
	}
	p, err := provider.Find("relay")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := p.Fetched(); !ok {
		t.Fatal("add fetched no list")
	}
}
