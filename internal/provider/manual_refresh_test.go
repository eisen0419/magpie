package provider

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/catalog"
)

// kiloModels, clineFeed and fetchDecide each keep a cap of their own
// (10s, 8s, 20s). A manual refresh that already has a deadline is allowed
// past that cap; a background caller is not. The request's deadline is
// what the vendor is waited on, so the test reads it instead of sleeping
// the real cap.

func testHome(t *testing.T) {
	t.Helper()
	h := t.TempDir()
	t.Setenv("HOME", h)
	t.Setenv("USERPROFILE", h)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(h, ".config"))
	t.Setenv("XDG_CACHE_HOME", filepath.Join(h, ".cache"))
	t.Setenv("DSH_HOME", "")
}

// deadlineLeft is how long the first list request was allowed to run.
// The server's request context does not carry the client's deadline, so
// the outgoing client's does: the Transport notes it. ask uses srv and
// returns the error the fetch returned. body is what the vendor answers,
// at once: the deadline is the assertion, not a wait.
func deadlineLeft(t *testing.T, body string, parent time.Duration, manual bool, ask func(ctx context.Context, srv *httptest.Server) error) time.Duration {
	t.Helper()
	var mu sync.Mutex
	var left time.Duration
	old := http.DefaultClient.Transport
	base := old
	if base == nil {
		base = http.DefaultTransport
	}
	http.DefaultClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if dl, ok := r.Context().Deadline(); ok {
			mu.Lock()
			if left == 0 {
				left = time.Until(dl)
			}
			mu.Unlock()
		}
		return base.RoundTrip(r)
	})
	t.Cleanup(func() { http.DefaultClient.Transport = old })
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	ctx, cancel := context.WithTimeout(context.Background(), parent)
	defer cancel()
	if manual {
		ctx = catalog.WithManualRefresh(ctx)
	}
	if err := ask(ctx, srv); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if left == 0 {
		t.Fatal("the list request had no deadline")
	}
	return left
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func assertListDeadline(t *testing.T, left, cap, caller time.Duration, manual bool) {
	t.Helper()
	if manual {
		// past the function's own cap, and still within the caller's
		if left < cap+5*time.Second || left > caller {
			t.Fatalf("manual refresh left %s, want the caller's ~%s past the %s cap", left, caller, cap)
		}
		return
	}
	const slop = 2 * time.Second
	if left < cap-slop || left > cap+slop {
		t.Fatalf("background list left %s, want about %s", left, cap)
	}
}

const manualKiloList = `{"data":[{"id":"kilo-auto/free","name":"Auto Free","isFree":true,"supported_parameters":["tools"]}]}`
const manualClineList = `{"free":[{"id":"cline-free/deepseek-v4.1-flash","name":"Flash"}],"clinePass":[]}`
const manualDecideList = `{"object":"list","data":[{"id":"jev-latest"}]}`
const manualCloudflareAccounts = `{"success":true,"result":[{"id":"acc9"}]}`

func TestKiloModelsHonorsManualRefresh(t *testing.T) {
	const caller = 30 * time.Second
	left := deadlineLeft(t, manualKiloList, caller, true, func(ctx context.Context, srv *httptest.Server) error {
		p := Provider{ID: "kilo-manual", Chat: srv.URL + "/api/openrouter", Key: "k"}
		_, _, err := p.kiloModels(ctx)
		return err
	})
	assertListDeadline(t, left, 10*time.Second, caller, true)
}

func TestKiloModelsKeepsItsListCap(t *testing.T) {
	const caller = 2 * time.Minute
	left := deadlineLeft(t, manualKiloList, caller, false, func(ctx context.Context, srv *httptest.Server) error {
		p := Provider{ID: "kilo-bg", Chat: srv.URL + "/api/openrouter", Key: "k"}
		_, _, err := p.kiloModels(ctx)
		return err
	})
	assertListDeadline(t, left, 10*time.Second, caller, false)
}

func TestClineFeedHonorsManualRefresh(t *testing.T) {
	const caller = 30 * time.Second
	left := deadlineLeft(t, manualClineList, caller, true, func(ctx context.Context, srv *httptest.Server) error {
		p := Provider{ID: "cline-manual", Chat: srv.URL + "/api/v1"}
		_, _, err := p.clineFeed(ctx)
		return err
	})
	assertListDeadline(t, left, 8*time.Second, caller, true)
}

func TestClineFeedKeepsItsListCap(t *testing.T) {
	const caller = 2 * time.Minute
	left := deadlineLeft(t, manualClineList, caller, false, func(ctx context.Context, srv *httptest.Server) error {
		p := Provider{ID: "cline-bg", Chat: srv.URL + "/api/v1"}
		_, _, err := p.clineFeed(ctx)
		return err
	})
	assertListDeadline(t, left, 8*time.Second, caller, false)
}

func TestFetchDecideHonorsManualRefresh(t *testing.T) {
	testHome(t)
	const caller = 45 * time.Second
	left := deadlineLeft(t, manualDecideList, caller, true, func(ctx context.Context, srv *httptest.Server) error {
		p := Provider{ID: "decide-manual", Name: "Decide", Key: "k", Decide: srv.URL + "/v1"}
		_, err := p.fetchDecide(ctx)
		return err
	})
	assertListDeadline(t, left, 20*time.Second, caller, true)
}

func TestFetchDecideKeepsItsListCap(t *testing.T) {
	testHome(t)
	const caller = 2 * time.Minute
	left := deadlineLeft(t, manualDecideList, caller, false, func(ctx context.Context, srv *httptest.Server) error {
		p := Provider{ID: "decide-bg", Name: "Decide", Key: "k", Decide: srv.URL + "/v1"}
		_, err := p.fetchDecide(ctx)
		return err
	})
	assertListDeadline(t, left, 20*time.Second, caller, false)
}

func TestCloudflareAccountHonorsManualRefresh(t *testing.T) {
	testHome(t)
	const caller = 45 * time.Second
	left := deadlineLeft(t, manualCloudflareAccounts, caller, true, func(ctx context.Context, srv *httptest.Server) error {
		p := Provider{ID: "cf-manual", Name: "Cloudflare", Key: "cf-manual-key", Decide: srv.URL + "/client/v4"}
		_, err := p.cloudflareAccount(ctx)
		return err
	})
	assertListDeadline(t, left, 20*time.Second, caller, true)
}

func TestCloudflareAccountKeepsItsListCap(t *testing.T) {
	testHome(t)
	const caller = 2 * time.Minute
	left := deadlineLeft(t, manualCloudflareAccounts, caller, false, func(ctx context.Context, srv *httptest.Server) error {
		p := Provider{ID: "cf-bg", Name: "Cloudflare", Key: "cf-bg-key", Decide: srv.URL + "/client/v4"}
		_, err := p.cloudflareAccount(ctx)
		return err
	})
	assertListDeadline(t, left, 20*time.Second, caller, false)
}

// A custom OpenAI Responses provider with no Models URL lists /v1/models,
// then asks it again with client_version (codexEntries). The second ask
// only adds Codex entries. A stall there must not use up a manual
// refresh's deadline: the plain list, which answered past listBound,
// is the list, and the caller's context is still alive for the steps
// after the fetch (noteRemoteHub, fetchDecide).
func TestCodexEntriesProbeLeavesTheManualDeadline(t *testing.T) {
	testHome(t)
	t.Setenv("PATH", t.TempDir())
	t.Cleanup(catalog.SetListBoundForTest(50 * time.Millisecond))
	var probe atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		if r.URL.Query().Get("client_version") != "" {
			probe.Store(true)
			<-r.Context().Done()
			return
		}
		timer := time.NewTimer(150 * time.Millisecond)
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
	if err := Save(Provider{ID: "relay", Name: "Relay", Key: "sk", Responses: srv.URL + "/v1"}); err != nil {
		t.Fatal(err)
	}
	p, err := Find("relay")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	ctx = catalog.WithManualRefresh(ctx)
	start := time.Now()
	ms, err := p.Fetch(ctx)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("primary list: %v", err)
	}
	if len(ms) != 1 || ms[0].ID != "gpt-4.1" {
		t.Fatalf("models %+v", ms)
	}
	if !probe.Load() {
		t.Fatal("the client_version probe was not asked")
	}
	if ctx.Err() != nil {
		t.Fatalf("parent expired before the steps after the fetch: %v (took %s)", ctx.Err(), elapsed)
	}
	if elapsed > 1500*time.Millisecond {
		t.Fatalf("fetch took %s; the optional probe used the caller's deadline", elapsed)
	}
}
