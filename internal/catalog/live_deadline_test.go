package catalog

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// A list the user asked for waits out the caller's deadline. A list asked
// in the background, or with no deadline, still ends at listBound. The
// durations are the 8s cap and the 11s answer of #7409, shrunk so the
// test doesn't sleep either.

func setListBound(t *testing.T, d time.Duration) {
	t.Helper()
	t.Cleanup(SetListBoundForTest(d))
}

// lateList answers one model after delay, or sooner when the request is
// cancelled. delay is how long the vendor takes; the test picks it past
// listBound and inside the caller's deadline, or far past both.
func lateList(t *testing.T, delay time.Duration, body string) *httptest.Server {
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
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

const openAIList = `{"object":"list","data":[{"id":"gpt-4.1","name":"GPT-4.1"}]}`
const geminiList = `{"models":[{"name":"models/gemini-2.5-pro","displayName":"Gemini 2.5 Pro","supportedGenerationMethods":["generateContent"]}]}`

func TestManualRefreshWaitsPastListBound(t *testing.T) {
	setListBound(t, 50*time.Millisecond)
	srv := lateList(t, 150*time.Millisecond, openAIList)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	ctx = WithManualRefresh(ctx)
	ms, err := FetchURL(ctx, srv.URL+"/v1/models", "k", false, nil)
	if err != nil {
		t.Fatalf("manual refresh with a deadline past listBound: %v", err)
	}
	if len(ms) != 1 || ms[0].ID != "gpt-4.1" {
		t.Fatalf("models %+v", ms)
	}
}

func TestManualGeminiListWaitsPastListBound(t *testing.T) {
	setListBound(t, 50*time.Millisecond)
	srv := lateList(t, 150*time.Millisecond, geminiList)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	ctx = WithManualRefresh(ctx)
	ms, _, err := FetchGemini(ctx, srv.URL+"/v1beta", "k", nil)
	if err != nil {
		t.Fatalf("manual Gemini list: %v", err)
	}
	if len(ms) != 1 || ms[0].ID != "gemini-2.5-pro" {
		t.Fatalf("models %+v", ms)
	}
}

func TestManualRefreshStopsAtCallerDeadline(t *testing.T) {
	// longer than the caller, so only the caller's deadline can stop it
	setListBound(t, 2*time.Second)
	srv := lateList(t, 5*time.Second, openAIList)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	ctx = WithManualRefresh(ctx)
	start := time.Now()
	_, err := FetchURL(ctx, srv.URL+"/v1/models", "k", false, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want the caller's deadline", err)
	}
	if time.Since(start) > time.Second {
		t.Fatalf("waited %s after a 50ms deadline", time.Since(start))
	}
}

func TestManualRefreshStopsWhenCallerCancels(t *testing.T) {
	setListBound(t, 2*time.Second)
	srv := lateList(t, 5*time.Second, openAIList)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ctx = WithManualRefresh(ctx)
	go func() {
		time.Sleep(40 * time.Millisecond)
		cancel()
	}()
	start := time.Now()
	_, err := FetchURL(ctx, srv.URL+"/v1/models", "k", false, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want the caller's cancel", err)
	}
	if time.Since(start) > time.Second {
		t.Fatalf("waited %s after cancel", time.Since(start))
	}
}

func TestManualBoundListKeepsTheCallerDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	ctx = WithManualRefresh(ctx)
	got, stop := BoundList(ctx, 50*time.Millisecond)
	defer stop()
	want, _ := ctx.Deadline()
	have, ok := got.Deadline()
	if !ok || !have.Equal(want) {
		t.Fatalf("deadline %v, want the caller's %v", have, want)
	}
	stop()
	if ctx.Err() != nil {
		t.Fatal("BoundList's cancel ended the caller")
	}
}

func TestManualBoundListWithoutDeadlineUsesFallback(t *testing.T) {
	ctx := WithManualRefresh(context.Background())
	got, stop := BoundList(ctx, 50*time.Millisecond)
	defer stop()
	start := time.Now()
	<-got.Done()
	if waited := time.Since(start); waited > time.Second {
		t.Fatalf("waited %s, want the fallback", waited)
	}
}

func TestBackgroundListBoundUsesFallback(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	got, stop := BoundList(ctx, 50*time.Millisecond)
	defer stop()
	start := time.Now()
	<-got.Done()
	if waited := time.Since(start); waited > time.Second {
		t.Fatalf("waited %s, want the fallback", waited)
	}
	if ctx.Err() != nil {
		t.Fatal("fallback ended the caller")
	}
}

func TestBackgroundListBoundUsesListBoundWhenFallbackIsZero(t *testing.T) {
	setListBound(t, 50*time.Millisecond)
	got, stop := BoundList(context.Background(), 0)
	defer stop()
	start := time.Now()
	<-got.Done()
	if waited := time.Since(start); waited > time.Second {
		t.Fatalf("waited %s, want listBound", waited)
	}
}

func TestManualProbeBoundStaysWithinListBound(t *testing.T) {
	setListBound(t, 50*time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	ctx = WithManualRefresh(ctx)
	probe, stop := ProbeBound(ctx)
	defer stop()
	start := time.Now()
	<-probe.Done()
	if waited := time.Since(start); waited > time.Second {
		t.Fatalf("probe waited %s, want listBound", waited)
	}
	if ctx.Err() != nil {
		t.Fatalf("probe expired the caller: %v", ctx.Err())
	}
}

func TestManualRefreshWithoutDeadlineStaysWithinListBound(t *testing.T) {
	// WithManualRefresh on a context that has no deadline of its own
	// still ends at listBound: a manual refresh that forgot one cannot hang.
	setListBound(t, 50*time.Millisecond)
	srv := lateList(t, 3*time.Second, openAIList)
	ctx := WithManualRefresh(context.Background())
	if _, ok := ctx.Deadline(); ok {
		t.Fatal("the test context has a deadline")
	}
	start := time.Now()
	_, err := FetchURL(ctx, srv.URL+"/v1/models", "k", false, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("manual refresh with no deadline: got %v, want listBound", err)
	}
	if waited := time.Since(start); waited > time.Second {
		t.Fatalf("manual refresh with no deadline waited %s, past listBound", waited)
	}
}

func TestBackgroundListStaysWithinListBound(t *testing.T) {
	setListBound(t, 50*time.Millisecond)
	srv := lateList(t, 3*time.Second, openAIList)
	start := time.Now()
	_, err := FetchURL(context.Background(), srv.URL+"/v1/models", "k", false, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("background refresh with no deadline: got %v, want listBound", err)
	}
	if waited := time.Since(start); waited > time.Second {
		t.Fatalf("background refresh waited %s, past listBound", waited)
	}
}

func TestBackgroundListWithALongDeadlineStaysWithinListBound(t *testing.T) {
	// KeepListsFresh gives the whole refetch a minute. That deadline must
	// not lift the per-request cap.
	setListBound(t, 50*time.Millisecond)
	srv := lateList(t, 3*time.Second, openAIList)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	start := time.Now()
	_, err := FetchURL(ctx, srv.URL+"/v1/models", "k", false, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("background refresh under a long deadline: got %v, want listBound", err)
	}
	if waited := time.Since(start); waited > time.Second {
		t.Fatalf("background refresh waited %s, past listBound", waited)
	}
}
