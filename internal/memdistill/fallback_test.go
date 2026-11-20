package memdistill

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/sessions"
	"github.com/yetone/magpie/internal/usage"
)

// A session listed from an agent's own files, whose agent has no transcript
// reader, is read from the gateway's record when one of its gateway keys
// holds one (zcode's request id is the bare id, not the database id). The
// recording's newest turn is the last turn for the quiet wait and the mark,
// so a conversation that went on through the gateway is read again. Hermes
// and Alma are listed read-only and have no reader, so they are read the
// same way. A read-only session that has a reader (Cursor's editor; here
// the claude file) stays on that file. Nothing recorded is left out and
// not marked.
func TestNativeSessionsWithoutReaderUseGateway(t *testing.T) {
	sb := newSandbox(t)
	read = transcriptOf
	sb.at = time.Now()
	on(t, "")
	if err := sessions.SetGatewayRecording(true, false); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })

	keys := sessions.GatewayKeys(sessions.Session{Agent: "zcode", ID: "sess_zcode1"})
	if !slices.Contains(keys, "zcode|zcode1") || !slices.Contains(keys, "zcode|sess_zcode1") {
		t.Fatalf("gateway keys %v", keys)
	}
	fileLast := sb.at.Add(-5 * time.Hour)
	recordPair(t, "zcode", "zcode1", sb.at.Add(-150*time.Minute), "zcode: we release with make ship", "zcode: pnpm only")
	recordPair(t, "hermes", "h1", sb.at.Add(-140*time.Minute), "hermes: we release with make ship", "hermes: pnpm only")
	recordPair(t, "dsh", "d1", sb.at.Add(-130*time.Minute), "dsh: we release with make ship", "dsh: pnpm only")
	recordPair(t, "droid", "r1", sb.at.Add(-120*time.Minute), "droid: we release with make ship", "droid: pnpm only")
	recordPair(t, "alma", "a1", sb.at.Add(-110*time.Minute), "alma: we release with make ship", "alma: pnpm only")
	recordPair(t, "grok", "g1", sb.at.Add(-10*time.Minute), "grok: we release with make ship", "grok: still going")
	recordPair(t, "claude", "c1", sb.at.Add(-100*time.Minute), "gw: we release with make ship", "gw: pnpm only")

	z := sb.session("sess_zcode1", 0)
	z.Agent, z.Transcript, z.Last = "zcode", false, fileLast
	h := sb.session("h1", 0)
	h.Agent, h.Transcript, h.ReadOnly, h.Last = "hermes", false, true, fileLast
	d := sb.session("d1", 0)
	d.Agent, d.Transcript, d.Last = "dsh", false, fileLast
	r := sb.session("r1", 0)
	r.Agent, r.Transcript, r.Last = "droid", false, fileLast
	a := sb.session("a1", 0)
	a.Agent, a.Transcript, a.ReadOnly, a.Last = "alma", false, true, fileLast
	none := sb.session("none", 0)
	none.Agent, none.Transcript, none.Last = "cline", false, fileLast
	g := sb.session("g1", 0)
	g.Agent, g.Transcript, g.Last = "grok", false, fileLast
	c := sb.session("c1", 0)
	c.Agent, c.Path, c.ReadOnly, c.Last = "claude", writeClaude(t, sb.at.Add(-100*time.Minute), "claude: we release with make ship", "claude: pnpm only"), true, sb.at.Add(-100*time.Minute)

	// already read at the file's last turn; the gateway went on after that
	st, _ := Load()
	st.Done = map[string]time.Time{"zcode/sess_zcode1": fileLast}
	if err := save(st); err != nil {
		t.Fatal(err)
	}

	ss, _ := due(State{Since: sb.at.Add(-Back), Done: map[string]time.Time{"zcode/sess_zcode1": fileLast}}, sb.at)
	got := ids(ss)
	want := []string{"c1", "a1", "r1", "d1", "h1", "sess_zcode1"}
	if !slices.Equal(got, want) {
		t.Fatalf("due %v, want %v", got, want)
	}
	for _, s := range ss {
		if s.ID == "sess_zcode1" && !s.Last.Equal(sb.at.Add(-150*time.Minute).Add(time.Minute)) {
			t.Fatalf("zcode last %v, want the gateway's newer turn", s.Last)
		}
		if (s.ID == "h1" || s.ID == "a1") && !s.ReadOnly {
			t.Fatalf("read-only session lost its mark: %+v", s)
		}
		if s.ID == "c1" && !s.ReadOnly {
			t.Fatalf("reader session lost read-only: %+v", s)
		}
		if s.ID == "none" || s.ID == "g1" {
			t.Fatalf("not due: %+v", s)
		}
	}

	res, err := Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != len(want) {
		t.Fatalf("%+v %v %q", res, err, sb.asked)
	}
	wantAsked := []string{
		"auto/m User: claude: we release with make ship",
		"auto/m User: alma: we release with make ship",
		"auto/m User: droid: we release with make ship",
		"auto/m User: dsh: we release with make ship",
		"auto/m User: hermes: we release with make ship",
		"auto/m User: zcode: we release with make ship",
	}
	if !slices.Equal(sb.asked, wantAsked) {
		t.Fatalf("asked %q", sb.asked)
	}
	for _, line := range sb.asked {
		if strings.Contains(line, "gw:") || strings.Contains(line, "grok:") {
			t.Fatalf("read the wrong conversation: %q", sb.asked)
		}
	}
	st, _ = Load()
	if st.Error != "" {
		t.Fatalf("error %q", st.Error)
	}
	for _, id := range []string{"cline/none", "grok/g1"} {
		if _, ok := st.Done[id]; ok {
			t.Fatalf("%s marked: %+v", id, st.Done)
		}
	}
	for _, id := range []struct {
		key string
		at  time.Time
	}{
		{"hermes/h1", sb.at.Add(-140 * time.Minute)},
		{"alma/a1", sb.at.Add(-110 * time.Minute)},
	} {
		if got, ok := st.Done[id.key]; !ok || !got.Equal(id.at.Add(time.Minute)) {
			t.Fatalf("%s mark %v", id.key, st.Done[id.key])
		}
	}
	if z, ok := st.Done["zcode/sess_zcode1"]; !ok || !z.Equal(sb.at.Add(-150*time.Minute).Add(time.Minute)) {
		t.Fatalf("zcode mark %v", st.Done["zcode/sess_zcode1"])
	}
	if res, err := Run(context.Background(), "auto/m", sb.complete); err != nil || res.Read != 0 || len(sb.asked) != len(wantAsked) {
		t.Fatalf("read again: %+v %v %q", res, err, sb.asked)
	}
}

// A native session the Sessions page lists is still one row when the
// gateway also counted it, and with no reader it is read from the
// recording, not marked readable for the file reader.
func TestListedNativeIsNotAlsoAGatewayRow(t *testing.T) {
	sb := newSandbox(t)
	list, read = listed, transcriptOf
	sb.at = time.Now()
	on(t, "")
	if err := sessions.SetGatewayRecording(true, false); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })
	restoreUsage(t)

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	at := sb.at.Add(-3 * time.Hour).UTC().Truncate(time.Second)
	id := "droidgw1"
	dir := filepath.Join(home, ".factory", "sessions")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		os.RemoveAll(filepath.Join(home, ".factory"))
		sessions.Reset()
	})
	body := strings.Join([]string{
		fmt.Sprintf(`{"type":"session_start","id":"%s","title":"Ship","cwd":"/work/droid"}`, id),
		fmt.Sprintf(`{"type":"message","id":"m1","timestamp":"%s","message":{"role":"user","content":[{"type":"text","text":"file: we release with make ship"}]}}`, at.Format(time.RFC3339Nano)),
		fmt.Sprintf(`{"type":"message","id":"m2","timestamp":"%s","message":{"role":"user","content":[{"type":"text","text":"file: pnpm only"}]}}`, at.Add(time.Minute).Format(time.RFC3339Nano)),
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(dir, id+".jsonl"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	sessions.Reset()
	recordPair(t, "droid", id, sb.at.Add(-90*time.Minute), "droid: we release with make ship", "droid: pnpm only")
	recordPair(t, "droid", "onlygw", sb.at.Add(-2*time.Hour), "only: we release with make ship", "only: pnpm only")
	for i, sid := range []string{id, "onlygw"} {
		usage.Append(usage.Record{Time: sb.at.Add(-time.Duration(3-i) * time.Hour), Agent: "droid", Session: sid, Provider: "p", Model: "m", Input: 10, Output: 2, Status: 200})
	}

	var native, gateway int
	for _, s := range listed() {
		if s.Agent != "droid" {
			continue
		}
		switch s.ID {
		case id:
			native++
			if s.Gateway || s.Transcript || s.ReadOnly {
				t.Fatalf("listed native %+v", s)
			}
		case "onlygw":
			gateway++
			if !s.Gateway {
				t.Fatalf("gateway-only %+v", s)
			}
		default:
			t.Fatalf("extra droid %+v", s)
		}
	}
	if native != 1 || gateway != 1 {
		t.Fatalf("native %d gateway %d", native, gateway)
	}
	res, err := Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 2 {
		t.Fatalf("%+v %v %q", res, err, sb.asked)
	}
	if !slices.Equal(sb.asked, []string{
		"auto/m User: droid: we release with make ship",
		"auto/m User: only: we release with make ship",
	}) {
		t.Fatalf("asked %q", sb.asked)
	}
}

// A turn saved after the row was found due is not part of that read, and
// does not count as quiet yet. The mark stays the turn that was quiet, so
// the new one is read once it has been quiet itself.
func TestTurnSavedAfterDueWaitsUntilQuiet(t *testing.T) {
	sb := newSandbox(t)
	sb.at = time.Now()
	on(t, "")
	if err := sessions.SetGatewayRecording(true, false); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })
	var users []string
	sb.reply = func(user string) (string, error) {
		users = append(users, user)
		return "[]", nil
	}
	gw := sb.at.Add(-2 * time.Hour)
	recordPair(t, "droid", "d1", gw, "droid: we release with make ship", "droid: pnpm only")
	d := sb.session("d1", 0)
	d.Agent, d.Transcript, d.Last = "droid", false, gw.Add(-time.Hour)
	saved := false
	read = func(s sessions.Session) (sessions.Transcript, error) {
		if s.ID == "d1" && !saved {
			saved = true
			recordUser(t, "droid", "d1", sb.at, "still going: not quiet yet")
		}
		return transcriptOf(s)
	}
	res, err := Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 1 || len(users) != 1 || !strings.Contains(users[0], "droid: we release with make ship") || strings.Contains(users[0], "still going") {
		t.Fatalf("%+v %v %q", res, err, users)
	}
	st, _ := Load()
	if got := st.Done["droid/d1"]; !got.Equal(gw.Add(time.Minute)) {
		t.Fatalf("mark %v, want the turn that was quiet", got)
	}
	ss, _ := due(State{Since: sb.at.Add(-Back), Done: st.Done}, sb.at)
	if slices.Contains(ids(ss), "d1") {
		t.Fatalf("the turn saved during the read is already due: %+v", ss)
	}
	sb.at = sb.at.Add(Quiet + time.Minute)
	res, err = Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 1 || len(users) != 2 || !strings.Contains(users[1], "still going") {
		t.Fatalf("after quiet: %+v %v %q", res, err, users)
	}
}

// The file's time is not the mark when the transcript is only the gateway
// recording. A later gateway turn that is still at or before the file's
// time is read once it is quiet; marking the file's time would hide it.
func TestDoneFollowsGatewayTurnNotTheFile(t *testing.T) {
	sb := newSandbox(t)
	read = transcriptOf
	sb.at = time.Now()
	on(t, "")
	if err := sessions.SetGatewayRecording(true, false); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })
	var users []string
	sb.reply = func(user string) (string, error) {
		users = append(users, user)
		return "[]", nil
	}
	fileLast := sb.at.Add(-2 * time.Hour)
	gw := sb.at.Add(-3 * time.Hour)
	recordPair(t, "zcode", "zcode1", gw, "zcode: we release with make ship", "zcode: pnpm only")
	z := sb.session("sess_zcode1", 0)
	z.Agent, z.Transcript, z.Last = "zcode", false, fileLast
	res, err := Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 1 || len(users) != 1 || !strings.Contains(users[0], "zcode: we release with make ship") {
		t.Fatalf("%+v %v %q", res, err, users)
	}
	st, _ := Load()
	if got, want := st.Done["zcode/sess_zcode1"], gw.Add(time.Minute); !got.Equal(want) {
		t.Fatalf("mark %v, want the gateway turn %v, not the file %v", got, want, fileLast)
	}
	mid := fileLast.Add(-30 * time.Minute)
	recordUser(t, "zcode", "zcode1", mid, "caught up: said before the file's time")
	res, err = Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 1 || len(users) != 2 || !strings.Contains(users[1], "caught up") {
		t.Fatalf("turn behind the file was skipped: %+v %v %q", res, err, users)
	}
}

// A ZCode recording first read as the gateway's bare id is the same
// recording when the files arrive as sess_…. It is not read again.
func TestZCodeBareIDIsNotDistilledAgainAsSess(t *testing.T) {
	sb := newSandbox(t)
	read = transcriptOf
	sb.at = time.Now()
	on(t, "")
	if err := sessions.SetGatewayRecording(true, false); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })
	at := sb.at.Add(-2 * time.Hour)
	recordPair(t, "zcode", "abc", at, "zcode: we release with make ship", "zcode: pnpm only")
	g := sb.session("abc", 0)
	g.Agent, g.Gateway, g.Transcript, g.ReadOnly, g.Last = "zcode", true, true, true, at.Add(time.Minute)
	res, err := Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 1 || len(sb.asked) != 1 {
		t.Fatalf("first: %+v %v %q", res, err, sb.asked)
	}
	sb.list = nil
	z := sb.session("sess_abc", 0)
	z.Agent, z.Transcript, z.Last = "zcode", false, at.Add(-time.Hour)
	res, err = Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 0 || len(sb.asked) != 1 {
		t.Fatalf("same recording again: %+v %v %q", res, err, sb.asked)
	}
}

// A request that started earlier than the newest turn (so its turn.Time is
// older) and finished saving between due() and transcriptOf() must not be
// included in the already-snapshotted read, and must still trigger the next
// pass even though the maximum turn timestamp did not advance.
func TestEarlierTurnSavedBeforeReadWaitsForNextPass(t *testing.T) {
	sb := newSandbox(t)
	sb.at = time.Now()
	on(t, "")
	if err := sessions.SetGatewayRecording(true, false); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })
	var users []string
	sb.reply = func(user string) (string, error) {
		users = append(users, user)
		return "[]", nil
	}
	gw := sb.at.Add(-2 * time.Hour)
	recordPair(t, "droid", "d1", gw, "droid: we release with make ship", "droid: pnpm only")
	d := sb.session("d1", 0)
	d.Agent, d.Transcript, d.Last = "droid", false, gw.Add(-time.Hour)
	saved := false
	read = func(s sessions.Session) (sessions.Transcript, error) {
		if s.ID == "d1" && !saved {
			saved = true
			recordUser(t, "droid", "d1", gw.Add(-5*time.Minute), "earlier start saved before read")
		}
		return transcriptOf(s)
	}
	res, err := Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 1 || len(users) != 1 || !strings.Contains(users[0], "droid: we release with make ship") || strings.Contains(users[0], "earlier start saved before read") {
		t.Fatalf("run 1: %+v %v %q", res, err, users)
	}
	res, err = Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 1 || len(users) != 2 || !strings.Contains(users[1], "earlier start saved before read") {
		t.Fatalf("run 2: %+v %v %q", res, err, users)
	}
	res, err = Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 0 || len(users) != 2 {
		t.Fatalf("run 3 should be a no-op: %+v %v %q", res, err, users)
	}
}

// A request that started earlier than the newest turn and finished saving
// after a run completed must not be skipped by a timestamp-only Done mark.
func TestEarlierTurnSavedAfterReadStillTriggersNextPass(t *testing.T) {
	sb := newSandbox(t)
	read = transcriptOf
	sb.at = time.Now()
	on(t, "")
	if err := sessions.SetGatewayRecording(true, false); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })
	var users []string
	sb.reply = func(user string) (string, error) {
		users = append(users, user)
		return "[]", nil
	}
	gw := sb.at.Add(-2 * time.Hour)
	recordPair(t, "droid", "d1", gw, "droid: we release with make ship", "droid: pnpm only")
	d := sb.session("d1", 0)
	d.Agent, d.Transcript, d.Last = "droid", false, gw.Add(-time.Hour)
	res, err := Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 1 || len(users) != 1 {
		t.Fatalf("run 1: %+v %v %q", res, err, users)
	}
	recordUser(t, "droid", "d1", gw.Add(-5*time.Minute), "earlier start saved after run")
	res, err = Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 1 || len(users) != 2 || !strings.Contains(users[1], "earlier start saved after run") {
		t.Fatalf("run 2: %+v %v %q", res, err, users)
	}
	res, err = Run(context.Background(), "auto/m", sb.complete)
	if err != nil || res.Read != 0 || len(users) != 2 {
		t.Fatalf("run 3 should be a no-op: %+v %v %q", res, err, users)
	}
}

// A corrupt or unreadable gateway recording is unknown, never done: once
// repaired, even with the same turn timestamps, the next run distills it.
func TestUnreadableFallbackRecordingIsNotMarkedDone(t *testing.T) {
	t.Run("corrupt before due", func(t *testing.T) {
		sb := newSandbox(t)
		read = transcriptOf
		sb.at = time.Now()
		on(t, "")
		if err := sessions.SetGatewayRecording(true, false); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })
		gw := sb.at.Add(-2 * time.Hour)
		recordPair(t, "droid", "d1", gw, "droid: we release with make ship", "droid: pnpm only")
		snap, ok := sessions.GatewayHearer()("droid", "d1")
		if !ok || len(snap.Files) != 1 {
			t.Fatalf("expected 1 file, got %+v ok=%v", snap, ok)
		}
		path := snap.Files[0].Path
		orig, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(append([]byte{}, orig...), []byte("{broken\n")...), 0o600); err != nil {
			t.Fatal(err)
		}
		d := sb.session("d1", 0)
		d.Agent, d.Transcript, d.Last = "droid", false, gw.Add(-time.Hour)
		res, err := Run(context.Background(), "auto/m", sb.complete)
		if err != nil || res.Read != 0 {
			t.Fatalf("corrupt run: %+v %v", res, err)
		}
		if st, _ := Load(); !st.Done["droid/d1"].IsZero() {
			t.Fatalf("corrupt recording was marked Done: %+v", st.Done)
		}
		if err := os.WriteFile(path, orig, 0o600); err != nil {
			t.Fatal(err)
		}
		res, err = Run(context.Background(), "auto/m", sb.complete)
		if err != nil || res.Read != 1 {
			t.Fatalf("repaired run: %+v %v", res, err)
		}
	})

	t.Run("corrupt between due and read", func(t *testing.T) {
		sb := newSandbox(t)
		sb.at = time.Now()
		on(t, "")
		if err := sessions.SetGatewayRecording(true, false); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })
		gw := sb.at.Add(-2 * time.Hour)
		recordPair(t, "droid", "d1", gw, "droid: we release with make ship", "droid: pnpm only")
		snap, ok := sessions.GatewayHearer()("droid", "d1")
		if !ok || len(snap.Files) != 1 {
			t.Fatalf("expected 1 file, got %+v ok=%v", snap, ok)
		}
		path := snap.Files[0].Path
		orig, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		d := sb.session("d1", 0)
		d.Agent, d.Transcript, d.Last = "droid", false, gw.Add(-time.Hour)
		corrupted := false
		read = func(s sessions.Session) (sessions.Transcript, error) {
			if s.ID == "d1" && !corrupted {
				corrupted = true
				if err := os.WriteFile(path, []byte("{broken\n"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			return transcriptOf(s)
		}
		res, err := Run(context.Background(), "auto/m", sb.complete)
		if err != nil || res.Read != 0 {
			t.Fatalf("corrupt read run: %+v %v", res, err)
		}
		if st, _ := Load(); !st.Done["droid/d1"].IsZero() {
			t.Fatalf("read error marked recording Done: %+v", st.Done)
		}
		if err := os.WriteFile(path, orig, 0o600); err != nil {
			t.Fatal(err)
		}
		res, err = Run(context.Background(), "auto/m", sb.complete)
		if err != nil || res.Read != 1 {
			t.Fatalf("repaired run: %+v %v", res, err)
		}
	})
}

// A subagent's gateway recording belongs to that subagent, not to the main
// session (sessions.GatewaySame excludes Subs). A newer subagent turn must
// never stand in for the main conversation or mark it Done, and a session
// whose only gateway recording is a subagent's is not due.
func TestSubagentRecordingDoesNotMaskMainSession(t *testing.T) {
	t.Run("newer subagent does not hide main recording", func(t *testing.T) {
		sb := newSandbox(t)
		read = transcriptOf
		sb.at = time.Now()
		on(t, "")
		if err := sessions.SetGatewayRecording(true, false); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })

		mainAt := sb.at.Add(-3 * time.Hour)
		subAt := sb.at.Add(-2 * time.Hour)
		recordPair(t, "droid", "d1", mainAt, "main: keep the gateway config minimal", "main: follow up")
		recordPair(t, "droid", "sub1", subAt, "sub: grep the repo", "sub: second turn")
		d := sb.session("d1", 0)
		d.Agent, d.Transcript, d.Last = "droid", false, sb.at.Add(-4*time.Hour)
		d.Subs = []string{"sub1"}

		res, err := Run(context.Background(), "auto/m", sb.complete)
		if err != nil || res.Read != 1 {
			t.Fatalf("run 1: %+v %v", res, err)
		}
		if len(sb.asked) != 1 || !strings.Contains(sb.asked[0], "main: keep the gateway config minimal") || strings.Contains(sb.asked[0], "sub: grep the repo") {
			t.Fatalf("sent wrong recording: %q", sb.asked)
		}
		res, err = Run(context.Background(), "auto/m", sb.complete)
		if err != nil || res.Read != 0 {
			t.Fatalf("run 2 re-read: %+v %v", res, err)
		}
	})
	t.Run("subagent-only recording does not make main session due", func(t *testing.T) {
		sb := newSandbox(t)
		read = transcriptOf
		sb.at = time.Now()
		on(t, "")
		if err := sessions.SetGatewayRecording(true, false); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { sessions.SetGatewayRecording(false, true) })

		recordPair(t, "droid", "sub2", sb.at.Add(-2*time.Hour), "sub only: one", "sub only: two")
		d := sb.session("d2", 0)
		d.Agent, d.Transcript, d.Last = "droid", false, sb.at.Add(-3*time.Hour)
		d.Subs = []string{"sub2"}

		got, _ := due(State{Since: sb.at.Add(-24 * time.Hour)}, sb.at)
		if len(got) != 0 {
			t.Fatalf("subagent recording made main session due: %v", ids(got))
		}
	})
}

func ids(ss []sessions.Session) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = s.ID
	}
	return out
}

func recordUser(t *testing.T, agent, id string, at time.Time, text string) {
	t.Helper()
	turn := sessions.GatewayTurn{
		Agent: agent, Session: id, Time: at, Status: 200,
		Input:  []sessions.Part{{Role: "user", Kind: "text", Text: text}},
		Output: []sessions.Part{{Role: "assistant", Kind: "text", Text: "Done."}},
	}
	if err := sessions.SaveGatewayTurn(turn); err != nil {
		t.Fatal(err)
	}
}

func recordPair(t *testing.T, agent, id string, at time.Time, first, second string) {
	t.Helper()
	answer := []sessions.Part{{Role: "assistant", Kind: "text", Text: "Done."}}
	one := []sessions.Part{{Role: "user", Kind: "text", Text: first}}
	two := append(append([]sessions.Part{}, one...), answer...)
	two = append(two, sessions.Part{Role: "user", Kind: "text", Text: second})
	for i, in := range [][]sessions.Part{one, two} {
		turn := sessions.GatewayTurn{
			Agent: agent, Session: id, Time: at.Add(time.Duration(i) * time.Minute), Status: 200,
			Input: append([]sessions.Part{{Role: "system", Kind: "text", Text: "sys"}}, in...), Output: answer,
		}
		if err := sessions.SaveGatewayTurn(turn); err != nil {
			t.Fatal(err)
		}
	}
}

func writeClaude(t *testing.T, at time.Time, first, second string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "claude.jsonl")
	line := func(role, text string, at time.Time) string {
		if role == "user" {
			return fmt.Sprintf(`{"type":"user","timestamp":"%s","message":{"role":"user","content":%q}}`, at.Format(time.RFC3339Nano), text)
		}
		return fmt.Sprintf(`{"type":"assistant","timestamp":"%s","message":{"role":"assistant","content":[{"type":"text","text":%q}]}}`, at.Format(time.RFC3339Nano), text)
	}
	body := strings.Join([]string{
		line("user", first, at),
		line("assistant", "Done.", at.Add(time.Second)),
		line("user", second, at.Add(2*time.Second)),
		line("assistant", "Done.", at.Add(3*time.Second)),
	}, "\n") + "\n"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func restoreUsage(t *testing.T) {
	t.Helper()
	p := usage.Path()
	b, err := os.ReadFile(p)
	missing := os.IsNotExist(err)
	if err != nil && !missing {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if missing {
			os.Remove(p)
			return
		}
		if err := os.WriteFile(p, b, 0o600); err != nil {
			t.Errorf("restore usage: %v", err)
		}
	})
}
