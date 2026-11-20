// Package memdistill reads the agents' finished conversations and asks a
// model what in them is worth remembering: a preference the user stated,
// a rule learned from a correction, a fact of the project the code
// doesn't say. What it finds waits in memory as a candidate (To review)
// until the user keeps it; nothing it proposes reaches an agent before.
//
// It runs when switched on (State.On), on sessions quiet for Quiet, each
// once for what it held when read: a session that goes on is read again
// later, whole. The hourly run reads at most PerRun; one the user asked
// for (ReadAll) reads every one due, and Stop ends it. The model is the one
// picked, else the caller's automatic choice. The request goes through
// magpie's own gateway (Complete), so masking and routing apply as to any
// agent's request.
package memdistill

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/yetone/magpie/internal/appdir"
	"github.com/yetone/magpie/internal/filelock"
	"github.com/yetone/magpie/internal/lastgood"
	"github.com/yetone/magpie/internal/memory"
	"github.com/yetone/magpie/internal/sessions"
	"github.com/yetone/magpie/internal/usage"
)

// Complete sends one request to model, asked to reason at effort ("" the
// model's own default): a system text and one user message, and returns
// the reply's text.
type Complete func(ctx context.Context, model, effort, system, user string) (string, error)

const (
	// Quiet is how long a session has had no turn before it is read: one
	// still going is read when it stops
	Quiet = 30 * time.Minute
	// PerRun caps the sessions an hourly run sends to the model: more
	// than a busy hour brings, so only a backlog waits for the next run.
	// A run the user asked for (ReadAll) reads every one due
	// (RealZeroP27 on X: at 10 an hour, 400 took 40 hours)
	PerRun = 30
	// Back is how far back a run looks the first time it is switched on,
	// when no range is picked (State.Days 0)
	Back = 7 * 24 * time.Hour
	// All is State.Days for every conversation listed: the latest
	// Listed, read at once when picked (ReadAll), else PerRun an hour
	All = -1
	// Listed is how many of the latest sessions a run looks through
	Listed = 400
	// MinTurns is the fewest things the user said for a session to be read
	MinTurns = 2
	// MaxPerSession caps what one session may propose
	MaxPerSession = 5
	// budget is the transcript's size sent, in characters
	budget = 24_000
	// keepMarks is how long a read session's mark is kept
	keepMarks = 30 * 24 * time.Hour
)

// Agent is the agent id the model requests are made as (gateway.TokenFor),
// so Usage counts them apart, and whose sessions are never read.
const Agent = "memory"

// State is distill.json.
type State struct {
	On bool `json:"on,omitempty"`
	// Model is the model picked; "" is the automatic choice
	Model string `json:"model,omitempty"`
	// Days is how far back reading looks, in days, from when it was
	// switched on or the range was picked: 0 is Back, All every
	// conversation listed (Ranges)
	Days int `json:"days,omitempty"`
	// Effort is the reasoning level the model is asked at, reading and
	// tidying, fitted to its own levels as it is sent; "" its default
	Effort string `json:"effort,omitempty"`
	// Since is when it was switched on or the range picked, less the
	// range: older sessions aren't read. Zero with All
	Since time.Time `json:"since,omitzero"`
	// Done is each read session's last turn when it was read, by
	// agent/id
	Done map[string]time.Time `json:"done,omitempty"`
	// DoneFiles is each gateway-backed session's snapshotted file sizes
	// (day/file -> byte length) when it was read, by agent/id, so a turn
	// appended after due() — even one stamped earlier than Done — still
	// triggers a re-read.
	DoneFiles map[string]map[string]int64 `json:"doneFiles,omitempty"`
	// the last run: when, with which model, how many sessions read, how
	// many memories proposed, and what stopped it
	Last      time.Time `json:"last,omitzero"`
	LastModel string    `json:"lastModel,omitempty"`
	Read      int       `json:"read,omitempty"`
	Added     int       `json:"added,omitempty"`
	Error     string    `json:"error,omitempty"`
	// Left is how many were due and not read when the last run ended:
	// stopped, failed or past PerRun
	Left int `json:"left,omitempty"`
	// Dream is the nightly tidy switched on (dream.go), DreamSince when
	Dream      bool      `json:"dream,omitempty"`
	DreamSince time.Time `json:"dreamSince,omitzero"`
	// the last tidy that went through: when, with which model, and what it
	// changed, undone or not; DreamError is what stopped the last one tried
	Dreamt     time.Time       `json:"dreamt,omitzero"`
	DreamModel string          `json:"dreamModel,omitempty"`
	DreamError string          `json:"dreamError,omitempty"`
	Changes    []memory.Change `json:"changes,omitempty"`
	Undone     bool            `json:"undone,omitempty"`
}

// Path is distill.json; a variable for tests.
var Path = func() string { return appdir.Join(appdir.Config(), "distill.json") }

var (
	now  = time.Now
	list = listed
	read = transcriptOf
	// running is held through a run in this process, and the file lock
	// across magpies
	running sync.Mutex
	busy    atomic.Bool
	// stop ends the run going (Stop), stopped says it was asked to
	stopMu  sync.Mutex
	stop    context.CancelFunc
	stopped atomic.Bool
	// doing and of are the run's progress: sessions gone through, of
	// how many
	doing, of atomic.Int64
)

// Running says a run is going in this process.
func Running() bool { return busy.Load() }

// Progress is how many sessions the run going has gone through, of how
// many it will, read or passed over; 0, 0 when none is going or it is
// still listing them.
func Progress() (int, int) { return int(doing.Load()), int(of.Load()) }

// Stop ends the run going in this process, if any: the request in flight
// is dropped and its session read next time. It isn't an error.
func Stop() {
	stopMu.Lock()
	defer stopMu.Unlock()
	if stop != nil {
		stopped.Store(true)
		stop()
	}
}

// ErrRunning is a run asked for while one is going.
var ErrRunning = errors.New("already reading conversations")

// Load reads distill.json; a missing file is the default (off). One that
// can't be read is an error, never the default: a run doesn't start, and
// a change isn't written over it.
func Load() (State, error) {
	var s State
	b, err := lastgood.Read(Path(), lastgood.Into[State])
	if errors.Is(err, os.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return s, err
	}
	if err := json.Unmarshal(b, &s); err != nil {
		return s, err
	}
	return s, nil
}

func save(s State) error {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	p := Path()
	if err := os.MkdirAll(filepath.Dir(p), 0o700); err != nil {
		return err
	}
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, p); err != nil {
		return err
	}
	return lastgood.Keep(p, lastgood.Into[State], 0o600)
}

// change applies fn to distill.json under its lock.
func change(fn func(*State)) (State, error) {
	unlock, held := filelock.Lock(Path()+".lock", 10*time.Second)
	if !held {
		return State{}, errors.New("distill.json is held by another magpie")
	}
	defer unlock()
	s, err := Load()
	if err != nil {
		return s, err
	}
	fn(&s)
	return s, save(s)
}

// Ranges are the ranges, in days, reading can be set to look back: Back's
// seven first, All last.
var Ranges = []int{7, 30, 90, 365, All}

// since is the oldest a session read may be when days is picked at t.
func since(days int, t time.Time) time.Time {
	switch days {
	case All:
		return time.Time{}
	case 0:
		return t.Add(-Back).UTC()
	}
	return t.Add(-time.Duration(days) * 24 * time.Hour).UTC()
}

// Set switches distilling on or off and picks its model ("" automatic).
// Switched on from off, it looks back the range picked (Back by default)
// from now; switched off, the run going stops.
func Set(on bool, model string) (State, error) {
	if !on {
		Stop()
	}
	return change(func(s *State) {
		if on && !s.On {
			s.Since = since(s.Days, now())
		}
		s.On, s.Model = on, strings.TrimSpace(model)
	})
}

// SetRange picks how far back reading looks: one of Ranges, from now. A
// conversation read already isn't read again for a longer range; one
// older than a shorter range isn't read.
func SetRange(days int) (State, error) {
	if !slices.Contains(Ranges, days) {
		return State{}, fmt.Errorf("a range of %v days, not %d", Ranges, days)
	}
	return change(func(s *State) {
		if days == 7 {
			days = 0 // Back, kept as before a range could be picked
		}
		s.Days, s.Since = days, since(days, now())
	})
}

// SetEffort picks the reasoning level the model is asked at, "" for its
// own default. A level the model doesn't have is fitted to the nearest it
// has when sent.
func SetEffort(effort string) (State, error) {
	effort = strings.ToLower(strings.TrimSpace(effort))
	if effort != "" && !slices.Contains(efforts, effort) {
		return State{}, fmt.Errorf("a reasoning level of %v, not %q", efforts, effort)
	}
	return change(func(s *State) { s.Effort = effort })
}

// efforts are the levels SetEffort takes (gateway's effortRank).
var efforts = []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}

// Result is what one run did.
type Result struct {
	Read  int
	Added int
}

// Run reads up to PerRun of the sessions due, if switched on, with auto
// when the state names no model. A model that fails stops the run: the
// sessions not read yet are read next time.
func Run(ctx context.Context, auto string, complete Complete) (Result, error) {
	return runUpTo(ctx, auto, complete, PerRun)
}

// ReadAll is Run over every session due, for a run the user asked for:
// switched on, a model or a longer range picked, Read now. Stop ends it.
func ReadAll(ctx context.Context, auto string, complete Complete) (Result, error) {
	return runUpTo(ctx, auto, complete, Listed)
}

func runUpTo(ctx context.Context, auto string, complete Complete, limit int) (Result, error) {
	if !running.TryLock() {
		return Result{}, ErrRunning
	}
	defer running.Unlock()
	busy.Store(true)
	defer busy.Store(false)
	unlock, held := filelock.Lock(Path()+".run", 0)
	if !held {
		return Result{}, ErrRunning
	}
	defer unlock()
	st, err := Load()
	if err != nil || !st.On {
		return Result{}, err
	}
	ctx, cancel := context.WithCancel(ctx)
	stopMu.Lock()
	stop = cancel
	stopped.Store(false)
	stopMu.Unlock()
	defer func() {
		stopMu.Lock()
		stop = nil
		stopMu.Unlock()
		cancel()
		doing.Store(0)
		of.Store(0)
	}()
	model := st.Model
	if model == "" {
		model = auto
	}
	var res ran
	var runErr error
	if model == "" {
		runErr = errors.New("no model to read conversations with: pick one")
	} else {
		res, runErr = run(ctx, st, model, st.Effort, complete, limit)
	}
	if stopped.Load() && errors.Is(runErr, context.Canceled) {
		runErr = nil
	}
	_, err = change(func(s *State) {
		s.Last, s.LastModel, s.Read, s.Added, s.Error, s.Left = now().UTC(), model, res.Read, res.Added, "", res.left
		if runErr != nil {
			s.Error = runErr.Error()
		}
		if s.Done == nil {
			s.Done = map[string]time.Time{}
		}
		for k, at := range res.done {
			s.Done[k] = at
		}
		if len(res.doneFiles) > 0 && s.DoneFiles == nil {
			s.DoneFiles = map[string]map[string]int64{}
		}
		for k, files := range res.doneFiles {
			s.DoneFiles[k] = files
		}
		for k, at := range s.Done {
			if forget(at, s.Since, res.oldest) {
				delete(s.Done, k)
				delete(s.DoneFiles, k)
			}
		}
	})
	return res.Result, errors.Join(runErr, err)
}

type ran struct {
	Result
	done      map[string]time.Time
	doneFiles map[string]map[string]int64
	// oldest is the last turn of the oldest session listed, zero when
	// none was
	oldest time.Time
	// left is how many due weren't gone through
	left int
}

// listed are the latest Listed conversations as the Sessions page lists
// them: the agents' own files and the sessions only magpie's gateway
// recorded (WorkBuddy's, an agent's on another computer), each once
// (sessions.GatewayKeys). Reading only the files left every agent seen
// only through the gateway out of memory (bxgz233 on X: 1000+ WorkBuddy
// sessions, none read).
func listed() []sessions.Session {
	native := sessions.List(Listed)
	keys := map[string]bool{}
	for _, s := range native {
		for _, k := range sessions.GatewayKeys(s) {
			keys[k] = true
		}
	}
	all := sessions.MergeExternal(native, usage.ListedGateway(keys, Listed))
	return all[:min(len(all), Listed)]
}

// gwOf holds each gateway-backed session's pinned recording snapshot from
// due(). transcriptOf reads strictly within those file byte bounds, and
// remember records those sizes in State.DoneFiles so later appends — even
// turns stamped earlier than Done — wait for and trigger the next pass.
var (
	gwMu sync.Mutex
	gwOf map[string]sessions.GatewaySnapshot
)

func noteGateway(s sessions.Session, snap sessions.GatewaySnapshot) {
	gwMu.Lock()
	if gwOf == nil {
		gwOf = map[string]sessions.GatewaySnapshot{}
	}
	gwOf[key(s)] = snap
	gwMu.Unlock()
}

func gatewayNoted(s sessions.Session) (sessions.GatewaySnapshot, bool) {
	gwMu.Lock()
	defer gwMu.Unlock()
	snap, ok := gwOf[key(s)]
	return snap, ok
}

// transcriptOf reads s where it is kept. A gateway-only session is what
// the gateway recorded of it. So is a native session whose agent has no
// transcript reader, when one of its gateway keys holds a recording
// (recordedTurn): the file can't be read, and the gateway row was merged
// away because the native one won. Read-only is the same when the agent
// has no reader (Hermes, Alma). A read-only session with a reader
// (Cursor's editor) stays on that reader. due's snapshot pins the exact
// recording files and byte offsets that made the row due, so any turn
// appended after due() waits for the next pass. Any other native session
// is its agent's file.
func transcriptOf(s sessions.Session) (sessions.Transcript, error) {
	if snap, ok := gatewayNoted(s); ok {
		return sessions.GatewayTranscriptSnapshot(snap)
	}
	if s.Gateway {
		return sessions.GatewayTranscript(s.Agent, s.ID)
	}
	if snap, ok := recordedTurn(s, sessions.GatewayHearer()); ok {
		return sessions.GatewayTranscriptSnapshot(snap)
	}
	return sessions.TranscriptOf(s)
}

// recordedTurn is the recording snapshot the gateway kept under one of s's
// own gateway names (sessions.GatewaySame). Subagent and copy keys are
// excluded so a subagent's recording never stands in for or masks the main
// session. None when s is read from its own file or is itself a gateway
// row. A read-only session with a reader (Cursor's editor) has one, so it
// stays there. One without (Hermes, Alma) is read from the gateway when
// one of its own keys was recorded. heard is sessions.GatewayHearer, stated
// once for the pass.
func recordedTurn(s sessions.Session, heard func(string, string) (sessions.GatewaySnapshot, bool)) (best sessions.GatewaySnapshot, ok bool) {
	if s.Gateway || s.Transcript || heard == nil {
		return
	}
	for _, k := range sessions.GatewaySame(s) {
		a, i, cut := strings.Cut(k, "/")
		if !cut || a == "" || i == "" {
			continue
		}
		snap, yes := heard(a, i)
		if !yes {
			continue
		}
		if !ok || snap.Last.After(best.Last) {
			best, ok = snap, true
		}
	}
	return
}

// forget says a read session's mark, its last turn at, can go: the
// session is older than since, so it is never read again, or older than
// keepMarks and than every session listed (oldest), so it was deleted or
// fell off the listing, which it comes back to only with a turn after
// the mark. A mark kept only keepMarks would have a session older than
// that, inside the range, read again every hour.
func forget(at, since, oldest time.Time) bool {
	return at.Before(since) || now().Sub(at) > keepMarks && !oldest.IsZero() && at.Before(oldest)
}

// due are the sessions to read now, the latest first, and the last turn
// of the oldest listed. A native session with no transcript reader is
// due when the gateway recorded it, read-only or not. The transcript is
// only that recording, so the quiet wait and the mark are the recording's
// newest turn, not the agent's file. The noted snapshot pins the exact
// files and byte offsets observed here, so any turn appended afterwards
// waits for the next pass.
func due(st State, t time.Time) ([]sessions.Session, time.Time) {
	var out []sessions.Session
	var oldest time.Time
	heard := sessions.GatewayHearer()
	gwMu.Lock()
	gwOf = map[string]sessions.GatewaySnapshot{}
	gwMu.Unlock()
	for _, s := range list() {
		var backed bool
		if s.Gateway {
			if snap, ok := heard(s.Agent, s.ID); ok {
				noteGateway(s, snap)
			}
		} else if snap, ok := recordedTurn(s, heard); ok {
			backed = true
			s.Last = snap.Last
			noteGateway(s, snap)
		}
		if oldest.IsZero() || s.Last.Before(oldest) {
			oldest = s.Last
		}
		switch {
		case s.Agent == Agent, !s.Transcript && !backed,
			s.ParentSession != "",
			t.Sub(s.Last) < Quiet, s.Last.Before(st.Since):
			continue
		}
		if marked(st, s) {
			continue
		}
		out = append(out, s)
	}
	slices.SortStableFunc(out, func(a, b sessions.Session) int { return b.Last.Compare(a.Last) })
	return out, oldest
}

func key(s sessions.Session) string { return s.Agent + "/" + s.ID }

// remember marks s read at its last turn (and snapshotted file sizes, when
// gateway-backed) under every name its own gateway recording goes by
// (sessions.GatewaySame), so a ZCode row listed later as sess_… is not
// read again after the bare request id already was.
func remember(done map[string]time.Time, doneFiles map[string]map[string]int64, s sessions.Session) {
	snap, hasSnap := gatewayNoted(s)
	for _, k := range sessions.GatewaySame(s) {
		done[k] = s.Last
		if hasSnap {
			doneFiles[k] = snap.Sizes()
		}
	}
}

// snapshotCovered says every file in snap was already read at the same byte
// length. An empty seen map is a pre-upgrade mark with no file snapshot and
// falls back to the Done timestamp check.
func snapshotCovered(snap sessions.GatewaySnapshot, seen map[string]int64) bool {
	if len(seen) == 0 {
		return true
	}
	for _, f := range snap.Files {
		if n, ok := seen[f.Key]; !ok || f.Size != n {
			return false
		}
	}
	return true
}

// marked says s's recording was already read at a turn at least as new as
// the one it holds now — and, when gateway-backed, at the same snapshotted
// file sizes — under this row's id or another name of the same recording.
func marked(st State, s sessions.Session) bool {
	snap, hasSnap := gatewayNoted(s)
	for _, k := range sessions.GatewaySame(s) {
		at, ok := st.Done[k]
		if !ok || s.Last.After(at) {
			continue
		}
		if hasSnap && !snapshotCovered(snap, st.DoneFiles[k]) {
			continue
		}
		return true
	}
	return false
}

func run(ctx context.Context, st State, model, effort string, complete Complete, limit int) (ran, error) {
	r := ran{done: map[string]time.Time{}, doneFiles: map[string]map[string]int64{}}
	ss, oldest := due(st, now())
	r.oldest = oldest
	r.left = len(ss)
	of.Store(int64(len(ss)))
	for _, s := range ss {
		if r.Read >= limit {
			break
		}
		if ctx.Err() != nil {
			return r, ctx.Err()
		}
		r.left--
		doing.Add(1)
		tr, err := read(s)
		if err != nil {
			// A failed read is unknown, not done: leave gateway-backed
			// recordings unmarked so repairing or settling them retries
			// even when the newest turn timestamp stays the same.
			if _, gw := gatewayNoted(s); !gw && !s.Gateway {
				remember(r.done, r.doneFiles, s)
			}
			continue
		}
		text, turns := transcript(tr)
		if turns < MinTurns {
			remember(r.done, r.doneFiles, s)
			continue
		}
		proj := memory.ProjectOf(s.Cwd)
		reply, err := complete(ctx, model, effort, prompt, request(text, known(proj), proj))
		if err != nil {
			r.left++ // read again next time
			if ctx.Err() != nil {
				return r, ctx.Err()
			}
			return r, fmt.Errorf("%s: %w", model, err)
		}
		r.Read++
		remember(r.done, r.doneFiles, s)
		r.Added += keep(parse(reply), s, proj, said(tr))
	}
	return r, nil
}

// transcript is what was said, user and assistant text alone (no tool
// calls, results or thinking), each turn named, within budget: the start
// and the end kept when it is longer. turns counts the user's.
func transcript(t sessions.Transcript) (string, int) {
	var lines []string
	turns := 0
	for _, p := range t.Parts {
		if p.Kind != "text" || p.Role != "user" && p.Role != "assistant" {
			continue
		}
		who, max := "User", 4000
		if p.Role == "assistant" {
			who, max = "Assistant", 1200
		} else {
			turns++
		}
		lines = append(lines, who+": "+cut(p.Text, max))
	}
	s := strings.Join(lines, "\n\n")
	if len(s) > budget {
		head, tail := budget/3, budget-budget/3
		s = cutBytes(s[:head]) + "\n\n[…]\n\n" + strings.ToValidUTF8(s[len(s)-tail:], "")
	}
	return s, turns
}

func cut(s string, max int) string {
	if utf8.RuneCountInString(s) <= max {
		return s
	}
	return string([]rune(s)[:max]) + "…"
}

func cutBytes(s string) string { return strings.ToValidUTF8(s, "") }

// known are the memories the model is shown, so it proposes none of them
// again: the active and waiting ones everyone has, and the project's.
func known(proj *memory.Project) []string {
	ms, _, err := memory.All()
	if err != nil {
		return nil
	}
	var out []string
	for _, m := range ms {
		if m.Status != memory.Active && m.Status != memory.Candidate {
			continue
		}
		if m.Scope == memory.Global || proj != nil && m.Scope == "project:"+proj.Key {
			out = append(out, "- "+oneLine(m.Text))
		}
	}
	if len(out) > 80 {
		out = out[len(out)-80:]
	}
	return out
}

// said is everything the user wrote, as letters and digits alone, for
// telling a quote of theirs from words they never wrote.
func said(t sessions.Transcript) string {
	var b strings.Builder
	for _, p := range t.Parts {
		if p.Kind == "text" && p.Role == "user" {
			b.WriteString(bare(p.Text))
		}
	}
	return b.String()
}

// bare is s lower-cased with everything but letters and digits taken out.
func bare(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return unicode.ToLower(r)
		}
		return -1
	}, s)
}

func oneLine(s string) string { return strings.Join(strings.Fields(s), " ") }

// prompt names neither the app reading nor any vendor: it goes to the
// model's vendor as written.
const prompt = `You read a conversation between a user and a coding assistant and write down what is worth remembering in later conversations, with this user or in this project.

Keep only what stays true beyond this conversation and that a later assistant couldn't learn by reading the code:
- profile: who the user is (role, languages they speak, what they know)
- preference: how the user wants things done, stated or shown by a correction
- rule: a rule learned from a mistake the user corrected, with why
- fact: something true of the project that the code doesn't say
- reference: where something is (a host, a dashboard, a URL, a document)
- procedure: steps that worked for a task likely to come again
- episode: a decision that was made, and why

Each item rests on words the user wrote in this conversation: quote them exactly. Leave out what only the assistant said, such as rules it follows from its own instructions or notes.

A rule or convention of one project belongs in the project's own instruction files and docs, which the user keeps up to date, not here: leave it out unless it is about how this user works everywhere. Never keep an item that disagrees with what the project's files say in the conversation.

Leave out: the task itself and its progress, anything about this conversation alone, code details visible in the repository, what the project's instruction files or docs say, secrets (keys, tokens, passwords), and anything already in "Already known". When nothing is worth keeping, answer [].

Answer with a JSON array alone, at most 5 items:
[{"text": "one short sentence, in the language the user wrote in", "kind": "preference", "scope": "global or project", "why": "what in the conversation shows it", "quote": "the user's own words that show it, copied exactly, short"}]
scope is "project" when it holds only in this project, "global" when it holds for the user everywhere.`

func request(text string, known []string, proj *memory.Project) string {
	var b strings.Builder
	if proj != nil {
		fmt.Fprintf(&b, "Project: %s\n\n", proj.Name)
	}
	b.WriteString("Already known:\n")
	if len(known) == 0 {
		b.WriteString("(nothing)\n")
	}
	for _, k := range known {
		b.WriteString(k + "\n")
	}
	b.WriteString("\nConversation:\n\n")
	b.WriteString(text)
	return b.String()
}

// found is one memory the model proposed.
type found struct {
	Text  string `json:"text"`
	Kind  string `json:"kind"`
	Scope string `json:"scope"`
	Why   string `json:"why"`
	Quote string `json:"quote"`
}

// parse reads the reply's JSON array, in a code fence or after words; a
// reply without one is nothing found.
func parse(reply string) []found {
	i, j := strings.Index(reply, "["), strings.LastIndex(reply, "]")
	if i < 0 || j < i {
		return nil
	}
	var out []found
	if json.Unmarshal([]byte(reply[i:j+1]), &out) != nil {
		return nil
	}
	return out
}

// keep saves what was found as candidates, each from s, and counts the
// new ones. Each rests on words the user wrote (in their, as bare gives
// it): one without is the assistant's own instructions or notes said back,
// or made up, and is dropped. A kind not known is a fact; a project scope
// without a project is everyone's.
func keep(fs []found, s sessions.Session, proj *memory.Project, their string) int {
	n := 0
	for i, f := range fs {
		if i >= MaxPerSession {
			break
		}
		text := strings.TrimSpace(f.Text)
		if text == "" || utf8.RuneCountInString(text) > 400 {
			continue
		}
		if q := bare(f.Quote); q == "" || !strings.Contains(their, q) {
			continue
		}
		kind := strings.ToLower(strings.TrimSpace(f.Kind))
		if !slices.Contains(memory.Kinds, kind) {
			kind = memory.Fact
		}
		scope, p := memory.Global, (*memory.Project)(nil)
		if strings.EqualFold(strings.TrimSpace(f.Scope), "project") && proj != nil {
			scope, p = "project:"+proj.Key, proj
		}
		why, status := cut(strings.TrimSpace(f.Why), 300), memory.Candidate
		src := &memory.Source{Agent: s.Agent, Session: s.ID, Dir: s.Cwd, Quote: cut(strings.TrimSpace(f.Quote), 300)}
		_, added, err := memory.Save(memory.Input{Text: &text, Kind: &kind, Scope: &scope, Why: &why, Status: &status}, memory.FromInferred, src, p)
		if err == nil && added {
			n++
		}
	}
	return n
}

// Keep runs Run first after first, then every interval, until ctx ends,
// and Dream after it when one is due (Dreams); auto gives the automatic
// model at each run. Each tick applies memory's retention first
// (memory.Prune), distilling on or off.
func Keep(ctx context.Context, first, interval time.Duration, auto func() string, complete Complete) {
	t := time.NewTimer(first)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		t.Reset(interval)
		// retention holds whether distilling is on or not: memories saved
		// by agents leave candidates and archived ones too
		if _, err := memory.Prune(); err != nil {
			log.Println("memory: retention:", err)
		}
		// switched off (the default): nothing to read, nor a model to
		// ask. auto picks one from every provider's models, and that
		// loaded the whole catalog (3.6 MB) for a run that read nothing
		if st, err := Load(); err == nil && !st.On {
			continue
		}
		if _, err := Run(ctx, auto(), complete); err != nil && !errors.Is(err, ErrRunning) {
			log.Println("memory: reading conversations:", err) // and in the Memory tab
		}
		if st, err := Load(); err == nil && Dreams(st, now()) {
			if _, err := Dream(ctx, auto(), complete); err != nil && !errors.Is(err, ErrRunning) {
				log.Println("memory: tidying:", err)
			}
		}
	}
}
