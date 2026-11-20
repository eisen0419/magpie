package sessions

// Gateway conversations are a separate, opt-in local store, not native agent
// files or the usage ledger. IDs are hashed before they become path components.

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/yetone/magpie/internal/redact"
	"github.com/yetone/magpie/internal/settings"
)

const GatewayRetention = 7 * 24 * time.Hour
const gatewayTurnMax = 1 << 20
const gatewayReadMax = 128
const gatewayStoreMax = 256 << 20
const gatewaySessionMax = gatewayStoreMax

var gatewayMu sync.Mutex
var gatewayGeneration uint64 = 1

// GatewayTurn is one captured client-facing exchange. Input may include replayed
// history; Output is only this request's reply. It is not a native session.
type GatewayTurn struct {
	Agent   string    `json:"agent"`
	Session string    `json:"session"`
	Time    time.Time `json:"time"`
	Model   string    `json:"model"`
	Status  int       `json:"status"`
	Input   []Part    `json:"input"`
	Output  []Part    `json:"output"`
	Cut     bool      `json:"cut,omitempty"`
}

func gatewayDir() string { return filepath.Join(settings.Dir(), "gateway-conversations") }

func gatewayIdentity(agent, id string) string {
	b, _ := json.Marshal([]string{agent, id})
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// SaveGatewayTurn keeps a bounded, scrubbed exchange. Recording is checked again
// under the same lock as disable-and-clear, so in-flight requests cannot refill it.
func SaveGatewayTurn(turn GatewayTurn) error {
	return saveGatewayTurn(turn, GatewayRecordingGeneration())
}

// GatewayRecordingGeneration returns the current consent generation. A
// request captures it when recording starts and must present the same value
// when saving, so clearing and re-enabling recording cannot resurrect old data.
func GatewayRecordingGeneration() uint64 {
	gatewayMu.Lock()
	defer gatewayMu.Unlock()
	return gatewayGeneration
}

func SaveGatewayTurnGeneration(turn GatewayTurn, generation uint64) error {
	return saveGatewayTurn(turn, generation)
}

func saveGatewayTurn(turn GatewayTurn, generation uint64) error {
	gatewayMu.Lock()
	allowed := settings.Shared().GatewayConversations && generation == gatewayGeneration
	gatewayMu.Unlock()
	if !allowed || turn.Agent == "" || turn.Session == "" {
		return nil
	}
	if turn.Time.IsZero() {
		turn.Time = time.Now()
	}
	if turn.Time.Before(time.Now().Add(-GatewayRetention)) {
		return nil
	}
	bound := func(parts []Part) []Part {
		left := 64 << 10 // Each direction gets its own budget; history cannot crowd out the reply.
		out := []Part{}
		for _, p := range parts {
			if left <= 0 {
				turn.Cut = true
				break
			}
			p.Text = string(redact.ScrubJSON([]byte(p.Text)))
			p.Name = redact.Scrub(p.Name)
			if len(p.Text) > min(left, 64<<10) {
				n := min(left, 64<<10)
				for n > 0 && !utf8.ValidString(p.Text[:n]) {
					n--
				}
				p.Cut += utf8.RuneCountInString(p.Text[n:])
				p.Text = p.Text[:n]
				turn.Cut = true
			}
			left -= len(p.Text) + len(p.Name) + 128
			out = append(out, p)
		}
		return out
	}
	turn.Input = bound(turn.Input)
	turn.Output = bound(turn.Output)
	b, err := json.Marshal(turn)
	if err != nil {
		return err
	}
	if len(b) > gatewayTurnMax {
		return errors.New("gateway conversation exceeds storage limit")
	}
	gatewayMu.Lock()
	defer gatewayMu.Unlock()
	if generation != gatewayGeneration || !settings.Shared().GatewayConversations {
		return nil
	}
	dir := filepath.Join(gatewayDir(), turn.Time.UTC().Format(time.DateOnly), gatewayIdentity(turn.Agent, turn.Session))
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	// Append one bounded line; the cost is independent of other sessions and
	// of this session's earlier turns. Cleanup runs outside the request path.
	// Not O_APPEND: on Windows that opens the file for appending only, and
	// the truncate below, which drops a cut-off last line, is refused
	// ("Access is denied"), so every later turn of the session failed.
	// The line is written at the end found instead; gatewayMu keeps
	// another save from writing between.
	path := filepath.Join(dir, "turns.jsonl")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	end, err := gatewayCompleteSize(f, info.Size())
	if err != nil {
		return err
	}
	if end+int64(len(b))+1 > gatewaySessionMax {
		return errors.New("gateway session exceeds storage limit")
	}
	// A process interrupted mid-append must not poison all later turns.
	if end != info.Size() {
		if err := f.Truncate(end); err != nil {
			return err
		}
	}
	_, err = f.WriteAt(append(b, '\n'), end)
	return errors.Join(err, f.Close())
}

// Ignore at most one unfinished turn; earlier complete lines remain intact.
func gatewayCompleteSize(f *os.File, size int64) (int64, error) {
	if size == 0 {
		return 0, nil
	}
	var last [1]byte
	if _, err := f.ReadAt(last[:], size-1); err != nil {
		return 0, err
	}
	if last[0] == '\n' {
		return size, nil
	}
	n := min(size, int64(gatewayTurnMax+1))
	tail := make([]byte, int(n))
	if _, err := f.ReadAt(tail, size-n); err != nil {
		return 0, err
	}
	if i := bytes.LastIndexByte(tail, '\n'); i >= 0 {
		return size - n + int64(i) + 1, nil
	}
	if n == size {
		return 0, nil
	}
	return 0, errors.New("unreadable gateway conversation")
}

// SetGatewayRecording changes only this machine's consent. Clear disables first.
func SetGatewayRecording(on, clear bool) error {
	gatewayMu.Lock()
	defer gatewayMu.Unlock()
	if clear || !on {
		gatewayGeneration++
	}
	if err := settings.SetGatewayConversations(on && !clear); err != nil {
		return err
	}
	if clear {
		// deleted sessions' text in magpie's trash is recorded text too
		return errors.Join(os.RemoveAll(gatewayDir()), clearGatewayTrash(nil))
	}
	return nil
}

// HasGatewayConversations says whether any gateway conversation text is kept,
// so the Sessions page offers to delete it only when there is something to
// delete. A folder that can't be read counts as holding some: the reader can
// still ask to clear it.
func HasGatewayConversations() bool {
	if paths, _ := gatewayTrashFiles(); len(paths) > 0 {
		return true
	}
	errFound := errors.New("found")
	err := filepath.WalkDir(gatewayDir(), func(_ string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type().IsRegular() {
			return errFound
		}
		return nil
	})
	return err != nil && !errors.Is(err, os.ErrNotExist)
}

// PruneGatewayConversations removes expired date buckets even while recording is
// off. Reads enforce the exact cutoff too, between periodic cleanup passes.
func PruneGatewayConversations(now time.Time) error {
	gatewayMu.Lock()
	root, generation := gatewayDir(), gatewayGeneration
	// a deleted session's text in magpie's trash expires as it would have
	trashErr := clearGatewayTrash(func(d string) bool {
		date, err := time.Parse(time.DateOnly, d)
		return err == nil && !date.Add(24*time.Hour).After(now.Add(-GatewayRetention))
	})
	gatewayMu.Unlock()
	if trashErr != nil {
		return trashErr
	}
	type saved struct {
		path    string
		info    os.FileInfo
		expired bool
	}
	var files []saved
	var total int64
	// Walking and sorting never hold the lock needed by a response's save.
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		if err != nil {
			return err
		}
		if !d.Type().IsRegular() {
			return nil
		}
		info, err := d.Info()
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		date, err := time.Parse(time.DateOnly, strings.Split(rel, string(filepath.Separator))[0])
		expired := err == nil && !date.Add(24*time.Hour).After(now.Add(-GatewayRetention))
		files = append(files, saved{path, info, expired})
		total += info.Size()
		return nil
	})
	if err != nil {
		return err
	}
	sort.Slice(files, func(i, j int) bool {
		if files[i].expired != files[j].expired {
			return files[i].expired
		}
		return files[i].info.ModTime().Before(files[j].info.ModTime())
	})
	for _, f := range files {
		if !f.expired && total <= gatewayStoreMax {
			continue
		}
		gatewayMu.Lock()
		if generation != gatewayGeneration || root != gatewayDir() {
			gatewayMu.Unlock()
			return nil
		}
		current, err := os.Stat(f.path)
		if err == nil && os.SameFile(current, f.info) && current.Size() == f.info.Size() && current.ModTime().Equal(f.info.ModTime()) {
			err = os.Remove(f.path)
			if err == nil {
				total -= f.info.Size()
				_ = os.Remove(filepath.Dir(f.path))
				_ = os.Remove(filepath.Dir(filepath.Dir(f.path)))
			}
		}
		gatewayMu.Unlock()
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

// GatewayFile pins one file in a gateway recording at the moment it was
// observed: for turns.jsonl, Size is the complete-line byte offset; for a
// legacy per-turn .json file, Size is the file's byte length.
type GatewayFile struct {
	Key  string
	Path string
	Size int64
}

// GatewaySnapshot pins a gateway recording as observed by GatewayHearer:
// the owning gateway key (Agent, ID), the newest valid turn timestamp
// within retention (Last), and the exact files and byte offsets present
// when observed (Files).
type GatewaySnapshot struct {
	Agent      string
	ID         string
	Last       time.Time
	Files      []GatewayFile
	generation uint64
}

// Sizes returns a fresh map of each pinned file's Key to its observed Size.
func (s GatewaySnapshot) Sizes() map[string]int64 {
	if len(s.Files) == 0 {
		return nil
	}
	out := make(map[string]int64, len(s.Files))
	for _, f := range s.Files {
		out[f.Key] = f.Size
	}
	return out
}

// GatewayTranscript reads recent captured exchanges. Only a full matching
// previous history prefix is removed, never equal text found elsewhere in a chat.
func GatewayTranscript(agent, id string) (Transcript, error) {
	out := Transcript{Parts: []Part{}, Source: "gateway"}
	if agent == "" || id == "" {
		return out, errors.New("no such session")
	}
	gatewayMu.Lock()
	defer gatewayMu.Unlock()
	now := time.Now()
	var files []GatewayFile
	for d := 0; d <= 7; d++ {
		day := now.UTC().AddDate(0, 0, -d).Format(time.DateOnly)
		dir := filepath.Join(gatewayDir(), day, gatewayIdentity(agent, id))
		entries, err := os.ReadDir(dir)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return out, err
		}
		for _, e := range entries {
			if e.Type().IsRegular() && (strings.HasSuffix(e.Name(), ".json") || e.Name() == "turns.jsonl") {
				files = append(files, GatewayFile{
					Key:  day + "/" + e.Name(),
					Path: filepath.Join(dir, e.Name()),
				})
			}
		}
	}
	return readGatewayFiles(agent, id, files, now, false)
}

// GatewayTranscriptSnapshot reads only the files and byte bounds pinned in
// snap. Turns appended after snap was observed — even one whose request
// started earlier than snap.Last — lie beyond snap's bounds and wait for
// the next pass.
func GatewayTranscriptSnapshot(snap GatewaySnapshot) (Transcript, error) {
	out := Transcript{Parts: []Part{}, Source: "gateway"}
	if snap.Agent == "" || snap.ID == "" {
		return out, errors.New("no such session")
	}
	gatewayMu.Lock()
	defer gatewayMu.Unlock()
	if snap.generation != 0 && snap.generation != gatewayGeneration {
		return out, errors.New("gateway recording changed")
	}
	return readGatewayFiles(snap.Agent, snap.ID, snap.Files, time.Now(), true)
}

func readGatewayFiles(agent, id string, files []GatewayFile, now time.Time, pinned bool) (Transcript, error) {
	out := Transcript{Parts: []Part{}, Source: "gateway"}
	var turns []GatewayTurn
	keep := func(turn GatewayTurn) {
		if turn.Agent != agent || turn.Session != id || turn.Time.Before(now.Add(-GatewayRetention)) {
			return
		}
		turns = append(turns, turn)
		if len(turns) > gatewayReadMax*2 {
			sort.SliceStable(turns, func(i, j int) bool { return turns[i].Time.Before(turns[j].Time) })
			turns = slices.Clone(turns[len(turns)-gatewayReadMax:])
			out.Cut = true
		}
	}
	for _, gf := range files {
		f, err := os.Open(gf.Path)
		if err != nil {
			return out, err
		}
		if strings.HasSuffix(gf.Path, ".jsonl") {
			var info os.FileInfo
			info, err = f.Stat()
			if err != nil {
				f.Close()
				return out, err
			}
			if info.Size() > gatewaySessionMax {
				f.Close()
				return out, errors.New("gateway session exceeds storage limit")
			}
			var end int64
			if pinned {
				if gf.Size <= 0 || info.Size() < gf.Size {
					f.Close()
					return out, errors.New("gateway recording changed")
				}
				var last [1]byte
				if _, err = f.ReadAt(last[:], gf.Size-1); err != nil {
					f.Close()
					return out, err
				}
				if last[0] != '\n' {
					f.Close()
					return out, errors.New("gateway recording changed")
				}
				end = gf.Size
			} else {
				end, err = gatewayCompleteSize(f, info.Size())
				if err != nil {
					f.Close()
					return out, err
				}
				out.Cut = out.Cut || end != info.Size()
			}
			scan := bufio.NewScanner(io.LimitReader(f, end))
			scan.Buffer(make([]byte, 64<<10), gatewayTurnMax+1)
			for scan.Scan() {
				var turn GatewayTurn
				if json.Unmarshal(scan.Bytes(), &turn) != nil {
					f.Close()
					return out, errors.New("unreadable gateway conversation")
				}
				keep(turn)
			}
			err = scan.Err()
		} else {
			// Read the original per-turn files until their retention expires.
			if pinned {
				var info os.FileInfo
				info, err = f.Stat()
				if err != nil {
					f.Close()
					return out, err
				}
				if info.Size() != gf.Size {
					f.Close()
					return out, errors.New("gateway recording changed")
				}
			}
			var data []byte
			data, err = io.ReadAll(io.LimitReader(f, gatewayTurnMax+1))
			var turn GatewayTurn
			if err == nil && (len(data) > gatewayTurnMax || json.Unmarshal(data, &turn) != nil) {
				err = errors.New("unreadable gateway conversation")
			}
			if err == nil {
				keep(turn)
			}
		}
		f.Close()
		if err != nil {
			return out, err
		}
	}
	sort.SliceStable(turns, func(i, j int) bool { return turns[i].Time.Before(turns[j].Time) })
	if len(turns) > gatewayReadMax {
		turns = turns[len(turns)-gatewayReadMax:]
		out.Cut = true
	}
	var previous, system []Part
	for _, turn := range turns {
		out.Captured++
		out.Cut = out.Cut || turn.Cut
		// Context is kept independently because it is repeated before every
		// request's conversation, rather than being a new conversational turn.
		var context, input []Part
		for _, p := range turn.Input {
			if p.Role == "system" {
				context = append(context, p)
			} else {
				input = append(input, p)
			}
		}
		if !slices.Equal(context, system) {
			for _, p := range context {
				if !out.add(true, p) {
					return out, nil
				}
			}
			system = context
		}
		fresh := input
		if len(previous) > 0 && len(input) >= len(previous) && slices.Equal(input[:len(previous)], previous) {
			fresh = input[len(previous):]
		}
		for _, p := range fresh {
			if !out.add(true, p) {
				return out, nil
			}
		}
		for _, p := range turn.Output {
			if !out.add(false, p) {
				return out, nil
			}
		}
		previous = append(slices.Clone(input), turn.Output...)
	}
	return out, nil
}

// gatewayUnheard is the answer when the store isn't there: asking about a
// session reads no folder of it.
func gatewayUnheard(string, string) (GatewaySnapshot, bool) { return GatewaySnapshot{}, false }

// GatewayHearer reports, for each agent and session id, a pinned snapshot of
// what magpie recorded of it within retention, and whether any valid turn was
// recorded. The store is stated once: with nothing recorded, a session costs
// no further read (a listing that stated each session's folder was most of
// the Sessions page's time, #2278). A turn is stamped when its request starts
// and appended when the request finishes, so the newest time is the maximum
// over every completed turn across turns.jsonl and all legacy .json files,
// not the last line or newest mtime. Any unreadable or corrupt file makes
// the recording unknown (false) rather than partially valid. One pass
// remembers each session it already opened.
func GatewayHearer() func(agent, id string) (GatewaySnapshot, bool) {
	root := gatewayDir()
	st, err := os.Stat(root)
	if err != nil || !st.IsDir() {
		return gatewayUnheard
	}
	var mu sync.Mutex
	type hit struct {
		snap GatewaySnapshot
		ok   bool
	}
	cache := map[string]hit{}
	return func(agent, id string) (GatewaySnapshot, bool) {
		if agent == "" || id == "" {
			return GatewaySnapshot{}, false
		}
		k := agent + "\x00" + id
		mu.Lock()
		if v, ok := cache[k]; ok {
			mu.Unlock()
			return v.snap, v.ok
		}
		mu.Unlock()
		snap, ok := gatewayHeard(root, agent, id)
		mu.Lock()
		if v, seen := cache[k]; seen {
			mu.Unlock()
			return v.snap, v.ok
		}
		cache[k] = hit{snap, ok}
		mu.Unlock()
		return snap, ok
	}
}

func gatewayHeard(root, agent, id string) (GatewaySnapshot, bool) {
	if agent == "" || id == "" {
		return GatewaySnapshot{}, false
	}
	// The generation is the only shared state the scan needs. Directory
	// reads and parsing stay outside gatewayMu, which a response's save
	// also takes; a clear that bumps the generation drops this answer.
	gatewayMu.Lock()
	generation := gatewayGeneration
	gatewayMu.Unlock()
	now := time.Now().UTC()
	cutoff := now.Add(-GatewayRetention)
	var best time.Time
	var files []GatewayFile
	ok := false
	for d := 0; d <= 7; d++ {
		day := now.AddDate(0, 0, -d).Format(time.DateOnly)
		dir := filepath.Join(root, day, gatewayIdentity(agent, id))
		entries, err := os.ReadDir(dir)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return GatewaySnapshot{}, false
		}
		for _, e := range entries {
			if !e.Type().IsRegular() {
				continue
			}
			name := e.Name()
			if name != "turns.jsonl" && !strings.HasSuffix(name, ".json") {
				continue
			}
			p := filepath.Join(dir, name)
			at, size, yes, err := gatewayFileHeard(p, agent, id, cutoff)
			if err != nil {
				return GatewaySnapshot{}, false
			}
			if yes {
				files = append(files, GatewayFile{Key: day + "/" + name, Path: p, Size: size})
				if !ok || at.After(best) {
					best, ok = at, true
				}
			}
		}
	}
	gatewayMu.Lock()
	cleared := generation != gatewayGeneration
	gatewayMu.Unlock()
	if cleared || !ok {
		return GatewaySnapshot{}, false
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Key < files[j].Key })
	return GatewaySnapshot{
		Agent:      agent,
		ID:         id,
		Last:       best,
		Files:      files,
		generation: generation,
	}, true
}

// gatewayFileHeard returns the newest valid turn time and pinned byte length
// of path, or an error if the file cannot be read or contains corrupt JSON.
func gatewayFileHeard(path, agent, id string, cutoff time.Time) (time.Time, int64, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return time.Time{}, 0, false, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return time.Time{}, 0, false, err
	}
	if strings.HasSuffix(path, ".jsonl") {
		return gatewayJSONLHeard(f, info.Size(), agent, id, cutoff)
	}
	at, ok, err := gatewayJSONHeard(f, info.Size(), agent, id, cutoff)
	return at, info.Size(), ok, err
}

func gatewayJSONHeard(f *os.File, size int64, agent, id string, cutoff time.Time) (time.Time, bool, error) {
	if size <= 0 || size > gatewayTurnMax {
		return time.Time{}, false, errors.New("unreadable gateway conversation")
	}
	buf := make([]byte, int(size))
	if _, err := io.ReadFull(f, buf); err != nil {
		return time.Time{}, false, err
	}
	return heardTurn(buf, agent, id, cutoff)
}

func gatewayJSONLHeard(f *os.File, size int64, agent, id string, cutoff time.Time) (time.Time, int64, bool, error) {
	if size > gatewaySessionMax {
		return time.Time{}, 0, false, errors.New("gateway session exceeds storage limit")
	}
	end, err := gatewayCompleteSize(f, size)
	if err != nil {
		return time.Time{}, 0, false, err
	}
	if end == 0 {
		return time.Time{}, 0, false, nil
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return time.Time{}, 0, false, err
	}
	scan := bufio.NewScanner(io.LimitReader(f, end))
	scan.Buffer(make([]byte, 64<<10), gatewayTurnMax+1)
	var best time.Time
	ok := false
	for scan.Scan() {
		at, yes, err := heardTurn(scan.Bytes(), agent, id, cutoff)
		if err != nil {
			return time.Time{}, 0, false, err
		}
		if yes && (!ok || at.After(best)) {
			best, ok = at, true
		}
	}
	if err := scan.Err(); err != nil {
		return time.Time{}, 0, false, err
	}
	return best, end, ok, nil
}

// heardTurn is buf's turn time when it is agent|id's and still kept, or an
// error if buf is not valid JSON.
func heardTurn(buf []byte, agent, id string, cutoff time.Time) (time.Time, bool, error) {
	var turn struct {
		Agent   string    `json:"agent"`
		Session string    `json:"session"`
		Time    time.Time `json:"time"`
	}
	if json.Unmarshal(buf, &turn) != nil {
		return time.Time{}, false, errors.New("unreadable gateway conversation")
	}
	if turn.Agent != agent || turn.Session != id || turn.Time.IsZero() || turn.Time.Before(cutoff) {
		return time.Time{}, false, nil
	}
	return turn.Time, true, nil
}
