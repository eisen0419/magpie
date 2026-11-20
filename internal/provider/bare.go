package provider

import (
	"strings"

	"github.com/yetone/magpie/internal/settings"
)

// Model ids without the provider (settings.BareModelIDs; love1sbug on X):
// a client moved from another proxy (CLIProxyAPI) asks for gpt-5.6-sol,
// not codex/gpt-5.6-sol, and picks its model from /v1/models. The gateway
// has always taken a bare id (GroupFor, then Resolve); with the setting on,
// /v1/models lists each model by its bare id as well, after the
// provider/model ids, which stay as they were. Off by default: a list
// nobody asked to grow keeps its ids.
//
// A bare id is listed only where a request for it goes, as the gateway
// routes it: two providers' gpt-5.6-sol is one row, the routing group magpie
// found for it (its members in the Providers order, failed over between),
// or with found groups off the first provider in the Providers order. A
// bare id that is a catalog id already, or that the gateway would take to
// a model the list doesn't show (a gateway key held to other models, an
// id like anthropic/claude-x that names a provider of the user's), isn't
// listed by it. A routing group the user made is listed by its own id as
// well (fast for group/fast, glm-5.2 for group/glm-5.2), where a
// request for that id reaches the group. A group magpie found is listed
// by the model (gpt-6.2-astra for auto/gpt-6.2-astra) the same way, which
// is the id a request for it reaches the group by, including when the
// group lists its members alone so no provider row carries that id.

// BareModelIDs says whether /v1/models lists models by their bare ids too.
func BareModelIDs() bool { return settings.Shared().BareModelIDs }

// BareOnly says whether /v1/models lists a model by its bare id in place
// of the provider/model ids it stands for (settings.BareModelIDsOnly;
// qQianye on X), not beside them. See OnlyBare.
func BareOnly() bool {
	s := settings.Shared()
	return s.BareModelIDs && s.BareModelIDsOnly
}

// SetBareModelIDs turns settings.BareModelIDs on, beside the provider/model
// ids, or off.
func SetBareModelIDs(on bool) error { return SetBareIDs(on, false) }

// SetBareIDs sets settings.BareModelIDs to on and BareModelIDsOnly to only
// (which off clears). The list is built for each request, so the next one
// has the bare ids, or goes without them; no agent's file names a model by
// one, so none is written again.
func SetBareIDs(on, only bool) error {
	only = on && only
	s := settings.Load()
	if s.BareModelIDs == on && s.BareModelIDsOnly == only {
		return nil
	}
	s.BareModelIDs, s.BareModelIDsOnly = on, only
	return settings.Save(s)
}

// Bare is a row a list adds for a bare id: ID, and Of, the index in the
// list of the entry a request for it reaches.
type Bare struct {
	ID string
	Of int
}

// BareEntries are the rows a list of shown adds for the bare ids, in the
// order the models first appear in shown, each once, then the routing
// groups no model's bare id reaches, by the id a request for them uses
// (groupBare).
func BareEntries(shown []Entry) []Bare {
	defer Hold()()
	ids := make(map[string]bool, len(shown))
	for _, e := range shown {
		ids[e.ID] = true
	}
	entries := providerEntries()
	all := groupsIn(entries)
	var out []Bare
	seen := map[string]bool{}
	reached := map[int]bool{}
	for _, e := range shown {
		m := e.Model
		if e.Group != "" || m == "" || seen[m] || ids[m] {
			continue
		}
		seen[m] = true
		if i := bareTarget(shown, all, entries, m); i >= 0 {
			out = append(out, Bare{ID: m, Of: i})
			reached[i] = true
		}
	}
	// then each routing group that no model's bare id reaches: one the
	// user made, by its own id, and one magpie found, by its model
	for j, e := range shown {
		if e.Group == "" || reached[j] {
			continue
		}
		if b, ok := groupBare(shown, all, entries, ids, seen, j); ok {
			seen[b.ID] = true
			out = append(out, b)
		}
	}
	return out
}

// groupBare is the Bare for the routing group at shown[j] (小八 on Discord:
// group/fast is long to type into a client), by the id a request for it
// reaches it by. A group the user made, listed as group/<id>, by its own
// id, fast. A group magpie found, listed as auto/<model>, by that model
// (gpt-6.2-astra for auto/gpt-6.2-astra; #7380, Sakugar): the id GroupFor
// takes to the group, which a provider's row of the model would have
// carried. Only where that is unambiguous: not by an id listed already or
// given to another row, and not when a request for it reaches some other
// row (a group the user made with the model's id wins GroupFor, so the
// auto group stays auto/<model>).
//
// A group named for a model a provider serves (glm-5.2) is listed so
// too: a request for glm-5.2 reaches the group, GroupFor before Resolve,
// whether or not the provider's glm-5.2 row is shown. When it is, that
// row's bare id reaches the group first (BareEntries' reached); when it
// isn't (the group lists its members alone, Group.Unlists; a provider
// kept for routing groups; the agent's hidden models), the group's own
// row carries the id, or Only left group/glm-5.2 (小八 on Discord). The
// same holds for a found group: while a member's row is shown it carries
// the model's bare id, and when none is, the auto/<model> row does.
func groupBare(shown []Entry, all []Group, entries []Entry, ids, seen map[string]bool, j int) (Bare, bool) {
	id := groupBareID(shown[j])
	if id == "" || strings.Contains(id, "/") || ids[id] || seen[id] {
		return Bare{}, false
	}
	// a request for id has to reach this row, not another group that
	// goes by the same model, and not a provider's model
	if bareTarget(shown, all, entries, id) != j {
		return Bare{}, false
	}
	return Bare{ID: id, Of: j}, true
}

// groupBareID is the id groupBare lists the group by: its own id for a
// group listed as group/<id>, the model for one listed as auto/<model>.
// "" when the row is not a group's to list that way.
func groupBareID(e Entry) string {
	if e.Group == "" {
		return ""
	}
	if model, ok := strings.CutPrefix(e.ID, AutoPrefix); ok && model != "" {
		return model
	}
	if !e.Named || e.ID != GroupPrefix+e.Group {
		return ""
	}
	return e.Group
}

// bareTarget is the index in shown of the entry a request for the bare id
// reaches: the routing group GroupFor gives it, else the provider's model
// Resolve does; -1 when shown has neither.
func bareTarget(shown []Entry, all []Group, entries []Entry, id string) int {
	if !strings.Contains(id, "/") {
		if gid, ok := groupForIn(all, id); ok {
			for i, e := range shown {
				if e.ID == gid || e.Group != "" && GroupPrefix+e.Group == gid {
					return i
				}
			}
			return -1
		}
	}
	p, model, ok := resolveIn(entries, id)
	if !ok {
		return -1
	}
	for i, e := range shown {
		if e.Group == "" && e.Provider.ID == p.ID && e.Model == model {
			return i
		}
	}
	return -1
}

// BareEntry is the entry of shown a client asking for a model by its bare
// id is told of (GET /v1/models/{id}), its ID the bare id.
func BareEntry(shown []Entry, id string) (Entry, bool) {
	for _, b := range BareEntries(shown) {
		if b.ID == id {
			e := shown[b.Of]
			e.ID = id
			return e, true
		}
	}
	return Entry{}, false
}

// OnlyBare is the list of shown with bare ids alone (BareOnly): each Bare
// of bares in the place of the first row it stands for, and the other
// rows it stands for left out. A Bare stands for the row a request for it
// reaches (a routing group found for the model, auto/glm-5.3, is that
// model's under another prefix), and for a provider's row of the same
// model when the request reaches that model: by that group, or by the
// provider first in the Providers order serving it by that id. A row it
// doesn't reach (a vendor/model id that names another provider of the
// user's, so a request for it goes there) and any other group's row stay,
// by their ids, so no model is lost from the list. Each row is its index
// in shown, and the bare id it is listed by, or "".
func OnlyBare(shown []Entry, bares []Bare) (rows []int, ids []string) {
	by := make(map[string]int, len(bares))
	reached := make(map[int]int, len(bares))
	for i, b := range bares {
		by[b.ID] = i
		reached[b.Of] = i
	}
	placed := make([]bool, len(bares))
	for j, e := range shown {
		i, ok := reached[j]
		if !ok && e.Group == "" {
			if i, ok = by[e.Model]; ok {
				to := shown[bares[i].Of]
				ok = to.Group != "" || to.Model == e.Model
			}
		}
		switch {
		case !ok:
			rows, ids = append(rows, j), append(ids, "")
		case !placed[i]:
			placed[i] = true
			rows, ids = append(rows, bares[i].Of), append(ids, bares[i].ID)
		}
	}
	return rows, ids
}
