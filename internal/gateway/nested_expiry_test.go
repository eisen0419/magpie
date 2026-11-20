package gateway

import (
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/access"
	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
)

// #7325 (shen-joe): a group left on Quota first ("") is weighed with its
// parent seat by seat. A group that sets its own routing is tried whole,
// even when that routing is the parent's. group/c is listed before group/a,
// and deepseek-v4.1-flash is not seat 0 of group/a (the 10-day sibling is).
//
// The flash seat is one Credits window, Of 500, Used 10, so it stays in the
// free band (under SpentShare, 98). RunsOut is the tranche left now: 100
// credits in 6h, then 50 in 12h, then 1000 in 3d. The gateway sees one
// RunsOut at a time, not the three packages together.
func TestNestedGroupSpendsTheSoonestQuotaAcrossItsGroups(t *testing.T) {
	now := time.Now()
	credits := func(resets, runsOut time.Duration, left float64) provider.Allowance {
		w := provider.Limit{
			Name: "Credits", Used: 10, Resets: now.Add(resets), Span: 30 * 24 * time.Hour,
			Of: 500, Amount: 50, Unit: "credits",
		}
		if runsOut > 0 {
			w.RunsOut, w.RunsOutLeft = now.Add(runsOut), left
		}
		return provider.Allowance{w}
	}
	const (
		flash   = "deepseek-v4.1-flash"
		sibling = "deepseek-v4.1-flash-1017"
		glm     = "glm-5.3"
	)
	setup := func(t *testing.T, aRouting string) (*Server, map[string]provider.Allowance) {
		t.Helper()
		fresh(t)
		share := map[string]provider.Allowance{
			"sk-late": credits(10*24*time.Hour, 0, 0),
			"sk-soon": credits(28*24*time.Hour, 6*time.Hour, 100),
			"sk-day":  credits(24*time.Hour, 0, 0),
		}
		keyWindows(t, share)
		vendor := &keyed{}
		serveOn(t, "late", "sk-late", []string{sibling}, vendor)
		serveOn(t, "soon", "sk-soon", []string{flash}, vendor)
		serveOn(t, "day", "sk-day", []string{glm}, vendor)
		for _, id := range []string{"late", "soon", "day"} {
			if err := provider.SetCountFree(id, true); err != nil {
				t.Fatal(err)
			}
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "A", Routing: aRouting,
			Members: []string{"late/" + sibling, "soon/" + flash},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{Name: "C", Members: []string{"day/" + glm}}); err != nil {
			t.Fatal(err)
		}
		return New(), share
	}
	models := func(t *testing.T, s *Server, session, routing string) []string {
		t.Helper()
		parent, _, ok := provider.FindGroup("group/parent")
		if !ok {
			if err := provider.SaveGroup(provider.Group{
				Name: "Parent", Members: []string{"group/c", "group/a"}, Routing: routing,
			}); err != nil {
				t.Fatal(err)
			}
		} else if parent.Routing != routing {
			parent.Routing = routing
			if err := provider.SaveGroup(parent); err != nil {
				t.Fatal(err)
			}
		}
		code, body := postAs(t, s, session, `{"model":"group/parent","messages":[{"role":"user","content":"hi"}]}`)
		if code != 200 {
			t.Fatalf("%d %s", code, body)
		}
		r := s.trace.routes[len(s.trace.routes)-1]
		var got []string
		for _, w := range r.Order {
			got = append(got, w.Model)
		}
		return got
	}
	soonest := []string{flash, glm, sibling}
	dayFirst := []string{glm, flash, sibling}

	t.Run("expiring", func(t *testing.T) {
		s, share := setup(t, "")
		// 100 of 500 lost in 6h. Expiry is the deadline, so the flash
		// beats the 1-day glm, which beats the 10-day sibling.
		if got := models(t, s, "e1", provider.FreeExpiring); !slices.Equal(got, soonest) {
			t.Fatalf("6h: %v, want %v", got, soonest)
		}
		// the 6h tranche is gone; 50 credits remain, lost in 12h, Used
		// still 10 (free band). 12h still beats 1d.
		share["sk-soon"] = credits(28*24*time.Hour, 12*time.Hour, 50)
		if got := models(t, s, "e2", provider.FreeExpiring); !slices.Equal(got, soonest) {
			t.Fatalf("12h: %v, want %v", got, soonest)
		}
		// what remains is the 3d package (1000 credits, share capped at
		// the window). The 1-day glm is now the soonest.
		share["sk-soon"] = credits(28*24*time.Hour, 3*24*time.Hour, 1000)
		if got := models(t, s, "e3", provider.FreeExpiring); !slices.Equal(got, dayFirst) {
			t.Fatalf("3d: %v, want %v", got, dayFirst)
		}
	})

	t.Run("pace reported amounts", func(t *testing.T) {
		// Pace ranks by loss per hour, not by the earliest deadline.
		// Day seat: (100-10)/24 = 3.75/h. Sibling: 90/(10*24) = 0.375/h.
		// Flash, RunsOutLeft/Of capped at 100, per hour until RunsOut:
		//   100/500 in 6h = 20/6 ≈ 3.33/h, under the day (3.33 < 0.9×3.75)
		//   50/500 in 12h = 10/12 ≈ 0.83/h, still under the day
		//   1000/500 in 3d, share capped at 100, = 100/72 ≈ 1.39/h, still
		//   under the day
		// So the day leads every reported tranche. Flattening does not
		// change that comparison; it keeps the 10-day sibling from being
		// a separate question only when the flash's own rate leads.
		s, share := setup(t, "")
		if got := models(t, s, "p1", provider.Pace); !slices.Equal(got, dayFirst) {
			t.Fatalf("100/500 in 6h (3.33/h): %v, want %v", got, dayFirst)
		}
		share["sk-soon"] = credits(28*24*time.Hour, 12*time.Hour, 50)
		if got := models(t, s, "p2", provider.Pace); !slices.Equal(got, dayFirst) {
			t.Fatalf("50/500 in 12h (0.83/h): %v, want %v", got, dayFirst)
		}
		share["sk-soon"] = credits(28*24*time.Hour, 3*24*time.Hour, 1000)
		if got := models(t, s, "p3", provider.Pace); !slices.Equal(got, dayFirst) {
			t.Fatalf("1000/500 in 3d (1.39/h): %v, want %v", got, dayFirst)
		}
	})

	t.Run("pace higher hourly loss", func(t *testing.T) {
		// 200 of 500 in 6h is 40/6 ≈ 6.67/h, over the day's 3.75/h
		// (3.75 < 0.9×6.67). Flattened, the flash leads and the 10-day
		// sibling (0.375/h) stays behind the glm. Tried whole, group/a
		// would ride that sibling ahead of the glm with the flash.
		s, share := setup(t, "")
		share["sk-soon"] = credits(28*24*time.Hour, 6*time.Hour, 200)
		if got := models(t, s, "fast", provider.Pace); !slices.Equal(got, soonest) {
			t.Fatalf("200/500 in 6h (6.67/h): %v, want %v", got, soonest)
		}
	})

	t.Run("free keeps listed order", func(t *testing.T) {
		s, _ := setup(t, "")
		// Free first spends free seats in listed order. It does not sort
		// by expiry, so the 6-hour flash is last: group/c, then group/a's
		// 10-day seat, then the flash. 免费优先 was not an expiry bug.
		got := models(t, s, "free", provider.FreeFirst)
		want := []string{glm, sibling, flash}
		if !slices.Equal(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
	})

	t.Run("a group that sets its own routing stays whole", func(t *testing.T) {
		s, _ := setup(t, provider.Ordered)
		// a is In order, 10-day seat first, so the 6-hour seat stays behind it.
		// c, left on Quota first, is a seat of the parent and its day comes first.
		got := models(t, s, "order", provider.FreeExpiring)
		want := []string{glm, sibling, flash}
		if !slices.Equal(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
	})

	t.Run("expiring inside expiring stays whole", func(t *testing.T) {
		s, _ := setup(t, provider.FreeExpiring)
		// a set its own Free quota by expiry, the same as the parent, so
		// it is one unit: its own expiry puts the 6h flash ahead of the
		// 10-day sibling, and that sibling rides with it ahead of the glm.
		// Flattened, the glm would sit between them.
		got := models(t, s, "same", provider.FreeExpiring)
		want := []string{flash, sibling, glm}
		if !slices.Equal(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
	})
}

// Free first orders free seats by listed order, not by expiry. Flattening
// still matters when a default child has a paid tail: tried whole, that
// tail rides ahead of a sibling child's free seat. group/a is listed
// before group/c here, which is the order that shows it.
func TestNestedGroupFreeKeepsAPaidTailBehindASibling(t *testing.T) {
	fresh(t)
	vendor := &keyed{}
	serveOn(t, "soon", "sk-soon", []string{"deepseek-v4.1-flash"}, vendor)
	serveOn(t, "paid", "sk-paid", []string{"claude-paid"}, vendor)
	serveOn(t, "day", "sk-day", []string{"glm-5.3"}, vendor)
	for _, id := range []string{"soon", "day"} {
		if err := provider.SetCountFree(id, true); err != nil {
			t.Fatal(err)
		}
	}
	if err := provider.SetModelPrice("paid/claude-paid", &catalog.Price{Input: 1, Output: 2}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name:    "A",
		Members: []string{"soon/deepseek-v4.1-flash", "paid/claude-paid"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{Name: "C", Members: []string{"day/glm-5.3"}}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		Name: "Parent", Members: []string{"group/a", "group/c"}, Routing: provider.FreeFirst,
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	code, body := postAs(t, s, "paidtail", `{"model":"group/parent","messages":[{"role":"user","content":"hi"}]}`)
	if code != 200 {
		t.Fatalf("%d %s", code, body)
	}
	var got []string
	for _, w := range s.trace.routes[len(s.trace.routes)-1].Order {
		got = append(got, w.Model)
	}
	want := []string{"deepseek-v4.1-flash", "glm-5.3", "claude-paid"}
	if !slices.Equal(got, want) {
		t.Fatalf("order %v, want %v", got, want)
	}
}

// In turn inside In turn stays two units. The outer group turns among the
// child groups; each child turns among its own seats. Flattened, the second
// request would move to the second seat of one pool (b, then c, d, a)
// instead of to the other child.
func TestRotateInsideRotateStaysWhole(t *testing.T) {
	fresh(t)
	for _, id := range []string{"ra", "rb", "rc", "rd"} {
		serveOn(t, id, "k"+id, []string{"m"}, &keyed{})
	}
	if err := provider.SaveGroup(provider.Group{
		ID: "rota7325", Name: "Rota", Members: []string{"ra/m", "rb/m"}, Routing: provider.Rotate,
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		ID: "rotb7325", Name: "Rotb", Members: []string{"rc/m", "rd/m"}, Routing: provider.Rotate,
	}); err != nil {
		t.Fatal(err)
	}
	if err := provider.SaveGroup(provider.Group{
		ID: "rotout7325", Name: "Rotout",
		Members: []string{"group/rota7325", "group/rotb7325"}, Routing: provider.Rotate,
	}); err != nil {
		t.Fatal(err)
	}
	s := New()
	order := func() string {
		r := s.trace.routes[len(s.trace.routes)-1]
		var out []string
		for _, w := range r.Order {
			out = append(out, w.Provider+"["+strings.Join(w.Via, ",")+"]")
		}
		return strings.Join(out, " ")
	}
	want := []string{
		"ra[rota7325] rb[rota7325] rc[rotb7325] rd[rotb7325]",
		"rd[rotb7325] rc[rotb7325] rb[rota7325] ra[rota7325]",
	}
	for i, w := range want {
		code, body := postAs(t, s, "rot"+string(rune('a'+i)), `{"model":"group/rotout7325","messages":[{"role":"user","content":"hi"}]}`)
		if code != 200 {
			t.Fatalf("%d %s", code, body)
		}
		if o := order(); o != w {
			t.Fatalf("request %d: %s, want %s", i+1, o, w)
		}
	}
}

// A group in the group in the group, each left on Quota first, joins the
// same pool: the soonest seat is the deepest one's. The same holds for
// Use before expiry, which here ranks the 6h reset (90/6 = 15/h) ahead of
// the day (3.75/h) and that ahead of the 10-day seat (0.375/h).
func TestNestedGroupThreeDeepSpendsTheSoonest(t *testing.T) {
	for _, routing := range []string{provider.FreeExpiring, provider.Pace} {
		t.Run(routing, func(t *testing.T) {
			now := time.Now()
			window := func(in time.Duration) provider.Allowance {
				return provider.Allowance{{Name: "Credits", Used: 10, Resets: now.Add(in), Span: 30 * 24 * time.Hour, Of: 500, Amount: 50, Unit: "credits"}}
			}
			fresh(t)
			keyWindows(t, map[string]provider.Allowance{
				"sk-soon": window(6 * time.Hour),
				"sk-day":  window(24 * time.Hour),
				"sk-late": window(10 * 24 * time.Hour),
			})
			vendor := &keyed{}
			serveOn(t, "soon", "sk-soon", []string{"deepseek-v4.1-flash"}, vendor)
			serveOn(t, "day", "sk-day", []string{"glm-5.3"}, vendor)
			serveOn(t, "late", "sk-late", []string{"deepseek-v4.1-flash-1017"}, vendor)
			for _, id := range []string{"soon", "day", "late"} {
				if err := provider.SetCountFree(id, true); err != nil {
					t.Fatal(err)
				}
			}
			if err := provider.SaveGroup(provider.Group{Name: "Inner", Members: []string{"soon/deepseek-v4.1-flash"}}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SaveGroup(provider.Group{Name: "A", Members: []string{"group/inner", "late/deepseek-v4.1-flash-1017"}}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SaveGroup(provider.Group{Name: "C", Members: []string{"day/glm-5.3"}}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SaveGroup(provider.Group{
				Name: "Parent", Members: []string{"group/c", "group/a"}, Routing: routing,
			}); err != nil {
				t.Fatal(err)
			}
			s := New()
			code, body := postAs(t, s, "deep", `{"model":"group/parent","messages":[{"role":"user","content":"hi"}]}`)
			if code != 200 {
				t.Fatalf("%d %s", code, body)
			}
			var got []string
			for _, w := range s.trace.routes[len(s.trace.routes)-1].Order {
				got = append(got, w.Model)
			}
			want := []string{"deepseek-v4.1-flash", "glm-5.3", "deepseek-v4.1-flash-1017"}
			if !slices.Equal(got, want) {
				t.Fatalf("order %v, want %v", got, want)
			}
		})
	}
}

// A gateway key that changes the outer group's routing applies to default
// nested children at plan time and in trace Subs, not only against the saved
// parent. A key pooling a non-pooling parent joins default children; a key
// changing a pooling parent updates Subs and ordering to the key's routing;
// a key setting a non-pooling routing leaves default children whole on Quota
// first (""); an explicit child keeps its own routing either way; a Manual
// parent keeps its pick.
func TestKeyRoutingWithNestedGroups(t *testing.T) {
	now := time.Now()
	credits := func(resets, runsOut time.Duration, left float64) provider.Allowance {
		w := provider.Limit{
			Name: "Credits", Used: 10, Resets: now.Add(resets), Span: 30 * 24 * time.Hour,
			Of: 500, Amount: 50, Unit: "credits",
		}
		if runsOut > 0 {
			w.RunsOut, w.RunsOutLeft = now.Add(runsOut), left
		}
		return provider.Allowance{w}
	}
	run := func(t *testing.T, parentRouting, childARouting, keyRouting, pick string, soonLeft float64) ([]string, map[string]string) {
		t.Helper()
		fresh(t)
		keyWindows(t, map[string]provider.Allowance{
			"sk-soon": credits(30*24*time.Hour, 6*time.Hour, soonLeft),
			"sk-late": credits(10*24*time.Hour, 0, 0),
			"sk-day":  credits(24*time.Hour, 0, 0),
		})
		vendor := &keyed{}
		serveOn(t, "soon", "sk-soon", []string{"deepseek-v4.1-flash"}, vendor)
		serveOn(t, "late", "sk-late", []string{"deepseek-v4.1-flash-1017"}, vendor)
		serveOn(t, "day", "sk-day", []string{"glm-5.3"}, vendor)
		for _, id := range []string{"soon", "late", "day"} {
			if err := provider.SetCountFree(id, true); err != nil {
				t.Fatal(err)
			}
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "A", Members: []string{"late/deepseek-v4.1-flash-1017", "soon/deepseek-v4.1-flash"}, Routing: childARouting,
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "C", Members: []string{"day/glm-5.3"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "Parent", Members: []string{"group/c", "group/a"}, Routing: parentRouting, Pick: pick,
		}); err != nil {
			t.Fatal(err)
		}
		keys, secrets := newCaller(t, "Key")
		if _, err := access.Update("routing-key", access.Change{Key: keys[0].ID, Routing: keyRouting}); err != nil {
			t.Fatal(err)
		}
		s := New()
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"group/parent","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+secrets[0])
		req.RemoteAddr = "192.168.1.20:5000"
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body.String())
		}
		last := s.trace.routes[len(s.trace.routes)-1]
		var order []string
		for _, w := range last.Order {
			order = append(order, w.Model)
		}
		subs := map[string]string{}
		if last.Group != nil {
			for _, sub := range last.Group.Subs {
				subs[sub.ID] = sub.Routing
			}
		}
		return order, subs
	}

	for _, savedParent := range []string{"", provider.Ordered, provider.Weighted} {
		name := savedParent
		if name == "" {
			name = "smart"
		}
		t.Run("key pools saved "+name+" parent", func(t *testing.T) {
			got, subs := run(t, savedParent, "", provider.FreeExpiring, "", 100)
			want := []string{"deepseek-v4.1-flash", "glm-5.3", "deepseek-v4.1-flash-1017"}
			if !slices.Equal(got, want) {
				t.Fatalf("order %v, want %v", got, want)
			}
			if subs["a"] != provider.FreeExpiring || subs["c"] != provider.FreeExpiring {
				t.Fatalf("Subs routing = %v, want a=%q c=%q", subs, provider.FreeExpiring, provider.FreeExpiring)
			}
		})
	}

	t.Run("key changes pooling parent to another pooling strategy", func(t *testing.T) {
		// On 100 of 500 in 6h (3.33/h < day 3.75/h), Pace leads with glm-5.3;
		// saved parent FreeExpiring would lead with deepseek-v4.1-flash.
		got, subs := run(t, provider.FreeExpiring, "", provider.Pace, "", 100)
		want := []string{"glm-5.3", "deepseek-v4.1-flash", "deepseek-v4.1-flash-1017"}
		if !slices.Equal(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
		if subs["a"] != provider.Pace || subs["c"] != provider.Pace {
			t.Fatalf("Subs routing = %v, want a=%q c=%q", subs, provider.Pace, provider.Pace)
		}
		// On 200 of 500 in 6h (6.67/h > day 3.75/h), Pace puts flash first and
		// keeps the 10-day sibling behind the day (default child joins Pace pool).
		got, subs = run(t, provider.FreeExpiring, "", provider.Pace, "", 200)
		want = []string{"deepseek-v4.1-flash", "glm-5.3", "deepseek-v4.1-flash-1017"}
		if !slices.Equal(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
		if subs["a"] != provider.Pace || subs["c"] != provider.Pace {
			t.Fatalf("Subs routing = %v, want a=%q c=%q", subs, provider.Pace, provider.Pace)
		}
	})

	for _, keyNonPool := range []struct {
		name, key string
		wantOrder []string
	}{
		// When late is paid with tier-0 Used (5) and soon is free with tier-1 Used (92,
		// under SpentShare=98), FreeExpiring puts free soon first while Quota first ("")
		// puts tier-0 late before tier-1 soon.
		{"order", provider.Ordered, []string{"glm-5.3", "deepseek-v4.1-flash-1017", "deepseek-v4.1-flash"}},
		{"smart", access.RoutingSmart, []string{"glm-5.3", "deepseek-v4.1-flash-1017", "deepseek-v4.1-flash"}},
		{"weight", provider.Weighted, []string{"glm-5.3", "deepseek-v4.1-flash-1017", "deepseek-v4.1-flash"}},
	} {
		t.Run("key changes pooling parent to non-pooling "+keyNonPool.name, func(t *testing.T) {
			fresh(t)
			keyWindows(t, map[string]provider.Allowance{
				"sk-soon": {provider.Limit{Name: "Credits", Used: 92, Resets: now.Add(30 * 24 * time.Hour), Span: 30 * 24 * time.Hour, Of: 500, Amount: 50, Unit: "credits"}},
				"sk-late": {provider.Limit{Name: "Credits", Used: 5, Resets: now.Add(10 * 24 * time.Hour), Span: 30 * 24 * time.Hour, Of: 500, Amount: 50, Unit: "credits"}},
				"sk-day":  {provider.Limit{Name: "Credits", Used: 30, Resets: now.Add(24 * time.Hour), Span: 30 * 24 * time.Hour, Of: 500, Amount: 50, Unit: "credits"}},
			})
			vendor := &keyed{}
			serveOn(t, "soon", "sk-soon", []string{"deepseek-v4.1-flash"}, vendor)
			serveOn(t, "late", "sk-late", []string{"deepseek-v4.1-flash-1017"}, vendor)
			serveOn(t, "day", "sk-day", []string{"glm-5.3"}, vendor)
			if err := provider.SetCountFree("soon", true); err != nil {
				t.Fatal(err)
			}
			if err := provider.SetCountFree("day", true); err != nil {
				t.Fatal(err)
			}
			if err := provider.SaveGroup(provider.Group{
				Name: "A", Members: []string{"late/deepseek-v4.1-flash-1017", "soon/deepseek-v4.1-flash"},
			}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SaveGroup(provider.Group{
				Name: "C", Members: []string{"day/glm-5.3"},
			}); err != nil {
				t.Fatal(err)
			}
			if err := provider.SaveGroup(provider.Group{
				Name: "Parent", Members: []string{"group/c", "group/a"}, Routing: provider.FreeExpiring,
			}); err != nil {
				t.Fatal(err)
			}
			keys, secrets := newCaller(t, "Key")
			if _, err := access.Update("routing-key", access.Change{Key: keys[0].ID, Routing: keyNonPool.key}); err != nil {
				t.Fatal(err)
			}
			s := New()
			rec := httptest.NewRecorder()
			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"group/parent","messages":[{"role":"user","content":"hi"}]}`))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer "+secrets[0])
			req.RemoteAddr = "192.168.1.20:5000"
			s.Handler().ServeHTTP(rec, req)
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body.String())
			}
			last := s.trace.routes[len(s.trace.routes)-1]
			var got []string
			for _, w := range last.Order {
				got = append(got, w.Model)
			}
			subs := map[string]string{}
			for _, sub := range last.Group.Subs {
				subs[sub.ID] = sub.Routing
			}
			if !slices.Equal(got, keyNonPool.wantOrder) {
				t.Fatalf("order %v, want %v (group=%+v)", got, keyNonPool.wantOrder, last.Group)
			}
			if subs["a"] != "" || subs["c"] != "" {
				t.Fatalf("Subs routing = %v, want a=\"\" c=\"\"", subs)
			}
		})
	}

	t.Run("explicit child stays whole under key pooling parent", func(t *testing.T) {
		got, subs := run(t, provider.Ordered, provider.FreeExpiring, provider.FreeExpiring, "", 100)
		want := []string{"deepseek-v4.1-flash", "deepseek-v4.1-flash-1017", "glm-5.3"}
		if !slices.Equal(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
		if subs["a"] != provider.FreeExpiring || subs["c"] != provider.FreeExpiring {
			t.Fatalf("Subs routing = %v, want a=%q c=%q", subs, provider.FreeExpiring, provider.FreeExpiring)
		}
	})

	t.Run("three-deep default joins key pool and middle explicit stays whole", func(t *testing.T) {
		fresh(t)
		keyWindows(t, map[string]provider.Allowance{
			"sk-soon": credits(30*24*time.Hour, 6*time.Hour, 100),
			"sk-late": credits(10*24*time.Hour, 0, 0),
			"sk-day":  credits(24*time.Hour, 0, 0),
		})
		vendor := &keyed{}
		serveOn(t, "soon", "sk-soon", []string{"deepseek-v4.1-flash"}, vendor)
		serveOn(t, "late", "sk-late", []string{"deepseek-v4.1-flash-1017"}, vendor)
		serveOn(t, "day", "sk-day", []string{"glm-5.3"}, vendor)
		for _, id := range []string{"soon", "late", "day"} {
			if err := provider.SetCountFree(id, true); err != nil {
				t.Fatal(err)
			}
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "Inner", Members: []string{"late/deepseek-v4.1-flash-1017", "soon/deepseek-v4.1-flash"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "Mid", Members: []string{"group/inner"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "Parent", Members: []string{"day/glm-5.3", "group/mid"}, Routing: provider.Ordered,
		}); err != nil {
			t.Fatal(err)
		}
		keys, secrets := newCaller(t, "Key")
		if _, err := access.Update("routing-key", access.Change{Key: keys[0].ID, Routing: provider.FreeExpiring}); err != nil {
			t.Fatal(err)
		}
		ask := func(msg string) ([]string, map[string]string) {
			s := New()
			rec := httptest.NewRecorder()
			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"group/parent","messages":[{"role":"user","content":"`+msg+`"}]}`))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer "+secrets[0])
			req.RemoteAddr = "192.168.1.20:5000"
			s.Handler().ServeHTTP(rec, req)
			if rec.Code != 200 {
				t.Fatalf("%d %s", rec.Code, rec.Body.String())
			}
			last := s.trace.routes[len(s.trace.routes)-1]
			var order []string
			for _, w := range last.Order {
				order = append(order, w.Model)
			}
			subs := map[string]string{}
			for _, sub := range last.Group.Subs {
				subs[sub.ID] = sub.Routing
			}
			return order, subs
		}
		got, subs := ask("three-deep-default")
		want := []string{"deepseek-v4.1-flash", "glm-5.3", "deepseek-v4.1-flash-1017"}
		if !slices.Equal(got, want) {
			t.Fatalf("3-deep default under key expiring: order %v, want %v", got, want)
		}
		if subs["mid"] != provider.FreeExpiring || subs["inner"] != provider.FreeExpiring {
			t.Fatalf("3-deep default Subs = %v, want mid=%q inner=%q", subs, provider.FreeExpiring, provider.FreeExpiring)
		}

		// Middle group set explicitly to FreeExpiring stays one unit under a key
		// pooling with Pace, while its default child Inner still joins Mid's FreeExpiring.
		if err := provider.SaveGroup(provider.Group{
			ID: "mid", Name: "Mid", Members: []string{"group/inner"}, Routing: provider.FreeExpiring,
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := access.Update("routing-key", access.Change{Key: keys[0].ID, Routing: provider.Pace}); err != nil {
			t.Fatal(err)
		}
		got, subs = ask("three-deep-middle-explicit")
		// Mid's head is flash (6h expiry); under Pace (3.33/h < day 3.75/h),
		// day leads, then Mid as one unit (flash then late).
		want = []string{"glm-5.3", "deepseek-v4.1-flash", "deepseek-v4.1-flash-1017"}
		if !slices.Equal(got, want) {
			t.Fatalf("3-deep middle explicit under key pace: order %v, want %v, subs=%v", got, want, subs)
		}
		if subs["mid"] != provider.FreeExpiring || subs["inner"] != provider.FreeExpiring {
			t.Fatalf("3-deep middle explicit Subs = %v, want mid=%q inner=%q", subs, provider.FreeExpiring, provider.FreeExpiring)
		}
	})

	t.Run("manual parent keeps its pick and child routing", func(t *testing.T) {
		got, subs := run(t, provider.Manual, "", provider.FreeExpiring, "group/a", 100)
		// Under Quota first (""), soon (6h RunsOut) leads late (10d) inside group/a,
		// while Manual keeps only group/a and leaves Subs["a"] as "".
		want := []string{"deepseek-v4.1-flash", "deepseek-v4.1-flash-1017"}
		if !slices.Equal(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
		if subs["a"] != "" {
			t.Fatalf("Subs routing = %v, want a=\"\"", subs)
		}
	})

	t.Run("embeddings trace group and Subs follow key routing", func(t *testing.T) {
		fresh(t)
		keyWindows(t, map[string]provider.Allowance{
			"sk-soon": credits(30*24*time.Hour, 6*time.Hour, 100),
			"sk-late": credits(10*24*time.Hour, 0, 0),
			"sk-day":  credits(24*time.Hour, 0, 0),
		})
		vendor := &keyed{}
		serveOn(t, "soon", "sk-soon", []string{"deepseek-v4.1-flash"}, vendor)
		serveOn(t, "late", "sk-late", []string{"deepseek-v4.1-flash-1017"}, vendor)
		serveOn(t, "day", "sk-day", []string{"glm-5.3"}, vendor)
		for _, id := range []string{"soon", "late", "day"} {
			if err := provider.SetCountFree(id, true); err != nil {
				t.Fatal(err)
			}
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "A", Members: []string{"late/deepseek-v4.1-flash-1017", "soon/deepseek-v4.1-flash"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "C", Members: []string{"day/glm-5.3"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := provider.SaveGroup(provider.Group{
			Name: "Parent", Members: []string{"group/c", "group/a"}, Routing: provider.Ordered,
		}); err != nil {
			t.Fatal(err)
		}
		keys, secrets := newCaller(t, "Key")
		if _, err := access.Update("routing-key", access.Change{Key: keys[0].ID, Routing: provider.FreeExpiring}); err != nil {
			t.Fatal(err)
		}
		s := New()
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{"model":"group/parent","input":"hi"}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+secrets[0])
		req.RemoteAddr = "192.168.1.20:5000"
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body.String())
		}
		last := s.trace.routes[len(s.trace.routes)-1]
		if last.Group == nil || last.Group.Routing != provider.FreeExpiring {
			t.Fatalf("embeddings trace Group = %+v, want Routing=%q", last.Group, provider.FreeExpiring)
		}
		subs := map[string]string{}
		for _, sub := range last.Group.Subs {
			subs[sub.ID] = sub.Routing
		}
		if subs["a"] != provider.FreeExpiring || subs["c"] != provider.FreeExpiring {
			t.Fatalf("embeddings trace Subs = %v, want a=%q c=%q", subs, provider.FreeExpiring, provider.FreeExpiring)
		}
		if len(vendor.tried) == 0 || vendor.tried[0] != "sk-soon" {
			t.Fatalf("embeddings vendor first key = %v, want sk-soon", vendor.tried)
		}
	})
}
