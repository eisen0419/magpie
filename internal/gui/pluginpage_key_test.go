package gui

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/yetone/magpie/internal/settings"
)

// TestPluginPageKeysStayPut: two installs of one package (a folder and
// another copy, as a path and an @latest row do) used to take -2 in
// plugins.json's order, so an update that reordered the file renamed the
// page and a stored magpie.navHidden no longer matched. The keys follow
// the specs, and a plugin that is the only one of its name keeps the key
// it has always had. Which copy replaces a built-in stays the first in
// the file.
func TestPluginPageKeysStayPut(t *testing.T) {
	h := t.TempDir()
	t.Setenv("HOME", h)
	t.Setenv("USERPROFILE", h)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(h, ".config"))
	t.Setenv("XDG_CACHE_HOME", filepath.Join(h, ".cache"))
	t.Setenv("XDG_DATA_HOME", filepath.Join(h, ".local", "share"))
	t.Setenv("MAGPIE_BUN", "")
	t.Setenv("PATH", "")
	a := filepath.Join(h, "src", "a-glance")
	b := filepath.Join(h, "src", "b-glance")
	notes := filepath.Join(h, "src", "notes")
	copyTree(t, "testdata/page-plugin", a)
	copyTree(t, "testdata/page-plugin", b)
	copyTree(t, "testdata/panel-plugin", notes)
	pkg := []byte(`{"name":"magpie-usage-glance","version":"0.1.0","magpie":{"page":{"title":"Usage glance","file":"page/index.html","replaces":"usage"},"permissions":["usage:summary"]}}`)
	for _, dir := range []string{a, b} {
		if err := os.WriteFile(filepath.Join(dir, "package.json"), pkg, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(settings.Dir(), 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(rows []map[string]any) {
		t.Helper()
		raw, err := json.Marshal(map[string]any{"plugins": rows})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(settings.Dir(), "plugins.json"), raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	row := func(spec string, off bool) map[string]any {
		m := map[string]any{"spec": spec}
		if off {
			m["off"] = true
		}
		return m
	}
	// a sorts before b. The bare key is always a's; b is -2. notes, the
	// only plugin of its name, keeps the key a single install has.
	want := map[string]string{a: "plugin-magpie-usage-glance", b: "plugin-magpie-usage-glance-2"}
	var got map[string]string
	for _, order := range [][]string{{b, a, notes}, {a, notes, b}, {notes, b, a}} {
		rows := make([]map[string]any, len(order))
		for i, spec := range order {
			rows[i] = row(spec, false)
		}
		write(rows)
		pages, panels := windowParts()
		got = map[string]string{}
		var replacer string
		for _, p := range pages {
			got[p.Spec] = p.Key
			if p.Replaces != "" {
				if replacer != "" || p.Replaces != "usage" {
					t.Errorf("order %v: replaces %+v, already %s", order, p, replacer)
				}
				replacer = p.Spec
			}
		}
		firstGlance := a
		for _, spec := range order {
			if spec == a || spec == b {
				firstGlance = spec
				break
			}
		}
		if replacer != firstGlance {
			t.Errorf("order %v: %s replaces usage, want the first glance %s", order, replacer, firstGlance)
		}
		if got[a] != want[a] || got[b] != want[b] {
			t.Errorf("order %v: keys %v, want %v", order, got, want)
		}
		if len(panels) != 3 || panels[0].Key != "plugin-magpie-provider-notes-0" || panels[1].Key != "plugin-magpie-provider-notes-1" || panels[2].Key != "plugin-magpie-provider-notes-2" {
			t.Errorf("order %v: notes panels changed: %+v", order, panels)
		}
	}
	if got == nil {
		t.Fatal("no order ran")
	}
	// one install, whichever copy, keeps the bare key
	for _, spec := range []string{a, b} {
		write([]map[string]any{row(spec, false)})
		pages, _ := windowParts()
		if len(pages) != 1 || pages[0].Key != "plugin-magpie-usage-glance" || pages[0].Spec != spec {
			t.Errorf("alone %s: %+v", spec, pages)
		}
	}
	// the copy that is off takes no key, so the one left keeps the bare key
	write([]map[string]any{row(b, false), row(a, true)})
	pages, _ := windowParts()
	if len(pages) != 1 || pages[0].Spec != b || pages[0].Key != "plugin-magpie-usage-glance" {
		t.Errorf("only b on: %+v", pages)
	}
}
