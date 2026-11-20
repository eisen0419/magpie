package library

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/gateway"
)

// A Library server whose URL is this magpie's own hub is not a server the
// hub gives out. Listed, it is asked for tools, and that ask is the hub
// again (#7305). Loopback (localhost, 127.0.0.1, ::1), the host
// MAGPIE_ADDR pins, the host MAGPIE_PUBLIC_URL names, and this gateway's
// port, are this hub. Another host, or another port, is another magpie.
// The server's name is not what decides: only a server called magpie keeps
// the agents on their own entries.
func TestHubLeavesOutItsOwnHub(t *testing.T) {
	h := hubSandbox(t)
	write(t, filepath.Join(h, ".claude.json"), `{"mcpServers": {}}`)
	t.Setenv("MAGPIE_ADDR", "198.51.100.8:3425")
	t.Setenv("MAGPIE_PUBLIC_URL", "http://nas.lan:3425")
	if gateway.Port() != "3425" {
		t.Fatalf("port %s", gateway.Port())
	}
	// 192.0.2.20 is TEST-NET-1, not an address of this computer.
	const far = "http://192.0.2.20:3425/mcp/magpie/hub"
	keep := []Server{
		{Name: "far", Transport: "http", URL: far},
		{Name: "otherport", Transport: "http", URL: "http://127.0.0.1:9999/mcp/magpie/hub"},
		{Name: "pinnedelse", Transport: "http", URL: "http://198.51.100.8:9999/mcp/magpie/hub"},
		{Name: "publicelse", Transport: "http", URL: "http://nas.lan:443/mcp/magpie/hub"},
		{Name: "hubbub", Transport: "http", URL: "http://127.0.0.1:3425/mcp/magpie/hubbub"},
		{Name: "plain", Transport: "http", URL: "https://example.com/mcp"},
		{Name: "fs", Transport: "stdio", Command: "npx", Args: []string{"fs"}},
	}
	drop := []Server{
		{Name: "loop", Transport: "http", URL: "http://127.0.0.1:3425/mcp/magpie/hub/agy"},
		{Name: "all", Transport: "http", URL: "http://127.0.0.1:3425/mcp/magpie/hub"},
		{Name: "slash", Transport: "http", URL: "http://127.0.0.1:3425/mcp/magpie/hub/"},
		{Name: "query", Transport: "http", URL: "http://127.0.0.1:3425/mcp/magpie/hub?x=1"},
		{Name: "local", Transport: "http", URL: "http://localhost:3425/mcp/magpie/hub"},
		{Name: "caps", Transport: "http", URL: "http://LocalHost:3425/mcp/magpie/hub/"},
		{Name: "v6", Transport: "http", URL: "http://[::1]:3425/mcp/magpie/hub/claude"},
		{Name: "pinned", Transport: "http", URL: "http://198.51.100.8:3425/mcp/magpie/hub"},
		{Name: "public", Transport: "http", URL: "http://nas.lan:3425/mcp/magpie/hub/opencode"},
		{Name: "sseown", Transport: "sse", URL: "http://127.0.0.1:3425/mcp/magpie/hub"},
		{Name: "unspec", Transport: "http", URL: "http://0.0.0.0:3425/mcp/magpie/hub"},
	}
	for i := range keep {
		keep[i].Agents = []string{"claude"}
		ok(t)(SaveServer("", keep[i]))
	}
	for i := range drop {
		drop[i].Agents = []string{"claude"}
		ok(t)(SaveServer("", drop[i]))
	}
	ok(t)(SetMCPHub(true))

	var got []string
	for _, s := range (theHub{}).Servers("claude") {
		got = append(got, s.Name)
	}
	slices.Sort(got)
	var want []string
	for _, s := range keep {
		want = append(want, s.Name)
	}
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("hub servers %v, want %v", got, want)
	}
	if hubOf(t, "claude") != "on" {
		t.Fatalf("claude's hub %q, want on: a server pointed at this hub under another name is not one called magpie", hubOf(t, "claude"))
	}
}

// A server the library calls magpie still keeps every agent's own entries,
// including when its URL is this hub. The name is what is taken.
func TestHubNameTakenStillWhenTheURLIsThisHub(t *testing.T) {
	h := hubSandbox(t)
	t.Setenv("MAGPIE_ADDR", "127.0.0.1:3425")
	t.Setenv("MAGPIE_PUBLIC_URL", "")
	write(t, filepath.Join(h, ".claude.json"), `{"mcpServers": {}}`)
	ok(t)(SaveServer("", Server{Name: "magpie", Transport: "http", URL: "http://127.0.0.1:3425/mcp/magpie/hub/agy", Agents: []string{"claude"}}))
	ok(t)(SetMCPHub(true))
	if hubOf(t, "claude") != hubTaken {
		t.Fatalf("claude's hub %q, want %q", hubOf(t, "claude"), hubTaken)
	}
	e, err := targetByID("claude").MCP.entries()
	if err != nil {
		t.Fatal(err)
	}
	if u, _ := e["magpie"]["url"].(string); u != "http://127.0.0.1:3425/mcp/magpie/hub/agy" {
		t.Fatalf("the library's magpie was replaced: %v", e["magpie"])
	}
}

// tools/list on the hub does not ask a server at this hub, so the names
// don't nest. It does ask a hub at another port, and a URL on this port
// that is not the hub.
func TestHubDoesNotAskItsOwnHub(t *testing.T) {
	hubSandbox(t)
	t.Setenv("MAGPIE_PUBLIC_URL", "")
	var selfHits, bubHits, farHits atomic.Int32
	self := httptest.NewServer(countHub(t, &selfHits, &bubHits))
	t.Cleanup(self.Close)
	far := httptest.NewServer(countHub(t, &farHits, nil))
	t.Cleanup(far.Close)
	u, err := url.Parse(self.URL)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("MAGPIE_ADDR", u.Host)
	fake := fakeStdio(t, "ok", nil)
	fake.Agents = []string{"claude"}
	ok(t)(SaveServer("", *fake))
	ok(t)(SaveServer("", Server{Name: "self", Transport: "http", URL: self.URL + gateway.HubMCPPath + "agy", Agents: []string{"claude"}}))
	ok(t)(SaveServer("", Server{Name: "all", Transport: "http", URL: self.URL + gateway.HubMCPAll, Agents: []string{"claude"}}))
	ok(t)(SaveServer("", Server{Name: "hubbub", Transport: "http", URL: self.URL + "/mcp/magpie/hubbub", Agents: []string{"claude"}}))
	ok(t)(SaveServer("", Server{Name: "remote", Transport: "http", URL: far.URL + gateway.HubMCPAll, Agents: []string{"claude"}}))
	ok(t)(SetMCPHub(true))

	got := hubNames((theHub{}).Tools(context.Background(), "claude"))
	for _, name := range []string{"self/a", "all/a"} {
		if slices.Contains(got, name) {
			t.Errorf("listed %s from this hub: %v", name, got)
		}
	}
	for _, name := range []string{"fake/a", "hubbub/a", "remote/a"} {
		if !slices.Contains(got, name) {
			t.Errorf("missing %s: %v", name, got)
		}
	}
	if _, err := (theHub{}).Call(context.Background(), "claude", "self", "a", json.RawMessage(`{}`)); err == nil {
		t.Fatal("the hub ran a server pointed at itself")
	}
	if selfHits.Load() != 0 {
		t.Errorf("this hub was asked %d times", selfHits.Load())
	}
	if bubHits.Load() == 0 || farHits.Load() == 0 {
		t.Errorf("hubbub hits %d, remote hits %d", bubHits.Load(), farHits.Load())
	}
}

// X-Magpie-Router used to drop only a server with Bridge set. A plain
// Library entry at the hub the request came in on is the same loop, under
// a host that isn't loopback and isn't this gateway's configured port: the
// dial set the header, and the request came back here. A hub at another
// host:port is still listed and run. A bridge server is still left out.
func TestHubFromRouterDropsAPlainHub(t *testing.T) {
	hubSandbox(t)
	t.Setenv("MAGPIE_ADDR", "127.0.0.1:1")
	t.Setenv("MAGPIE_PUBLIC_URL", "")
	var aliasHits, farHits atomic.Int32
	alias := httptest.NewServer(countHub(t, &aliasHits, nil))
	t.Cleanup(alias.Close)
	farSrv := httptest.NewServer(countHub(t, &farHits, nil))
	t.Cleanup(farSrv.Close)
	u, err := url.Parse(alias.URL)
	if err != nil {
		t.Fatal(err)
	}
	fake := fakeStdio(t, "ok", nil)
	fake.Agents = []string{"claude"}
	ok(t)(SaveServer("", *fake))
	ok(t)(SaveServer("", Server{Name: "alias", Transport: "http", URL: alias.URL + gateway.HubMCPPath + "agy", Agents: []string{"claude"}}))
	ok(t)(SaveServer("", Server{Name: "far", Transport: "http", URL: farSrv.URL + gateway.HubMCPAll, Agents: []string{"claude"}}))
	ok(t)(SaveServer("", Server{Name: "home", Transport: "http", URL: farSrv.URL + gateway.HubMCPAll, Bridge: "home", Agents: []string{"claude"}}))
	ok(t)(SetMCPHub(true))
	h := theHub{}
	plain := context.Background()
	if !slices.Contains(hubNames(h.Tools(plain, "claude")), "alias/a") {
		t.Fatal("the plain entry was not listed for an agent")
	}
	aliasAt := aliasHits.Load()
	routed := gateway.WithRouterHost(plain, u.Host)
	got := hubNames(h.Tools(routed, "claude"))
	if slices.Contains(got, "alias/a") || slices.Contains(got, "home/a") {
		t.Fatalf("a router was given %v", got)
	}
	if !slices.Contains(got, "far/a") || !slices.Contains(got, "fake/a") {
		t.Fatalf("a router lost another magpie or a command server: %v", got)
	}
	if aliasHits.Load() != aliasAt {
		t.Fatalf("the hub asked itself %d more times under X-Magpie-Router", aliasHits.Load()-aliasAt)
	}
	if _, err := h.Call(routed, "claude", "alias", "a", json.RawMessage(`{}`)); err == nil {
		t.Fatal("the hub ran the server pointed at the hub that answered")
	}
	res, err := h.Call(plain, "claude", "alias", "a", json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(hubText(t, res), "remote") {
		t.Fatalf("an agent did not run the same server: %s", hubText(t, res))
	}
}

// An agent's own entry at this hub is not one to bring into the Library:
// the hub wrote it, or the user pointed it at this magpie, and importing
// it is how the hub ends up inside itself. A hub at another host or port,
// and any other URL, is still found and imported.
func TestHubFoundSkipsItsOwnHub(t *testing.T) {
	h := hubSandbox(t)
	t.Setenv("MAGPIE_ADDR", "198.51.100.8:3425")
	t.Setenv("MAGPIE_PUBLIC_URL", "http://nas.lan:3425")
	write(t, filepath.Join(h, ".claude.json"), `{
  "mcpServers": {
    "mine": {"command": "my-mcp"},
    "loop": {"type": "http", "url": "http://127.0.0.1:3425/mcp/magpie/hub/agy"},
    "local": {"url": "http://localhost:3425/mcp/magpie/hub"},
    "caps": {"url": "http://LocalHost:3425/mcp/magpie/hub/"},
    "v6": {"url": "http://[::1]:3425/mcp/magpie/hub/claude"},
    "pinned": {"url": "http://198.51.100.8:3425/mcp/magpie/hub"},
    "public": {"url": "http://nas.lan:3425/mcp/magpie/hub/opencode"},
    "magpie": {"url": "http://127.0.0.1:3425/mcp/magpie/hub/claude"},
    "query": {"url": "http://127.0.0.1:3425/mcp/magpie/hub?x=1"},
    "sseown": {"type": "sse", "url": "http://127.0.0.1:3425/mcp/magpie/hub"},
    "office": {"url": "http://192.0.2.20:3425/mcp/magpie/hub"},
    "otherport": {"url": "http://127.0.0.1:9999/mcp/magpie/hub"},
    "hubbub": {"url": "http://127.0.0.1:3425/mcp/magpie/hubbub"},
    "web": {"url": "https://example.com/mcp"}
  }
}
`)
	got := foundNames(t)
	want := []string{"hubbub", "mine", "office", "otherport", "web"}
	if !slices.Equal(got, want) {
		t.Fatalf("found %v, want %v", got, want)
	}
	if _, err := ImportServer("loop"); err == nil {
		t.Fatal("imported a server pointed at this hub")
	}
	ok(t)(ImportServer("office"))
	v, err := Read(nil)
	if err != nil {
		t.Fatal(err)
	}
	var office *Server
	for _, s := range v.Servers {
		if s.Name == "office" {
			office = s.Server
		}
	}
	if office == nil || office.URL != "http://192.0.2.20:3425/mcp/magpie/hub" {
		t.Fatalf("office imported as %+v", office)
	}
}

// A reverse-proxied MAGPIE_PUBLIC_URL (no explicit port, or with a base path
// prefix) and LAZYCAT_APP_DOMAIN reach this gateway on port 443 even while the
// gateway listens on 3425. ownHub drops those URLs and keeps another magpie on
// a different host; remoteHeaders still marks a path-prefixed public hub as a
// router call; and sameHub stops a portless Host (implicit 80/443) without
// hiding a different port or another host.
func TestHubLeavesOutPublicProxyAndPortlessRouterHost(t *testing.T) {
	t.Run("portless_public_url", func(t *testing.T) {
		h := hubSandbox(t)
		t.Setenv("MAGPIE_ADDR", "127.0.0.1:3425")
		t.Setenv("MAGPIE_PUBLIC_URL", "https://magpie.example.com")
		t.Setenv("LAZYCAT_APP_DOMAIN", "")
		write(t, filepath.Join(h, ".claude.json"), `{
  "mcpServers": {
    "proxy": {"url": "https://magpie.example.com/mcp/magpie/hub"},
    "proxyagent": {"url": "https://magpie.example.com/mcp/magpie/hub/claude"},
    "other": {"url": "https://other.example.com/mcp/magpie/hub"}
  }
}`)
		if found := foundNames(t); !slices.Equal(found, []string{"other"}) {
			t.Fatalf("found %v, want [other]", found)
		}
		if _, err := ImportServer("proxy"); err == nil {
			t.Fatal("imported a server pointed at this reverse-proxied public hub")
		}
		ok(t)(SaveServer("", Server{Name: "proxy", Transport: "http", URL: "https://magpie.example.com/mcp/magpie/hub", Agents: []string{"claude"}}))
		ok(t)(SaveServer("", Server{Name: "proxyagent", Transport: "http", URL: "https://magpie.example.com/mcp/magpie/hub/agy", Agents: []string{"claude"}}))
		ok(t)(SaveServer("", Server{Name: "other", Transport: "http", URL: "https://other.example.com/mcp/magpie/hub", Agents: []string{"claude"}}))
		ok(t)(SaveServer("", Server{Name: "otherport", Transport: "http", URL: "https://magpie.example.com:8443/mcp/magpie/hub", Agents: []string{"claude"}}))
		ok(t)(SaveServer("", Server{Name: "hubbub", Transport: "http", URL: "https://magpie.example.com/mcp/magpie/hubbub", Agents: []string{"claude"}}))
		ok(t)(SetMCPHub(true))
		var got []string
		for _, s := range (theHub{}).Servers("claude") {
			got = append(got, s.Name)
		}
		slices.Sort(got)
		want := []string{"hubbub", "other", "otherport"}
		if !slices.Equal(got, want) {
			t.Fatalf("hub servers %v, want %v", got, want)
		}
		if _, err := (theHub{}).Call(context.Background(), "claude", "proxy", "a", json.RawMessage(`{}`)); err == nil {
			t.Fatal("the hub ran a server pointed at its reverse-proxied public URL")
		}
		if !ownHub(&Server{Name: "caps", Transport: "http", URL: "HTTPS://Magpie.Example.com/mcp/magpie/hub/agy"}) {
			t.Fatal("ownHub missed a portless public URL in another case")
		}
		if !ownHub(&Server{Name: "query", Transport: "sse", URL: "https://magpie.example.com/mcp/magpie/hub?x=1"}) {
			t.Fatal("ownHub missed a public hub URL with a query")
		}
		t.Setenv("MAGPIE_PUBLIC_URL", "http://magpie.example.com")
		if !ownHub(&Server{Name: "http", Transport: "http", URL: "http://magpie.example.com/mcp/magpie/hub"}) {
			t.Fatal("ownHub missed a portless http MAGPIE_PUBLIC_URL")
		}
		if ownHub(&Server{Name: "https", Transport: "http", URL: "https://magpie.example.com/mcp/magpie/hub"}) {
			t.Fatal("ownHub dropped https while the public URL is http")
		}
		t.Setenv("MAGPIE_PUBLIC_URL", "https://magpie.example.com:8443")
		if !ownHub(&Server{Name: "pubport", Transport: "http", URL: "https://magpie.example.com:8443/mcp/magpie/hub"}) {
			t.Fatal("ownHub missed a public URL whose port is not the gateway's")
		}
		if ownHub(&Server{Name: "otherport", Transport: "http", URL: "https://magpie.example.com/mcp/magpie/hub"}) {
			t.Fatal("ownHub dropped the public host on the scheme default while the public URL names :8443")
		}
	})

	t.Run("prefixed_public_url", func(t *testing.T) {
		hubSandbox(t)
		t.Setenv("MAGPIE_ADDR", "127.0.0.1:3425")
		t.Setenv("MAGPIE_PUBLIC_URL", "https://magpie.example.com/magpie")
		t.Setenv("LAZYCAT_APP_DOMAIN", "")
		self := Server{Name: "prefixed", Transport: "http", URL: "https://magpie.example.com/magpie/mcp/magpie/hub/agy", Agents: []string{"claude"}}
		sibling := Server{Name: "sibling", Transport: "http", URL: "https://magpie.example.com/magpie2/mcp/magpie/hub", Agents: []string{"claude"}}
		other := Server{Name: "other", Transport: "http", URL: "https://other.example.com/magpie/mcp/magpie/hub", Agents: []string{"claude"}}
		traversal := Server{Name: "tools", Transport: "http", URL: "https://magpie.example.com/magpie/mcp/magpie/hub/../tools", Agents: []string{"claude"}}
		if !ownHub(&self) {
			t.Fatal("ownHub missed path-prefixed MAGPIE_PUBLIC_URL")
		}
		if ownHub(&sibling) {
			t.Fatal("ownHub dropped sibling prefix /magpie2")
		}
		if ownHub(&other) {
			t.Fatal("ownHub dropped different host")
		}
		if ownHub(&traversal) {
			t.Fatal("ownHub dropped non-hub path /mcp/magpie/hub/../tools")
		}
		hdr, fail := remoteHeaders(context.Background(), &self)
		if fail != nil {
			t.Fatalf("remoteHeaders failed: %+v", fail)
		}
		if hdr.Get(gateway.RouterHeader) != "1" {
			t.Fatalf("remoteHeaders did not set %s for path-prefixed public hub", gateway.RouterHeader)
		}
		plain := &Server{Name: "plain", Transport: "http", URL: "https://other.example.com/mcp/magpie/hub"}
		hdr, fail = remoteHeaders(context.Background(), plain)
		if fail != nil || hdr.Get(gateway.RouterHeader) != "1" {
			t.Fatalf("remoteHeaders on a remote hub: %v %v", hdr, fail)
		}
		hdr, fail = remoteHeaders(context.Background(), &sibling)
		if fail != nil || hdr.Get(gateway.RouterHeader) != "" {
			t.Fatalf("remoteHeaders marked a sibling prefix: %v %v", hdr, fail)
		}
		if !sameHub(self.URL, "magpie.example.com") {
			t.Fatal("sameHub missed a path-prefixed public hub on a portless Host")
		}
		if sameHub(sibling.URL, "magpie.example.com") || sameHub(other.URL, "magpie.example.com") || sameHub(traversal.URL, "magpie.example.com") {
			t.Fatal("sameHub hid a sibling prefix, another host, or a cleaned non-hub path")
		}
	})

	t.Run("lazycat_domain", func(t *testing.T) {
		hubSandbox(t)
		t.Setenv("MAGPIE_ADDR", "127.0.0.1:3425")
		t.Setenv("MAGPIE_PUBLIC_URL", "")
		t.Setenv("LAZYCAT_APP_DOMAIN", "magpie.heiyu.space")
		if !ownHub(&Server{Name: "lc", Transport: "http", URL: "https://magpie.heiyu.space/mcp/magpie/hub/claude"}) {
			t.Fatal("ownHub missed LAZYCAT_APP_DOMAIN")
		}
		if ownHub(&Server{Name: "other", Transport: "http", URL: "https://other.heiyu.space/mcp/magpie/hub/claude"}) {
			t.Fatal("ownHub dropped another LazyCat domain")
		}
	})

	t.Run("portless_router_host", func(t *testing.T) {
		hubSandbox(t)
		t.Setenv("MAGPIE_ADDR", "127.0.0.1:3425")
		t.Setenv("MAGPIE_PUBLIC_URL", "")
		t.Setenv("LAZYCAT_APP_DOMAIN", "")
		if !sameHub("https://magpie.example.com/mcp/magpie/hub", "magpie.example.com") {
			t.Fatal("sameHub missed portless HTTPS proxy Host")
		}
		if !sameHub("http://my-nas.local/mcp/magpie/hub/agy", "my-nas.local") {
			t.Fatal("sameHub missed portless HTTP Host")
		}
		if !sameHub("http://127.0.0.1/mcp/magpie/hub", "localhost") {
			t.Fatal("sameHub missed loopback portless Host")
		}
		if !sameHub("http://[::1]:80/mcp/magpie/hub", "[::1]") {
			t.Fatal("sameHub missed bracketed IPv6 portless Host")
		}
		if !sameHub("http://[::1]/mcp/magpie/hub", "::1") {
			t.Fatal("sameHub missed unbracketed IPv6 portless Host")
		}
		if sameHub("http://my-nas.local:9999/mcp/magpie/hub", "my-nas.local") {
			t.Fatal("sameHub dropped non-default port under portless Host")
		}
		if sameHub("https://other.example.com/mcp/magpie/hub", "magpie.example.com") {
			t.Fatal("sameHub dropped different host under portless Host")
		}
		if sameHub("https://magpie.example.com/mcp/magpie/hub/../tools", "magpie.example.com") {
			t.Fatal("sameHub dropped cleaned non-hub path")
		}
		if sameHub("http://my-nas.local:443/mcp/magpie/hub", "my-nas.local") {
			t.Fatal("sameHub treated http on :443 as a portless Host")
		}
		if sameHub("https://my-nas.local:80/mcp/magpie/hub", "my-nas.local") {
			t.Fatal("sameHub treated https on :80 as a portless Host")
		}
		if !sameHub("http://my-nas.local:80/mcp/magpie/hub", "my-nas.local") {
			t.Fatal("sameHub missed an explicit http default port")
		}
		if !sameHub("https://magpie.example.com:443/mcp/magpie/hub", "Magpie.Example.com") {
			t.Fatal("sameHub missed an explicit https default port")
		}
		if !sameHub("http://127.0.0.1:3425/mcp/magpie/hub", "127.0.0.1:3425") {
			t.Fatal("sameHub missed an explicit host:port")
		}
		if sameHub("http://192.0.2.20:3425/mcp/magpie/hub", "127.0.0.1:3425") {
			t.Fatal("sameHub dropped a remote hub on the same explicit port")
		}
		if !sameHub("http://[::1]:3425/mcp/magpie/hub", "[::1]:3425") {
			t.Fatal("sameHub missed a bracketed IPv6 host:port")
		}
		if sameHub("http://127.0.0.1:3425/mcp/magpie/hub/../tools", "127.0.0.1:3425") {
			t.Fatal("sameHub treated /mcp/magpie/hub/../tools as the hub")
		}
		if !sameHub("http://127.0.0.1:3425/mcp/./magpie/hub", "127.0.0.1:3425") {
			t.Fatal("sameHub missed /mcp/./magpie/hub")
		}
		if ownHub(&Server{Name: "tools", Transport: "http", URL: "http://127.0.0.1:3425/mcp/magpie/hub/../tools"}) {
			t.Fatal("ownHub treated a cleaned non-hub path as this hub")
		}
		if !ownHub(&Server{Name: "dot", Transport: "http", URL: "http://127.0.0.1:3425/mcp/./magpie/hub"}) {
			t.Fatal("ownHub missed a cleaned hub path")
		}
		routed := gateway.WithRouterHost(context.Background(), "magpie.example.com")
		if !routerLoop(routed, &Server{Name: "proxy", Transport: "http", URL: "https://magpie.example.com/mcp/magpie/hub/agy"}) {
			t.Fatal("routerLoop missed portless Host")
		}
		if routerLoop(routed, &Server{Name: "other", Transport: "http", URL: "https://other.example.com/mcp/magpie/hub"}) {
			t.Fatal("routerLoop dropped another host under portless Host")
		}
	})
}

func hubOf(t *testing.T, id string) string {
	t.Helper()
	v, err := Read(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range v.Agents {
		if a.ID == id {
			return a.Hub
		}
	}
	t.Fatalf("no %s in the view", id)
	return ""
}

func foundNames(t *testing.T) []string {
	t.Helper()
	v, err := Read(nil)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, f := range v.FoundServers {
		out = append(out, f.Server.Name)
	}
	slices.Sort(out)
	return out
}

// countHub is a streamable HTTP MCP server. Hits on a path ending in
// hubbub go to bub, when it is set; every other path goes to hits.
func countHub(t *testing.T, hits, bub *atomic.Int32) http.Handler {
	t.Helper()
	seen := http.Header{}
	inner := mcpHandler(t, false, &seen)
	var mu sync.Mutex
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		path := strings.TrimRight(r.URL.Path, "/")
		if bub != nil && strings.HasSuffix(path, "hubbub") {
			bub.Add(1)
		} else {
			hits.Add(1)
		}
		inner(w, r)
	})
}
