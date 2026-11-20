package gateway

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Here is this computer: loopback and its names, an unspecified address,
// the host MAGPIE_ADDR pins, and an address on an interface that is up.
// Another address is not, including one in a documentation range.
func TestHubMCPHereIsThisComputer(t *testing.T) {
	old := hostIPs
	hostIPs = func() []net.IP { return []net.IP{net.ParseIP("203.0.113.9")} }
	t.Cleanup(func() { hostIPs = old })
	t.Setenv("MAGPIE_ADDR", "198.51.100.8:3425")
	for _, host := range []string{"localhost", "LocalHost", "127.0.0.1", "::1", "0.0.0.0", "::", "198.51.100.8", "203.0.113.9"} {
		if !Here(host) {
			t.Errorf("Here(%q) = false", host)
		}
	}
	for _, host := range []string{"", "192.0.2.20", "203.0.113.10", "nas.lan", "example.com"} {
		if Here(host) {
			t.Errorf("Here(%q) = true", host)
		}
	}
}

// A request carrying X-Magpie-Router is one the library can tell came back
// to this hub: the host it arrived on is on the context, and a request
// without the header carries none.
func TestHubMCPRouterHeaderCarriesTheHost(t *testing.T) {
	fresh(t)
	f := &routerHostHub{fakeHub: &fakeHub{on: true, tools: map[string][]HubTool{"": {
		{Server: "fs", Tool: "read", Spec: hubSpec("read")},
	}}}}
	old := MCPHub
	MCPHub = f
	t.Cleanup(func() { MCPHub = old })
	h := lanGuard(New().Handler())
	body := `{"jsonrpc":"2.0","id":1,"method":"tools/list"}`
	post := func(host string, hdr ...string) {
		t.Helper()
		r := httptest.NewRequest(http.MethodPost, HubMCPAll, strings.NewReader(body))
		r.RemoteAddr, r.Host = here, host
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Accept", "application/json, text/event-stream")
		for i := 0; i+1 < len(hdr); i += 2 {
			r.Header.Set(hdr[i], hdr[i+1])
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
	}
	post("nas.lan:3425")
	if f.host != "" {
		t.Fatalf("no header: host %q", f.host)
	}
	post("nas.lan:3425", RouterHeader, "1")
	if f.host != "nas.lan:3425" {
		t.Fatalf("router host %q, want nas.lan:3425", f.host)
	}
	post("magpie.example.com", RouterHeader, "1")
	if f.host != "magpie.example.com" {
		t.Fatalf("portless router host %q, want magpie.example.com", f.host)
	}
}

// routerHostHub records the host a router request carried.
type routerHostHub struct {
	*fakeHub
	host string
}

func (h *routerHostHub) Tools(ctx context.Context, agent string) []HubTool {
	h.host = RouterHost(ctx)
	return h.fakeHub.Tools(ctx, agent)
}
