package gateway

import (
	"github.com/yetone/magpie/internal/access"
	"github.com/yetone/magpie/internal/provider"
)

// keyRouted is g as a request with who's gateway key routes it: by the
// key's own routing (access.Key.Routing, justforlxz on X) when it sets
// one, else by the group's. A manual group sends to the member picked on
// its card whatever the key says. A group in g that sets its own routing
// keeps it, even when it matches the key's, and a provider in it routed
// its own way keeps its own (planLevel). A group left on Quota first is
// weighed seat by seat with the key's routing; a provider on Quota first,
// or routed as the group is, is pooled with it. The key's routing, as the
// group's, picks between what stays whole.
func keyRouted(who access.Identity, g provider.Group) provider.Group {
	switch {
	case who.Routing == "" || g.Routing == provider.Manual:
	case who.Routing == access.RoutingSmart:
		g.Routing = ""
	default:
		g.Routing = who.Routing
	}
	return g
}
