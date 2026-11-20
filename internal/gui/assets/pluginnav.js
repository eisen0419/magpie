// Plugin tabs are built after the window's first navHides: start.js runs
// it, then /api/plugin-pages answers and pluginpages.js load builds the
// tabs. This applies the same hides and the reader's order to those tabs,
// and opens a plugin page the window was left on — not one the sidebar
// leaves out, unless this load was asked for it (?view=). #7335.

// afterPages runs once the plugin tabs are in #nav. opened so a later
// load (a new language) doesn't pull the reader back to the kept page.
export function afterPages(pages, kept) {
  navHides();
  navSubs();
  window.renderNavPages?.();
  if (afterPages.opened) return;
  afterPages.opened = true;
  const asked = params.get("view") || "";
  const want = asked || kept || "";
  if (!pages.has(want)) return;
  // ?view= opens it even when the sidebar leaves it out, as a built-in
  // page's link does. A kept page does not.
  if (!asked && navMine().includes(want)) return;
  // only from the landing the window chose before the tabs existed:
  // Agents, or Providers when Agents is hidden and nothing was asked
  if (asked || view === "agents" || (view === "providers" && navMine().includes("agents"))) show(want);
}
