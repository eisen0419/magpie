// Run with Node's test runner and Playwright on the module path; see README.md.
// Hide accounts masks account identities, not a model picker's alias
// (#7407). A dated option's alias is the undated model id
// (claude-sonnet-4-5-20250929 → claude-sonnet-4-5). noteAccounts used to
// take every unnamed alias, so the found group's auto/<model> was blurred
// on its card (.mdl) and in the editor's "Agents pick it as" hint. An
// address, a vendor's account name and a plugin's name for an account
// stay hidden; the name the user gave an account stays in sight (#3656).
// Every language, Chromium and WebKit.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { test } = require("node:test");
const { chromium, webkit } = require("playwright");

const assets = path.resolve(__dirname, "../assets");
const MODEL = "claude-sonnet-4-5";
const DATED = "claude-sonnet-4-5-20250929";
const REF = "auto/" + MODEL;
const MAIL = "ada@example.com";
const WHO = "韩梅梅";
const PHONE = "138****5678";
const OWN = "Work laptop";
const boot = 15000;

const agent = {
  id: "claude", name: "Claude Code", path: "/t/c.json", icon: "claudecode-color", wired: true,
  fields: [{
    key: "model", label: "model", value: MODEL,
    options: [
      { value: MODEL, label: "Claude Sonnet 4.5", ref: "anthropic/" + MODEL, note: "" },
      { value: DATED, label: "Claude Sonnet 4.5", alias: MODEL, ref: "anthropic/" + MODEL, note: "" },
    ],
  }],
};
const info = (id, provider, name) => ({ id, ready: true, provider, name, model: "Claude Sonnet 4.5", icon: "generic" });
const groups = {
  found: true, pools: [], deciders: [],
  models: [
    { id: "anthropic/" + MODEL, name: "Claude Sonnet 4.5", providerName: "Anthropic", icon: "generic" },
    { id: "openai/" + MODEL, name: "Claude Sonnet 4.5", providerName: "OpenAI", icon: "generic" },
  ],
  groups: [{
    id: "auto-" + MODEL, name: "Claude Sonnet 4.5", ref: REF, auto: true, ready: true,
    members: ["anthropic/" + MODEL, "openai/" + MODEL],
    memberInfo: [info("anthropic/" + MODEL, "anthropic", "Anthropic"), info("openai/" + MODEL, "openai", "OpenAI")],
  }],
};
const base = { icon: "generic", chat: "", responses: "", anthropic: "", catalog: "", agents: [], fallback: [], headers: {}, keyList: [], proxy: "", models: [] };
const providers = {
  providers: [
    { ...base, id: "antigravity", name: "Antigravity", icon: "antigravity-color",
      account: { agent: "antigravity", agentName: "Antigravity", user: MAIL, plan: "Pro", logins: [{ user: MAIL, plan: "Pro", active: true, on: true }] },
      accountNames: { [MAIL]: OWN } },
    { ...base, id: "fakeco", name: "FakeCo",
      account: { agent: "fakeco", agentName: "FakeCo", user: WHO, plan: "Team", logins: [{ user: WHO, plan: "Team", active: true, on: true }] },
      accountUsers: { [WHO]: PHONE } },
  ],
  presets: [], excluded: [], gateway: { running: true, window: true },
};
const quotas = [
  { provider: "antigravity", name: "Antigravity", plan: "Pro", user: MAIL, alias: OWN, named: true, windows: [{ name: "Gemini", used: 5 }] },
  { provider: "fakeco", name: "FakeCo", plan: "Team", user: WHO, alias: PHONE, windows: [{ name: "Credits", used: 10 }] },
];

function serve(lang) {
  const settings = { theme: "light", lang };
  const state = { agents: [agent], profiles: [], settings };
  return async (route) => {
    const url = new URL(route.request().url());
    const json = (data) => route.fulfill({ json: data });
    if (url.pathname === "/boot.js") return route.fulfill({ contentType: "text/javascript", body: `window.bootPrefs = {lang:${JSON.stringify(lang)},theme:"light",web:false};` });
    if (url.pathname === "/wails/runtime.js") return route.fulfill({ contentType: "text/javascript", body: "export const Window = {};" });
    if (url.pathname === "/api/state") return json(state);
    if (url.pathname === "/api/settings") return json(settings);
    if (url.pathname === "/api/providers") return json(providers);
    if (url.pathname === "/api/usage/quotas") return json(quotas);
    if (url.pathname === "/api/groups") return json(groups);
    if (url.pathname === "/api/plugins") return json({ plugins: [] });
    if (url.pathname === "/api/gateway/trace") {
      if (url.searchParams.get("wait")) return new Promise(() => {});
      return json({ mine: true, now: new Date().toISOString(), seq: 0, totals: { requests: 0, rerouted: 0, errors: 0 }, routes: [] });
    }
    if (url.pathname === "/api/gateway/history") return json({ cut: false, days: [], routes: [] });
    if (url.pathname.startsWith("/api/")) return json({});
    const file = path.join(assets, url.pathname === "/" ? "index.html" : url.pathname);
    const contentType = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png" }[path.extname(file)];
    try { await route.fulfill({ body: await fs.readFile(file), contentType }); } catch { await route.fulfill({ status: 404, body: "" }); }
  };
}

for (const engine of (process.env.BROWSER ? [process.env.BROWSER] : ["chromium", "webkit"])) {
  for (const lang of ["en", "zh", "zh-TW", "ja", "de"]) {
    test(`${engine} ${lang}: Hide accounts leaves auto/<model> readable and still hides accounts`, async (t) => {
      const browser = await (engine === "webkit" ? webkit.launch() : chromium.launch({ channel: "chromium" }));
      t.after(() => browser.close());
      const page = await (await browser.newContext({ viewport: { width: 1100, height: 800 }, reducedMotion: "reduce" })).newPage();
      page.setDefaultTimeout(5000);
      page.setDefaultNavigationTimeout(60000);
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));
      const seen = (pathname) => page.waitForResponse((r) => new URL(r.url()).pathname === pathname, { timeout: 60000 });
      const stateP = seen("/api/state");
      const quotasP = seen("/api/usage/quotas");
      const providersP = seen("/api/providers");
      await page.route("**/*", serve(lang));
      await page.goto("http://magpie.test/?view=groups");
      const card = page.locator('.rt-group[data-id="auto-claude-sonnet-4-5"]');
      const mdl = card.locator(".mdl");
      await mdl.waitFor({ timeout: boot });
      await Promise.all([stateP, quotasP, providersP]);
      await page.evaluate((text) => {
        const d = document.createElement("div");
        d.id = "sample";
        d.textContent = text;
        document.querySelector("#view-groups").append(d);
      }, `${MAIL} · ${WHO} · ${PHONE} · ${OWN}`);
      await page.evaluate(() => hideAccounts.set(true));
      await page.waitForFunction(() => document.querySelector("#sample .pii"));

      assert.equal(await mdl.locator(".pii").count(), 0, "the group's ref is blurred: " + await mdl.innerHTML());
      assert.equal(await mdl.textContent(), REF);
      await card.locator(".nm").click();
      await page.locator(".rt-gedit").waitFor({ timeout: boot });
      const hints = await page.locator(".rt-gedit .hint").evaluateAll((hs) => hs.map((h) => ({
        text: h.textContent,
        raws: [...h.querySelectorAll(".pii")].map((p) => p.dataset.raw),
      })));
      const idHint = hints.find((h) => h.text.includes(REF) || h.raws.includes(MODEL) || h.raws.includes(REF));
      assert.ok(idHint, "no Agents-pick-it hint: " + JSON.stringify(hints));
      assert.deepEqual(idHint.raws, [], "the hint blurs the model: " + JSON.stringify(idHint));
      assert.ok(idHint.text.includes(REF), idHint.text);

      const sample = await page.locator("#sample").evaluate((e) => ({
        text: e.textContent,
        raws: [...e.querySelectorAll(".pii")].map((p) => p.dataset.raw),
      }));
      assert.ok(sample.raws.includes(MAIL), "the address stays hidden: " + JSON.stringify(sample));
      assert.ok(sample.raws.includes(WHO), "the account's name stays hidden: " + JSON.stringify(sample));
      assert.ok(sample.raws.includes(PHONE), "the plugin's name for the account stays hidden: " + JSON.stringify(sample));
      assert.ok(!sample.raws.includes(OWN) && sample.text.includes(OWN), "the user's own name stays in sight: " + JSON.stringify(sample));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth), 0, "no sideways scroll");
      assert.deepEqual(errors, []);
    });
  }
}
