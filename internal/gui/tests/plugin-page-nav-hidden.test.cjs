// Run with Node's test runner and Playwright on the module path; see README.md.
// A plugin page hidden in Settings › General › Sidebar pages came back after
// every update (#7335): its tab is built when /api/plugin-pages answers,
// after start.js has already run navHides. The tab stays hidden, a kept
// order still holds with the tab in a built-in's place, a kept hidden
// plugin page is not reopened, and ?view= still opens it. Chromium and
// WebKit. No backend: the API is faked.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { test } = require("node:test");
const { chromium, webkit } = require("playwright");

const assets = path.resolve(__dirname, "../assets");
const BASE = "http://magpie.test/";
const ORDER = ["routing", "providers", "agents", "sessions", "library", "groups", "gateway", "usage", "masking", "plugins"];

const pages = [
  { spec: "/x/glance", name: "glance", key: "plugin-glance", title: "Glance", reads: [], url: "/_magpie/plugin/aa/index.html" },
  { spec: "/x/repl", name: "repl", key: "plugin-repl", title: "Instead of Usage", reads: [], replaces: "usage", url: "/_magpie/plugin/bb/index.html" },
];

function server(held) {
  return async (route) => {
    const url = new URL(route.request().url());
    const json = (d) => route.fulfill({ json: d });
    if (url.pathname === "/navorder.js" && held) await held.promise;
    if (url.pathname === "/boot.js") return route.fulfill({ contentType: "text/javascript", body: `window.bootPrefs = {lang:"en",theme:"light",web:false};` });
    if (url.pathname === "/wails/runtime.js") return route.fulfill({ contentType: "text/javascript", body: "export const Window = {};" });
    if (url.pathname === "/api/state") return json({ agents: [], profiles: [], settings: { lang: "en", theme: "light" } });
    if (url.pathname === "/api/settings") return json({ theme: "light", lang: "en", version: "0.1.400", gateway: "http://127.0.0.1:3425", web: false, gatewayOn: false, visionModels: [], imageGenModels: [], lanURLs: [] });
    if (url.pathname === "/api/providers") return json({ providers: [], presets: [], excluded: [], gateway: { running: true, window: true } });
    if (url.pathname === "/api/plugins") return json({ plugins: [] });
    if (url.pathname === "/api/usage/quotas") return json([]);
    if (url.pathname === "/api/groups") return json({ groups: [], models: [] });
    if (url.pathname === "/api/plugin-pages") return json(pages);
    if (url.pathname === "/api/plugin-panels") return json([]);
    if (url.pathname.startsWith("/_magpie/plugin/")) return route.fulfill({ contentType: "text/html", body: "<!doctype html><title>p</title>" });
    if (url.pathname === "/api/window/fit") return route.fulfill({ status: 204 });
    if (url.pathname.startsWith("/api/")) return json({});
    const file = path.join(assets, url.pathname === "/" ? "index.html" : url.pathname);
    const contentType = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png" }[path.extname(file)];
    try { await route.fulfill({ body: await fs.readFile(file), contentType }); } catch { await route.fulfill({ status: 404, body: "" }); }
  };
}

function hold() {
  let go;
  const promise = new Promise((r) => { go = r; });
  return { promise, go };
}

async function open(browser, { held, store, search = "" } = {}) {
  const context = await browser.newContext({ viewport: { width: 1100, height: 700 }, reducedMotion: "reduce" });
  const page = await context.newPage();
  page.setDefaultTimeout(8000);
  page.setDefaultNavigationTimeout(60000);
  page.errors = [];
  page.on("pageerror", (e) => page.errors.push(e.message));
  // a plugin page's frame is sandboxed without storage: the same script
  // runs there and must not throw
  if (store) await page.addInitScript((s) => { try { for (const [k, v] of Object.entries(s)) localStorage.setItem(k, v); } catch { /* a sandboxed frame has none */ } }, store);
  await page.route("**/*", server(held));
  await page.goto(BASE + search, { waitUntil: "domcontentloaded" });
  return page;
}

const hidden = (page, view) => page.evaluate((v) => document.querySelector(`#nav button[data-view="${v}"]`)?.hidden, view);
const on = (page) => page.evaluate(() => document.querySelector("#nav button.on")?.dataset.view || view);

for (const engine of (process.env.BROWSER ? [process.env.BROWSER] : ["chromium", "webkit"])) {
  test(`${engine}: a hidden plugin page stays out, and a late order keeps a replacement with its built-in`, async (t) => {
    const browser = await (engine === "webkit" ? webkit.launch() : chromium.launch({ channel: "chromium" }));
    t.after(() => browser.close());
    const held = hold();
    const page = await open(browser, {
      held,
      store: {
        "magpie.navHidden": JSON.stringify(["plugin-glance"]),
        "magpie.navOrder": JSON.stringify(ORDER),
      },
    });
    await page.waitForSelector("#nav button[data-view='plugin-glance']", { state: "attached" });
    // navorder.js is still held, so this hide is the one the plugin tabs
    // apply themselves, not the one start.js already ran
    await page.waitForFunction(() => document.querySelector("#nav button[data-view='plugin-glance']")?.hidden === true);
    assert.equal(await hidden(page, "plugin-repl"), false, "the replacement is not hidden");
    held.go();
    await page.waitForFunction((want) => [...document.querySelectorAll("#nav > button[data-view]:not([data-plugin-page])")].map((b) => b.dataset.view).join() === want.join(), ORDER);
    const seq = await page.evaluate(() => [...document.querySelectorAll("#nav > button[data-view]")].map((b) => b.dataset.view));
    assert.equal(seq[0], "routing", "the reader's order");
    assert.equal(seq[seq.indexOf("usage") + 1], "plugin-repl", "the replacement stays on Usage's row: " + seq.join());
    assert.equal(await hidden(page, "plugin-glance"), true, "still hidden once the order is in");
    assert.equal(await page.locator("#nav .nav-gl[data-plugin-pages]").isHidden(), true, "a group whose only page is hidden leaves out its name");
    await page.evaluate(() => show("settings"));
    await page.waitForSelector("#navPagesRow:not([hidden]) .nav-pages-pick");
    assert.match(await page.locator(".nav-pages-pick").innerText(), /1 hidden/);
    assert.deepEqual(page.errors, []);
    await page.context().close();
  });

  test(`${engine}: a kept hidden plugin page stays shut, and ?view= still opens it`, async (t) => {
    const browser = await (engine === "webkit" ? webkit.launch() : chromium.launch({ channel: "chromium" }));
    t.after(() => browser.close());
    const shut = await open(browser, { store: { "magpie.navHidden": JSON.stringify(["plugin-glance"]), "magpie.lastView": "plugin-glance" } });
    await shut.waitForSelector("#nav button[data-view='plugin-glance']", { state: "attached" });
    await shut.waitForFunction(() => document.querySelector("#nav button[data-view='plugin-glance']").hidden);
    assert.equal(await on(shut), "agents", "a kept hidden plugin page is not reopened");
    assert.deepEqual(shut.errors, []);
    await shut.context().close();

    const asked = await open(browser, { search: "?view=plugin-glance", store: { "magpie.navHidden": JSON.stringify(["plugin-glance"]) } });
    await asked.waitForSelector("#nav button[data-view='plugin-glance']", { state: "attached" });
    await asked.waitForFunction(() => view === "plugin-glance" && document.querySelector("#nav button[data-view='plugin-glance']").hidden);
    assert.equal(await on(asked), "plugin-glance", "?view= opens a hidden plugin page");
    assert.deepEqual(asked.errors, []);
    await asked.context().close();

    const back = await open(browser, { store: { "magpie.lastView": "plugin-glance" } });
    await back.waitForFunction(() => view === "plugin-glance");
    assert.equal(await hidden(back, "plugin-glance"), false);
    assert.deepEqual(back.errors, []);
    await back.context().close();

    // Agents hidden: the window lands on Providers, then the kept plugin page
    const agentsOut = await open(browser, { store: { "magpie.navHidden": JSON.stringify(["agents"]), "magpie.lastView": "plugin-glance" } });
    await agentsOut.waitForFunction(() => view === "plugin-glance");
    assert.equal(await hidden(agentsOut, "agents"), true);
    assert.deepEqual(agentsOut.errors, []);
    await agentsOut.context().close();
  });
}
