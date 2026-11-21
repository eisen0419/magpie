// Run with Node's test runner and Playwright on the module path; see README.md.
// Several provider auth.options with placement "top" and descriptions stack
// under the accounts at their own height, with the same padding and
// alignment as the block under the models, and none of them overlaps
// Models (#7353). One top option does the same. The window header stays
// the title bar. Both engines, 1100 and 420 wide.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { test } = require("node:test");
const { chromium, webkit } = require("playwright");

const assets = path.resolve(__dirname, "../assets");

const about = "Draws the plugin's own page beside the chat and keeps its tools on that page, so the description wraps under the label instead of staying on one line.";
const panel = { key: "panel", label: "Panel", description: about, placement: "top", on: false };
const beta = { key: "beta", label: "Beta models", description: "Beta models may be withdrawn without notice. Leave this off unless you mean to try a model the vendor has not finished.", placement: "top", on: false };
const trace = { key: "trace", label: "Wire trace", description: "Writes each request the plugin sends, headers included, into the log the provider's own page shows.", placement: "top", on: false };
const free = { key: "free_only", label: "Free models only", description: "Hides models the plugin marks paid.", on: false };
const CASES = { three: [panel, beta, trace, free], one: [panel] };

function serve(options) {
  const plugin = { id: "wbpro", pid: "wbpro", name: "WorkBuddy Pro", icon: "generic", spec: "workbuddy-pro", signedIn: true, models: 2,
    methods: [{ type: "api", label: "API key" }], options };
  const payload = { providers: [{ id: "wbpro", name: "WorkBuddy Pro", icon: "generic", chat: "plugin://wbpro/v1", responses: "", anthropic: "", catalog: "",
    models: [{ id: "m1", name: "M1", on: true }, { id: "m2", name: "M2", on: true }], agents: [], fallback: [], headers: {}, keyList: [], key: {}, proxy: "", maxConcurrency: null,
    account: { agent: "wbpro", agentName: "WorkBuddy Pro", agentIcon: "generic", user: "a@fake",
      logins: [{ user: "a@fake", active: true, on: true }, { user: "b@fake", on: true }] } }],
  presets: [], excluded: [], gateway: { running: true, window: true }, plugins: [plugin] };
  return async (route) => {
    const url = new URL(route.request().url());
    const json = (data) => route.fulfill({ json: data });
    if (url.pathname === "/boot.js") return route.fulfill({ contentType: "text/javascript", body: 'window.bootPrefs = {lang:"en",theme:"light",web:false};' });
    if (url.pathname === "/wails/runtime.js") return route.fulfill({ contentType: "text/javascript", body: "export const Window = {};" });
    if (url.pathname === "/api/state") return json({ agents: [], profiles: [], settings: { lang: "en", theme: "light" } });
    if (url.pathname === "/api/providers") return json(payload);
    if (url.pathname === "/api/groups") return json({ groups: [], pools: [] });
    if (url.pathname.startsWith("/api/")) return json({});
    const file = path.join(assets, url.pathname === "/" ? "index.html" : url.pathname);
    const contentType = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png" }[path.extname(file)];
    try { await route.fulfill({ body: await fs.readFile(file), contentType }); } catch { await route.fulfill({ status: 404, body: "" }); }
  };
}

for (const engine of (process.env.BROWSER ? [process.env.BROWSER] : ["chromium", "webkit"])) {
  test(engine + ": top plugin options keep their own height", async (t) => {
    const browser = await (engine === "webkit" ? webkit.launch() : chromium.launch({ channel: "chromium" }));
    t.after(() => browser.close());
    for (const width of [1100, 420]) for (const which of Object.keys(CASES)) {
      await t.test(`${width} ${which}`, async () => {
        const ctx = await browser.newContext({ viewport: { width, height: 800 }, reducedMotion: "reduce" });
        const page = await ctx.newPage();
        page.setDefaultTimeout(5000);
        page.setDefaultNavigationTimeout(60000);
        const errors = [];
        page.on("pageerror", (e) => errors.push(e.message));
        await page.route("**/*", serve(CASES[which]));
        await page.goto("http://magpie.test/?view=providers");
        await page.locator(".row.provider", { hasText: "WorkBuddy Pro" }).click();
        const ed = page.locator("#modal .editor");
        await ed.locator(".plugin-options").first().waitFor();
        await page.waitForTimeout(300); // the editor drawn again as what it asked for comes in

        const got = await ed.evaluate((ed) => {
          const r = (el) => { const b = el.getBoundingClientRect(); return { top: b.top, bottom: b.bottom, left: b.left, right: b.right, height: b.height }; };
          const laid = (block) => {
            const cs = getComputedStyle(block);
            const rows = [...block.querySelectorAll(":scope > .plugin-option")].map((row) => ({ key: row.dataset.key, ...r(row) }));
            const gap = parseFloat(cs.rowGap || cs.gap) || 0;
            const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
            const content = rows.reduce((h, row) => h + row.height, 0) + gap * Math.max(0, rows.length - 1);
            const box = r(block);
            const hit = (a, b) => a.top < b.bottom - 0.5 && a.bottom > b.top + 0.5 && a.left < b.right - 0.5 && a.right > b.left + 0.5;
            return {
              keys: rows.map((row) => row.key),
              boxH: box.height, boxBottom: box.bottom,
              lastRowBottom: rows.at(-1)?.bottom ?? null,
              content, padY,
              pad: [cs.paddingTop, cs.paddingRight, cs.paddingBottom, cs.paddingLeft],
              align: cs.alignItems, dir: cs.flexDirection, display: cs.display,
              stacked: rows.every((row, i) => i === 0 || row.top >= rows[i - 1].bottom - 1),
              inside: rows.every((row) => row.top >= box.top - 1 && row.bottom <= box.bottom + 1),
              hits(other) { return other ? rows.filter((row) => hit(row, other)).map((row) => row.key) : []; },
            };
          };
          const e = ed.querySelector(":scope > .ebody") || ed;
          const blocks = [...e.querySelectorAll(".plugin-options")];
          const models = [...e.querySelectorAll("label")].find((l) => l.textContent === "Models");
          const modelsLabelBox = models ? r(models) : null;
          const modelsWrap = models?.nextElementSibling || null;
          const modelsField = modelsWrap?.matches?.(".models") ? modelsWrap : modelsWrap?.querySelector?.(".models") || null;
          const modelsBox = modelsField ? r(modelsField) : null;
          const header = document.querySelector("header.top");
          const head = header.getBoundingClientRect();
          const macWindow = document.body.classList.contains("window") && document.body.classList.contains("mac");
          const top = laid(blocks[0]);
          const under = blocks[1] ? laid(blocks[1]) : null;
          return {
            headerH: head.height,
            headerCss: getComputedStyle(header).height,
            macWindow,
            modelsLabelBox,
            modelsBox,
            top: { ...top, overModelsLabel: top.hits(modelsLabelBox), overModels: top.hits(modelsBox) },
            under: under && { keys: under.keys, pad: under.pad, align: under.align, dir: under.dir, display: under.display, boxH: under.boxH, content: under.content, inside: under.inside },
          };
        });

        const top = got.top;
        assert.deepEqual(top.keys, which === "three" ? ["panel", "beta", "trace"] : ["panel"], JSON.stringify(top.keys));
        assert.ok(got.modelsLabelBox && got.modelsBox, "Models label and control must be rendered");
        assert.ok(Math.abs(top.boxH - (top.content + top.padY)) <= 2, `top options use their own height ${top.content}, not the title bar ${got.headerH} (${got.headerCss}); box is ${top.boxH}; padding ${top.pad}; align ${top.align}; over Models ${top.overModels}`);
        assert.deepEqual(top.pad, ["0px", "0px", "0px", "0px"], `top options take the editor's padding, not the title bar's: ${top.pad}`);
        assert.ok(top.align === "normal" || top.align === "stretch", `top options align with the stack, not the title bar: ${top.align}`);
        assert.equal(top.dir, "column", top.dir);
        assert.equal(top.display, "flex", top.display);
        assert.ok(top.stacked, "the options stack one under another: " + JSON.stringify(top));
        assert.ok(top.inside, `each option sits inside the stack (box ${top.boxH}, content ${top.content})`);
        assert.ok(top.lastRowBottom !== null && top.lastRowBottom <= got.modelsLabelBox.top + 0.5,
          `last top option bottom (${top.lastRowBottom}) must stay above Models label top (${got.modelsLabelBox.top})`);
        assert.ok(top.boxBottom <= got.modelsLabelBox.top + 0.5,
          `top options block bottom (${top.boxBottom}) must stay above Models label top (${got.modelsLabelBox.top})`);
        assert.deepEqual(top.overModelsLabel, [], `options overlap Models label: ${top.overModelsLabel}`);
        assert.deepEqual(top.overModels, [], `options overlap Models: ${top.overModels}`);
        if (which === "three") {
          assert.ok(top.boxH > got.headerH + 4, `three options are taller than the title bar: box ${top.boxH}, header ${got.headerH}`);
          assert.ok(got.under, "the options under the models are still drawn");
          assert.deepEqual(got.under.keys, ["free_only"]);
          assert.deepEqual(got.under.pad, top.pad, `the top stack's padding matches the one under the models: ${top.pad} / ${got.under.pad}`);
          assert.equal(got.under.align, top.align);
          assert.equal(got.under.dir, top.dir);
          assert.ok(got.under.inside && Math.abs(got.under.boxH - got.under.content) <= 2, JSON.stringify(got.under));
        }
        const expectedHeaderH = got.macWindow ? 50 : 46;
        assert.ok(Math.abs(got.headerH - expectedHeaderH) <= 0.5, `the title bar stays its expected height ${expectedHeaderH}px: got ${got.headerH} (${got.headerCss})`);
        assert.deepEqual(errors, []);
        await ctx.close();
      });
    }
  });
}
