// A plugin that forwards every header it was given, on the transport the
// request's path names. The host is what takes magpie's own headers off
// (#7415); this plugin never drops one itself. After each call it checks
// the caller's Request, Headers, list, Map and options still hold them.
import http from "node:http"
import https from "node:https"
import http2 from "node:http2"
import { get as httpGet, request as httpRequest } from "node:http"
import { request as httpsRequest } from "node:https"
import { createRequire } from "node:module"
import { WebSocket as wsEsmNamed } from "ws"
import undiciDefault, { fetch as undiciEsmFetch, request as undiciEsmRequest, WebSocket as UndiciEsmWebSocket } from "undici"

const required = createRequire(import.meta.url)("node:http")
const require = createRequire(import.meta.url)

const MAGPIE = ["x-magpie-work-dir", "x-magpie-work-git", "x-magpie-fallback", "x-magpie-session", "x-magpie-subagent", "x-magpie-effort", "x-magpie-caller-agent", "x-magpie-caller-name", "x-magpie-caller-via"]

function titleCase(name) {
  return name.split("-").map((p) => (p ? p[0].toUpperCase() + p.slice(1) : p)).join("-")
}

function pairsFrom(headers, cased) {
  const out = []
  for (const [k, v] of headers) out.push([cased ? titleCase(k) : k, v])
  return out
}

function objectFrom(pairs) {
  const o = {}
  for (const [k, v] of pairs) o[k] = v
  return o
}

function flatFrom(pairs) {
  const out = []
  for (const [k, v] of pairs) out.push(k, v)
  return out
}

function namesOf(value) {
  if (value instanceof Request) return namesOf(value.headers)
  if (typeof Headers === "function" && value instanceof Headers) {
    const out = []
    for (const [k] of value) out.push(String(k).toLowerCase())
    return out
  }
  if (value instanceof Map) {
    const out = []
    for (const [k] of value) out.push(String(k).toLowerCase())
    return out
  }
  if (Array.isArray(value)) {
    if (value.length && Array.isArray(value[0])) return value.map((e) => String(e?.[0] ?? "").toLowerCase())
    const out = []
    for (let i = 0; i + 1 < value.length; i += 2) out.push(String(value[i]).toLowerCase())
    return out
  }
  if (value && typeof value === "object") return Object.keys(value).map((k) => k.toLowerCase())
  return []
}

function intact(value) {
  const have = new Set(namesOf(value))
  const missing = MAGPIE.filter((k) => !have.has(k))
  return missing.length ? "mutated, lost " + missing.join(",") : ""
}

async function kept(res, ...values) {
  for (const v of values) {
    const why = intact(v)
    if (why) return new Response(why, { status: 500 })
  }
  return res
}

function nodeDone(req, ended) {
  return new Promise((resolve, reject) => {
    req.on("response", (res) => {
      res.resume()
      res.on("end", () => resolve(new Response("ok", { status: res.statusCode })))
    })
    req.on("error", reject)
    if (!ended) req.end()
  })
}

function awaitOpen(ws) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error("websocket timed out")), 5000)
    let settled = false
    const finish = (fn) => {
      if (settled) return
      settled = true
      clearTimeout(t)
      fn()
    }
    const opened = () => finish(() => {
      try { ws.close() } catch {}
      resolve(new Response("ok"))
    })
    const failed = () => finish(() => reject(new Error("websocket failed")))
    if (typeof ws.addEventListener === "function") {
      ws.addEventListener("open", opened)
      ws.addEventListener("error", failed)
      return
    }
    if (typeof ws.on === "function") {
      ws.on("open", opened)
      ws.on("error", failed)
      return
    }
    failed()
  })
}

function openWebSocket(url, headers) {
  return awaitOpen(new WebSocket(url, { headers }))
}

function h2Done(origin, path, headers, trailers) {
  return new Promise((resolve, reject) => {
    const s = http2.connect(origin, { rejectUnauthorized: false })
    const fail = (e) => {
      s.destroy()
      reject(e instanceof Error ? e : new Error(String(e?.message || e)))
    }
    s.on("error", fail)
    const req = trailers ? s.request(headers, { waitForTrailers: true }) : s.request(headers)
    req.on("error", fail)
    if (trailers) {
      req.on("wantTrailers", () => {
        try {
          req.sendTrailers(trailers)
        } catch (e) {
          fail(e)
        }
      })
    }
    const body = []
    let status = 0
    req.on("response", (h) => (status = h[":status"]))
    req.on("data", (c) => body.push(c))
    req.on("end", () => {
      s.close()
      resolve(new Response(Buffer.concat(body), { status: status || 502 }))
    })
    req.end(trailers ? "hello" : undefined)
  })
}

function watchSensitive(headers) {
  let saw = false
  Object.defineProperty(headers, http2.sensitiveHeaders, {
    configurable: true,
    get() {
      saw = true
      return ["authorization"]
    },
  })
  return () => saw
}

async function readUndici(res) {
  if (res.body == null) return ""
  if (typeof res.body.text === "function") return res.body.text()
  const chunks = []
  for await (const c of res.body) chunks.push(Buffer.from(c))
  return Buffer.concat(chunks)
}

export async function StripPlugin() {
  return {
    config: async (cfg) => {
      cfg.provider = cfg.provider ?? {}
      cfg.provider.stripco = {
        name: "StripCo",
        npm: "@ai-sdk/openai-compatible",
        api: "https://strip.invalid/v1",
        models: { "strip-1": { name: "Strip One", limit: { context: 1000, output: 100 } } },
      }
    },
    auth: {
      provider: "stripco",
      methods: [{ type: "api", label: "API key" }],
      loader: async () => ({
        apiKey: "k-strip",
        async fetch(url, init) {
          const u = new URL(url)
          const mode = u.pathname.slice(1)
          if (mode === "bun-desc") {
            const d = Object.getOwnPropertyDescriptor(Bun, "fetch")
            return new Response(JSON.stringify({ configurable: !!d?.configurable, writable: !!d?.writable, same: Bun.fetch === globalThis.fetch }))
          }
          const got = init.headers instanceof Headers ? init.headers : new Headers(init.headers)
          const cased = mode.endsWith("-case")
          const pairs = pairsFrom(got, cased)
          const object = objectFrom(pairs)
          const tls = { rejectUnauthorized: false }
          try {
            if (mode === "fetch-headers") {
              const h = new Headers(pairs)
              return kept(await fetch(url, { headers: h }), h)
            }
            if (mode === "fetch-tuples") return kept(await fetch(url, { headers: pairs }), pairs)
            if (mode === "fetch-request") {
              const req = new Request(url, { headers: pairs })
              return kept(await fetch(req), req)
            }
            if (mode === "fetch-url") return kept(await fetch(new URL(url), { headers: object }), object)
            if (mode === "fetch-object" || mode === "fetch-case" || mode === "fetch-redirect") return kept(await fetch(url, { headers: object }), object)
            if (mode === "fetch-oneshot-clean") {
              const clean = new Headers([["x-own", got.get("x-own")], ["authorization", got.get("authorization")]])
              return await fetch(url, { headers: clean.entries() })
            }
            if (mode === "fetch-proto-init") {
              const proto = { method: "POST", body: "proto-body", headers: object }
              const opts = Object.create(proto)
              const res = await fetch(url, opts)
              if (Object.prototype.hasOwnProperty.call(opts, "headers") || proto.headers !== object) return new Response("proto init mutated", { status: 500 })
              return kept(res, object)
            }
            if (mode === "fetch-frozen-proto") {
              const proto = Object.freeze({ method: "POST", body: "proto-body", headers: object })
              const opts = Object.create(proto)
              return kept(await fetch(url, opts), object)
            }
            if (mode === "fetch-request-init") {
              const req = new Request(url, { method: "POST", body: "proto-body", headers: pairs })
              return kept(await fetch(url, req), req)
            }
            if (mode === "fetch-request-undef-init") {
              const req = new Request(url, { method: "POST", body: "proto-body", headers: pairs })
              return kept(await fetch(req, { method: undefined, headers: undefined, body: undefined }), req)
            }
            if (mode === "fetch-post-override-get-req") {
              const postReq = new Request(url, { method: "POST", body: "proto-body", headers: pairs })
              const getReq = new Request(url, { method: "GET", headers: pairs })
              return kept(await fetch(postReq, getReq), postReq, getReq)
            }
            if (mode === "fetch-req-obj") {
              const reqObj = { url, method: "POST", body: "proto-body", headers: object }
              const res = await fetch(reqObj)
              if (reqObj.headers !== object) return new Response("request object mutated", { status: 500 })
              return kept(res, object)
            }
            if (mode === "fetch-req-obj-manual") {
              const reqObj = { url, headers: object }
              const res = await fetch(reqObj, { redirect: "manual" })
              if (reqObj.headers !== object) return new Response("request object mutated", { status: 500 })
              return kept(res, object)
            }
            if (mode === "fetch-iter-tuples-clean") {
              const list = [["x-own", got.get("x-own")].values(), ["authorization", got.get("authorization")].values()]
              return await fetch(url, { headers: list })
            }
            if (mode === "fetch-consumed-req-body") {
              const req = new Request(url, { method: "POST", headers: pairs, body: "consumed-old" })
              await req.text()
              return kept(await fetch(req, { body: "proto-body" }), pairs)
            }
            if (mode === "fetch-consumed-req-headers") {
              const req = new Request(url, { method: "POST", headers: pairs, body: "consumed-old" })
              await req.text()
              return kept(await fetch(req, { headers: object, body: "proto-body" }), object)
            }
            if (mode === "ws-headers") {
              const h = new Headers(pairs)
              return kept(await openWebSocket(url, h), h)
            }
            if (mode === "ws-tuples") return kept(await openWebSocket(url, pairs), pairs)
            if (mode === "ws-map") {
              const map = new Map(pairs)
              return kept(await openWebSocket(url, map), map)
            }
            if (mode === "ws-entries") {
              const h = new Headers(pairs)
              return kept(await openWebSocket(url, h.entries()), h)
            }
            if (mode === "ws-iterable") {
              const list = pairs.map((pair) => pair.slice())
              const iterable = { [Symbol.iterator]: () => list[Symbol.iterator]() }
              return kept(await openWebSocket(url, iterable), list)
            }
            if (mode === "ws-dup-iterable") {
              const list = [...pairs.map((pair) => pair.slice()), ["x-dup", "a"], ["X-Dup", "b"]]
              const iterable = { [Symbol.iterator]: () => list[Symbol.iterator]() }
              return kept(await openWebSocket(url, iterable), list)
            }
            if (mode === "ws-inherited") {
              const proto = { headers: object, protocols: ["chat"] }
              const opts = Object.create(proto)
              const res = await awaitOpen(new WebSocket(url, opts))
              if (Object.prototype.hasOwnProperty.call(opts, "headers") || proto.headers !== object) return new Response("inherited options mutated", { status: 500 })
              return kept(res, object)
            }
            if (mode === "ws-setter-proto") {
              let setterCalled = false
              class WsOpts {
                get headers() { return object }
                set headers(_) { setterCalled = true }
                get protocols() { return ["chat"] }
              }
              const opts = new WsOpts()
              const res = await awaitOpen(new WebSocket(url, opts))
              if (setterCalled) return new Response("prototype setter invoked", { status: 500 })
              return kept(res, object)
            }
            if (mode === "ws-buffer-names") {
              const list = pairs.map(([k, v]) => [Buffer.from(k), v])
              return kept(await openWebSocket(url, list), list)
            }
            if (mode === "ws-boxed-string-names") {
              const list = pairs.map(([k, v]) => [new String(k), v])
              const iterable = { [Symbol.iterator]: () => list[Symbol.iterator]() }
              return kept(await openWebSocket(url, iterable), list)
            }
            if (mode === "ws-iter-tuples") {
              const list = pairs.map((p) => p.values())
              return kept(await openWebSocket(url, list), pairs)
            }
            if (mode === "ws-mixed-tuples") {
              const list = [pairs[0], ...pairs.slice(1).map((p) => p.values())]
              return kept(await openWebSocket(url, list), pairs)
            }
            if (mode === "ws-3arg-undef") {
              const opts = { headers: object }
              const res = await awaitOpen(new WebSocket(url, opts, undefined))
              if (opts.headers !== object) return new Response("ws options mutated", { status: 500 })
              return kept(res, object)
            }
            if (mode === "ws-http") {
              const ws = new http.WebSocket(url, { headers: object })
              if (!(ws instanceof WebSocket)) return new Response("http.WebSocket instanceof WebSocket failed", { status: 500 })
              return kept(await awaitOpen(ws), object)
            }
            if (mode === "ws-object" || mode === "ws-case") return kept(await openWebSocket(url, object), object)
            if (mode === "ws-require" || mode === "ws-require-3" || mode === "ws-require-direct" || mode === "ws-require-3-undef" || mode === "ws-esm-named") {
              const Ws = mode === "ws-esm-named" ? wsEsmNamed : mode === "ws-require-direct" || mode === "ws-require-3-undef" ? require("ws") : require("ws").WebSocket
              const opts = { headers: object }
              const ws = mode === "ws-require-3" ? new Ws(url, ["chat"], opts) : mode === "ws-require-3-undef" ? new Ws(url, opts, undefined) : new Ws(url, opts)
              if (opts.headers !== object) return new Response("ws options mutated", { status: 500 })
              return kept(await awaitOpen(ws), object)
            }
            if (mode === "http-url") return kept(await nodeDone(http.request(url, { headers: object })), object)
            if (mode === "http-opts") return kept(await nodeDone(http.request({ hostname: u.hostname, port: u.port, path: u.pathname, headers: object })), object)
            if (mode === "http-get") return kept(await nodeDone(http.get(url, { headers: object }), true), object)
            if (mode === "http-set") {
              const req = http.request(url)
              for (const [k, v] of pairs) req.setHeader(k, v)
              return kept(await nodeDone(req), pairs)
            }
            if (mode === "http-append") {
              const req = http.request(url)
              for (const [k, v] of pairs) req.appendHeader(k, v)
              return kept(await nodeDone(req), pairs)
            }
            if (mode === "http-setheaders") {
              const h = new Headers(pairs)
              const req = http.request(url)
              req.setHeaders(h)
              return kept(await nodeDone(req), h)
            }
            if (mode === "http-tuples") return kept(await nodeDone(http.request(url, { headers: pairs })), pairs)
            if (mode === "http-flat") {
              const flat = flatFrom(pairs)
              return kept(await nodeDone(http.request(url, { headers: flat })), flat)
            }
            if (mode === "http-flat-array-val") {
              const flat = ["Host", u.host, "x-multi", ["a", "b"], ...flatFrom(pairs)]
              return kept(await nodeDone(http.request(url, { headers: flat })), flat)
            }
            if (mode === "http-named") return kept(await nodeDone(httpRequest(url, { headers: object })), object)
            if (mode === "http-get-named") return kept(await nodeDone(httpGet(url, { headers: object }), true), object)
            if (mode === "http-require") return kept(await nodeDone(required.request(url, { headers: object })), object)
            if (mode === "http-case") return kept(await nodeDone(http.request(url, { headers: object })), object)
            if (mode === "http-client") return kept(await nodeDone(new http.ClientRequest(url, { headers: object })), object)
            if (mode === "http-store") {
              const req = http.request(url)
              req._headers = object
              return kept(await nodeDone(req), object)
            }
            if (mode === "http-trailers") {
              const trailers = { "x-trail": "kept" }
              for (const [k, v] of pairs) if (MAGPIE.includes(String(k).toLowerCase())) trailers[k] = v
              const req = http.request(url, { method: "POST" })
              for (const [k, v] of pairs) req.setHeader(k, v)
              req.setHeader("Trailer", ["x-trail", ...MAGPIE].join(", "))
              req.addTrailers(trailers)
              req.write("hello")
              return kept(await nodeDone(req), trailers, pairs)
            }
            if (mode === "http-agent") {
              const agent = new http.Agent({ keepAlive: true })
              try {
                return await kept(await nodeDone(http.request(url, { agent, headers: object })), object)
              } finally {
                agent.destroy()
              }
            }
            if (mode === "https-url" || mode === "https-case") return kept(await nodeDone(https.request(url, { headers: object, ...tls })), object)
            if (mode === "https-named") return kept(await nodeDone(httpsRequest(url, { headers: object, ...tls })), object)
            if (mode === "https-get") return kept(await nodeDone(https.get(url, { headers: object, ...tls }), true), object)
            if (mode === "https-set") {
              const req = https.request(url, tls)
              for (const [k, v] of pairs) req.setHeader(k, v)
              return kept(await nodeDone(req), pairs)
            }
            if (mode === "http2-flat") {
              const flat = [":method", "GET", ":path", u.pathname, ...flatFrom(pairs)]
              return kept(await h2Done(u.origin, u.pathname, flat), flat)
            }
            if (mode === "http2-flat-array-val") {
              const flat = [":method", "GET", ":path", u.pathname, "x-multi", ["a", "b"], ...flatFrom(pairs)]
              return kept(await h2Done(u.origin, u.pathname, flat), flat)
            }
            if (mode === "http2-sensitive" || mode === "http2-sensitive-flat") {
              const headers = mode === "http2-sensitive-flat"
                ? [":method", "GET", ":path", u.pathname, ...flatFrom(pairs)]
                : { ":method": "GET", ":path": u.pathname, ...object }
              const saw = watchSensitive(headers)
              const res = await h2Done(u.origin, u.pathname, headers)
              if (!saw()) return new Response("sensitiveHeaders dropped", { status: 500 })
              return kept(res, headers)
            }
            if (mode === "http2-trailers") {
              const trailers = { "x-trail": "kept" }
              for (const [k, v] of pairs) if (MAGPIE.includes(String(k).toLowerCase())) trailers[k] = v
              const saw = watchSensitive(trailers)
              const headers = { ":method": "POST", ":path": u.pathname, trailer: ["x-trail", ...MAGPIE].join(", "), ...object }
              const res = await h2Done(u.origin, u.pathname, headers, trailers)
              if (!saw()) return new Response("sensitiveHeaders dropped on trailers", { status: 500 })
              return kept(res, headers, trailers)
            }
            if (mode === "http2-object" || mode === "http2-case") {
              const headers = { ":method": "GET", ":path": u.pathname, ...object }
              return kept(await h2Done(u.origin, u.pathname, headers), headers)
            }
            if (mode === "bun-session" || mode === "bun-session-detached" || mode === "bun-session-oneshot-clean" || mode === "bun-session-proto-init" || mode === "bun-session-req-obj" || mode === "bun-session-consumed-req-body") {
              if (typeof Bun.FetchSession !== "function") return new Response("no FetchSession", { status: 500 })
              const session = new Bun.FetchSession()
              try {
                if (mode === "bun-session-detached") {
                  let invalidThrew = false
                  try {
                    void Bun.FetchSession.prototype.fetch
                  } catch (e) {
                    invalidThrew = e instanceof TypeError
                  }
                  if (!invalidThrew) return new Response("prototype.fetch did not throw TypeError", { status: 500 })
                  const { fetch: detached } = session
                  if (session.fetch !== detached) return new Response("session.fetch not cached", { status: 500 })
                  const res = await detached(url, { headers: object })
                  const body = await res.text()
                  return kept(new Response(body, { status: res.status, headers: res.headers }), object)
                }
                if (mode === "bun-session-consumed-req-body") {
                  const req = new Request(url, { method: "POST", headers: pairs, body: "consumed-old" })
                  await req.text()
                  const res = await session.fetch(req, { body: "proto-body" })
                  const body = await res.text()
                  return kept(new Response(body, { status: res.status, headers: res.headers }), pairs)
                }
                if (mode === "bun-session-oneshot-clean") {
                  const clean = new Headers([["x-own", got.get("x-own")], ["authorization", got.get("authorization")]])
                  const res = await session.fetch(url, { headers: clean.entries() })
                  return new Response(await res.text(), { status: res.status, headers: res.headers })
                }
                if (mode === "bun-session-req-obj") {
                  const reqObj = { url, headers: object }
                  const res = await session.fetch(reqObj)
                  const body = await res.text()
                  if (reqObj.headers !== object) return new Response("request object mutated", { status: 500 })
                  return kept(new Response(body, { status: res.status, headers: res.headers }), object)
                }
                if (mode === "bun-session-proto-init") {
                  const proto = { method: "POST", body: "proto-body", headers: object }
                  const opts = Object.create(proto)
                  const res = await session.fetch(url, opts)
                  const body = await res.text()
                  if (Object.prototype.hasOwnProperty.call(opts, "headers") || proto.headers !== object) return new Response("proto init mutated", { status: 500 })
                  return kept(new Response(body, { status: res.status, headers: res.headers }), object)
                }
                const res = await session.fetch(url, { headers: object })
                const body = await res.text()
                return kept(new Response(body, { status: res.status, headers: res.headers }), object)
              } finally {
                try { session.close() } catch {}
              }
            }
            if (mode.startsWith("undici-")) {
              const undici = mode.startsWith("undici-default-") ? undiciDefault : require("undici")
              if (mode === "undici-fetch" || mode === "undici-default-fetch") {
                if (typeof globalThis.fetch.preconnect === "function" && typeof undici.fetch.preconnect !== "function") {
                  return new Response("undici.fetch.preconnect missing", { status: 500 })
                }
                return kept(await undici.fetch(url, { headers: object }), object)
              }
              if (mode === "undici-esm-fetch") return kept(await undiciEsmFetch(url, { headers: object }), object)
              if (mode === "undici-fetch-oneshot-clean") {
                const clean = new Headers([["x-own", got.get("x-own")], ["authorization", got.get("authorization")]])
                return await undici.fetch(url, { headers: clean.entries() })
              }
              if (mode === "undici-fetch-proto-init") {
                const proto = { method: "POST", body: "proto-body", headers: object }
                const opts = Object.create(proto)
                const res = await undici.fetch(url, opts)
                if (Object.prototype.hasOwnProperty.call(opts, "headers") || proto.headers !== object) return new Response("proto init mutated", { status: 500 })
                return kept(res, object)
              }
              if (mode === "undici-fetch-req-obj") {
                const reqObj = { url, headers: object }
                const res = await undici.fetch(reqObj)
                if (reqObj.headers !== object) return new Response("request object mutated", { status: 500 })
                return kept(res, object)
              }
              if (mode === "undici-fetch-consumed-req-body") {
                const req = new Request(url, { method: "POST", headers: pairs, body: "consumed-old" })
                await req.text()
                return kept(await undici.fetch(req, { body: "proto-body" }), pairs)
              }
              if (mode === "undici-request" || mode === "undici-default-request" || mode === "undici-esm-request") {
                const fn = mode === "undici-esm-request" ? undiciEsmRequest : undici.request
                const res = await fn(url, { method: "GET", headers: object })
                const body = await readUndici(res)
                return kept(new Response(body, { status: res.statusCode }), object)
              }
              if (mode === "undici-request-proto-init") {
                const proto = { method: "POST", body: "proto-body", headers: object }
                const opts = Object.create(proto)
                const res = await undici.request(url, opts)
                const body = await readUndici(res)
                if (Object.prototype.hasOwnProperty.call(opts, "headers") || proto.headers !== object) return new Response("proto init mutated", { status: 500 })
                return kept(new Response(body, { status: res.statusCode }), object)
              }
              if (mode === "undici-request-setter-proto") {
                let setterCalled = false
                class ReqOpts {
                  get method() { return "POST" }
                  get body() { return "proto-body" }
                  get headers() { return object }
                  set headers(_) { setterCalled = true }
                }
                const res = await undici.request(url, new ReqOpts())
                const body = await readUndici(res)
                if (setterCalled) return new Response("prototype setter invoked", { status: 500 })
                return kept(new Response(body, { status: res.statusCode }), object)
              }
              if (mode === "undici-request-buffer-names") {
                const list = pairs.map(([k, v]) => [Buffer.from(k), v])
                const res = await undici.request(url, { method: "GET", headers: list })
                const body = await readUndici(res)
                return kept(new Response(body, { status: res.statusCode }), list)
              }
              if (mode === "undici-request-iter-tuples") {
                const list = pairs.map((p) => p.values())
                const res = await undici.request(url, { method: "GET", headers: list })
                const body = await readUndici(res)
                return kept(new Response(body, { status: res.statusCode }), pairs)
              }
              if (mode === "undici-request-dup-iterable") {
                const list = [...pairs.map((pair) => pair.slice()), ["x-dup", "a"], ["X-Dup", "b"]]
                const iterable = { [Symbol.iterator]: () => list[Symbol.iterator]() }
                const res = await undici.request(url, { method: "GET", headers: iterable })
                const body = await readUndici(res)
                return kept(new Response(body, { status: res.statusCode }), list)
              }
              if (mode === "undici-ws-3-undef") {
                const opts = { headers: object }
                const res = await awaitOpen(new undici.WebSocket(url, opts, undefined))
                if (opts.headers !== object) return new Response("undici ws options mutated", { status: 500 })
                return kept(res, object)
              }
              if (mode === "undici-esm-ws") return kept(await awaitOpen(new UndiciEsmWebSocket(url, { headers: object })), object)
              return kept(await awaitOpen(new undici.WebSocket(url, { headers: object })), object)
            }
            return new Response("unknown mode " + mode, { status: 404 })
          } catch (e) {
            return new Response(String(e?.stack ?? e), { status: 502 })
          }
        },
      }),
    },
  }
}
