import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the measured optimization matrix", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /VLM Speed Lab/);
  assert.match(html, /5\.08/);
  assert.match(html, /149\.2/);
  assert.match(html, /Source-resolution control/);
  assert.match(html, /Compiled execution/);
  assert.match(html, /Exact vs 01b/);
  assert.match(html, /Corrupt repeat/);
  assert.match(html, /88\.351s → 6\.858s/);
  assert.doesNotMatch(html, /GPU run pending|end-to-end run pending/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/);
});

test("keeps measured and planned work visually honest", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(page, /p50 \/ p95 ms/);
  assert.match(page, /Not measured/);
  assert.match(page, /status: "regression"/);
  assert.match(page, /status: "rejected"/);
  assert.match(page, /status: "planned"/);
  assert.match(page, /4\/4 concepts/);
  assert.match(page, /Semantic preservation and exact bytes/);
  assert.match(css, /\.comparison-table-wrap \{ overflow-x:auto/);
  assert.match(css, /\.status-measured/);
  assert.match(css, /\.status-rejected/);
  assert.match(css, /\.status-planned/);
});
