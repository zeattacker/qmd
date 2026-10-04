/**
 * rerank-off.test.ts - QMD_RERANK=off kill switch
 *
 * With the switch set, no search path may call a reranker (remote or local
 * GGUF), results must keep the fused RRF order exactly like rerank:false, and
 * the MCP `query` tool must default rerank to false and ignore rerank:true.
 * Everything here runs without real models (CI-safe).
 */

import { describe, test, expect, beforeAll, afterAll, afterEach, vi } from "vitest";
import { mkdtemp, writeFile, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { createStore, type QMDStore } from "../src/index.js";
import { LlamaCpp, isRerankDisabled, type LLM, type RerankDocument } from "../src/llm.js";
import { RemoteLLM } from "../src/llm-remote.js";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.js";

type ToolsListResponse = {
  result: { tools: { name: string; inputSchema: { properties: Record<string, { default?: unknown }> } }[] };
};
type QueryCallResponse = { result: { structuredContent: { results: { file: string }[] } } };

const origRerankEnv = process.env.QMD_RERANK;

function restoreRerankEnv(): void {
  if (origRerankEnv === undefined) delete process.env.QMD_RERANK;
  else process.env.QMD_RERANK = origRerankEnv;
}

let testDir: string;
let docsDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-rerank-off-"));
  docsDir = join(testDir, "docs");
  await mkdir(docsDir, { recursive: true });
  await writeFile(join(docsDir, "auth.md"), "# Authentication\n\nAuthentication uses JWT tokens for session management.\n");
  await writeFile(join(docsDir, "login.md"), "# Login flow\n\nUsers log in with email and password; a session token is issued.\n");
  await writeFile(join(docsDir, "api.md"), "# API Reference\n\n### POST /login\nAuthenticate a user and return a JWT.\n");
  await writeFile(join(docsDir, "meeting.md"), "# Standup\n\nAuthentication feature is on track; session refresh pending.\n");
});

afterAll(async () => {
  restoreRerankEnv();
  await rm(testDir, { recursive: true, force: true });
});

afterEach(() => {
  restoreRerankEnv();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("isRerankDisabled", () => {
  test("off-like values disable reranking (case/whitespace-insensitive)", () => {
    for (const v of ["off", "OFF", " false ", "0", "no", "none", "disable", "disabled"]) {
      expect(isRerankDisabled(v)).toBe(true);
    }
  });

  test("unset, empty, or on-like values keep reranking enabled", () => {
    for (const v of [undefined, "", "on", "true", "1", "yes"]) {
      expect(isRerankDisabled(v)).toBe(false);
    }
  });
});

describe("search pipelines with QMD_RERANK=off", () => {
  let store: QMDStore;

  beforeAll(async () => {
    store = await createStore({
      dbPath: join(testDir, "pipeline.sqlite"),
      config: { collections: { docs: { path: docsDir, pattern: "**/*.md" } } },
    });
    await store.update();

    // Model-free fake so the pipelines run without GGUFs; rerank scores invert
    // input order so a reranked list would differ from the RRF order.
    const fake = {
      embedModelName: "fake-embed",
      rerankModelName: "fake-rerank",
      generateModelName: "fake-generate",
      embedBatch: async (texts: string[]) => texts.map(() => null),
      expandQuery: async () => [],
      rerank: async (_q: string, docs: RerankDocument[]) => ({
        results: docs.map((d, i) => ({ file: d.file, score: (i + 1) / docs.length, index: i })),
        model: "fake-rerank",
      }),
    };
    store.internal.llm = fake as unknown as LLM;
  });

  afterAll(async () => {
    await store.close();
  });

  const queries = [
    { type: "lex" as const, query: "authentication session" },
    { type: "lex" as const, query: "login JWT" },
  ];
  // Scores are part of the comparison: the RRF-only path scores 1/rank, any
  // rerank blend would change them even when the order happens to match.
  const ranked = (results: { file: string; score: number }[]) => results.map(r => [r.file, r.score]);

  // The rerank stage (cache lookup + LLM) is store.rerank; spying on it counts
  // stage entries even when results would come from the rerank cache.
  test("control: without the switch, rerank:true enters the rerank stage", async () => {
    delete process.env.QMD_RERANK;
    const stage = vi.spyOn(store.internal, "rerank");
    await store.search({ queries, rerank: true });
    expect(stage).toHaveBeenCalled();
  });

  test("structured search: rerank:true and default skip the rerank stage and match rerank:false exactly", async () => {
    delete process.env.QMD_RERANK;
    const baseline = ranked(await store.search({ queries, rerank: false }));
    expect(baseline.length).toBeGreaterThan(1);

    process.env.QMD_RERANK = "off";
    const stage = vi.spyOn(store.internal, "rerank");
    const forced = ranked(await store.search({ queries, rerank: true }));
    const defaulted = ranked(await store.search({ queries }));
    expect(stage).not.toHaveBeenCalled();
    expect(forced).toEqual(baseline);
    expect(defaulted).toEqual(baseline);
  });

  test("hybrid query (plain string): skips the rerank stage and matches rerank:false exactly", async () => {
    delete process.env.QMD_RERANK;
    const baseline = ranked(await store.search({ query: "authentication session", rerank: false }));
    expect(baseline.length).toBeGreaterThan(0);

    process.env.QMD_RERANK = "off";
    const stage = vi.spyOn(store.internal, "rerank");
    const forced = ranked(await store.search({ query: "authentication session", rerank: true }));
    expect(stage).not.toHaveBeenCalled();
    expect(forced).toEqual(baseline);
  });
});

describe("reranker backends with QMD_RERANK=off", () => {
  const docs: RerankDocument[] = [{ file: "a.md", text: "alpha" }];

  test("LlamaCpp.rerank refuses before initializing llama.cpp or loading a GGUF", async () => {
    process.env.QMD_RERANK = "off";
    const ensureLlama = vi.spyOn(LlamaCpp.prototype as unknown as { ensureLlama: () => Promise<unknown> }, "ensureLlama");
    const llm = new LlamaCpp();
    await expect(llm.rerank("q", docs)).rejects.toThrow(/Reranking is disabled/);
    expect(ensureLlama).not.toHaveBeenCalled();
  });

  test("RemoteLLM with QMD_RERANK_URL makes no HTTP call and returns no neutral scores", async () => {
    process.env.QMD_RERANK = "off";
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const llm = new RemoteLLM({ embedUrl: "http://embed.invalid", rerankUrl: "http://rerank.invalid" });
    await expect(llm.rerank("q", docs)).rejects.toThrow(/Reranking is disabled/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("RemoteLLM without QMD_RERANK_URL never falls back to the local reranker", async () => {
    process.env.QMD_RERANK = "off";
    const localRerank = vi.spyOn(LlamaCpp.prototype, "rerank");
    const llm = new RemoteLLM({ embedUrl: "http://embed.invalid" });
    await expect(llm.rerank("q", docs)).rejects.toThrow(/Reranking is disabled/);
    expect(localRerank).not.toHaveBeenCalled();
  });
});

describe("MCP query tool with QMD_RERANK=off", () => {
  const origIndexPath = process.env.INDEX_PATH;
  const origConfigDir = process.env.QMD_CONFIG_DIR;
  let handle: HttpServerHandle;
  let sessionId: string | null = null;

  async function mcp<T>(body: object): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Accept": "application/json, text/event-stream",
    };
    if (sessionId) headers["mcp-session-id"] = sessionId;
    const res = await fetch(`http://localhost:${handle.port}/mcp`, { method: "POST", headers, body: JSON.stringify(body) });
    sessionId = res.headers.get("mcp-session-id") ?? sessionId;
    // Our own server's JSON-RPC response; tests assert on the fields they read.
    return (await res.json()) as T;
  }

  beforeAll(async () => {
    const dbPath = join(testDir, "mcp.sqlite");
    const config = { collections: { docs: { path: docsDir, pattern: "**/*.md" } } };
    const seed = await createStore({ dbPath, config });
    await seed.update();
    await seed.close();

    const configDir = join(testDir, "mcp-config");
    await mkdir(configDir, { recursive: true });
    await writeFile(join(configDir, "index.yml"), YAML.stringify(config));
    process.env.INDEX_PATH = dbPath;
    process.env.QMD_CONFIG_DIR = configDir;
    process.env.QMD_RERANK = "off";

    handle = await startMcpHttpServer(0, { quiet: true });
    await mcp({
      jsonrpc: "2.0", id: 1, method: "initialize",
      params: { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "test", version: "1.0" } },
    });
  });

  afterAll(async () => {
    await handle?.stop();
    if (origIndexPath !== undefined) process.env.INDEX_PATH = origIndexPath;
    else delete process.env.INDEX_PATH;
    if (origConfigDir !== undefined) process.env.QMD_CONFIG_DIR = origConfigDir;
    else delete process.env.QMD_CONFIG_DIR;
  });

  test("tools/list advertises rerank default false", async () => {
    const json = await mcp<ToolsListResponse>({ jsonrpc: "2.0", id: 2, method: "tools/list", params: {} });
    const query = json.result.tools.find(t => t.name === "query");
    expect(query?.inputSchema.properties.rerank?.default).toBe(false);
  });

  test("explicit rerank:true returns the same order as rerank:false without touching a reranker", async () => {
    process.env.QMD_RERANK = "off";
    const localRerank = vi.spyOn(LlamaCpp.prototype, "rerank");
    const call = async (args: Record<string, unknown>) => {
      const json = await mcp<QueryCallResponse>({
        jsonrpc: "2.0", id: 3, method: "tools/call",
        params: { name: "query", arguments: { searches: [{ type: "lex", query: "authentication session" }], ...args } },
      });
      return json.result.structuredContent.results.map(r => r.file);
    };
    const forced = await call({ rerank: true });
    const disabled = await call({ rerank: false });
    expect(forced.length).toBeGreaterThan(0);
    expect(forced).toEqual(disabled);
    expect(localRerank).not.toHaveBeenCalled();
  });
});
