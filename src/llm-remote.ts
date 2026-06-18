/**
 * llm-remote.ts - Remote LLM backend for QMD
 *
 * Re-applied onto v2.5.3 (adapted from fork commit fbe21b5).
 *
 * Delegates embedding and reranking to external OpenAI-compatible HTTP servers
 * while keeping query expansion via local node-llama-cpp (GGUF).
 *
 * Enable via environment variables:
 *   QMD_EMBED_URL=http://embed-server:8000    (OpenAI /v1/embeddings)
 *   QMD_RERANK_URL=http://rerank-server:8001  (POST /v1/reranks)
 *   QMD_EMBED_REMOTE_MODEL=Qwen/Qwen3-Embedding-0.6B  (optional)
 *   QMD_RERANK_REMOTE_MODEL=Qwen/Qwen3-Reranker-0.6B  (optional)
 *
 * When QMD_EMBED_URL is set, embeddings bypass node-llama-cpp entirely.
 * When QMD_RERANK_URL is set, reranking bypasses node-llama-cpp entirely.
 * Query expansion (generate) always uses local GGUF via LlamaCpp.
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  RerankDocument,
  RerankDocumentResult,
  RerankOptions,
  RerankResult,
  Queryable,
} from "./llm.js";

import { LlamaCpp, type LlamaCppConfig } from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

export type RemoteLLMConfig = {
  /** Base URL for OpenAI-compatible embeddings endpoint (e.g. http://embed:8000) */
  embedUrl?: string;
  /** Base URL for reranking endpoint (e.g. http://rerank:8001) */
  rerankUrl?: string;
  /** Model name to send in embed requests (default: Qwen/Qwen3-Embedding-0.6B) */
  embedModel?: string;
  /** Model name to send in rerank requests (default: Qwen/Qwen3-Reranker-0.6B) */
  rerankModel?: string;
  /** Timeout in ms for HTTP requests (default: 30000) */
  timeoutMs?: number;
  /** LlamaCpp config for local query expansion (generate) */
  llamaCppConfig?: LlamaCppConfig;
};

// =============================================================================
// Remote rerank truncation budgets
// =============================================================================
// The remote rerank server (llama.cpp ranking backend) has a fixed per-slot
// context (commonly 2048 tokens) and returns 400 when the full prompt
// (system+instruct template + query + document) exceeds it. RemoteLLM has no
// local tokenizer (that is the whole point of remote mode), so we truncate by
// characters before POSTing. Defaults are sized to keep query+doc+template
// under a 2048-token slot even for dense (non-English) text at ~3 chars/token:
//   doc 4000 chars (~1333 tok) + query 400 chars (~133 tok) + template (~120 tok)
//   ≈ 1586 tokens, well under 2048.
// Bump these (and the server's n_ctx) if your reranker slot has more room.
const RERANK_MAX_DOC_CHARS = (() => {
  const v = parseInt(process.env.QMD_RERANK_MAX_DOC_CHARS ?? "", 10);
  return Number.isFinite(v) && v > 0 ? v : 4000;
})();
const RERANK_MAX_QUERY_CHARS = (() => {
  const v = parseInt(process.env.QMD_RERANK_MAX_QUERY_CHARS ?? "", 10);
  return Number.isFinite(v) && v > 0 ? v : 400;
})();

// =============================================================================
// RemoteLLM Implementation
// =============================================================================

export class RemoteLLM implements LLM {
  private readonly embedUrl: string | null;
  private readonly rerankUrl: string | null;
  private readonly embedModelName_: string;
  private readonly rerankModelName_: string;
  private readonly timeoutMs: number;

  /** Local LlamaCpp for query expansion (generate). Lazy-initialized. */
  private llamaCpp: LlamaCpp | null = null;
  private readonly llamaCppConfig: LlamaCppConfig;

  constructor(config: RemoteLLMConfig = {}) {
    this.embedUrl = config.embedUrl || process.env.QMD_EMBED_URL || null;
    this.rerankUrl = config.rerankUrl || process.env.QMD_RERANK_URL || null;
    this.embedModelName_ = config.embedModel || process.env.QMD_EMBED_REMOTE_MODEL || "Qwen/Qwen3-Embedding-0.6B";
    this.rerankModelName_ = config.rerankModel || process.env.QMD_RERANK_REMOTE_MODEL || "Qwen/Qwen3-Reranker-0.6B";
    this.timeoutMs = config.timeoutMs ?? 30_000;
    this.llamaCppConfig = config.llamaCppConfig ?? {};
  }

  /** Whether remote embedding is configured */
  get hasRemoteEmbed(): boolean { return !!this.embedUrl; }
  /** Whether remote reranking is configured */
  get hasRemoteRerank(): boolean { return !!this.rerankUrl; }

  // Model-name getters — store.ts reads these to pick fingerprints/defaults.
  get embedModelName(): string {
    return this.embedUrl ? this.embedModelName_ : this.ensureLlamaCpp().embedModelName;
  }
  get rerankModelName(): string {
    return this.rerankUrl ? this.rerankModelName_ : this.ensureLlamaCpp().rerankModelName;
  }
  get generateModelName(): string {
    return this.ensureLlamaCpp().generateModelName;
  }

  /**
   * Get or create the local LlamaCpp instance for query expansion.
   * Only loaded when generate()/expandQuery() (or a local fallback) is needed.
   */
  private ensureLlamaCpp(): LlamaCpp {
    if (!this.llamaCpp) {
      this.llamaCpp = new LlamaCpp(this.llamaCppConfig);
    }
    return this.llamaCpp;
  }

  // ===========================================================================
  // Embedding — remote HTTP
  // ===========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    if (!this.embedUrl) {
      // Fallback to local LlamaCpp
      return this.ensureLlamaCpp().embed(text, options);
    }

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      const response = await fetch(`${this.embedUrl}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input: text,
          model: this.embedModelName_,
        }),
        signal: controller.signal,
      }).finally(() => clearTimeout(timer));

      if (!response.ok) {
        console.error(`Remote embed failed: ${response.status} ${response.statusText}`);
        return null;
      }

      const data = await response.json() as {
        data: Array<{ embedding: number[] }>;
        model: string;
      };

      if (!data.data?.[0]?.embedding) {
        console.error("Remote embed: unexpected response format");
        return null;
      }

      return {
        embedding: data.data[0].embedding,
        model: data.model || this.embedModelName_,
      };
    } catch (error) {
      console.error("Remote embed error:", error);
      return null;
    }
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    if (!this.embedUrl) {
      return this.ensureLlamaCpp().embedBatch(texts, options);
    }

    if (texts.length === 0) return [];

    // Split into sub-batches to avoid OOM on GPU embed server.
    // Adaptive batch size: smaller batches for longer texts to prevent CUDA OOM.
    const totalChars = texts.reduce((sum, t) => sum + t.length, 0);
    const avgChars = totalChars / texts.length;
    const REMOTE_BATCH_SIZE = avgChars > 2000 ? 8 : avgChars > 1000 ? 16 : 32;
    if (texts.length > REMOTE_BATCH_SIZE) {
      const results: (EmbeddingResult | null)[] = [];
      for (let i = 0; i < texts.length; i += REMOTE_BATCH_SIZE) {
        const chunk = texts.slice(i, i + REMOTE_BATCH_SIZE);
        const chunkResults = await this.embedBatch(chunk, options);
        results.push(...chunkResults);
      }
      return results;
    }

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs * 2); // longer for batch

      const response = await fetch(`${this.embedUrl}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input: texts,
          model: this.embedModelName_,
        }),
        signal: controller.signal,
      }).finally(() => clearTimeout(timer));

      if (!response.ok) {
        console.error(`Remote embedBatch failed: ${response.status}`);
        return texts.map(() => null);
      }

      const data = await response.json() as {
        data: Array<{ index: number; embedding: number[] }>;
        model: string;
      };

      // Map by index to preserve order
      const resultMap = new Map<number, number[]>();
      for (const item of data.data) {
        resultMap.set(item.index, item.embedding);
      }

      return texts.map((_, i) => {
        const embedding = resultMap.get(i);
        if (!embedding) return null;
        return { embedding, model: data.model || this.embedModelName_ };
      });
    } catch (error) {
      console.error("Remote embedBatch error:", error);
      return texts.map(() => null);
    }
  }

  // ===========================================================================
  // Reranking — remote HTTP
  // ===========================================================================

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    if (!this.rerankUrl) {
      return this.ensureLlamaCpp().rerank(query, documents, options);
    }

    if (documents.length === 0) {
      return { results: [], model: this.rerankModelName_ };
    }

    const neutralResults = (): RerankDocumentResult[] =>
      documents.map((d, i) => ({ file: d.file, score: 0.5, index: i }));

    // Truncate query and each document to fit the remote reranker's context
    // window. Without a local tokenizer we cap by characters; oversized inputs
    // otherwise make the ranking backend return 400 (then we lose reranking).
    const safeQuery =
      query.length > RERANK_MAX_QUERY_CHARS ? query.slice(0, RERANK_MAX_QUERY_CHARS) : query;
    const safeDocs = documents.map(d =>
      d.text.length > RERANK_MAX_DOC_CHARS ? d.text.slice(0, RERANK_MAX_DOC_CHARS) : d.text
    );

    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      const response = await fetch(`${this.rerankUrl}/v1/reranks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: safeQuery,
          documents: safeDocs,
          model: this.rerankModelName_,
        }),
        signal: controller.signal,
      }).finally(() => clearTimeout(timer));

      if (!response.ok) {
        console.error(`Remote rerank failed: ${response.status}`);
        // Fallback: return documents in original order with neutral scores
        return { results: neutralResults(), model: this.rerankModelName_ };
      }

      const data = await response.json() as {
        results: Array<{ index: number; relevance_score: number }>;
        model: string;
      };

      return {
        results: data.results.map(r => ({
          file: documents[r.index]?.file ?? "",
          score: r.relevance_score,
          index: r.index,
        })),
        model: data.model || this.rerankModelName_,
      };
    } catch (error) {
      console.error("Remote rerank error:", error);
      return { results: neutralResults(), model: this.rerankModelName_ };
    }
  }

  // ===========================================================================
  // Query expansion & generation — always local LlamaCpp
  // ===========================================================================

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    return this.ensureLlamaCpp().generate(prompt, options);
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean; intent?: string }
  ): Promise<Queryable[]> {
    return this.ensureLlamaCpp().expandQuery(query, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // For remote models, assume they exist if URL is configured
    if (model === this.embedModelName_ && this.embedUrl) {
      return { name: model, exists: true };
    }
    if (model === this.rerankModelName_ && this.rerankUrl) {
      return { name: model, exists: true };
    }
    return this.ensureLlamaCpp().modelExists(model);
  }

  async dispose(): Promise<void> {
    if (this.llamaCpp) {
      await this.llamaCpp.dispose();
      this.llamaCpp = null;
    }
  }
}

// =============================================================================
// Factory: choose RemoteLLM or LlamaCpp based on env
// =============================================================================

/**
 * Returns true if any remote LLM env vars are configured.
 */
export function isRemoteLLMConfigured(): boolean {
  return !!(process.env.QMD_EMBED_URL || process.env.QMD_RERANK_URL);
}

/**
 * Create the appropriate LLM instance based on environment configuration.
 * If QMD_EMBED_URL or QMD_RERANK_URL is set, returns RemoteLLM.
 * Otherwise returns LlamaCpp (default).
 */
export function createLLM(config: LlamaCppConfig & Pick<RemoteLLMConfig, "embedUrl" | "rerankUrl" | "timeoutMs"> = {}): LLM {
  if (isRemoteLLMConfigured()) {
    // Remote model *names* are sourced from QMD_EMBED_REMOTE_MODEL /
    // QMD_RERANK_REMOTE_MODEL inside the RemoteLLM constructor. The local
    // LlamaCpp (used for query expansion only) still receives the GGUF config.
    return new RemoteLLM({
      embedUrl: config.embedUrl,
      rerankUrl: config.rerankUrl,
      timeoutMs: config.timeoutMs,
      llamaCppConfig: {
        embedModel: config.embedModel,
        generateModel: config.generateModel,
        rerankModel: config.rerankModel,
        modelCacheDir: config.modelCacheDir,
        expandContextSize: config.expandContextSize,
        inactivityTimeoutMs: config.inactivityTimeoutMs,
        disposeModelsOnInactivity: config.disposeModelsOnInactivity,
      },
    });
  }

  return new LlamaCpp(config);
}
