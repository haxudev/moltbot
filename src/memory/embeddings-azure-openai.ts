import { requireApiKey, resolveApiKeyForProvider } from "../agents/model-auth.js";
import type { EmbeddingProvider, EmbeddingProviderOptions } from "./embeddings.js";

export type AzureOpenAiEmbeddingClient = {
  baseUrl: string;
  headers: Record<string, string>;
  model: string;
};

function normalizeAzureEmbeddingBaseUrl(raw: string): URL {
  const trimmed = raw.trim();
  if (!trimmed) {
    throw new Error(
      "Azure OpenAI embeddings require memorySearch.remote.baseUrl (including ?api-version=...).",
    );
  }
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error(
      `Invalid Azure OpenAI embeddings baseUrl: ${JSON.stringify(trimmed)}. Expected a full https URL.`,
    );
  }
  if (parsed.protocol !== "https:") {
    throw new Error(
      `Invalid Azure OpenAI embeddings baseUrl protocol: ${parsed.protocol}. Expected https.`,
    );
  }
  if (!parsed.hostname) {
    throw new Error(`Invalid Azure OpenAI embeddings baseUrl host in: ${JSON.stringify(trimmed)}.`);
  }
  // Normalize path: ensure it ends with /embeddings, without breaking query string.
  const path = parsed.pathname.replace(/\/+$/, "");
  if (!/\/embeddings$/i.test(path)) {
    parsed.pathname = `${path || ""}/embeddings`;
  } else {
    parsed.pathname = `${path}`;
  }
  return parsed;
}

function assertHasApiVersion(url: URL): void {
  // Azure OpenAI requires api-version query parameter.
  if (!url.searchParams.has("api-version")) {
    throw new Error(
      [
        "Azure OpenAI embeddings baseUrl is missing required query parameter api-version.",
        "Set memorySearch.remote.baseUrl to include it, for example:",
        "https://{resource}.openai.azure.com/openai/deployments/{deployment}/embeddings?api-version=2023-05-15",
      ].join(" "),
    );
  }
}

function normalizeModel(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed.startsWith("azure-openai/")) {
    return trimmed.slice("azure-openai/".length);
  }
  return trimmed;
}

export async function resolveAzureOpenAiEmbeddingClient(
  options: EmbeddingProviderOptions,
): Promise<AzureOpenAiEmbeddingClient> {
  const remote = options.remote;
  const remoteApiKey = remote?.apiKey?.trim();
  const remoteBaseUrl = remote?.baseUrl?.trim();

  const apiKey = remoteApiKey
    ? remoteApiKey
    : requireApiKey(
        await resolveApiKeyForProvider({
          provider: "azure-openai",
          cfg: options.config,
          agentDir: options.agentDir,
        }),
        "azure-openai",
      );

  if (!remoteBaseUrl) {
    throw new Error(
      [
        "Azure OpenAI embeddings require memorySearch.remote.baseUrl.",
        "Expected an Azure deployments embeddings URL including api-version, for example:",
        "https://{resource}.openai.azure.com/openai/deployments/{deployment}/embeddings?api-version=2023-05-15",
      ].join(" "),
    );
  }

  const baseUrl = remoteBaseUrl;
  const headerOverrides = Object.assign({}, remote?.headers);
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    // Azure OpenAI uses api-key header, not Bearer auth.
    "api-key": apiKey,
    ...headerOverrides,
  };

  const model = normalizeModel(options.model);

  return { baseUrl, headers, model };
}

export async function createAzureOpenAiEmbeddingProvider(
  options: EmbeddingProviderOptions,
): Promise<{ provider: EmbeddingProvider; client: AzureOpenAiEmbeddingClient }> {
  const client = await resolveAzureOpenAiEmbeddingClient(options);
  const url = normalizeAzureEmbeddingBaseUrl(client.baseUrl);
  assertHasApiVersion(url);

  const embed = async (input: string[]): Promise<number[][]> => {
    if (input.length === 0) {
      return [];
    }

    const body: Record<string, unknown> = { input };
    // Some Azure deployments accept model in body, some ignore it.
    if (client.model) {
      body.model = client.model;
    }

    const res = await fetch(url.toString(), {
      method: "POST",
      headers: client.headers,
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`azure openai embeddings failed: ${res.status} ${text}`);
    }
    const payload = (await res.json()) as {
      data?: Array<{ embedding?: number[] }>;
    };
    const data = payload.data ?? [];
    return data.map((entry) => entry.embedding ?? []);
  };

  return {
    provider: {
      id: "azure-openai",
      model: client.model,
      embedQuery: async (text) => {
        const [vec] = await embed([text]);
        return vec ?? [];
      },
      embedBatch: embed,
    },
    client,
  };
}
