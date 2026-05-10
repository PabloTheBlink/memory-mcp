# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run build        # Compile TypeScript → dist/
npm run dev          # Run MCP server directly via tsx (no build needed)
npm run review       # Run the review/debug HTTP server (src/review-server.ts)
npm run maintenance  # Run maintenance pass (requires built dist/)
npm run maintenance:force  # Force maintenance even if run recently
```

There are no tests. The server is validated by connecting it to an MCP client.

## Architecture

This is a **Model Context Protocol (MCP) server** that exposes an associative memory system backed by SQLite (`data/memory.db`). It runs over stdio and is designed to be connected to an AI agent as a long-term memory tool.

### Data model

The graph lives in SQLite with three tables:
- **nodes** — concepts, each with a text label, a 768-dim embedding (stored as JSON), a `strength` (0–1, Ebbinghaus decay), and `access_count`
- **edges** — undirected weighted associations between nodes; type is one of `causal | temporal | semantic | episodic`
- **meta** — key/value store for `active_context` and `last_consolidation` / `last_maintenance` timestamps

Edges are stored with canonical ordering (`min(from,to)` first) to keep them undirected.

### Source modules

| File | Role |
|------|------|
| `graph.ts` | All SQLite access — schema init, CRUD for nodes/edges/meta |
| `embeddings.ts` | Calls local Ollama (`nomic-embed-text` model) for embeddings; cosine similarity helpers |
| `activation.ts` | BFS spreading activation from seed nodes, decaying by `weight × decayFactor` per hop |
| `context.ts` | Context hub logic — context nodes are labeled `[ctx:name]` and act as episodic binding hubs |
| `decay.ts` | Ebbinghaus decay: `newStrength = strength × e^(-days/strength)`; prunes nodes/edges below 0.05 |
| `maintenance.ts` | Orchestrates full maintenance: decay → semantic linking → auto-merge duplicates → orphan pruning |
| `index.ts` | MCP server entry point — registers and handles the 6 tools |
| `review-server.ts` | Separate HTTP server for inspecting/debugging graph state |

### Key design decisions

**Embeddings require Ollama running locally** on `http://localhost:11434` with the `nomic-embed-text` model pulled. All tool calls that create or recall nodes will fail if Ollama is unreachable.

**Context nodes** are regular graph nodes with the `[ctx:]` prefix. They act as hubs — memories formed in a context get a weak `episodic` edge to the context node (0.05 boost), which biases spreading activation toward context-relevant memories during recall. The active context defaults to the git repo name.

**`memory_recall` scoring** blends three signals: semantic similarity (0.5), spreading activation (0.35), and node strength (0.15). The similarity threshold for recall is intentionally low (0.35) to handle cross-language queries.

**Maintenance** is rate-limited to once per hour by default (checked via `meta.last_maintenance`). The auto-merge step collapses near-duplicate nodes using both embedding similarity (≥0.97) and text similarity (≥0.65 Levenshtein ratio), keeping the node with higher `access_count`.
