# Memory MCP
Associative memory (SQLite) for agents.

## Tools
- `memory_activate`: Create/activate concept. Use for new knowledge.
- `memory_associate`: Link concepts (causal, temporal, semantic, episodic).
- `memory_recall`: Search by similarity + activation. Cross-language (0.35 threshold).
- `memory_set_context`: Hub-based recall bias (e.g. `user`, `project:name`).
- `memory_status`: Stats & top nodes.
- `memory_maintenance`: Auto-merge (0.97 sim), link, and prune (<0.05 strength).

## Patterns
- **Episodic binding**: Activating concepts in a context links them to that context node (`[ctx:name]`).
- **Recall scoring**: Semantic (0.45) + Activation (0.30) + Strength (0.15) + Importance (0.10).
- **Maintenance**: Rate-limited (1/hr). Merges duplicates using embedding + Levenshtein.

## Setup
Requires local Ollama (`nomic-embed-text`) at `http://localhost:11434`.

## Commands
`npm run dev` (stdio), `npm run review` (HTTP inspector), `npm run maintenance:force`.
