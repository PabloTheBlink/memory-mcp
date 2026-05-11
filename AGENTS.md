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
No requires external dependencies like Ollama. It uses local embeddings via `@xenova/transformers` (model: `nomic-embed-text` or `all-MiniLM-L6-v2`).

## Protocolo de Inicio (MANDATORIO)
Antes de realizar cualquier tarea o responder a la primera petición del usuario en cada nueva sesión, el agente DEBE:
1. Ejecutar `memory_status` para identificar el contexto activo y los nodos principales.
2. Ejecutar `memory_recall` con una consulta sobre "preferencias de desarrollo, reglas de estilo y perfil del usuario".
3. Sincronizar su comportamiento y decisiones técnicas con la información recuperada de la memoria.

## Comandos
`npm run dev` (stdio), `npm run review` (HTTP inspector), `npm run maintenance:force`.
