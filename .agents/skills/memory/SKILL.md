---
name: memory
description: "Protocolo de Memoria y Persistencia para Pablo usando herramientas MCP de memoria. Úsalo cuando necesites guardar, recuperar o gestionar memoria entre sesiones: inicio de sesión (status + recall), aprender nuevos conceptos o reglas del usuario, asociar nodos, reconstruir historial, o hacer mantenimiento de memoria al cierre."
---

# Protocolo de Memoria y Persistencia

## Inicio de sesión (ejecutar siempre en orden)

1. `memory_status` — ver estadísticas y contexto activo
2. `memory_set_context` — con `"project:bal_isquad"` u otro contexto activo (vacío = auto-detección)
3. `memory_recall` — query: `"preferencias de desarrollo, reglas de estilo y perfil de Pablo"`
4. Adaptar tono y decisiones técnicas a lo recuperado

## Durante la sesión

| Situación | Herramienta |
|---|---|
| Preferencia o regla explícita de Pablo | `memory_learn_rule` |
| Nuevo hecho / concepto / decisión técnica | `memory_activate` (importance > 0.7 para decisiones importantes) |
| Vincular dos conceptos | `memory_associate` (tipos: `causal`, `temporal`, `semantic`, `episodic`, `abstraction`) |
| Reconstruir historial / secuencia de decisiones | `memory_replay` desde el concepto de entrada |

**Nunca dejar nodos aislados**: asociar siempre a hubs `Pablo` o `project:bal_isquad`.

## Cierre de sesión

- `memory_consolidate` — siempre al cerrar (decaimiento Ebbinghaus + poda de débiles)
- `memory_maintenance` — solo si se crearon muchos nodos (rate-limited: 1/hora; incluye merge + re-linkado)

## Reglas de oro

- `memory_learn_rule` para reglas/preferencias explícitas; `memory_activate` para hechos
- Preguntar antes de guardar si hay dudas sobre una preferencia
- Nodos concisos y directos — estilo caveman cuando aplique
