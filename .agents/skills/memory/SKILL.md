---
name: memory
description: "Protocolo de Memoria y Persistencia para Pablo usando herramientas MCP de memoria. Úsalo cuando necesites guardar, recuperar o gestionar memoria entre sesiones: inicio de sesión (status + recall), aprender nuevos conceptos o reglas del usuario, asociar nodos, reconstruir historial, o hacer mantenimiento de memoria al cierre."
---

# Protocolo de Memoria y Persistencia (v1.1)

## Inicio de sesión (ejecutar siempre en orden)

1. `memory_status` — ver estadísticas y contexto activo
2. `memory_get_context_summary` — ver hubs conceptuales y actividad reciente en el contexto actual
3. `memory_recall` — query: `"preferencias de desarrollo, reglas de estilo y perfil de Pablo"`
4. `memory_suggest` — obtener sugerencias asociativas para arrancar con el contexto mental "caliente"

## Durante la sesión (Proactividad)

| Situación | Herramienta |
|---|---|
| Preferencia o regla explícita de Pablo | `memory_learn_rule` |
| Nuevo hecho / concepto / decisión técnica | `memory_activate` (importance > 0.7 para decisiones importantes) |
| Vincular dos conceptos | `memory_associate` (tipos: `causal`, `temporal`, `semantic`, `episodic`, `abstraction`) |
| Reconstruir historial / secuencia | `memory_replay` desde el concepto de entrada |
| ¿Qué más debería saber? | `memory_suggest` (proactivo) |
| Cambio de tarea o duda de contexto | `memory_get_context_summary` |

**Optimización de Memoria**: El sistema ahora autovíncula conceptos nuevos a nodos activos (`hot nodes`), creando una red episódica densa automáticamente.

## Cierre de sesión

- `memory_consolidate` — siempre al cerrar (decaimiento Ebbinghaus + poda de débiles)
- `memory_maintenance` — realizar mantenimiento profundo si la red ha crecido significativamente.

## Reglas de oro (Proactivas)

- **Anticipación**: Si detectas que vas a realizar una tarea repetitiva, usa `memory_learn_rule` para que Pablo no tenga que repetirlo en la siguiente sesión.
- **Descubrimiento**: Usa `memory_suggest` periódicamente para descubrir conexiones que no son obvias en la superficie pero están vinculadas semánticamente.
- **Higiene**: Los nodos ahora se limpian y asocian proactivamente en segundo plano durante el mantenimiento.
