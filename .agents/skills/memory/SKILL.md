# Skill: Protocolo de Memoria y Persistencia

Este protocolo es de obligado cumplimiento para asegurar la integridad de la memoria de Pablo y evitar redundancias o inconsistencias.

## 1. Fase de Inicio (Recuperación)
- **Status Obligatorio**: Ejecutar `memory_status` al primer turno de cada sesión.
- **Recall Contextual**: Realizar un `memory_recall` sobre "preferencias de desarrollo, reglas de estilo y perfil de Pablo".
- **Sincronización**: Adaptar el tono y las decisiones técnicas (stack, testing, herramientas) a lo recuperado.

## 2. Fase de Desarrollo (Activación)
- **Nuevos Conceptos**: Cada vez que se aprenda algo nuevo sobre un proyecto o sobre Pablo, usar `memory_activate`.
- **Importancia**: Asignar importancia > 0.7 a decisiones técnicas o reglas de estilo.
- **Vínculos**: No dejar nodos aislados. Asociar siempre nuevos conceptos a los hubs existentes (`Pablo`, `project:name`, etc.) mediante `memory_associate`.

## 3. Fase de Cierre (Verificación)
- **Integridad**: Antes de terminar la sesión, revisar si hay incongruencias entre lo aprendido hoy y lo que ya estaba en memoria.
- **Disparo Neuronal**: Asegurar que las herramientas de memoria estén disparando los eventos de visualización (`fireNode`).
- **Mantenimiento**: Si se han creado muchos nodos, ejecutar `memory_maintenance` para consolidar.

## 4. Reglas de Oro
- **No suponer**: Si hay dudas sobre una preferencia, preguntar antes de guardar.
- **Adaptabilidad**: Si el stack cambia (ej: de PHP a TS), actualizar las reglas de testing asociadas.
- **Brevitud**: Los nodos deben ser concisos y directos, siguiendo el estilo de comunicación de Pablo.
