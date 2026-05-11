# 🧠 Memory MCP: Sistema de Memoria Asociativa

Servidor MCP (Model Context Protocol) que implementa una memoria semántica y episódica para agentes de IA, utilizando un grafo de conocimiento pesado y embeddings locales.

## 🚀 Características principales

- **Memoria Semántica**: Recuperación basada en similitud vectorial (local con `@xenova/transformers`).
- **Activación Expandida (Spreading Activation)**: Recuperación de conceptos relacionados mediante saltos en el grafo.
- **Memoria Episódica y Temporal**: Enlace automático de conceptos que aparecen juntos en el tiempo.
- **Curva de Olvido (Ebbinghaus)**: Los conceptos pierden fuerza con el tiempo si no se refuerzan, evitando la saturación.
- **Dashboard de Visualización**: Interfaz web interactiva para ver la "neurona" del sistema en tiempo real.
- **Independiente del Modelo**: Funciona localmente sin necesidad de APIs externas para embeddings.

## 🛠️ Requisitos

- **Node.js**: Versión 18 o superior.
- **npm**: Gestor de paquetes.

## 📦 Instalación

1. Clona el repositorio o descarga los archivos.
2. Instala las dependencias:

```bash
npm install
```

*Nota: La primera vez que se ejecute, se descargará automáticamente el modelo de embeddings (`Xenova/all-MiniLM-L6-v2`), lo que puede tardar un momento dependiendo de tu conexión.*

## ⌨️ Comandos Disponibles

| Comando                     | Descripción                                                             |
| --------------------------- | ----------------------------------------------------------------------- |
| `npm run dev`               | Inicia el servidor MCP en modo desarrollo (Stdio).                      |
| `npm run build`             | Compila el proyecto TypeScript a JavaScript (`dist/`).                  |
| `npm run start`             | Ejecuta la versión compilada del servidor.                              |
| `npm run review`            | Inicia el servidor de visualización en `http://localhost:3131`.         |
| `npm run maintenance`       | Ejecuta el mantenimiento (fusión de duplicados, poda de memoria débil). |
| `npm run maintenance:force` | Fuerza el mantenimiento ignorando el límite de tiempo de 1 hora.        |

## 🔌 Configuración en Clientes MCP (como Claude Desktop)

Para usar este servidor en tu cliente MCP preferido, añade lo siguiente a tu configuración:

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": ["/Users/xx/memory-mcp/dist/index.js"]
    }
  }
}
```

## 🛠️ Herramientas Disponibles (Tools)

- `memory_activate`: Activa un concepto (lo crea si no existe) y lo vincula al contexto actual.
- `memory_associate`: Crea un enlace manual entre dos conceptos (causal, temporal, semántico).
- `memory_recall`: Busca memorias relevantes usando una consulta en lenguaje natural.
- `memory_status`: Muestra estadísticas del sistema y el contexto activo.
- `memory_set_context`: Cambia el contexto de trabajo (ej. "proyecto:x", "personal").
- `memory_maintenance`: Limpieza y optimización manual del grafo.
- `memory_replay`: Reconstruye una "narrativa" siguiendo los enlaces temporales desde un concepto.

## 📊 Visualización

Ejecuta `npm run review` y abre `http://localhost:3131` en tu navegador. Podrás ver:
- El grafo de neuronas y sus conexiones.
- Destellos visuales cuando una memoria es consultada o "disparada".
- Paneles de información detallada de cada concepto.
- Buscador de memorias.

---
*Desarrollado para potenciar la autonomía y continuidad de agentes inteligentes.*
