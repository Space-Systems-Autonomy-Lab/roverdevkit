import path from "node:path";
import { fileURLToPath } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(here, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // Forward backend calls during dev so the frontend can talk to
      // the FastAPI server without baking in an absolute URL. The
      // backend mounts routes at /healthz, /scenarios, /registry,
      // /predict, /version (no /api prefix yet); the proxy mirrors
      // that 1:1.
      "/healthz": "http://localhost:8000",
      "/version": "http://localhost:8000",
      "/scenarios": "http://localhost:8000",
      "/registry": "http://localhost:8000",
      "/predict": "http://localhost:8000",
    },
  },
});
