import type { ReactNode } from "react";

import { useHealth, useVersion } from "@/hooks/use-health";

/** Top-level layout: header with version + body slot. */
export function AppShell({ children }: { children: ReactNode }) {
  const { data: health } = useHealth();
  const { data: version } = useVersion();

  return (
    <div className="min-h-screen">
      <header className="border-b bg-[var(--color-card)]">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              roverdevkit · tradespace explorer
            </h1>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              Lunar micro-rover design space, powered by the W8 quantile-XGBoost
              surrogate over the corrected mission evaluator.
            </p>
          </div>
          <StatusBadge
            ok={health?.surrogate_loaded ?? false}
            apiVersion={version?.api_version ?? "—"}
            datasetVersion={version?.dataset_version ?? "—"}
          />
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-6">{children}</main>
      <footer className="mx-auto max-w-6xl px-6 py-6 text-xs text-[var(--color-muted-foreground)]">
        Phase 3 prototype · backend: FastAPI + XGBoost · frontend: React + Vite
        · Space Systems Autonomy Lab, Duke University.
      </footer>
    </div>
  );
}

function StatusBadge({
  ok,
  apiVersion,
  datasetVersion,
}: {
  ok: boolean;
  apiVersion: string;
  datasetVersion: string;
}) {
  return (
    <div className="flex items-center gap-3 text-xs">
      <span
        className={
          "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 " +
          (ok
            ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-200"
            : "bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-200")
        }
      >
        <span
          aria-hidden
          className={
            "h-1.5 w-1.5 rounded-full " +
            (ok ? "bg-emerald-500" : "bg-amber-500")
          }
        />
        {ok ? "Surrogate live" : "Surrogate degraded"}
      </span>
      <span className="text-[var(--color-muted-foreground)]">
        API {apiVersion} · dataset {datasetVersion}
      </span>
    </div>
  );
}
