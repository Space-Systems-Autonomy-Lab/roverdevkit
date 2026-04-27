import type { ReactNode } from "react";

import { AboutModelDialog } from "@/components/about-model-dialog";
import { Button } from "@/components/ui/button";
import { useHealth } from "@/hooks/use-health";

/** Top-level layout: header with status badge + body slot. */
export function AppShell({ children }: { children: ReactNode }) {
  const { data: health } = useHealth();

  return (
    <div className="min-h-screen">
      <header className="border-b bg-[var(--color-card)]">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              roverdevkit · tradespace explorer
            </h1>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              Predict mobility, mass, range, and energy margin for a candidate
              lunar micro-rover, with calibrated 90% prediction intervals.
            </p>
          </div>
          <StatusBadge ok={health?.surrogate_loaded ?? false} />
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-6">{children}</main>
      <footer className="mx-auto max-w-6xl px-6 py-6 text-xs text-[var(--color-muted-foreground)]">
        Space Systems Autonomy Lab · Duke University
      </footer>
    </div>
  );
}

function StatusBadge({ ok }: { ok: boolean }) {
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
        {ok ? "Model online" : "Model unavailable"}
      </span>
      <AboutModelDialog>
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-2.5 text-xs font-normal"
        >
          About this model
        </Button>
      </AboutModelDialog>
    </div>
  );
}
