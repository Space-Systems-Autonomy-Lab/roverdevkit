import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useDesignStore } from "@/store/design-store";
import { DESIGN_BOUNDS, type DesignVector } from "@/types/api";

/**
 * Twelve-field numeric form for the rover design vector.
 *
 * Each input enforces the schema's static bounds via min / max / step;
 * the backend's Pydantic schema is the canonical authority and will
 * 422 anything that slips through (e.g. typed-in NaN). The form
 * commits values to the Zustand store on every keystroke so the
 * "predict" mutation always sees the current draft.
 */
export function DesignForm({ disabled }: { disabled?: boolean }) {
  const design = useDesignStore((s) => s.design);
  const setDesignField = useDesignStore((s) => s.setDesignField);
  const resetDesign = useDesignStore((s) => s.resetDesign);

  const fields = Object.keys(DESIGN_BOUNDS) as (keyof DesignVector)[];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        {fields.map((key) => {
          const bounds = DESIGN_BOUNDS[key];
          const value = design[key];
          return (
            <div key={key} className="space-y-1">
              <Label
                htmlFor={key}
                className="flex items-center justify-between"
              >
                <span>{bounds.label}</span>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  {bounds.unit || "—"}
                </span>
              </Label>
              <Input
                id={key}
                type="number"
                inputMode="decimal"
                disabled={disabled}
                min={bounds.min}
                max={bounds.max}
                step={bounds.step}
                value={value}
                onChange={(e) => {
                  const raw = e.target.value;
                  const num = raw === "" ? Number.NaN : Number(raw);
                  if (key === "n_wheels") {
                    setDesignField("n_wheels", num === 4 ? 4 : 6);
                  } else if (key === "grouser_count") {
                    setDesignField(
                      "grouser_count",
                      Number.isFinite(num) ? Math.round(num) : 0,
                    );
                  } else {
                    (setDesignField as (k: typeof key, v: number) => void)(
                      key,
                      Number.isFinite(num) ? num : 0,
                    );
                  }
                }}
              />
              <p className="text-[0.65rem] text-[var(--color-muted-foreground)]">
                {bounds.description}{" "}
                <span className="opacity-70">
                  ({bounds.min}–{bounds.max})
                </span>
              </p>
            </div>
          );
        })}
      </div>
      <div className="flex justify-end">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={disabled}
          onClick={resetDesign}
        >
          Reset to defaults
        </Button>
      </div>
    </div>
  );
}
