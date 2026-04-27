import * as React from "react";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

/**
 * Tick mark on a slider track at a registry rover's value for the
 * field this slider controls. Coloured to match the rover's marker
 * on the prediction chart so the cross-component association is
 * preserved.
 */
export interface SliderTick {
  rover_name: string;
  value: number;
  color: string;
}

interface DesignSliderFieldProps {
  id: string;
  label: string;
  unit: string;
  description: string;
  min: number;
  max: number;
  step: number;
  value: number;
  ticks?: SliderTick[];
  /** Format the readout (e.g. integer formatting for grouser_count). */
  format?: (v: number) => string;
  /** Round / clamp to the field's domain (defaults to identity). */
  sanitize?: (v: number) => number;
  disabled?: boolean;
  onChange: (v: number) => void;
}

/**
 * One row of the design form: slider + editable numeric input,
 * plus optional coloured tick marks at registry rover values.
 *
 * Layout
 * ------
 * Two-row stack inside a single grid cell:
 *
 *   [label .................. unit]
 *   [slider track w/ ticks .. input]
 *   [description ...... (min–max)]
 *
 * The slider commits values continuously (matches the input box);
 * the input box is the precision escape hatch when scrubbing on the
 * track is too coarse.
 */
export function DesignSliderField({
  id,
  label,
  unit,
  description,
  min,
  max,
  step,
  value,
  ticks = [],
  format = (v) => formatDefault(v, step),
  sanitize = (v) => v,
  disabled,
  onChange,
}: DesignSliderFieldProps) {
  const handleSlider = React.useCallback(
    (v: number[]) => onChange(sanitize(v[0] ?? min)),
    [onChange, sanitize, min],
  );
  const handleInput = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const raw = e.target.value;
      if (raw === "") {
        onChange(sanitize(min));
        return;
      }
      const num = Number(raw);
      onChange(sanitize(Number.isFinite(num) ? num : min));
    },
    [onChange, sanitize, min],
  );

  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="flex items-center justify-between">
        <span>{label}</span>
        <span className="text-xs text-[var(--color-muted-foreground)]">
          {unit || "—"}
        </span>
      </Label>

      <div className="flex items-center gap-3">
        <SliderWithTicks
          id={id}
          min={min}
          max={max}
          step={step}
          value={value}
          ticks={ticks}
          disabled={disabled}
          onValueChange={handleSlider}
        />
        <Input
          aria-label={`${label} value`}
          type="number"
          inputMode="decimal"
          disabled={disabled}
          min={min}
          max={max}
          step={step}
          value={format(value)}
          onChange={handleInput}
          className="h-8 w-20 text-right tabular-nums"
        />
      </div>

      <p className="text-[0.65rem] text-[var(--color-muted-foreground)]">
        {description}{" "}
        <span className="opacity-70">
          ({format(min)}–{format(max)})
        </span>
      </p>
    </div>
  );
}

function formatDefault(v: number, step: number): string {
  if (!Number.isFinite(v)) return "0";
  // Pick a decimal precision from the step so 0.005 -> 3 places, 5 -> 0.
  const dp = Math.max(0, -Math.floor(Math.log10(step)));
  return v.toFixed(dp);
}

interface SliderWithTicksProps {
  id: string;
  min: number;
  max: number;
  step: number;
  value: number;
  ticks: SliderTick[];
  disabled?: boolean;
  onValueChange: (v: number[]) => void;
}

/**
 * Slider with absolute-positioned colored tick marks above the track.
 *
 * Ticks are read-only -- they live in a `pointer-events: none` layer
 * so they never steal focus or drag from the actual thumb. Hovering
 * a tick shows the rover name via the native `title` attribute,
 * which is good enough for a 4-rover registry without dragging in
 * another tooltip primitive.
 */
function SliderWithTicks({
  id,
  min,
  max,
  step,
  value,
  ticks,
  disabled,
  onValueChange,
}: SliderWithTicksProps) {
  const span = max - min;
  return (
    <div className="relative flex-1 py-2">
      <Slider
        id={id}
        min={min}
        max={max}
        step={step}
        value={[value]}
        disabled={disabled}
        onValueChange={onValueChange}
      />
      {ticks.length > 0 && span > 0 ? (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 h-full"
        >
          {ticks.map((tick) => {
            const clamped = Math.max(min, Math.min(max, tick.value));
            const pct = ((clamped - min) / span) * 100;
            const outOfRange = tick.value < min || tick.value > max;
            return (
              <span
                key={tick.rover_name}
                title={`${tick.rover_name}: ${tick.value}`}
                className={cn(
                  "absolute top-1/2 h-3 w-[2px] -translate-x-1/2 -translate-y-1/2 rounded",
                  outOfRange && "opacity-60",
                )}
                style={{
                  left: `${pct}%`,
                  backgroundColor: tick.color,
                }}
              />
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
