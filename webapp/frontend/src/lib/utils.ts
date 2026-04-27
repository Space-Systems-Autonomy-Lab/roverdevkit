import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Standard shadcn `cn` helper: merge Tailwind class strings safely. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
