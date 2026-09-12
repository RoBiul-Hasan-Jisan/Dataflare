import clsx from "clsx";
import { ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
}

export default function Button({
  variant = "primary",
  size = "md",
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded font-medium transition-colors disabled:opacity-45 disabled:cursor-not-allowed",
        size === "md" ? "px-4 py-2 text-[13.5px]" : "px-3 py-1.5 text-[12.5px]",
        variant === "primary" && "bg-flare text-white hover:bg-flare-bright",
        variant === "secondary" &&
          "bg-ink-600 text-paper border border-line hover:border-paper-faint",
        variant === "ghost" && "text-paper-dim hover:text-paper hover:bg-ink-700",
        variant === "danger" && "bg-signal-red/15 text-signal-red hover:bg-signal-red/25",
        className
      )}
      {...props}
    />
  );
}
