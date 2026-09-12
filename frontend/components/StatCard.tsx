import clsx from "clsx";
import { ReactNode } from "react";

export default function StatCard({
  label,
  value,
  suffix,
  tone = "default",
  icon,
}: {
  label: string;
  value: string | number;
  suffix?: string;
  tone?: "default" | "good" | "warn" | "bad";
  icon?: ReactNode;
}) {
  const toneColor = {
    default: "text-paper",
    good: "text-signal-teal",
    warn: "text-signal-amber",
    bad: "text-signal-red",
  }[tone];

  return (
    <div className="border border-line bg-ink-800/60 rounded px-4 py-3.5">
      <div className="flex items-center justify-between text-paper-faint text-[11.5px] mb-2">
        <span>{label}</span>
        {icon}
      </div>
      <div className={clsx("font-display text-[24px] font-semibold tabular", toneColor)}>
        {value}
        {suffix && <span className="text-[13px] text-paper-faint ml-1 font-body">{suffix}</span>}
      </div>
    </div>
  );
}
