import clsx from "clsx";
import { ReactNode } from "react";

export function Panel({
  children,
  className,
  title,
  subtitle,
  action,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <section className={clsx("border border-line bg-ink-800/60 rounded", className)}>
      {(title || action) && (
        <div className="flex items-start justify-between gap-4 px-5 pt-4 pb-3 border-b border-line-soft">
          <div>
            {title && (
              <h2 className="font-display text-[15px] font-medium text-paper">{title}</h2>
            )}
            {subtitle && <p className="text-[12.5px] text-paper-faint mt-0.5">{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}
