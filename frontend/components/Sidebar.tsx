"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import {
  LayoutDashboard,
  UploadCloud,
  ChartScatter,
  Cpu,
  Trophy,
  History,
} from "lucide-react";
import { useDataset } from "@/context/DatasetContext";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard, always: true },
  { href: "/upload", label: "Upload data", icon: UploadCloud, always: true },
  { href: "/explore", label: "Explore & EDA", icon: ChartScatter, always: false },
  { href: "/train", label: "Train a model", icon: Cpu, always: false },
  { href: "/results", label: "Results", icon: Trophy, always: false },
  { href: "/history", label: "History", icon: History, always: true },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { status } = useDataset();
  const hasData = status?.has_data ?? false;

  return (
    <aside className="w-[248px] shrink-0 border-r border-line bg-ink-950/60 hidden md:flex flex-col">
      <div className="h-16 flex items-center gap-2.5 px-6 border-b border-line">
        <svg width="22" height="22" viewBox="0 0 24 24" className="shrink-0">
          <path d="M12 1L14 9L22 11L14 13L12 21L10 13L2 11L10 9Z" fill="#FF5A1F" />
        </svg>
        <div className="leading-none">
          <div className="font-display font-semibold text-[17px] text-paper tracking-tight">
            DataFlare
          </div>
          <div className="text-[11px] text-paper-faint -mt-0.5">ml studio</div>
        </div>
      </div>

      <nav className="flex-1 py-4 px-3 space-y-0.5">
        {NAV.map((item) => {
          const active = pathname === item.href;
          const disabled = !item.always && !hasData;
          const Icon = item.icon;
          const content = (
            <div
              className={clsx(
                "flex items-center gap-3 rounded-md px-3 py-2.5 text-[13.5px] border-l-2 transition-colors",
                active
                  ? "border-flare bg-ink-700 text-paper font-medium"
                  : "border-transparent text-paper-dim hover:text-paper hover:bg-ink-800",
                disabled && "opacity-40 cursor-not-allowed hover:bg-transparent hover:text-paper-dim"
              )}
            >
              <Icon size={17} strokeWidth={1.75} />
              <span>{item.label}</span>
            </div>
          );
          return disabled ? (
            <div key={item.href} title="Upload a dataset first">
              {content}
            </div>
          ) : (
            <Link key={item.href} href={item.href}>
              {content}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-line text-[11px] text-paper-faint flex items-center gap-1.5">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-signal-teal" />
        {status?.has_data ? "Dataset loaded" : "Waiting for data"}
      </div>
    </aside>
  );
}
