"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, ChartScatter, Cpu, Trophy } from "lucide-react";
import { useDataset } from "@/context/DatasetContext";
import { api } from "@/lib/api";
import { DataInfo, Insight } from "@/lib/types";
import { Panel } from "@/components/Panel";
import StatCard from "@/components/StatCard";
import Dropzone from "@/components/Dropzone";
import { useRouter } from "next/navigation";
import { uploadFile } from "@/lib/upload";

export default function OverviewPage() {
  const { status, loading, refresh } = useDataset();
  const [info, setInfo] = useState<DataInfo | null>(null);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    if (!status?.has_data) return;
    api.get<DataInfo>("/api/data-info").then((r) => setInfo(r.data));
    api
      .get<{ insights: Insight[] }>("/api/insights")
      .then((r) => setInsights(r.data.insights ?? []));
  }, [status?.has_data]);

  async function handleFile(file: File) {
    setUploading(true);
    setError(null);
    try {
      await uploadFile(file);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  if (loading) return null;

  if (!status?.has_data) {
    return (
      <div className="max-w-2xl">
        <p className="text-[12.5px] tracking-wide text-flare mb-3 font-mono">no-code automl</p>
        <h1 className="font-display text-[34px] leading-[1.15] font-semibold text-paper mb-4">
          Turn a spreadsheet into a trained model in a few clicks.
        </h1>
        <p className="text-[15px] text-paper-dim mb-8 leading-relaxed">
          DataFlare profiles your data, runs a shortlist of algorithms against it, and hands
          you a ranked leaderboard — no notebooks, no pipeline code. Start by loading a
          dataset below.
        </p>
        <Dropzone onFile={handleFile} disabled={uploading} />
        {uploading && <p className="text-[13px] text-paper-faint mt-3">Uploading and profiling…</p>}
        {error && <p className="text-[13px] text-signal-red mt-3">{error}</p>}
        <p className="text-[13px] text-paper-faint mt-5">
          Don&apos;t have a file handy?{" "}
          <Link href="/upload" className="text-flare hover:text-flare-bright">
            Try a sample dataset
          </Link>{" "}
          instead.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-[24px] font-semibold text-paper">
          {status.dataset_name}
        </h1>
        <p className="text-[13px] text-paper-faint mt-1">
          Loaded and ready. Here&apos;s a quick read on data quality before you train.
        </p>
      </div>

      {info && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Rows" value={info.rows.toLocaleString()} />
          <StatCard label="Columns" value={info.columns} suffix={`${info.num_cols} numeric · ${info.cat_cols} categorical`} />
          <StatCard
            label="Missing values"
            value={`${info.null_pct}%`}
            tone={info.null_pct > 10 ? "bad" : info.null_pct > 0 ? "warn" : "good"}
          />
          <StatCard
            label="Data health"
            value={info.health_score}
            suffix="/ 100"
            tone={info.health_score >= 80 ? "good" : info.health_score >= 50 ? "warn" : "bad"}
          />
        </div>
      )}

      {insights.length > 0 && (
        <Panel title="Smart insights">
          <ul className="space-y-2.5">
            {insights.map((ins, i) => (
              <li key={i} className="flex gap-3 text-[13px]">
                <span
                  className={
                    "mt-1.5 shrink-0 w-1.5 h-1.5 rounded-full " +
                    (ins.type === "success"
                      ? "bg-signal-teal"
                      : ins.type === "warning"
                      ? "bg-signal-amber"
                      : ins.type === "error"
                      ? "bg-signal-red"
                      : "bg-signal-blue")
                  }
                />
                <span>
                  <span className="text-paper font-medium">{ins.title}: </span>
                  <span className="text-paper-dim">{ins.message}</span>
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      <div className="grid md:grid-cols-3 gap-4">
        <QuickLink
          href="/explore"
          icon={<ChartScatter size={18} strokeWidth={1.75} />}
          title="Explore the data"
          desc="Distributions, correlations, and column-level stats."
        />
        <QuickLink
          href="/train"
          icon={<Cpu size={18} strokeWidth={1.75} />}
          title="Train a model"
          desc="Pick a target column and let DataFlare compare algorithms."
        />
        <QuickLink
          href="/results"
          icon={<Trophy size={18} strokeWidth={1.75} />}
          title="View results"
          desc="Leaderboard, metrics, and model export."
        />
      </div>

      <p className="text-[12.5px] text-paper-faint">
        Want to start over?{" "}
        <button onClick={() => router.push("/upload")} className="text-flare hover:text-flare-bright">
          Load a different dataset
        </button>
        .
      </p>
    </div>
  );
}

function QuickLink({
  href,
  icon,
  title,
  desc,
}: {
  href: string;
  icon: React.ReactNode;
  title: string;
  desc: string;
}) {
  return (
    <Link
      href={href}
      className="group border border-line bg-ink-800/60 rounded p-5 hover:border-flare/60 transition-colors"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-flare">{icon}</span>
        <ArrowRight
          size={15}
          className="text-paper-faint group-hover:text-flare group-hover:translate-x-0.5 transition-all"
        />
      </div>
      <div className="font-display text-[14.5px] text-paper mb-1">{title}</div>
      <div className="text-[12.5px] text-paper-faint leading-relaxed">{desc}</div>
    </Link>
  );
}
