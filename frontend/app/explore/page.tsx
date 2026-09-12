"use client";

import { useEffect, useState } from "react";
import { useDataset } from "@/context/DatasetContext";
import { api } from "@/lib/api";
import { DataInfo, DataPreview } from "@/lib/types";
import { Panel } from "@/components/Panel";
import StatCard from "@/components/StatCard";
import DataTable from "@/components/DataTable";
import EmptyState from "@/components/EmptyState";
import PlotlyChart from "@/components/charts/PlotlyChart";
import { Loader2 } from "lucide-react";

type VizMap = Record<string, string>;

const CHART_LABELS: Record<string, string> = {
  correlation_heatmap: "Correlation matrix",
  distributions: "Feature distributions",
  boxplot_comparison: "Box plot comparison",
  violin_plots: "Violin plots",
  missing_pattern: "Missing value pattern",
  pca_projection: "PCA projection",
  tsne_projection: "t-SNE projection",
};

export default function ExplorePage() {
  const { status, loading } = useDataset();
  const [info, setInfo] = useState<DataInfo | null>(null);
  const [preview, setPreview] = useState<DataPreview | null>(null);
  const [viz, setViz] = useState<VizMap | null>(null);
  const [vizLoading, setVizLoading] = useState(false);
  const [tab, setTab] = useState<"overview" | "charts">("overview");

  useEffect(() => {
    if (!status?.has_data) return;
    api.get<DataInfo>("/api/data-info").then((r) => setInfo(r.data));
    api.get<DataPreview>("/api/data-preview", { params: { rows: 15 } }).then((r) => setPreview(r.data));
  }, [status?.has_data]);

  useEffect(() => {
    if (tab !== "charts" || viz || !status?.has_data) return;
    setVizLoading(true);
    api
      .get<VizMap>("/api/all-visualizations")
      .then((r) => setViz(r.data))
      .finally(() => setVizLoading(false));
  }, [tab, viz, status?.has_data]);

  if (loading) return null;
  if (!status?.has_data) return <EmptyState />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-[22px] font-semibold text-paper">Explore & EDA</h1>
        <p className="text-[13px] text-paper-faint mt-1">
          Understand your data before you train — types, gaps, and shape.
        </p>
      </div>

      <div className="flex gap-1 border-b border-line">
        {(["overview", "charts"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={
              "px-4 py-2 text-[13px] border-b-2 -mb-px transition-colors " +
              (tab === t
                ? "border-flare text-paper font-medium"
                : "border-transparent text-paper-faint hover:text-paper")
            }
          >
            {t === "overview" ? "Overview & preview" : "Charts"}
          </button>
        ))}
      </div>

      {tab === "overview" && info && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard label="Rows" value={info.rows.toLocaleString()} />
            <StatCard label="Columns" value={info.columns} />
            <StatCard
              label="Duplicate rows"
              value={info.duplicates}
              tone={info.duplicates > 0 ? "warn" : "good"}
            />
            <StatCard
              label="Missing cells"
              value={info.null_count.toLocaleString()}
              suffix={`${info.null_pct}%`}
              tone={info.null_pct > 10 ? "bad" : info.null_pct > 0 ? "warn" : "good"}
            />
          </div>

          <Panel title="Columns" subtitle={`${info.num_cols} numeric · ${info.cat_cols} categorical`}>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {info.column_names.map((c) => (
                <div
                  key={c}
                  className="flex items-center justify-between border border-line-soft rounded px-3 py-2 text-[12.5px]"
                >
                  <span className="text-paper truncate mr-2">{c}</span>
                  <span className="text-paper-faint font-mono text-[11px] shrink-0">
                    {info.dtypes[c]}
                  </span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Data preview" subtitle={preview ? `Showing 15 of ${preview.total_rows.toLocaleString()} rows` : undefined}>
            {preview ? (
              <DataTable columns={preview.columns} rows={preview.data} />
            ) : (
              <p className="text-[13px] text-paper-faint">Loading preview…</p>
            )}
          </Panel>
        </div>
      )}

      {tab === "charts" && (
        <div className="space-y-6">
          {vizLoading && (
            <p className="flex items-center gap-2 text-[13px] text-paper-faint">
              <Loader2 size={14} className="animate-spin" /> Building charts from your dataset…
            </p>
          )}
          {viz &&
            Object.entries(viz).map(([key, figJson]) => (
              <Panel key={key} title={CHART_LABELS[key] ?? key.replace(/_/g, " ")}>
                <PlotlyChart figureJson={figJson} />
              </Panel>
            ))}
          {viz && Object.keys(viz).length === 0 && (
            <p className="text-[13px] text-paper-faint">
              Not enough numeric columns to generate charts for this dataset.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
