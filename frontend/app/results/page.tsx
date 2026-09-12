"use client";

import { useEffect, useState } from "react";
import { Download, Trophy } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useDataset } from "@/context/DatasetContext";
import { api, API_BASE_URL } from "@/lib/api";
import { ResultsResponse } from "@/lib/types";
import { Panel } from "@/components/Panel";
import StatCard from "@/components/StatCard";
import DataTable from "@/components/DataTable";
import EmptyState from "@/components/EmptyState";
import Button from "@/components/Button";

export default function ResultsPage() {
  const { status, loading } = useDataset();
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!status?.has_results) return;
    api
      .get<ResultsResponse>("/api/results")
      .then((r) => setResults(r.data))
      .catch(() => setErr("No results available yet."));
  }, [status?.has_results]);

  if (loading) return null;

  if (!status?.has_data) return <EmptyState />;
  if (!status?.has_results) {
    return (
      <EmptyState
        title="No trained model yet"
        message="Head to Train a model to run your first comparison across algorithms."
      />
    );
  }
  if (!results) return null;

  const primaryMetric = results.num_columns[0];
  const modelCol = results.columns.includes("Model") ? "Model" : results.columns[0];
  const chartData = results.top_models.map((r) => ({
    name: String(r[modelCol]),
    value: Number(r[primaryMetric] ?? 0),
  }));

  const best = results.top_models[0];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-display text-[22px] font-semibold text-paper">Results</h1>
          <p className="text-[13px] text-paper-faint mt-1">
            Ranked by {primaryMetric} · {results.folds_used}-fold cross-validation
          </p>
        </div>
        <div className="flex gap-2">
          <a href={`${API_BASE_URL}/api/download-results`} target="_blank" rel="noreferrer">
            <Button variant="secondary" size="sm">
              <Download size={13} /> Results CSV
            </Button>
          </a>
          {results.model_id && (
            <a
              href={`${API_BASE_URL}/api/download-model/${results.model_id}`}
              target="_blank"
              rel="noreferrer"
            >
              <Button size="sm">
                <Download size={13} /> Best model (.pkl)
              </Button>
            </a>
          )}
        </div>
      </div>

      <Panel>
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded bg-flare/15 flex items-center justify-center shrink-0">
            <Trophy size={18} className="text-flare" />
          </div>
          <div>
            <div className="text-[12px] text-paper-faint mb-0.5">Best model</div>
            <div className="font-display text-[18px] text-paper font-semibold">
              {String(best?.[modelCol] ?? "—")}
            </div>
          </div>
        </div>
      </Panel>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {results.num_columns.slice(0, 4).map((m) => (
          <StatCard key={m} label={m} value={Number(results.best_metrics[m] ?? 0).toFixed(4)} />
        ))}
      </div>

      <Panel title={`Top models by ${primaryMetric}`}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1B2129" horizontal={false} />
            <XAxis type="number" stroke="#5C6774" fontSize={11.5} />
            <YAxis type="category" dataKey="name" stroke="#5C6774" fontSize={11.5} width={110} />
            <Tooltip
              contentStyle={{
                background: "#161B22",
                border: "1px solid #232B35",
                borderRadius: 5,
                fontSize: 12.5,
              }}
              labelStyle={{ color: "#E9EDF2" }}
              cursor={{ fill: "rgba(255,90,31,0.06)" }}
            />
            <Bar dataKey="value" fill="#FF5A1F" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Full leaderboard">
        <DataTable columns={results.columns} rows={results.top_models} highlightFirstRow />
      </Panel>

      {err && <p className="text-[13px] text-signal-red">{err}</p>}
    </div>
  );
}
