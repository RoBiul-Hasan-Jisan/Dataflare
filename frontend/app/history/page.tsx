"use client";

import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { useDataset } from "@/context/DatasetContext";
import { api, API_BASE_URL } from "@/lib/api";
import { HistoryEntry } from "@/lib/types";
import { Panel } from "@/components/Panel";
import Button from "@/components/Button";

export default function HistoryPage() {
  const { loading } = useDataset();
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    api.get<{ history: HistoryEntry[] }>("/api/history").then((r) => setHistory(r.data.history ?? []));
  }, []);

  if (loading) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-display text-[22px] font-semibold text-paper">Training history</h1>
          <p className="text-[13px] text-paper-faint mt-1">
            Every training run from this session, most recent last.
          </p>
        </div>
        {history.length > 0 && (
          <a href={`${API_BASE_URL}/api/download-history`} target="_blank" rel="noreferrer">
            <Button variant="secondary" size="sm">
              <Download size={13} /> Export CSV
            </Button>
          </a>
        )}
      </div>

      <Panel>
        {history.length === 0 ? (
          <p className="text-[13px] text-paper-faint py-6 text-center">
            No training runs yet in this session.
          </p>
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-[12.5px] border-collapse">
              <thead>
                <tr className="bg-ink-700/70 text-paper-dim">
                  {["Time", "Dataset", "Type", "Best model", "Score", "Rows", "Cols"].map((h) => (
                    <th key={h} className="text-left font-medium px-3 py-2 border-b border-line whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...history].reverse().map((h, i) => (
                  <tr key={i} className={i % 2 === 0 ? "" : "bg-ink-700/25"}>
                    <td className="px-3 py-2 border-b border-line-soft text-paper-faint whitespace-nowrap">{h.time}</td>
                    <td className="px-3 py-2 border-b border-line-soft text-paper whitespace-nowrap">{h.dataset}</td>
                    <td className="px-3 py-2 border-b border-line-soft whitespace-nowrap">
                      <span
                        className={
                          "px-2 py-0.5 rounded-sm text-[11px] font-medium " +
                          (h.problem_type === "classification"
                            ? "bg-signal-blue/15 text-signal-blue"
                            : "bg-signal-teal/15 text-signal-teal")
                        }
                      >
                        {h.problem_type}
                      </span>
                    </td>
                    <td className="px-3 py-2 border-b border-line-soft text-paper whitespace-nowrap">{h.best_model}</td>
                    <td className="px-3 py-2 border-b border-line-soft tabular text-paper whitespace-nowrap">{h.score}</td>
                    <td className="px-3 py-2 border-b border-line-soft tabular text-paper-faint whitespace-nowrap">{h.rows.toLocaleString()}</td>
                    <td className="px-3 py-2 border-b border-line-soft tabular text-paper-faint whitespace-nowrap">{h.cols}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
