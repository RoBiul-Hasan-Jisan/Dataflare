"use client";

import { useState } from "react";
import { Trash2, Loader2 } from "lucide-react";
import { api, apiErrorMessage } from "@/lib/api";
import { useDataset } from "@/context/DatasetContext";

export default function Topbar() {
  const { status, refresh } = useDataset();
  const [clearing, setClearing] = useState(false);

  async function handleClear() {
    setClearing(true);
    try {
      await api.post("/api/clear-session");
      await refresh();
    } catch (e) {
      alert(apiErrorMessage(e, "Couldn't clear the session."));
    } finally {
      setClearing(false);
    }
  }

  return (
    <header className="h-16 border-b border-line flex items-center justify-between px-6 md:px-10 bg-ink-950/40">
      <div className="text-[13px] text-paper-dim">
        {status?.has_data ? (
          <span>
            <span className="text-paper font-medium">{status.dataset_name}</span>
            <span className="mx-2 text-paper-faint">·</span>
            <span className="tabular">{status.rows.toLocaleString()}</span> rows
            <span className="mx-2 text-paper-faint">·</span>
            <span className="tabular">{status.columns}</span> columns
          </span>
        ) : (
          <span>No dataset loaded yet</span>
        )}
      </div>
      {status?.has_data && (
        <button
          onClick={handleClear}
          disabled={clearing}
          className="flex items-center gap-1.5 text-[12.5px] text-paper-faint hover:text-signal-red transition-colors disabled:opacity-50"
        >
          {clearing ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
          Clear session
        </button>
      )}
    </header>
  );
}
