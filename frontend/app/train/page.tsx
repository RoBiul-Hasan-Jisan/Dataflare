"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Sparkles } from "lucide-react";
import { useDataset } from "@/context/DatasetContext";
import { api, apiErrorMessage } from "@/lib/api";
import { DataInfo, DetectTargetResponse, TrainResponse } from "@/lib/types";
import { Panel } from "@/components/Panel";
import Button from "@/components/Button";
import EmptyState from "@/components/EmptyState";

export default function TrainPage() {
  const { status, loading, refresh } = useDataset();
  const [info, setInfo] = useState<DataInfo | null>(null);
  const [target, setTarget] = useState("");
  const [detect, setDetect] = useState<DetectTargetResponse | null>(null);
  const [trainSize, setTrainSize] = useState(0.8);
  const [fold, setFold] = useState(5);
  const [normalize, setNormalize] = useState(true);
  const [removeOutliers, setRemoveOutliers] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    if (!status?.has_data) return;
    api.get<DataInfo>("/api/data-info").then((r) => setInfo(r.data));
  }, [status?.has_data]);

  useEffect(() => {
    if (!target) {
      setDetect(null);
      return;
    }
    api
      .post<DetectTargetResponse>("/api/detect-target", { target })
      .then((r) => setDetect(r.data))
      .catch(() => setDetect(null));
  }, [target]);

  async function handleTrain() {
    if (!target) return;
    setTraining(true);
    setError(null);
    try {
      const res = await api.post<TrainResponse>("/api/train", {
        target,
        train_size: trainSize,
        fold,
        normalize,
        remove_outliers: removeOutliers,
        max_models: 999,
      });
      if (!res.data.success) throw new Error("Training did not complete.");
      await refresh();
      router.push("/results");
    } catch (e) {
      setError(apiErrorMessage(e, "Training failed. Try a different target or fewer folds."));
    } finally {
      setTraining(false);
    }
  }

  if (loading) return null;
  if (!status?.has_data) return <EmptyState />;

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="font-display text-[22px] font-semibold text-paper">Train a model</h1>
        <p className="text-[13px] text-paper-faint mt-1">
          Pick what you want to predict. DataFlare compares a shortlist of algorithms and
          ranks them for you.
        </p>
      </div>

      <Panel title="Target column">
        <select
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="w-full bg-ink-700 border border-line rounded px-3 py-2.5 text-[13.5px] text-paper"
        >
          <option value="">Select a column to predict…</option>
          {info?.column_names.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>

        {detect && (
          <div className="mt-3 flex items-center gap-2 text-[12.5px]">
            <span
              className={
                "px-2 py-0.5 rounded-sm font-medium " +
                (detect.problem_type === "classification"
                  ? "bg-signal-blue/15 text-signal-blue"
                  : "bg-signal-teal/15 text-signal-teal")
              }
            >
              {detect.type_label}
            </span>
            <span className="text-paper-faint">
              detected · {detect.unique_values} unique values
            </span>
          </div>
        )}
      </Panel>

      <Panel title="Training options">
        <div className="space-y-5">
          <Field label={`Train / test split — ${Math.round(trainSize * 100)}% train`}>
            <input
              type="range"
              min={0.5}
              max={0.95}
              step={0.05}
              value={trainSize}
              onChange={(e) => setTrainSize(parseFloat(e.target.value))}
              className="w-full accent-flare"
            />
          </Field>

          <Field label={`Cross-validation folds — ${fold}`}>
            <input
              type="range"
              min={2}
              max={10}
              step={1}
              value={fold}
              onChange={(e) => setFold(parseInt(e.target.value))}
              className="w-full accent-flare"
            />
          </Field>

          <div className="flex items-center justify-between">
            <div>
              <div className="text-[13px] text-paper">Normalize features</div>
              <div className="text-[12px] text-paper-faint">Recommended for most datasets</div>
            </div>
            <Toggle checked={normalize} onChange={setNormalize} />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <div className="text-[13px] text-paper">Remove outliers</div>
              <div className="text-[12px] text-paper-faint">Drops extreme values before training</div>
            </div>
            <Toggle checked={removeOutliers} onChange={setRemoveOutliers} />
          </div>
        </div>
      </Panel>

      <Button onClick={handleTrain} disabled={!target || training} className="w-full">
        {training ? (
          <>
            <Loader2 size={15} className="animate-spin" /> Training in progress — this can take a minute…
          </>
        ) : (
          <>
            <Sparkles size={15} /> Start training
          </>
        )}
      </Button>

      {error && <p className="text-[13px] text-signal-red">{error}</p>}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[13px] text-paper mb-2">{label}</label>
      {children}
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={
        "rounded-full relative transition-colors shrink-0 " +
        (checked ? "bg-flare" : "bg-ink-500")
      }
      style={{ height: 22, width: 40 }}
    >
      <span
        className="absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all"
        style={{ left: checked ? 20 : 3 }}
      />
    </button>
  );
}
