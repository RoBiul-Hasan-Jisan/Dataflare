"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CheckCircle2, Loader2 } from "lucide-react";
import { Panel } from "@/components/Panel";
import Dropzone from "@/components/Dropzone";
import Button from "@/components/Button";
import { api, apiErrorMessage } from "@/lib/api";
import { SampleResponse } from "@/lib/types";
import { useDataset } from "@/context/DatasetContext";
import { uploadFile } from "@/lib/upload";

const SAMPLES = [
  { id: "titanic", name: "Titanic", desc: "Passenger survival · classification", target: "Survived" },
  { id: "diamonds", name: "Diamonds", desc: "Price prediction · regression", target: "price" },
  { id: "iris", name: "Iris", desc: "Flower species · classification", target: "species" },
  { id: "tips", name: "Tips", desc: "Restaurant tipping · regression", target: "tip" },
  { id: "mpg", name: "Auto MPG", desc: "Fuel efficiency · regression", target: "mpg" },
];

export default function UploadPage() {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const { refresh } = useDataset();
  const router = useRouter();

  async function handleFile(file: File) {
    setBusy("file");
    setError(null);
    setSuccess(null);
    try {
      const res = await uploadFile(file);
      setSuccess(res.message);
      await refresh();
      setTimeout(() => router.push("/explore"), 700);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handleSample(sampleId: string) {
    setBusy(sampleId);
    setError(null);
    setSuccess(null);
    try {
      const res = await api.post<SampleResponse>("/api/load-sample", { sample: sampleId });
      setSuccess(`Loaded ${res.data.sample} (${res.data.rows} rows, ${res.data.columns} columns)`);
      await refresh();
      setTimeout(() => router.push("/explore"), 700);
    } catch (e) {
      setError(apiErrorMessage(e, "Couldn't load that sample dataset."));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="font-display text-[22px] font-semibold text-paper">Load a dataset</h1>
        <p className="text-[13px] text-paper-faint mt-1">
          Bring your own file, or start with a sample to see how DataFlare works.
        </p>
      </div>

      <Panel title="Upload your own">
        <Dropzone onFile={handleFile} disabled={busy === "file"} />
        {busy === "file" && (
          <p className="flex items-center gap-2 text-[13px] text-paper-faint mt-3">
            <Loader2 size={14} className="animate-spin" /> Uploading and profiling…
          </p>
        )}
      </Panel>

      <Panel title="Or try a sample dataset" subtitle="Good for a quick end-to-end test run">
        <div className="grid sm:grid-cols-2 gap-3">
          {SAMPLES.map((s) => (
            <div
              key={s.id}
              className="border border-line-soft rounded p-3.5 flex items-center justify-between"
            >
              <div>
                <div className="text-[13.5px] text-paper font-medium">{s.name}</div>
                <div className="text-[12px] text-paper-faint mt-0.5">{s.desc}</div>
              </div>
              <Button
                variant="secondary"
                size="sm"
                disabled={busy !== null}
                onClick={() => handleSample(s.id)}
              >
                {busy === s.id ? <Loader2 size={13} className="animate-spin" /> : "Load"}
              </Button>
            </div>
          ))}
        </div>
      </Panel>

      {success && (
        <p className="flex items-center gap-2 text-[13px] text-signal-teal">
          <CheckCircle2 size={15} /> {success} — heading to Explore…
        </p>
      )}
      {error && <p className="text-[13px] text-signal-red">{error}</p>}
    </div>
  );
}
