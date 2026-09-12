import Link from "next/link";
import { UploadCloud } from "lucide-react";

export default function EmptyState({
  title = "No dataset loaded",
  message = "Upload a CSV or Excel file, or load a sample dataset, to get started.",
}: {
  title?: string;
  message?: string;
}) {
  return (
    <div className="border border-dashed border-line rounded py-20 flex flex-col items-center justify-center text-center">
      <UploadCloud size={28} className="text-paper-faint mb-4" strokeWidth={1.5} />
      <h3 className="font-display text-[16px] text-paper mb-1.5">{title}</h3>
      <p className="text-[13px] text-paper-faint max-w-sm mb-5">{message}</p>
      <Link
        href="/upload"
        className="inline-flex items-center gap-2 px-4 py-2 rounded bg-flare text-white text-[13.5px] font-medium hover:bg-flare-bright transition-colors"
      >
        Go to upload
      </Link>
    </div>
  );
}
