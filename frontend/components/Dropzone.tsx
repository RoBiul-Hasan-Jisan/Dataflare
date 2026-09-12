"use client";

import { useCallback, useRef, useState } from "react";
import clsx from "clsx";
import { FileSpreadsheet, UploadCloud } from "lucide-react";

export default function Dropzone({
  onFile,
  disabled,
}: {
  onFile: (file: File) => void;
  disabled?: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      if (file) onFile(file);
    },
    [onFile, disabled]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={clsx(
        "border border-dashed rounded flex flex-col items-center justify-center text-center py-14 px-6 cursor-pointer transition-colors",
        dragging ? "border-flare bg-flare/5" : "border-line hover:border-paper-faint",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx,.xls"
        hidden
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
      />
      {dragging ? (
        <FileSpreadsheet size={26} className="text-flare mb-3" strokeWidth={1.5} />
      ) : (
        <UploadCloud size={26} className="text-paper-faint mb-3" strokeWidth={1.5} />
      )}
      <p className="text-[14px] text-paper mb-1">
        Drop a CSV or Excel file here, or click to browse
      </p>
      <p className="text-[12px] text-paper-faint">Supports .csv, .xlsx, .xls — up to 100MB</p>
    </div>
  );
}
