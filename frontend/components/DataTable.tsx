export default function DataTable({
  columns,
  rows,
  highlightFirstRow = false,
}: {
  columns: string[];
  rows: Record<string, unknown>[];
  highlightFirstRow?: boolean;
}) {
  function fmt(v: unknown) {
    if (v === null || v === undefined || v === "") return "—";
    if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(4);
    return String(v);
  }

  return (
    <div className="overflow-auto border border-line-soft rounded">
      <table className="w-full text-[12.5px] border-collapse">
        <thead>
          <tr className="bg-ink-700/70 text-paper-dim">
            {columns.map((c) => (
              <th
                key={c}
                className="text-left font-medium px-3 py-2 border-b border-line whitespace-nowrap"
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className={
                highlightFirstRow && i === 0
                  ? "bg-flare/10"
                  : i % 2 === 0
                  ? "bg-transparent"
                  : "bg-ink-700/25"
              }
            >
              {columns.map((c) => (
                <td key={c} className="px-3 py-2 border-b border-line-soft tabular whitespace-nowrap text-paper">
                  {fmt(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && (
        <div className="text-center py-8 text-paper-faint text-[13px]">No rows to show.</div>
      )}
    </div>
  );
}
