"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const DARK_COLORWAY = ["#FF5A1F", "#5B9CFF", "#31D8A6", "#F5B942", "#C084FC", "#FF7A45"];

export default function PlotlyChart({ figureJson, height = 420 }: { figureJson: string; height?: number }) {
  const fig = useMemo(() => {
    try {
      return JSON.parse(figureJson);
    } catch {
      return null;
    }
  }, [figureJson]);

  if (!fig) return null;

  const layout = {
    ...fig.layout,
    autosize: true,
    height,
    width: undefined,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#9BA6B4", family: "Inter, sans-serif", size: 11.5, ...(fig.layout?.font ?? {}) },
    title: fig.layout?.title
      ? { ...fig.layout.title, font: { color: "#E9EDF2", size: 14 } }
      : undefined,
    colorway: DARK_COLORWAY,
    xaxis: { ...fig.layout?.xaxis, gridcolor: "#1B2129", zerolinecolor: "#232B35", linecolor: "#232B35" },
    yaxis: { ...fig.layout?.yaxis, gridcolor: "#1B2129", zerolinecolor: "#232B35", linecolor: "#232B35" },
    margin: { t: fig.layout?.title ? 44 : 16, r: 16, l: 48, b: 40, ...(fig.layout?.margin ?? {}) },
    legend: { ...fig.layout?.legend, font: { color: "#9BA6B4" } },
  };

  return (
    <Plot
      data={fig.data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height }}
      useResizeHandler
    />
  );
}
