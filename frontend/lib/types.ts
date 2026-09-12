export interface SessionStatus {
  has_data: boolean;
  dataset_name: string | null;
  rows: number;
  columns: number;
  has_results: boolean;
}

export interface UploadResponse {
  success: boolean;
  filename: string;
  rows: number;
  columns: number;
  message: string;
}

export interface SampleResponse {
  success: boolean;
  sample: string;
  rows: number;
  columns: number;
  target_hint: string;
}

export interface DataInfo {
  rows: number;
  columns: number;
  num_cols: number;
  cat_cols: number;
  null_pct: number;
  null_count: number;
  duplicates: number;
  health_score: number;
  column_names: string[];
  num_columns: string[];
  cat_columns: string[];
  dtypes: Record<string, string>;
}

export interface DataPreview {
  columns: string[];
  data: Record<string, unknown>[];
  total_rows: number;
}

export interface Insight {
  type: "success" | "warning" | "info" | "error";
  icon?: string;
  title: string;
  message: string;
}

export interface DetectTargetResponse {
  problem_type: "classification" | "regression";
  unique_values: number;
  type_label: string;
}

export interface TrainRequest {
  target: string;
  train_size: number;
  fold: number;
  normalize: boolean;
  remove_outliers: boolean;
  max_models: number;
}

export interface TrainResponse {
  success: boolean;
  best_model: string;
  best_score: number;
  metric_name: string;
  elapsed: string;
  elapsed_seconds: number;
  trained_rows: number;
  warnings: string[];
  results: Record<string, unknown>[];
  results_columns: string[];
  model_id: string;
}

export interface ResultsResponse {
  exists: boolean;
  columns: string[];
  num_columns: string[];
  top_models: Record<string, unknown>[];
  best_metrics: Record<string, number>;
  training_time: number | null;
  model_id: string | null;
  folds_used: number;
}

export interface HistoryEntry {
  time: string;
  dataset: string;
  problem_type: string;
  best_model: string;
  score: number;
  rows: number;
  cols: number;
  model_id: string;
}

export interface ApiError {
  error: string;
  message?: string;
}
