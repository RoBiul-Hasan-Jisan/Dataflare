import axios from "axios";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true, // send/receive the Flask session cookie
});

export function apiErrorMessage(err: unknown, fallback = "Something went wrong."): string {
  if (axios.isAxiosError(err)) {
    const data = err.response?.data as { error?: string; message?: string } | undefined;
    if (data?.message) return data.message;
    if (data?.error) return humanizeErrorCode(data.error);
    if (err.code === "ERR_NETWORK") {
      return "Can't reach the DataFlare API. Make sure the backend server is running.";
    }
  }
  return fallback;
}

function humanizeErrorCode(code: string): string {
  const map: Record<string, string> = {
    memory_error: "The dataset is too large to train safely on this machine.",
    training_failed: "Training failed. Try a different target column or fewer folds.",
  };
  return map[code] ?? code.replace(/_/g, " ");
}
