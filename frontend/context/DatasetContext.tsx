"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";
import { api } from "@/lib/api";
import { SessionStatus } from "@/lib/types";

interface DatasetContextValue {
  status: SessionStatus | null;
  loading: boolean;
  refresh: () => Promise<void>;
}

const DatasetContext = createContext<DatasetContextValue>({
  status: null,
  loading: true,
  refresh: async () => {},
});

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<SessionStatus | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await api.get<SessionStatus>("/api/session-status");
      setStatus(res.data);
    } catch {
      setStatus({ has_data: false, dataset_name: null, rows: 0, columns: 0, has_results: false });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <DatasetContext.Provider value={{ status, loading, refresh }}>
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset() {
  return useContext(DatasetContext);
}
