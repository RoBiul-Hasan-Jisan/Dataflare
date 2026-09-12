import { api, apiErrorMessage } from "@/lib/api";
import { UploadResponse } from "@/lib/types";

export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await api.post<UploadResponse>("/api/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  } catch (e) {
    throw new Error(apiErrorMessage(e, "Upload failed."));
  }
}
