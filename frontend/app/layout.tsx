import type { Metadata } from "next";
import "./globals.css";
import { DatasetProvider } from "@/context/DatasetContext";
import Sidebar from "@/components/Sidebar";
import Topbar from "@/components/Topbar";

export const metadata: Metadata = {
  title: "DataFlare ML Studio",
  description: "Upload a dataset, explore it, and train models — no code required.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-body antialiased">
        <DatasetProvider>
          <div className="flex min-h-screen">
            <Sidebar />
            <div className="flex-1 min-w-0 flex flex-col">
              <Topbar />
              <main className="flex-1 px-6 md:px-10 py-8 max-w-[1400px] w-full mx-auto">
                {children}
              </main>
            </div>
          </div>
        </DatasetProvider>
      </body>
    </html>
  );
}
