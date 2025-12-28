import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { UploadButton } from "@/components/UploadButton";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Saree Virtual Try-On",
  description: "Generate photorealistic images of models wearing your saree designs",
};

import { Providers } from "./providers";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased`}>
        <Providers>
          {/* Header */}
          <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-8">
              <Link href="/gallery" className="flex items-center gap-2">
                <span className="text-xl font-semibold">Saree Try-On</span>
              </Link>

              <div className="relative">
                <UploadButton />
              </div>
            </div>
          </header>

          {/* Main content */}
          <main className="min-h-[calc(100vh-4rem)]">
            {children}
          </main>

          {/* Footer */}
          <footer className="border-t py-6 md:py-0">
            <div className="container mx-auto flex h-14 items-center justify-center px-4 md:px-8">
              <p className="text-sm text-muted-foreground">
                Built for deterministic saree visualization
              </p>
            </div>
          </footer>
        </Providers>
      </body>
    </html>
  );
}
