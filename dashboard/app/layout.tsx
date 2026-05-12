import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Smart Traffic Dashboard",
  description: "AI traffic-light operations dashboard for lane priority, learning progress, and manual zones."
};

export const viewport: Viewport = {
  themeColor: "#f6f7f9",
  colorScheme: "light"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
