import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  return {
    metadataBase: new URL(origin),
    title: { default: "VLM Speed Lab", template: "%s · VLM Speed Lab" },
    description: "Measured VLM speedups with reproducible quality evidence.",
    icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
    openGraph: {
      title: "VLM Speed Lab",
      description: "Make it faster. Prove it stayed good.",
      type: "website",
      url: origin,
      images: [{ url: `${origin}/og.png`, width: 1200, height: 630, alt: "VLM Speed Lab — measured, not marketed" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "VLM Speed Lab",
      description: "Make it faster. Prove it stayed good.",
      images: [`${origin}/og.png`],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body></html>;
}
