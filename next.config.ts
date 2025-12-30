import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Increase body size limit for API routes
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
  // Output standalone for deployment
  output: "standalone",
};

export default nextConfig;
