import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/api/artifacts/**',
      },
    ],
    // Allow unoptimized images in development
    unoptimized: process.env.NODE_ENV === 'development',
  },
};

export default nextConfig;
