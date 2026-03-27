import { resolve } from "node:path";

import solid from "vite-plugin-solid";
import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  build: {
    outDir: "milady/review_static",
    emptyOutDir: true,
    target: "es2022",
    minify: false,
    sourcemap: true,
    rollupOptions: {
      input: {
        review: resolve(__dirname, "review.html"),
      },
      output: {
        entryFileNames: "assets/[name].js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  plugins: [solid()],
});
