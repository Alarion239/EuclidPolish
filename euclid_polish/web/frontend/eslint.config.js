// ESLint flat config: typescript-eslint + react-hooks over the SPA source,
// its tests and the build config. `npm run lint` must stay error-free.
import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores(["node_modules/", "dist/", "coverage/"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: { ...globals.browser },
    },
    plugins: { "react-hooks": reactHooks },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "@typescript-eslint/no-unused-vars": ["error", {
        argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrors: "none",
      }],
    },
  },
  {
    // Legacy page bodies are replaced workspace by workspace in phase 3 (their
    // owners delete the dead code); until then dead locals there only warn.
    files: ["src/pages/**/*.{ts,tsx}"],
    rules: {
      "@typescript-eslint/no-unused-vars": ["warn", {
        argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrors: "none",
      }],
      "prefer-const": "warn",
    },
  },
  {
    files: ["vite.config.ts", "vitest.config.ts", "eslint.config.js", "test/**/*.{ts,tsx}"],
    languageOptions: { globals: { ...globals.node } },
  },
]);
