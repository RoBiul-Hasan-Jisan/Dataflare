import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0B0E13",
          900: "#0E1116",
          800: "#12161D",
          700: "#161B22",
          600: "#1D232C",
          500: "#262E39",
          400: "#3A4453",
        },
        line: {
          DEFAULT: "#232B35",
          soft: "#1B2129",
        },
        paper: {
          DEFAULT: "#E9EDF2",
          dim: "#9BA6B4",
          faint: "#5C6774",
        },
        flare: {
          DEFAULT: "#FF5A1F",
          bright: "#FF7A45",
          dim: "#8A3418",
        },
        signal: {
          teal: "#31D8A6",
          amber: "#F5B942",
          red: "#FF5D5D",
          blue: "#5B9CFF",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "sans-serif"],
        body: ["var(--font-body)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      borderRadius: {
        sm: "3px",
        DEFAULT: "5px",
        md: "6px",
        lg: "8px",
      },
      boxShadow: {
        none: "none",
      },
    },
  },
  plugins: [],
};
export default config;
