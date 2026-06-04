import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { loadTheme } from "./lib/theme";
import "./index.css";

// Apply the saved theme before first paint to avoid a flash.
document.documentElement.dataset.theme = loadTheme();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
