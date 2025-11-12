import React, { useState } from "react";
import Analyzer from "./components/Analyzer";
import Results from "./components/Results";

export default function App() {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>Skill Gap Analyzer</h1>
        <p className="sub">Upload resume PDF, paste JD — get gaps & course recommendations</p>
      </header>

      <main className="container">
        <div className="left">
          <Analyzer
            setAnalysis={setAnalysis}
            setLoading={setLoading}
            setError={setError}
          />
        </div>

        <aside className="right">
          <Results analysis={analysis} loading={loading} error={error} />
        </aside>
      </main>

      <footer className="footer">
        <small>Minor Project — Skill Gap Analyzer</small>
      </footer>
    </div>
  );
}