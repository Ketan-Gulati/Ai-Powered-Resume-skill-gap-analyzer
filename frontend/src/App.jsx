// src/App.jsx
import React, { useState } from "react";
import Analyzer from "./components/Analyzer";
import Results from "./components/Results";
import SkillChart from "./components/SkillChart";

export default function App() {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <header className="max-w-6xl mx-auto mb-6">
        <h1 className="text-2xl font-semibold">AI Powered Skill Gap Analyzer</h1>
        <p className="text-sm text-gray-600">
          Upload resume PDF, paste JD - get gaps & course recommendations
        </p>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        <section className="lg:col-span-2">
          <Analyzer
            setAnalysis={setAnalysis}
            setLoading={setLoading}
            setError={setError}
          />
          <div className="mt-6">
            <Results analysis={analysis} loading={loading} error={error} />
          </div>
        </section>

        <aside className="lg:col-span-1">
          <SkillChart
            matchedSkills={(analysis && analysis.matchedSkills) || []}
            recommendations={(analysis && analysis.recommendations) || []}
          />
        </aside>
      </main>

      <footer className="max-w-6xl mx-auto mt-8 text-center text-xs text-gray-500">
        Minor Project - Skill Gap Analyzer
      </footer>
    </div>
  );
}
