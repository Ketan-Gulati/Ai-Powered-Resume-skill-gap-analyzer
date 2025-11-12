import React, { useMemo } from "react";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
} from "chart.js";
import { Doughnut, Bar } from "react-chartjs-2";
import {
  FaCheckCircle,
  FaExclamationTriangle,
  FaBookOpen,
  FaChartLine,
} from "react-icons/fa";

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

export default function Results({ analysis, loading, error }) {
  if (loading)
    return (
      <div className="bg-white shadow-lg rounded-lg p-6 text-center text-indigo-600 font-semibold">
        Analyzing...
      </div>
    );
  if (error)
    return (
      <div className="bg-white shadow-lg rounded-lg p-6 text-center text-red-600 font-semibold">
        Error: {error}
      </div>
    );
  if (!analysis)
    return (
      <div className="bg-white shadow-lg rounded-lg p-6 text-center text-gray-500">
        Upload your resume and paste the job description to begin.
      </div>
    );

  const match = analysis.match || {};
  const recs =
    (analysis.recommendations && analysis.recommendations.recommended) || [];

  const ctxVal = match.context_similarity ?? 0;
  const doughnutData = {
    labels: ["Match", "Other"],
    datasets: [
      {
        data: [ctxVal, Math.max(0, 1 - ctxVal)],
        backgroundColor: ["#6366f1", "#e5e7eb"],
        borderWidth: 0,
      },
    ],
  };

  const skillPct = match.match_percent ?? 0;

  const topRecs = [...recs]
    .sort((a, b) => (b.score_percent || 0) - (a.score_percent || 0))
    .slice(0, 5);

  const barData = {
    labels: topRecs.map((r) => (r.title || "Course").slice(0, 25)),
    datasets: [
      {
        label: "Relevance",
        data: topRecs.map((r) => r.score_percent || 0),
        backgroundColor: "rgba(99,102,241,0.8)",
        borderRadius: 6,
      },
    ],
  };

  return (
    <div className="space-y-6">
      {/* Overview Card */}
      <div className="bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2 text-gray-800">
            <FaChartLine className="text-indigo-500" /> Overview
          </h3>
          <p className="text-sm text-gray-500">AI-Powered Semantic Matching</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Context Similarity Donut */}
          <div className="flex flex-col items-center">
            <div className="relative w-44 h-44">
              <Doughnut
                data={doughnutData}
                options={{
                  cutout: "78%",
                  plugins: { legend: { display: false } },
                }}
              />
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-bold text-gray-800">
                  {Math.round(ctxVal * 100)}%
                </span>
                <span className="text-gray-500 text-sm">Context</span>
              </div>
            </div>
          </div>

          {/* Skill Match Progress */}
          <div className="space-y-4">
            <div>
              <div className="flex items-center gap-2 text-indigo-600 font-semibold mb-1">
                <FaCheckCircle /> Skill Match
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-indigo-500 h-3 rounded-full transition-all duration-700"
                  style={{ width: `${skillPct}%` }}
                />
              </div>
              <div className="flex justify-between text-sm text-gray-500 mt-1">
                <span>Match</span>
                <span className="font-semibold text-gray-700">{skillPct}%</span>
              </div>
            </div>

            <div>
              <div className="flex items-center gap-2 text-amber-600 font-semibold mb-1">
                <FaExclamationTriangle /> Missing Skills
              </div>
              <div className="flex flex-wrap gap-2">
                {match.missing_skills?.length ? (
                  match.missing_skills.map((s, i) => (
                    <span
                      key={i}
                      className="px-3 py-1 text-sm bg-amber-100 text-amber-700 rounded-full font-medium"
                    >
                      {s}
                    </span>
                  ))
                ) : (
                  <span className="text-gray-400 text-sm">None</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Matched Skills */}
      <div className="bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition">
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2 text-green-600">
          <FaCheckCircle /> Matched Skills
        </h3>
        <div className="flex flex-wrap gap-2">
          {match.matched_skills?.length ? (
            match.matched_skills.map((s, i) => (
              <span
                key={i}
                className="px-3 py-1 text-sm bg-green-100 text-green-700 rounded-full font-medium"
              >
                {s}
              </span>
            ))
          ) : (
            <span className="text-gray-400 text-sm">None</span>
          )}
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition">
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2 text-indigo-600">
          <FaBookOpen /> Top Recommendations
        </h3>
        {topRecs.length ? (
          <>
            <div className="h-60 mb-4">
              <Bar
                data={barData}
                options={{
                  indexAxis: "y",
                  plugins: { legend: { display: false } },
                  scales: { x: { max: 100, ticks: { stepSize: 20 } } },
                  maintainAspectRatio: false,
                }}
              />
            </div>

            <div className="divide-y">
              {topRecs.map((r, i) => (
                <div
                  key={i}
                  className="py-3 flex justify-between items-center hover:bg-gray-50 px-2 rounded-lg"
                >
                  <div>
                    <div className="font-medium text-gray-800">
                      {r.title || "Course"}
                    </div>
                    <div className="text-sm text-gray-500">
                      {r.provider || "Online Platform"}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-indigo-600">
                      {r.score_percent || 0}%
                    </div>
                    {r.url && (
                      <a
                        href={r.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-sm text-indigo-500 font-medium hover:underline"
                      >
                        Open
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="text-gray-400 text-sm text-center py-4">
            No relevant recommendations found.
          </div>
        )}
      </div>
    </div>
  );
}