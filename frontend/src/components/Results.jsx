// src/components/Results.jsx
import React from "react";

export default function Results({ analysis = null, loading = false, error = "" }) {
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-3">Analysis Results</h2>
        <div className="text-sm text-gray-500">Waiting for results…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-3">Analysis Results</h2>
        <div className="text-sm text-red-600">Error: {error}</div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-lg font-semibold mb-3">Analysis Results</h2>
        <div className="text-sm text-gray-400">No analysis yet - submit a resume and job description.</div>
      </div>
    );
  }

  const matched = analysis.matchedSkills ?? [];
  const recs = analysis.recommendations ?? [];
  const raw = analysis.raw ?? {};
  const score = typeof analysis.score === "number" ? Math.round(analysis.score * 100) : 
                (raw.match && raw.match.match_percent ? Math.round(Number(raw.match.match_percent)) : null);

  // derive missing skills if available in raw
  let missing = [];
  if (raw.match && Array.isArray(raw.match.missing_skills)) {
    missing = raw.match.missing_skills;
  } else if (Array.isArray(analysis.missingSkills)) {
    missing = analysis.missingSkills;
  }

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-lg font-semibold mb-3">Analysis Results</h2>

      <div className="space-y-4">
        <div>
          <div className="text-sm text-gray-500 mb-1">Match Score</div>
          <div className="text-2xl font-bold text-indigo-600">
            {score !== null ? `${score}%` : "N/A"}
          </div>
        </div>

        <div>
          <div className="text-sm text-gray-500 mb-2">Matched Skills</div>
          {matched.length ? (
            <ul className="flex flex-wrap gap-2">
              {matched.map((s, i) => (
                <li
                  key={i}
                  className="bg-indigo-50 text-indigo-700 px-3 py-1 rounded-full text-xs border"
                >
                  {typeof s === "string" ? s : s?.name ?? JSON.stringify(s).slice(0, 30)}
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-xs text-gray-400">No matched skills found</div>
          )}
        </div>

        <div>
          <div className="text-sm text-gray-500 mb-2">Missing Skills</div>
          {missing.length ? (
            <ul className="flex flex-wrap gap-2">
              {missing.map((s, i) => (
                <li
                  key={i}
                  className="bg-red-50 text-red-700 px-3 py-1 rounded-full text-xs border"
                >
                  {typeof s === "string" ? s : s?.name ?? JSON.stringify(s).slice(0, 30)}
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-xs text-gray-400">No missing skills identified</div>
          )}
        </div>

        <div>
          <div className="text-sm text-gray-500 mb-2">Top Recommendations</div>
          {recs.length ? (
            <ol className="list-decimal pl-5 space-y-2 text-sm">
              {recs.map((r, i) => (
                <li key={i} className="space-y-0.5">
                  <div className="font-medium">
                    {r?.title ?? r?.name ?? (typeof r === "string" ? r : "Untitled")}
                  </div>
                  {r?.platform && <div className="text-xs text-gray-500">{r.platform}</div>}
                  {r?.desc && <div className="text-xs text-gray-600">{r.desc}</div>}
                  {r?.url && (
                    <div className="text-xs">
                      <a href={r.url} target="_blank" rel="noreferrer" className="text-indigo-600 underline">
                        View course
                      </a>
                    </div>
                  )}
                </li>
              ))}
            </ol>
          ) : (
            <div className="text-xs text-gray-400">No recommendations returned</div>
          )}
        </div>

        <div>
          <div className="text-sm text-gray-500 mb-2">Raw response (debug)</div>
          <pre className="bg-gray-50 p-3 rounded text-xs text-gray-700 overflow-auto max-h-48">
            {JSON.stringify(raw, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}
