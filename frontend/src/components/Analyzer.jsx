// src/components/Analyzer.jsx
import React, { useState, useRef } from "react";
import axios from "axios";
import Loader from "./Loader";
import Results from "./Results";
import SkillChart from "./SkillChart";

/**
 * Analyzer component
 * - Attempts proxy endpoint (/api/analyze) first (works with vite proxy)
 * - If proxy fails with network error, falls back to direct backend URL (http://127.0.0.1:9000/api/analyze)
 * - Sends file as 'resume_pdf' (matches backend)
 */

export default function Analyzer() {
  const fileRef = useRef(null);
  const [jdText, setJdText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [responseData, setResponseData] = useState(null);

  // endpoints:
  const PROXY_URL = `/api/analyze`; // preferred (Vite proxy)
  const DIRECT_URL = `http://127.0.0.1:9000/api/analyze`; // fallback (direct to backend)

  const buildForm = (file) => {
    const form = new FormData();
    if (file) form.append("resume_pdf", file); // backend expects resume_pdf
    form.append("jd_text", jdText ?? "");
    return form;
  };

  const postTo = async (url, form) => {
    return axios.post(url, form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 120000,
    });
  };

  const analyze = async (e) => {
    e?.preventDefault();
    setError("");
    setResponseData(null);

    const file = fileRef.current?.files?.[0] ?? null;
    if (!file && !jdText.trim()) {
      setError("Please upload a resume (PDF) or paste a job description.");
      return;
    }

    const form = buildForm(file);

    setLoading(true);

    // Try proxy first, then direct backend fallback if proxy fails with network/ECONNREFUSED
    try {
      let resp;
      try {
        resp = await postTo(PROXY_URL, form);
      } catch (proxyErr) {
        // If proxy error looks like network/proxy problem, try direct backend URL
        const code = proxyErr?.code || (proxyErr?.response?.status && String(proxyErr.response.status));
        const isNetworkError = proxyErr?.message?.toLowerCase()?.includes("network") || proxyErr?.code === "ECONNREFUSED";
        // log proxy failure
        console.warn("Proxy request failed, attempting direct backend. proxyErr:", proxyErr?.message || proxyErr);
        if (isNetworkError) {
          // fallback to direct backend
          resp = await postTo(DIRECT_URL, form);
        } else {
          // if not a network error (e.g., 4xx/5xx), rethrow to outer catch
          throw proxyErr;
        }
      }

      const final = resp.data ?? {};

      // Normalize backend response for frontend components
      const matchBlock = final.match ?? final.data?.match ?? {};
      const matchedSkills = matchBlock.matched_skills ?? matchBlock.matchedSkills ?? [];
      const missingSkills = matchBlock.missing_skills ?? matchBlock.missing ?? [];
      const matchPercent = matchBlock.match_percent ?? matchBlock.matchPercent ?? null;

      // recommendations may be in different spots
      let recs = [];
      if (Array.isArray(final.recommendations)) recs = final.recommendations;
      else if (Array.isArray(final.data?.recommendations)) recs = final.data.recommendations;
      else if (Array.isArray(final.data?.recommendations?.recommended)) recs = final.data.recommendations.recommended;
      else if (Array.isArray(final.recommendations?.recommended)) recs = final.recommendations.recommended;
      else recs = final.recommendations || final.data?.recommendations?.recommended || [];

      // ensure each rec has url field (empty string if missing)
      recs = (Array.isArray(recs) ? recs : []).map((r) => {
        if (!r) return {};
        return {
          title: r.title ?? r.name ?? "Untitled Course",
          url: r.url ?? r.link ?? r.course_url ?? r.CourseURL ?? "",
          desc: r.desc ?? r.description ?? r.CourseShortIntro ?? "",
          platform: r.platform ?? r.site ?? "",
          rating: r.rating ?? r.stars ?? null,
          duration: r.duration ?? "",
          score_percent: r.score_percent ?? r.score ?? null,
        };
      });

      setResponseData({
        matchedSkills,
        missingSkills,
        recommendations: recs,
        score: typeof matchPercent === "number" ? matchPercent / 100 : null,
        raw: final,
      });
    } catch (err) {
      // Present helpful messages depending on error type
      console.error("ANALYZE ERROR full:", err, err?.response?.data);
      // If the server replied with a JSON error (traceback etc.), show its .error or short message
      const serverData = err?.response?.data;
      if (serverData && typeof serverData === "object") {
        setError(serverData.error || serverData.message || "Server error — check console for details.");
        console.error("Server response body:", serverData);
      } else if (err?.code === "ECONNREFUSED" || (err?.message && err.message.toLowerCase().includes("network"))) {
        setError("Network error: could not reach backend. Backend may be down or proxy misconfigured.");
      } else {
        setError(err?.message || "Request failed — check console for details.");
      }
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    if (fileRef.current) fileRef.current.value = "";
    setJdText("");
    setResponseData(null);
    setError("");
  };

  return (
    <div className="p-6 max-w-6xl mx-auto app-root">

      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <form onSubmit={analyze} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="col-span-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Resume (PDF)
              </label>
              <input
                ref={fileRef}
                type="file"
                accept="application/pdf"
                className="block w-full text-sm text-gray-700"
              />
              <p className="text-xs text-gray-400 mt-2">
                Upload a resume PDF.
              </p>
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Job Description (paste)
              </label>
              <textarea
                value={jdText}
                onChange={(e) => setJdText(e.target.value)}
                rows={6}
                placeholder="Paste the job description or role here"
                className="w-full rounded-md border-gray-200 shadow-sm focus:ring-2 focus:ring-indigo-300 p-3 text-sm"
              />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="submit"
              className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition"
              disabled={loading}
            >
              {loading ? "Analyzing…" : "Analyze"}
            </button>

            <button
              type="button"
              onClick={resetAll}
              className="text-gray-700 px-3 py-2 rounded-md border hover:bg-gray-50 text-sm"
              disabled={loading}
            >
              Reset
            </button>

            {error && (
              <div className="ml-4 text-sm text-red-600" role="alert">
                {error}
              </div>
            )}
          </div>
        </form>
      </div>

      {loading && (
        <div className="mb-6">
          <Loader />
        </div>
      )}

      {responseData ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Results data={responseData} />
          </div>

          <div className="lg:col-span-1">
            <SkillChart
              matchedSkills={responseData.matchedSkills ?? []}
              recommendations={responseData.recommendations ?? []}
            />
          </div>
        </div>
      ) : (
        <div className="text-sm text-gray-500">No results yet - click Analyze to start.</div>
      )}
    </div>
  );
}
