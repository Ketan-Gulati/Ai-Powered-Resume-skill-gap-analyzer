// src/components/Analyzer.jsx
import React, { useState } from "react";
import axios from "axios";
import SkillChip from "./SkillChip";
import Loader from "./Loader";

export default function Analyzer({ onResult }) {
  const [jdText, setJdText] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [previewName, setPreviewName] = useState("");

  const handleFile = (e) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setPreviewName(f ? f.name : "");
  };

  const analyze = async (e) => {
    e.preventDefault();
    setErr("");
    if (!jdText.trim() && !file) {
      setErr("Paste a JD or upload a resume (or both).");
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      form.append("jd_text", jdText);
      if (file) form.append("resume_pdf", file);

      // change URL if backend runs on another port (eg http://localhost:9000/api/analyze)
      const resp = await axios.post("/api/analyze", form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });

      // expect backend to return { data: { match: {...}, recommendations: {...} } }
      const result = resp.data?.data ?? resp.data;
      onResult && onResult(result);
    } catch (error) {
      console.error(error);
      if (error.response?.data?.detail) setErr(error.response.data.detail);
      else if (error.message) setErr(error.message);
      else setErr("Unknown error. Check server logs.");
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setJdText("");
    setFile(null);
    setPreviewName("");
    setErr("");
    onResult && onResult(null);
  };

  return (
    <div className="w-full max-w-3xl bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold mb-4">Analyze Resume</h2>

      <form onSubmit={analyze} className="space-y-4">
        <label className="block text-sm font-medium text-gray-700">Resume (PDF)</label>
        <div className="flex items-center gap-3">
          <label className="inline-flex items-center px-4 py-2 bg-indigo-600 text-white rounded-md cursor-pointer hover:bg-indigo-700">
            Choose File
            <input
              type="file"
              accept="application/pdf"
              onChange={handleFile}
              className="hidden"
            />
          </label>
          <div className="text-sm text-gray-600">{previewName || "No file selected"}</div>
          {previewName && (
            <button
              type="button"
              onClick={() => { setFile(null); setPreviewName(""); }}
              className="ml-auto text-xs text-red-600 hover:underline"
            >
              Remove
            </button>
          )}
        </div>

        <label className="block text-sm font-medium text-gray-700">Job Description</label>
        <textarea
          value={jdText}
          onChange={(e) => setJdText(e.target.value)}
          rows={8}
          placeholder="Paste JD text here (e.g. 'React, Node.js, MongoDB, Docker...')"
          className="w-full rounded-md border-gray-200 shadow-sm focus:ring-2 focus:ring-indigo-300 p-3 bg-gray-50"
        />

        {err && <div className="text-sm text-red-600">{err}</div>}

        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={loading}
            className={`inline-flex items-center px-5 py-2 rounded-md text-white font-medium ${
              loading ? "bg-indigo-300 cursor-not-allowed" : "bg-indigo-600 hover:bg-indigo-700"
            }`}
          >
            {loading ? (
              <Loader/>
            ) : (
              "Analyze Resume"
            )}
          </button>

          <button
            type="button"
            onClick={clear}
            className="px-4 py-2 rounded-md border border-gray-200 bg-white text-gray-700 hover:bg-gray-50"
          >
            Reset
          </button>
        </div>

        <div className="mt-3">
          <div className="text-xs text-gray-500">Tip: paste JD and upload resume for best results.</div>
        </div>
      </form>
    </div>
  );
}