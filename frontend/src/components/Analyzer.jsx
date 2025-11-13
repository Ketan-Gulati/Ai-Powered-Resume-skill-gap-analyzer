import React, { useState, useRef } from "react";
import axios from "axios";
import Loader from "./Loader";
import {
  Upload,
  FileText,
  Sparkles,
  Wand2,
} from "lucide-react";

export default function Analyzer({ onResult }) {
  const [jdText, setJdText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef(null);

  const analyze = async (e) => {
    e.preventDefault();
    setError("");

    const file = fileRef.current?.files?.[0] ?? null;

    if (!file && !jdText.trim()) {
      setError("Upload a resume or paste a Job Description.");
      return;
    }

    const form = new FormData();
    if (file) form.append("resume_pdf", file);
    form.append("jd_text", jdText);

    setLoading(true);

    try {
      const response = await axios.post("/api/analyze", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.data) onResult(response.data);
    } catch (err) {
      console.log(err);
      setError("Backend error. Try again.");
    }
    setLoading(false);
  };

  return (
    <div className="px-6 py-10 max-w-5xl mx-auto">

      {/* HEADER SECTION */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-white tracking-tight mb-2 flex justify-center gap-2 items-center">
          <Sparkles className="text-indigo-400" size={34} />
          AI Skill Gap Analyzer
        </h1>

        <p className="text-gray-300 text-sm">
          Upload your resume or paste a JD — get insights powered by AI.
        </p>
      </div>

      {/* MAIN CARD */}
      <div className="
        bg-white/10 backdrop-blur-xl 
        border border-white/20 
        rounded-2xl p-8 shadow-xl
        transition-all duration-300
        hover:shadow-2xl hover:border-indigo-400/30
      ">
        <form onSubmit={analyze}>

          {/* GRID */}
          <div className="grid md:grid-cols-3 gap-8">

            {/* LEFT SIDE — FILE UPLOAD */}
            <div className="space-y-3">
              <label className="text-white font-semibold flex items-center gap-2 text-sm">
                <Upload size={18} /> Resume (PDF)
              </label>

              <div className="
                bg-white/5 border border-white/10 
                p-4 rounded-xl text-center cursor-pointer
                hover:border-indigo-400 hover:bg-white/10
                transition-all duration-200
              ">
                <input
                  ref={fileRef}
                  type="file"
                  accept="application/pdf"
                  className="hidden"
                  id="resumeUpload"
                />

                <label
                  htmlFor="resumeUpload"
                  className="block cursor-pointer text-gray-300 text-sm"
                >
                  <FileText className="mx-auto mb-3 text-indigo-300" size={32} />
                  Click to upload resume PDF
                </label>
              </div>

              <p className="text-xs text-gray-400">
                Uploading a resume is optional if you paste JD.
              </p>
            </div>

            {/* RIGHT SIDE — JOB DESCRIPTION */}
            <div className="md:col-span-2 space-y-3">
              <label className="text-white font-semibold flex items-center gap-2 text-sm">
                <Wand2 size={18} /> Job Description (paste)
              </label>

              <textarea
                value={jdText}
                onChange={(e) => setJdText(e.target.value)}
                rows={7}
                placeholder="Paste the job description here..."
                className="
                  w-full bg-white/10 rounded-xl 
                  border border-white/10 
                  text-gray-200 p-4 text-sm resize-none
                  focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400
                  placeholder-gray-400
                  transition-all
                "
              />
            </div>
          </div>

          {/* BUTTONS */}
          <div className="mt-8 flex items-center gap-4">
            <button
              type="submit"
              disabled={loading}
              className="
                px-6 py-3 rounded-xl text-white font-semibold 
                bg-gradient-to-r from-indigo-500 to-purple-600
                hover:from-indigo-600 hover:to-purple-700
                shadow-lg shadow-indigo-500/20 hover:shadow-purple-600/30
                transition-all duration-200 active:scale-95
              "
            >
              {loading ? "Analyzing..." : "Run Analysis"}
            </button>

            <button
              type="button"
              onClick={() => {
                fileRef.current.value = "";
                setJdText("");
                onResult(null);
                setError("");
              }}
              className="
                px-6 py-3 rounded-xl 
                bg-white/10 text-gray-300 
                border border-white/10
                hover:bg-white/20 hover:border-white/20
                transition-all duration-200
              "
            >
              Reset
            </button>

            {error && (
              <p className="text-red-400 text-sm ml-3">{error}</p>
            )}
          </div>
        </form>
      </div>

      {/* LOADER */}
      {loading && (
        <div className="mt-8">
          <Loader />
        </div>
      )}
    </div>
  );
}
