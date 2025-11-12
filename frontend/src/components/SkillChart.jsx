// src/components/SkillChart.jsx
import React, { useMemo } from "react";
import { Bar, Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Tooltip, Legend);

export default function SkillChart({ matchedSkills = [], recommendations = [] }) {
  const labels = useMemo(
    () =>
      matchedSkills.map((s) =>
        typeof s === "string" ? s : s?.name ?? s?.skill ?? "unknown"
      ),
    [matchedSkills]
  );

  const freq = useMemo(() => {
    const map = {};
    labels.forEach((l) => {
      const key = l || "unknown";
      map[key] = (map[key] || 0) + 1;
    });
    return Object.entries(map)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
  }, [labels]);

  const barData = useMemo(() => {
    const barLabels = freq.map((f) => f[0]);
    const barValues = freq.map((f) => f[1]);
    const colors = barLabels.map((_, i) => `hsl(${(i * 40) % 360} 70% 50%)`);
    return {
      labels: barLabels,
      datasets: [
        {
          label: "Matches",
          data: barValues,
          backgroundColor: colors,
          borderRadius: 6,
        },
      ],
    };
  }, [freq]);

  const pieData = useMemo(() => {
    const pieLabels = recommendations.length
      ? recommendations.map((r, i) => {
          if (typeof r === "string") return r.slice(0, 12);
          if (r && r.title) return r.title.slice(0, 24);
          return `rec-${i + 1}`;
        })
      : ["none"];
    const pieValues = recommendations.length ? recommendations.map(() => 1) : [1];
    const pieColors = pieLabels.map((_, i) => `hsl(${(i * 60) % 360} 70% 50%)`);
    return {
      labels: pieLabels,
      datasets: [
        {
          data: pieValues,
          backgroundColor: pieColors,
        },
      ],
    };
  }, [recommendations]);

  return (
    <div className="bg-white rounded-lg shadow-sm p-4">
      <h3 className="text-md font-medium mb-3">Visual Summary</h3>

      <div className="space-y-4">
        <div className="min-h-[180px]">
          {freq.length ? (
            <Bar data={barData} options={{ maintainAspectRatio: false }} />
          ) : (
            <div className="text-sm text-gray-400">No matched skills to show</div>
          )}
        </div>

        <div className="min-h-[180px]">
          {recommendations.length ? (
            <Pie data={pieData} options={{ maintainAspectRatio: false }} />
          ) : (
            <div className="text-sm text-gray-400">No recommendations to show</div>
          )}
        </div>
      </div>
    </div>
  );
}
