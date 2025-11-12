import React from "react";

export default function Loader({ size=36, text="Loading..." }) {
  return (
    <div style={{display:'flex',alignItems:'center',gap:12}}>
      <svg width={size} height={size} viewBox="0 0 50 50">
        <path fill="none" stroke="#5c4dff" strokeWidth="4" strokeLinecap="round"
          d="M25 5 a20 20 0 0 1 0 40 a20 20 0 0 1 0 -40"
          strokeDasharray="31.4 31.4">
          <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="1s" repeatCount="indefinite"/>
        </path>
      </svg>
      <div style={{fontSize:14,color:'#0f1724'}}>{text}</div>
    </div>
  );
}