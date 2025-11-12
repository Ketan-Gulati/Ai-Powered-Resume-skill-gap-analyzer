import React from "react";
export default function SkillChip({text, tone='missing'}) {
  const style = tone === 'missing' ? {background:'#fff7ed', color:'#8c4300'} : {background:'#def7ec', color:'#03543f'};
  return <div style={{padding:'6px 10px',borderRadius:18, fontWeight:700, fontSize:13, ...style}}>{text}</div>;
}