import React from 'react';
import { ShieldAlert, HardHat, Users, Film } from 'lucide-react';
import CountUp from './CountUp';
import './KpiRow.css';

export default function KpiRow({ summary, framesProcessed }) {
  const cards = [
    { label: 'Total Violations', value: summary?.total || 0, Icon: ShieldAlert, tone: 'red' },
    { label: 'No Helmet', value: summary?.NO_HELMET || 0, Icon: HardHat, tone: 'orange' },
    { label: 'Triple Riding', value: summary?.TRIPLE_RIDING || 0, Icon: Users, tone: 'purple' },
    { label: 'Frames Processed', value: framesProcessed || 0, Icon: Film, tone: 'blue' },
  ];

  return (
    <div className="kpi-container">
      <div className="kpi-strip">
        {cards.map(({ label, value, Icon, tone }) => (
          <div key={label} className={`kpi-block tone-${tone}`}>
            <div className="kpi-icon"><Icon size={20} /></div>
            <div className="kpi-value mono"><CountUp value={value} /></div>
            <div className="kpi-label">{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}