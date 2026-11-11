import React from 'react';
import { motion } from 'framer-motion';
import { Activity } from 'lucide-react';
import KpiRow from '../components/KpiRow';
import ViolationLog from '../components/ViolationLog';
import { useSettings } from '../contexts/SettingsContext';
import './Dashboard.css';

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12 } },
};
const item = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
};

export default function Dashboard() {
  const { summary, framesProcessed, violations, handleClearLog } = useSettings();

  return (
    <motion.div className="page-container" variants={container} initial="hidden" animate="show">
      <motion.div className="dashboard-header-band" variants={item}>
        <div>
          <h2 className="dashboard-title">Dashboard Overview</h2>
          <p className="dashboard-sub">Live traffic violations from your detection pipeline</p>
        </div>
        <span className="live-badge">
          <Activity size={14} /> LIVE
        </span>
      </motion.div>

      <motion.div variants={item}>
        <KpiRow summary={summary} framesProcessed={framesProcessed} />
      </motion.div>

      <motion.div className="dashboard-content" variants={item}>
        <ViolationLog violations={violations} onClear={handleClearLog} />
      </motion.div>
    </motion.div>
  );
}