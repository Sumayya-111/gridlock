import React, { useState, useEffect, useRef } from 'react';
import { checkHealth } from '../api/client';
import { connectJudgeWebSocket } from '../api/judgeClient';
import './TopNav.css';
import logo from '../assets/logo.png';

export default function TopNav() {
  const [apiOnline, setApiOnline] = useState(false);
  const [judgeOnline, setJudgeOnline] = useState(false);
  const judgeWsRef = useRef(null);

  useEffect(() => {
    const pollApi = async () => {
      try {
        await checkHealth();
        setApiOnline(true);
      } catch (e) {
        setApiOnline(false);
      }
    };
    pollApi();
    const apiInterval = setInterval(pollApi, 5000);

    const conn = connectJudgeWebSocket(
      () => setJudgeOnline(true),
      () => setJudgeOnline(false)
    );
    judgeWsRef.current = conn;

    const judgeStateCheck = setInterval(() => {
      setJudgeOnline(!!(conn.getState && conn.getState() === WebSocket.OPEN));
    }, 5000);

    return () => {
      clearInterval(apiInterval);
      clearInterval(judgeStateCheck);
      conn.close();
    };
  }, []);

  return (
    <header className="topnav">
      <div className="topnav-container topnav-simple">
        <div className="topnav-brand">
  <div className="brand-logo-tile">
    <img src={logo} alt="Logo" className="brand-logo-img" />
  </div>
  <div className="brand-subtext">AI-Based Traffic Violation Detection System</div>
</div>

        <div className="topnav-status">
          <div className="status-item">
            <span className="status-label">MAIN API</span>
            <span className={`status-dot ${apiOnline ? 'valid' : 'invalid'}`}></span>
          </div>
          <div className="status-item">
            <span className="status-label">LLM JUDGE</span>
            <span className={`status-dot ${judgeOnline ? 'valid' : 'invalid'}`}></span>
          </div>
          <div className="status-item">
            <span className="status-label">VISION MODEL</span>
            <span className="status-dot placeholder"></span>
          </div>
        </div>
      </div>
    </header>
  );
}