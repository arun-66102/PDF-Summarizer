import { useState, useEffect } from 'react';
import { getHealth, getModels } from '../api/client';

export default function Sidebar({
  model, setModel,
  contextLimit, setContextLimit,
  enableRouting, setEnableRouting,
  enableEmail, setEnableEmail,
  onNewChat,
  isOpen,
  onClose,
}) {
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getModels()
      .then((data) => {
        setModels(data.models || []);
        if (!model && data.default) setModel(data.default);
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // intentionally run once on mount

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && <div className="sidebar-overlay" onClick={onClose} />}

      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        {/* New Chat */}
        <button className="btn-new-chat" onClick={onNewChat}>
          ✦ New Chat
        </button>

        <hr />

        {/* Model selector */}
        <label htmlFor="model-select">Model</label>
        <select
          id="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          {models.map((m) => (
            <option key={m.key} value={m.key}>
              {m.label}
            </option>
          ))}
        </select>

        {/* Settings */}
        <details className="settings-expander" style={{ marginTop: '1rem' }}>
          <summary>⚙ Settings</summary>
          <div className="settings-body">
            <label htmlFor="context-slider">
              Context limit
            </label>
            <input
              id="context-slider"
              type="range"
              min={1000}
              max={8000}
              step={500}
              value={contextLimit}
              onChange={(e) => setContextLimit(Number(e.target.value))}
            />
            <div className="slider-value">{contextLimit}</div>

            <div className="checkbox-row">
              <input
                id="routing-toggle"
                type="checkbox"
                checked={enableRouting}
                onChange={(e) => setEnableRouting(e.target.checked)}
              />
              <label htmlFor="routing-toggle">Department routing</label>
            </div>

            <div className="checkbox-row">
              <input
                id="email-toggle"
                type="checkbox"
                checked={enableEmail}
                onChange={(e) => setEnableEmail(e.target.checked)}
              />
              <label htmlFor="email-toggle">Email delivery</label>
            </div>
          </div>
        </details>

        <hr />

        {/* Status badges */}
        {health && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <span className={`badge ${health.groq_connected ? 'badge-ok' : 'badge-err'}`}>
              ● {health.groq_connected ? 'Groq connected' : 'Groq key missing'}
            </span>
            {enableEmail && (
              <span className={`badge ${health.email_configured ? 'badge-ok' : 'badge-warn'}`}>
                ● {health.email_configured ? 'Email ready' : 'Email not set'}
              </span>
            )}
            <span className={`badge ${health.routing_available ? 'badge-ok' : 'badge-warn'}`}>
              ● {health.routing_available ? 'Routing ready' : 'Routing limited'}
            </span>
          </div>
        )}
      </aside>

      <style>{`
        .sidebar-overlay {
          display: none;
        }
        @media (max-width: 768px) {
          .sidebar-overlay {
            display: block;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.4);
            z-index: 99;
          }
        }
      `}</style>
    </>
  );
}
