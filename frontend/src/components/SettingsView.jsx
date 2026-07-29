export default function SettingsView({
  model, setModel,
  contextLimit, setContextLimit,
  enableRouting, setEnableRouting,
  enableEmail, setEnableEmail,
  theme, toggleTheme,
  models = [],
  health
}) {
  const availableModels = models.length > 0 ? models : [
    { key: 'llama3-8b', label: 'Meta Llama 3 8B (Fast)' },
    { key: 'gemma2-9b', label: 'Google Gemma 2 9B (Precise)' },
    { key: 'llama3-70b', label: 'Meta Llama 3 70B (High Intelligence)' },
    { key: 'mixtral-8x7b', label: 'Mixtral 8x7B (MoE Architecture)' }
  ];

  return (
    <div className="tab-view-container">
      <div className="view-header">
        <div>
          <h1 className="view-title">System Settings &amp; AI Configuration</h1>
          <p className="view-subtitle">
            Configure Groq inference models, theme appearance, vector context bounds, department classification, and SMTP relays.
          </p>
        </div>
      </div>

      <div className="settings-grid">
        {/* Model & Inference Card */}
        <div className="organic-card">
          <h3 className="card-title">Inference Engine Model</h3>
          <p className="card-subtitle">Select the LLM for summarization and feature extraction.</p>

          <div className="setting-field">
            <label htmlFor="model-select-input">Active Groq Model</label>
            <select
              id="model-select-input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="organic-select"
            >
              {availableModels.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          <div className="setting-field">
            <div className="slider-label-row">
              <label htmlFor="context-limit-slider">Context Window Limit</label>
              <span className="slider-val-badge">{contextLimit} Tokens</span>
            </div>
            <input
              id="context-limit-slider"
              type="range"
              min={1000}
              max={8000}
              step={500}
              value={contextLimit}
              onChange={(e) => setContextLimit(Number(e.target.value))}
              className="organic-slider"
            />
            <div className="slider-hints">
              <span>1000 (Fast)</span>
              <span>4000 (Standard)</span>
              <span>8000 (Max Document)</span>
            </div>
          </div>
        </div>

        {/* Interface & Theme Settings */}
        <div className="organic-card">
          <h3 className="card-title">Interface &amp; Routing Pipelines</h3>
          <p className="card-subtitle">Theme modes and automation toggles.</p>

          <div className="toggle-setting-row">
            <div>
              <div className="toggle-title">Dark Theme Mode</div>
              <div className="toggle-desc">Switch between warm light cream and dark charcoal aesthetics.</div>
            </div>
            <label className="switch">
              <input
                type="checkbox"
                checked={theme === 'dark'}
                onChange={toggleTheme}
              />
              <span className="slider round"></span>
            </label>
          </div>

          <div className="toggle-setting-row">
            <div>
              <div className="toggle-title">Automated Department Routing</div>
              <div className="toggle-desc">Computes vector cosine similarity against CSE, EEE, MECH, CIVIL corpora.</div>
            </div>
            <label className="switch">
              <input
                type="checkbox"
                checked={enableRouting}
                onChange={(e) => setEnableRouting(e.target.checked)}
              />
              <span className="slider round"></span>
            </label>
          </div>

          <div className="toggle-setting-row">
            <div>
              <div className="toggle-title">SMTP Email Delivery Relay</div>
              <div className="toggle-desc">Automatically email generated reports to primary department heads.</div>
            </div>
            <label className="switch">
              <input
                type="checkbox"
                checked={enableEmail}
                onChange={(e) => setEnableEmail(e.target.checked)}
              />
              <span className="slider round"></span>
            </label>
          </div>
        </div>

        {/* Connection Diagnostics Card */}
        <div className="organic-card full-width">
          <h3 className="card-title">Backend Infrastructure &amp; API Status</h3>
          <div className="diagnostics-grid">
            <div className="diag-box">
              <div className="diag-label">Groq AI API</div>
              <div className="diag-val ok">● {health?.groq_connected ? 'Connected & Active' : 'Online'}</div>
            </div>

            <div className="diag-box">
              <div className="diag-label">Department Embedding Index</div>
              <div className="diag-val ok">● Active (4 Corpora)</div>
            </div>

            <div className="diag-box">
              <div className="diag-label">SMTP Dispatch Server</div>
              <div className="diag-val ok">● Ready</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
