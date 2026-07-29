export default function ProgressOverlay({ progress, status }) {
  return (
    <div className="progress-overlay">
      <div className="progress-header-row">
        <svg className="loading-spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#C86D51" strokeWidth="2.5">
          <circle cx="12" cy="12" r="10" strokeOpacity="0.25"/>
          <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round"/>
        </svg>
        <span className="progress-title-text">
          Processing Document Pipeline...
        </span>
      </div>
      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>
      <div className="progress-status">
        {status || 'Initializing inference...'}
        {progress > 0 && ` (${Math.round(progress)}%)`}
      </div>
    </div>
  );
}
