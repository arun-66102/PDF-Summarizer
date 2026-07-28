export default function ProgressOverlay({ progress, status }) {
  return (
    <div className="progress-overlay">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span className="spin" style={{ display: 'inline-block' }}>⚙</span>
        <strong style={{ fontSize: '0.85rem', color: 'var(--burnt-orange)' }}>
          Processing...
        </strong>
      </div>
      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>
      <div className="progress-status">
        {status || 'Initializing...'}
        {progress > 0 && ` (${Math.round(progress)}%)`}
      </div>
    </div>
  );
}
