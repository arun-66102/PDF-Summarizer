export default function ResultCard({ result, source }) {
  if (!result) return null;

  const hasError = result.error && !result.routing;
  const routing = result.routing || {};
  const primaryDepts = routing.primary_departments || [];
  const summaryText = result.summary || '';

  const handleDownload = () => {
    const blob = new Blob([summaryText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `summary_${new Date().toISOString().slice(0, 19).replace(/[:-]/g, '')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (hasError) {
    return (
      <div className="result-section" style={{ borderColor: '#dc2626' }}>
        <h4>❌ Error</h4>
        <p>{result.summary || result.error || 'Unknown error'}</p>
      </div>
    );
  }

  return (
    <div>
      {/* Summary */}
      <div className="result-section">
        <h4>📝 Summary</h4>
        <div style={{ whiteSpace: 'pre-wrap' }}>{summaryText}</div>
      </div>

      {/* Routing */}
      {primaryDepts.length > 0 && (
        <div className="routing-section">
          <p>
            <strong>🎯 Routed to{routing.is_tie ? ' (tie)' : ''}:</strong>{' '}
            {primaryDepts.join(', ')}
          </p>
          <p className="caption">
            {routing.method || '?'} · confidence{' '}
            {(routing.confidence || 0).toFixed(3)}
          </p>
        </div>
      )}

      {/* Metrics */}
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-label">Length</div>
          <div className="metric-value">{result.text_length ?? '—'}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Chunks</div>
          <div className="metric-value">{result.chunks_processed ?? '—'}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Model</div>
          <div className="metric-value">{result.model_used ?? '—'}</div>
        </div>
      </div>

      {/* Download */}
      {summaryText && (
        <button className="btn-download" onClick={handleDownload}>
          💾 Download Summary
        </button>
      )}
    </div>
  );
}
