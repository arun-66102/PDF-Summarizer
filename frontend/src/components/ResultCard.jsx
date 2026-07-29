export default function ResultCard({ result, source: _source }) {
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
      <div className="result-card-container error-state">
        <div className="result-header">
          <h3 className="result-title error-text">Processing Error</h3>
        </div>
        <p className="error-body">{result.summary || result.error || 'Unknown processing error'}</p>
      </div>
    );
  }

  return (
    <div className="result-card-container">
      {/* Summary Header */}
      <div className="result-header">
        <div className="result-header-left">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#C86D51" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <polyline points="10 9 9 9 8 9"/>
          </svg>
          <h3 className="result-title">Executive Summary</h3>
        </div>
        {summaryText && (
          <button className="secondary-btn icon-btn" onClick={handleDownload}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Download Summary
          </button>
        )}
      </div>

      {/* Summary Body */}
      <div className="summary-body-text">{summaryText}</div>

      {/* Routing Target */}
      {primaryDepts.length > 0 && (
        <div className="routed-dept-box">
          <div>
            <div className="routing-label-text">
              Classification Engine ({routing.method || 'Cosine Similarity'})
            </div>
            <div className="routing-confidence-text">
              Confidence Score: {(routing.confidence || 0).toFixed(3)}
            </div>
          </div>
          <div className="routed-dept-tags">
            {primaryDepts.map((d) => (
              <span key={d} className="dept-pill-badge">
                {d}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Metrics Row */}
      <div className="metrics-row">
        <div className="metric-box">
          <div className="metric-box-info">
            <span className="metric-box-label">Character Length</span>
            <span className="metric-box-val">{result.text_length ?? '—'}</span>
          </div>
        </div>
        <div className="metric-box">
          <div className="metric-box-info">
            <span className="metric-box-label">Processed Chunks</span>
            <span className="metric-box-val">{result.chunks_processed ?? '—'}</span>
          </div>
        </div>
        <div className="metric-box">
          <div className="metric-box-info">
            <span className="metric-box-label">Active LLM Model</span>
            <span className="metric-box-val">{result.model_used ?? '—'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
