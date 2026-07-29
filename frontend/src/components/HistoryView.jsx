import { useState, useEffect } from 'react';
import { getHistory, sendEmail } from '../api/client';

export default function HistoryView({ history: propHistory = [], onResendEmail }) {
  const [selectedDept, setSelectedDept] = useState('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [activeModalItem, setActiveModalItem] = useState(null);
  const [realHistory, setRealHistory] = useState([]);
  const [dispatchStatus, setDispatchStatus] = useState({});

  useEffect(() => {
    getHistory()
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setRealHistory(data);
        }
      })
      .catch(() => {});
  }, []);

  const displayItems = realHistory.length > 0 ? realHistory : propHistory;

  const filteredItems = displayItems.filter((item) => {
    const routing = item.routing || item.result?.routing || {};
    const depts = routing.primary_departments || [];
    const matchesDept = selectedDept === 'ALL' || depts.includes(selectedDept);
    const textToSearch = `${item.docName || ''} ${item.summary || item.result?.summary || ''}`.toLowerCase();
    const matchesSearch = textToSearch.includes(searchTerm.toLowerCase());
    return matchesDept && matchesSearch;
  });

  const handleDownload = (item) => {
    const text = item.summary || item.result?.summary || '';
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `summary_${item.docName || 'routex_doc'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDispatchEmail = async (item) => {
    const routing = item.routing || item.result?.routing;
    const summaryText = item.summary || item.result?.summary;
    const pdfPath = item.file_path || item.result?.file_path;

    if (!routing || !summaryText) return;

    setDispatchStatus((prev) => ({ ...prev, [item.id]: 'Sending...' }));

    try {
      const res = await sendEmail(summaryText, routing, pdfPath);
      setDispatchStatus((prev) => ({
        ...prev,
        [item.id]: res.success ? `Sent: ${res.message}` : `Failed: ${res.message}`
      }));
    } catch (err) {
      setDispatchStatus((prev) => ({
        ...prev,
        [item.id]: `Error: ${err.message}`
      }));
    }
  };

  return (
    <div className="tab-view-container">
      <div className="view-header">
        <div>
          <h1 className="view-title">Document Processing History</h1>
          <p className="view-subtitle">
            Real-time audit log of all summarized documents and department email dispatch records.
          </p>
        </div>
      </div>

      {/* Filter Row */}
      <div className="history-filter-bar">
        <div className="dept-tabs-filter">
          {['ALL', 'CSE', 'EEE', 'MECH', 'CIVIL'].map((dept) => (
            <button
              key={dept}
              className={`filter-chip ${selectedDept === dept ? 'active' : ''}`}
              onClick={() => setSelectedDept(dept)}
            >
              {dept}
            </button>
          ))}
        </div>

        <div className="search-input-wrap">
          <svg className="search-icon-svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#736354" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <input
            type="text"
            placeholder="Search real-time history..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="organic-search-input"
          />
        </div>
      </div>

      {/* History List */}
      <div className="history-list">
        {filteredItems.length === 0 ? (
          <div className="organic-card" style={{ textAlign: 'center', padding: '40px' }}>
            <p style={{ color: 'var(--text-secondary)' }}>No processing records found yet. Process a PDF or text document to view real-time history.</p>
          </div>
        ) : (
          filteredItems.map((item, idx) => {
            const summaryText = item.summary || item.result?.summary || 'No summary text';
            const routing = item.routing || item.result?.routing || {};
            const depts = routing.primary_departments || [];
            const recipients = item.email_recipients || {};

            return (
              <div key={item.id || idx} className="history-card">
                <div className="history-card-header">
                  <div className="history-doc-info">
                    <div className="doc-icon-wrap">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#C86D51" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                      </svg>
                    </div>
                    <div>
                      <h4 className="doc-name">{item.docName || 'Text Content Processing'}</h4>
                      <span className="doc-timestamp">{item.time || 'Just now'}</span>
                    </div>
                  </div>

                  <div className="history-badges">
                    {depts.map((d) => (
                      <span key={d} className="dept-pill-badge">
                        {d}
                      </span>
                    ))}
                    <span className="model-pill">{item.model_used || 'llama3-8b'}</span>
                  </div>
                </div>

                <p className="history-summary-preview">
                  {summaryText.length > 220 ? `${summaryText.slice(0, 220)}...` : summaryText}
                </p>

                {/* Real-time Email Dispatch Info */}
                {Object.keys(recipients).length > 0 && (
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', backgroundColor: 'var(--bg-main)', padding: '8px 12px', borderRadius: 'var(--radius-md)' }}>
                    <strong>Email Recipients:</strong>{' '}
                    {Object.entries(recipients).map(([d, email]) => `${d} (${email})`).join(' · ')}
                  </div>
                )}

                {dispatchStatus[item.id] && (
                  <div style={{ fontSize: '12px', color: 'var(--accent-terracotta)', fontWeight: '600' }}>
                    {dispatchStatus[item.id]}
                  </div>
                )}

                <div className="history-actions-bar">
                  <button
                    className="history-action-btn"
                    onClick={() => setActiveModalItem(item)}
                  >
                    View Summary
                  </button>
                  <button
                    className="history-action-btn"
                    onClick={() => handleDownload(item)}
                  >
                    Download TXT
                  </button>
                  {depts.length > 0 && (
                    <button
                      className="history-action-btn highlight"
                      onClick={() => handleDispatchEmail(item)}
                    >
                      Dispatch Email Report
                    </button>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Modal */}
      {activeModalItem && (
        <div className="modal-backdrop" onClick={() => setActiveModalItem(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{activeModalItem.docName || 'Document Summary'}</h3>
              <button className="modal-close-btn" onClick={() => setActiveModalItem(null)}>
                ✕
              </button>
            </div>
            <div className="modal-body">
              <div className="modal-routing-info">
                <strong>Routed Departments &amp; Emails:</strong>{' '}
                {(activeModalItem.routing?.primary_departments || activeModalItem.result?.routing?.primary_departments || []).map(d => `${d} (${activeModalItem.email_recipients?.[d] || 'Configured Email'})`).join(', ')}
              </div>
              <div className="modal-text">
                {activeModalItem.summary || activeModalItem.result?.summary}
              </div>
            </div>
            <div className="modal-footer">
              <button className="secondary-btn" onClick={() => setActiveModalItem(null)}>
                Close
              </button>
              <button className="primary-action-btn" onClick={() => handleDownload(activeModalItem)}>
                Download Summary
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
