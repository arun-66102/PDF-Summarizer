import { useState, useRef, useEffect } from 'react';
import ResultCard from './ResultCard';
import ProgressOverlay from './ProgressOverlay';
import { processPdf, processText, sendEmail, streamProgress } from '../api/client';

export default function ChatArea({
  messages,
  attachedDocs,
  processing,
  progress,
  setProcessing,
  setProgress,
  addUserMessage,
  addAssistantMessage,
  attachDoc,
  removeDoc,
  clearDocs,
  // Settings
  model,
  contextLimit,
  enableRouting,
  enableEmail,
}) {
  const [inputText, setInputText] = useState('');
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, processing]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
      attachDoc(file);
    }
    e.target.value = '';
  };

  const handleSend = async () => {
    const hasDocs = attachedDocs.length > 0;
    const hasText = inputText.trim().length > 0;

    if (!hasDocs && !hasText) return;
    if (processing) return;

    const isPdfMode = hasDocs;
    const docName = isPdfMode ? attachedDocs[0].name : null;

    addUserMessage({
      text: hasText ? inputText.trim() : null,
      docName,
    });

    setInputText('');
    setProcessing(true);
    setProgress({ progress: 0, status: 'Initializing analysis...', done: false });

    try {
      let result;

      if (isPdfMode) {
        const doc = attachedDocs[0];
        result = await processPdf(doc.file, {
          model,
          contextLimit,
          enableRouting,
        });
      } else {
        result = await processText(inputText.trim(), {
          model,
          contextLimit,
          enableRouting,
        });
      }

      if (result.task_id) {
        streamProgress(result.task_id, (data) => {
          setProgress(data);
        });
      }

      setProgress({ progress: 100, status: 'Done', done: true });

      if (
        enableEmail &&
        isPdfMode &&
        result.routing?.primary_departments?.length > 0 &&
        result.file_path
      ) {
        try {
          const emailResult = await sendEmail(
            result.summary,
            result.routing,
            result.file_path
          );
          result._emailResult = emailResult;
        } catch {
          result._emailResult = { success: false, message: 'Email dispatch failed' };
        }
      }

      addAssistantMessage(result, isPdfMode ? 'pdf' : 'text');
      clearDocs();
    } catch (err) {
      addAssistantMessage(
        { summary: err.message || 'Processing failed', error: err.message },
        isPdfMode ? 'pdf' : 'text'
      );
    } finally {
      setProcessing(false);
      setProgress({ progress: 0, status: '', done: false });
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="upload-workspace">
      {messages.length === 0 && !processing ? (
        <div className="organic-card welcome-hero-card">
          {/* Brand Header with Logo */}
          <div className="hero-logo-banner">
            <img src="/logo.png" alt="RouteX AI" className="hero-logo-img" />
          </div>

          {/* PDF Drag and Drop Zone */}
          <div
            className="upload-card-zone"
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-icon-svg">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#C86D51" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="12" y1="18" x2="12" y2="12"/>
                <polyline points="9 15 12 12 15 15"/>
              </svg>
            </div>
            <h3 className="upload-prompt-title">
              Drop PDF file here or click to browse
            </h3>
            <p className="upload-prompt-sub">
              Upload research papers, technical specs, or reports for instant AI summarization &amp; department classification.
            </p>

            <button type="button" className="primary-action-btn" style={{ marginTop: '8px' }}>
              Select PDF File
            </button>
          </div>

          {/* Quick Action Suggestions */}
          <div className="suggestion-grid">
            <div
              className="suggestion-card"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="s-icon-svg">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#C86D51" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
              </div>
              <div className="s-title">Summarize PDF</div>
              <div className="s-desc">Extract bullet insights and key takeaways</div>
            </div>

            <div
              className="suggestion-card"
              onClick={() => document.getElementById('chat-input')?.focus()}
            >
              <div className="s-icon-svg">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6B8E7B" strokeWidth="2">
                  <circle cx="11" cy="11" r="8"/>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
              </div>
              <div className="s-title">Analyze Text</div>
              <div className="s-desc">Paste raw text or snippets for instant routing</div>
            </div>

            <div className="suggestion-card">
              <div className="s-icon-svg">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#D4A373" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <circle cx="12" cy="12" r="6"/>
                  <circle cx="12" cy="12" r="2"/>
                </svg>
              </div>
              <div className="s-title">Department Routing</div>
              <div className="s-desc">Auto-classify to CSE, EEE, MECH, or CIVIL</div>
            </div>

            <div className="suggestion-card">
              <div className="s-icon-svg">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#88A0A8" strokeWidth="2">
                  <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                  <polyline points="22,6 12,13 2,6"/>
                </svg>
              </div>
              <div className="s-title">Email Dispatch</div>
              <div className="s-desc">Send reports directly to department heads</div>
            </div>
          </div>
        </div>
      ) : (
        <div className="chat-messages-container">
          {messages.map((msg) =>
            msg.role === 'user' ? (
              <div key={msg.id} className="chat-msg user-msg-row">
                {msg.docName && (
                  <div className="attached-doc-badge">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                      <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    {msg.docName}
                  </div>
                )}
                {msg.text && (
                  <div className="user-bubble">
                    {msg.text}
                  </div>
                )}
                <span className="msg-time-stamp">{msg.time}</span>
              </div>
            ) : (
              <div key={msg.id} className="chat-msg assistant-msg-row">
                <div className="assistant-header-bar">
                  <img src="/logo.png" alt="RouteX" className="assistant-mini-logo" />
                  <span className="assistant-brand-tag">RouteX AI</span>
                  <span className="msg-time-stamp">{msg.time}</span>
                </div>
                <ResultCard result={msg.result} source={msg.source} />
                {msg.result?._emailResult && (
                  <div className={`email-status-banner ${msg.result._emailResult.success ? 'success' : 'warning'}`}>
                    {msg.result._emailResult.success ? 'Email report dispatched successfully' : 'Email dispatch skipped or failed'}
                  </div>
                )}
              </div>
            )
          )}

          {processing && (
            <div className="chat-msg assistant-msg-row">
              <div className="assistant-header-bar">
                <img src="/logo.png" alt="RouteX" className="assistant-mini-logo" />
                <span className="assistant-brand-tag">RouteX AI</span>
                <span className="msg-time-stamp">Processing...</span>
              </div>
              <ProgressOverlay progress={progress.progress} status={progress.status} />
            </div>
          )}

          <div ref={chatEndRef} />
        </div>
      )}

      {/* Input area */}
      <div className="text-input-card">
        {attachedDocs.length > 0 && (
          <div className="doc-chip-bar">
            {attachedDocs.map((doc, i) => (
              <div className="doc-chip" key={i}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                <strong>{doc.name}</strong> ({(doc.size / 1024).toFixed(1)} KB)
                <button className="chip-remove" onClick={() => removeDoc(i)}>
                  ✕
                </button>
              </div>
            ))}
            <button className="secondary-btn" onClick={() => handleSend()}>
              Process Attached PDF
            </button>
          </div>
        )}

        <div className="chat-input-row">
          <div className="btn-attach-wrap" title="Attach PDF Document">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#736354" strokeWidth="2">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="hidden-file-input"
            />
          </div>

          <textarea
            id="chat-input"
            placeholder="Paste text content or enter document processing instructions..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            disabled={processing}
            className="organic-textarea"
          />

          <button
            type="button"
            className="primary-action-btn send-btn"
            onClick={() => handleSend()}
            disabled={processing || (!inputText.trim() && attachedDocs.length === 0)}
          >
            Process &amp; Route
          </button>
        </div>
      </div>
    </div>
  );
}
