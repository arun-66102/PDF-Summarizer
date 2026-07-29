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

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, processing]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
      attachDoc(file);
    }
    // Reset input so same file can be re-selected
    e.target.value = '';
  };

  const handleSend = async (_forceProcessDocs = false) => {
    const hasDocs = attachedDocs.length > 0;
    const hasText = inputText.trim().length > 0;

    if (!hasDocs && !hasText) return;
    if (processing) return;

    const isPdfMode = hasDocs;
    const docName = isPdfMode ? attachedDocs[0].name : null;

    // Add user message
    addUserMessage({
      text: hasText ? inputText.trim() : null,
      docName,
    });

    setInputText('');
    setProcessing(true);
    setProgress({ progress: 0, status: 'Starting...', done: false });

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

      // If we have a task_id, stream progress updates
      if (result.task_id) {
        streamProgress(result.task_id, (data) => {
          setProgress(data);
        });
      }

      setProgress({ progress: 100, status: 'Done', done: true });

      // Handle email sending if enabled and routing has departments
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
          result._emailResult = { success: false, message: 'Email sending failed' };
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

  // ── Welcome screen ──
  if (messages.length === 0 && !processing) {
    return (
      <div className="main-content">
        <div className="chat-container">
          <div className="welcome-hero">
            <img
              src="/logo.png"
              alt="RouteX"
              className="logo-img"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'flex';
              }}
            />
            <div className="logo-fallback" style={{ display: 'none' }}>🤖</div>
            <h2>What can I help with?</h2>
            <p className="sub">
              Upload a PDF or paste text — I'll summarize, analyze, and route it.
            </p>
            <div className="suggestion-grid">
              <div className="suggestion-card" onClick={() => fileInputRef.current?.click()}>
                <div className="s-icon">📄</div>
                <div className="s-title">Summarize PDF</div>
                <div className="s-desc">Get a quick overview of any document</div>
              </div>
              <div className="suggestion-card" onClick={() => document.getElementById('chat-input')?.focus()}>
                <div className="s-icon">🔍</div>
                <div className="s-title">Analyze text</div>
                <div className="s-desc">Extract key insights and action items</div>
              </div>
              <div className="suggestion-card">
                <div className="s-icon">🎯</div>
                <div className="s-title">Route to department</div>
                <div className="s-desc">Classify content for specific teams</div>
              </div>
              <div className="suggestion-card">
                <div className="s-icon">📧</div>
                <div className="s-title">Email summary</div>
                <div className="s-desc">Send the results directly to your inbox</div>
              </div>
            </div>
          </div>
        </div>

        {/* Input area */}
        {renderInputArea()}
      </div>
    );
  }

  // ── Chat view ──
  return (
    <div className="main-content">
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg) =>
            msg.role === 'user' ? (
              <div key={msg.id} className="chat-msg">
                {msg.docName && (
                  <div className="doc-chip-bar" style={{ justifyContent: 'flex-end' }}>
                    <div className="doc-chip">📄 {msg.docName}</div>
                  </div>
                )}
                {msg.text && (
                  <>
                    <div className="user-bubble">
                      {msg.text.length > 500 ? msg.text.slice(0, 500) + '…' : msg.text}
                    </div>
                    <div className="user-time">{msg.time}</div>
                  </>
                )}
                {!msg.text && msg.docName && (
                  <div className="user-time" style={{ textAlign: 'right' }}>{msg.time}</div>
                )}
              </div>
            ) : (
              <div key={msg.id} className="chat-msg">
                <div className="assistant-header">
                  <span className="name">RouteX</span>
                  <span className="time">{msg.time}</span>
                </div>
                <ResultCard result={msg.result} source={msg.source} />
                {/* Email status */}
                {msg.result?._emailResult && (
                  <div className="email-section">
                    {msg.result._emailResult.success ? (
                      <p className="email-success">
                        📧 {msg.result._emailResult.message}
                      </p>
                    ) : (
                      <p className="email-warning">
                        ⚠️ {msg.result._emailResult.message}
                      </p>
                    )}
                  </div>
                )}
              </div>
            )
          )}

          {/* Processing indicator */}
          {processing && (
            <div className="chat-msg">
              <div className="assistant-header">
                <span className="name">RouteX</span>
                <span className="time">processing...</span>
              </div>
              <ProgressOverlay
                progress={progress.progress}
                status={progress.status}
              />
            </div>
          )}

          <div ref={chatEndRef} />
        </div>
      </div>

      {renderInputArea()}
    </div>
  );

  function renderInputArea() {
    return (
      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          {/* Attached doc chips */}
          {attachedDocs.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div className="doc-chip-bar">
                {attachedDocs.map((doc, i) => (
                  <div className="doc-chip" key={i}>
                    📄 {doc.name}{' '}
                    <span className="chip-size">
                      ({(doc.size / 1024).toFixed(1)} KB)
                    </span>
                    <button className="chip-remove" onClick={() => removeDoc(i)}>
                      ✕
                    </button>
                  </div>
                ))}
              </div>
              <button className="btn-process-docs" onClick={() => handleSend(true)}>
                ➤ Process {attachedDocs.length} document(s)
              </button>
            </div>
          )}

          {/* Input row */}
          <div className="chat-input-row">
            <div className="btn-attach">
              📎
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
              />
            </div>
            <textarea
              id="chat-input"
              placeholder="Message RouteX…"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={processing}
            />
            <button
              className="btn-send"
              onClick={() => handleSend()}
              disabled={processing || (!inputText.trim() && attachedDocs.length === 0)}
            >
              ➤
            </button>
          </div>
        </div>
      </div>
    );
  }
}
