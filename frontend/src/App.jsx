import { useState, useEffect } from 'react';
import { useChat } from './hooks/useChat';
import { getHealth, getModels } from './api/client';
import TopNav from './components/TopNav';
import ChatArea from './components/ChatArea';
import DashboardView from './components/DashboardView';
import DepartmentsView from './components/DepartmentsView';
import HistoryView from './components/HistoryView';
import SettingsView from './components/SettingsView';
import './styles/index.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');

  // Theme State (light | dark) with localStorage persistence
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('routex_theme') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('routex_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  // System Settings & Status
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState('llama3-8b');
  const [contextLimit, setContextLimit] = useState(4000);
  const [enableRouting, setEnableRouting] = useState(true);
  const [enableEmail, setEnableEmail] = useState(true);

  // Chat / Processing state
  const chat = useChat();

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getModels()
      .then((data) => {
        setModels(data.models || []);
        if (!model && data.default) setModel(data.default);
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleTestRoute = (deptCode) => {
    setActiveTab('upload');
  };

  return (
    <div className="organic-app-wrapper">
      {/* Top Bar with Navigation & Theme Toggle */}
      <TopNav
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        status={health}
        theme={theme}
        toggleTheme={toggleTheme}
      />

      {/* Main Content Area */}
      <main className="app-content-area">
        {activeTab === 'upload' && (
          <div className="tab-view-container">
            <div className="view-header">
              <div>
                <h1 className="view-title">Upload &amp; Analyze Document</h1>
                <p className="view-subtitle">
                  Upload a PDF file or paste text to generate instant summaries, extract insights, and auto-route to target departments.
                </p>
              </div>
            </div>

            <ChatArea
              {...chat}
              model={model}
              contextLimit={contextLimit}
              enableRouting={enableRouting}
              enableEmail={enableEmail}
            />
          </div>
        )}

        {activeTab === 'dashboard' && (
          <DashboardView
            history={chat.messages.filter(m => m.role === 'assistant' && m.result)}
            health={health}
            onNavigateToUpload={() => setActiveTab('upload')}
          />
        )}

        {activeTab === 'departments' && (
          <DepartmentsView onTestRoute={handleTestRoute} />
        )}

        {activeTab === 'history' && (
          <HistoryView
            history={chat.messages
              .filter(m => m.role === 'assistant' && m.result)
              .map((m) => ({
                id: m.id,
                time: m.time,
                docName: m.result.file_name || 'Document Processing',
                summary: m.result.summary,
                routing: m.result.routing,
                model_used: m.result.model_used,
                text_length: m.result.text_length,
                result: m.result
              }))}
          />
        )}

        {activeTab === 'settings' && (
          <SettingsView
            model={model}
            setModel={setModel}
            contextLimit={contextLimit}
            setContextLimit={setContextLimit}
            enableRouting={enableRouting}
            setEnableRouting={setEnableRouting}
            enableEmail={enableEmail}
            setEnableEmail={setEnableEmail}
            theme={theme}
            toggleTheme={toggleTheme}
            models={models}
            health={health}
          />
        )}
      </main>

      <footer className="organic-footer">
        <div className="footer-inner font-heading">
          <img src="/logo.png" alt="RouteX" className="footer-logo-img" />
          <span>RouteX AI Engine · Intelligent Document Processing &amp; Department Routing</span>
        </div>
      </footer>
    </div>
  );
}
