import { useState } from 'react';
import { useChat } from './hooks/useChat';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import './styles/index.css';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Settings
  const [model, setModel] = useState('llama3-8b');
  const [contextLimit, setContextLimit] = useState(4000);
  const [enableRouting, setEnableRouting] = useState(true);
  const [enableEmail, setEnableEmail] = useState(true);

  // Chat state
  const chat = useChat();

  return (
    <div className="app-layout">
      {/* Mobile menu button */}
      <button
        className="mobile-menu-btn"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Toggle menu"
      >
        ☰
      </button>

      <Sidebar
        model={model}
        setModel={setModel}
        contextLimit={contextLimit}
        setContextLimit={setContextLimit}
        enableRouting={enableRouting}
        setEnableRouting={setEnableRouting}
        enableEmail={enableEmail}
        setEnableEmail={setEnableEmail}
        onNewChat={chat.newChat}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <ChatArea
        {...chat}
        model={model}
        contextLimit={contextLimit}
        enableRouting={enableRouting}
        enableEmail={enableEmail}
      />

      {/* Footer */}
      <div className="app-footer">
        Powered by <span>Groq AI</span> · Intelligent Routing · Built by{' '}
        <span>RAG Retrievers</span>
      </div>
    </div>
  );
}
