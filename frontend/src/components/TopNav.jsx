export default function TopNav({ activeTab, setActiveTab, status, theme, toggleTheme }) {
  const navTabs = [
    { id: 'upload', label: 'Upload' },
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'departments', label: 'Departments' },
    { id: 'history', label: 'History' },
    { id: 'settings', label: 'Settings' }
  ];

  return (
    <header className="organic-topnav">
      <div className="topnav-inner">
        {/* Left Brand Identity with Logo */}
        <div className="brand-title-wrap" onClick={() => setActiveTab('upload')}>
          <img src="/logo.png" alt="RouteX Logo" className="brand-logo-img" />
          <span className="brand-name">RouteX</span>
        </div>

        {/* Center Navigation Links */}
        <nav className="topnav-tabs">
          {navTabs.map((tab) => (
            <button
              key={tab.id}
              className={`nav-tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
              {activeTab === tab.id && <span className="active-tab-indicator" />}
            </button>
          ))}
        </nav>

        {/* Right Actions: Theme Toggle & Status Badge */}
        <div className="topnav-right">
          <button
            className="theme-toggle-btn"
            onClick={toggleTheme}
            title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Theme`}
            aria-label="Toggle light/dark theme"
          >
            {theme === 'light' ? (
              /* Moon Icon for switching to dark mode */
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            ) : (
              /* Sun Icon for switching to light mode */
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5"/>
                <line x1="12" y1="1" x2="12" y2="3"/>
                <line x1="12" y1="21" x2="12" y2="23"/>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                <line x1="1" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="12" x2="23" y2="12"/>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
              </svg>
            )}
          </button>

          <div className="status-badge" title="Groq AI Engine Connection Status">
            <span className={`status-dot ${status?.groq_connected ? 'online' : 'offline'}`} />
            <span className="status-label">
              {status?.groq_connected ? 'Groq Online' : 'AI Engine'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
