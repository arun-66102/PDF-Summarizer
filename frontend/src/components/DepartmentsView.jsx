import { useState, useEffect } from 'react';
import { getStats } from '../api/client';

export default function DepartmentsView({ onTestRoute }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [deptEmails, setDeptEmails] = useState({
    CSE: "arun877865@gmail.com",
    EEE: "arunkumar7904334@gmail.com",
    MECH: "1989indhusri@gmail.com",
    CIVIL: "adhithiee2907@gmail.com"
  });

  useEffect(() => {
    getStats().then((data) => {
      if (data?.department_emails) {
        setDeptEmails(data.department_emails);
      }
    }).catch(() => {});
  }, []);

  const departments = [
    {
      code: "CSE",
      name: "Computer Science & Engineering",
      email: deptEmails.CSE || "arun877865@gmail.com",
      color: "#C86D51",
      description: "Software development, programming, artificial intelligence, machine learning, databases, cloud computing, cybersecurity, data science, and mobile applications.",
      keywords: ["python", "machine learning", "neural networks", "databases", "cybersecurity", "web dev", "docker", "algorithms"]
    },
    {
      code: "EEE",
      name: "Electrical & Electronics Engineering",
      email: deptEmails.EEE || "arunkumar7904334@gmail.com",
      color: "#6B8E7B",
      description: "Power systems, control systems, electrical machines, power electronics, embedded systems, signal processing, renewable energy, and microcontrollers.",
      keywords: ["power grid", "microcontrollers", "pcb design", "circuits", "embedded systems", "matlab", "transformers", "renewable energy"]
    },
    {
      code: "MECH",
      name: "Mechanical Engineering",
      email: deptEmails.MECH || "1989indhusri@gmail.com",
      color: "#D4A373",
      description: "Thermodynamics, fluid mechanics, heat transfer, CAD/CAM, robotics, mechatronics, manufacturing processes, and HVAC systems.",
      keywords: ["thermodynamics", "cad/cam", "robotics", "solidworks", "hvac", "manufacturing", "heat transfer", "turbines"]
    },
    {
      code: "CIVIL",
      name: "Civil Engineering",
      email: deptEmails.CIVIL || "adhithiee2907@gmail.com",
      color: "#88A0A8",
      description: "Structural engineering, construction planning, surveying, geotechnical engineering, concrete technology, urban planning, and environmental systems.",
      keywords: ["structural engineering", "concrete", "surveying", "urban planning", "bridges", "autocad civil", "soil mechanics", "infrastructure"]
    }
  ];

  const filteredDepts = departments.filter(d =>
    d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    d.code.toLowerCase().includes(searchQuery.toLowerCase()) ||
    d.keywords.some(k => k.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="tab-view-container">
      <div className="view-header">
        <div>
          <h1 className="view-title">Department Corpus Directory</h1>
          <p className="view-subtitle">
            Real-time semantic vector embeddings &amp; configured SMTP email dispatch addresses.
          </p>
        </div>
        <div className="search-input-wrap">
          <svg className="search-icon-svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#736354" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <input
            type="text"
            placeholder="Search departments or keywords..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="organic-search-input"
          />
        </div>
      </div>

      <div className="dept-cards-grid">
        {filteredDepts.map((dept) => (
          <div key={dept.code} className="dept-card">
            <div className="dept-card-top">
              <div className="dept-code-tag" style={{ backgroundColor: dept.color }}>
                {dept.code}
              </div>
              <span className="dept-status-dot-active" />
            </div>

            <h3 className="dept-card-name">{dept.name}</h3>
            <p className="dept-card-desc">{dept.description}</p>

            <div className="dept-email-bar">
              <span className="email-label">Configured Email:</span>
              <span className="email-val">{dept.email}</span>
            </div>

            <div className="dept-keywords-wrap">
              <span className="keywords-title">Corpus Key Topics:</span>
              <div className="keyword-chips">
                {dept.keywords.map((kw, i) => (
                  <span key={i} className="kw-chip">
                    {kw}
                  </span>
                ))}
              </div>
            </div>

            <button
              className="dept-test-btn"
              onClick={() => onTestRoute && onTestRoute(dept.code)}
            >
              Route Test Document &rarr;
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
