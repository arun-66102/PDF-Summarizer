export default function ServiceList({ onSelectService }) {
  const services = [
    {
      num: "01",
      title: "PDF SUMMARIZATION",
      tags: ["FAST PARSING", "PYPDF EXTRACTION", "CONTEXT AWARE"],
      desc: "Extract text, key insight bullets, structured takeaways, and executive summaries from complex document uploads in under 2 seconds."
    },
    {
      num: "02",
      title: "DEPARTMENT ROUTING",
      tags: ["SEMANTIC SIMILARITY", "VECTOR COSINE", "AUTO TARGETING"],
      desc: "Automatically determine optimal department endpoints (Engineering, Finance, HR, Legal, Ops) based on corpus embedding similarity."
    },
    {
      num: "03",
      title: "AUTOMATED EMAIL DISPATCH",
      tags: ["SMTP RELAY", "DOC ATTACHMENT", "AUDIT LOGS"],
      desc: "Directly compose and send formatted summary reports with original attached document links directly to assigned department heads."
    },
    {
      num: "04",
      title: "RAG & VECTOR EMBEDDINGS",
      tags: ["FAISS INDEX", "NOMIC / GEMINI", "K-NEAREST NEIGHBOR"],
      desc: "Persist document chunks into high-density vector space for sub-millisecond semantic search and context augmentation."
    }
  ];

  return (
    <section id="services" className="service-list-section">
      <div className="section-header-bar">
        <span className="section-badge">[ SYSTEM CAPABILITIES ]</span>
        <h2 className="section-title">BRUTALIST INFERENCE SUITE</h2>
      </div>

      <div className="service-list">
        {services.map((item) => (
          <div
            key={item.num}
            className="brutalist-service-card"
            onClick={() => onSelectService && onSelectService(item)}
          >
            {/* Index Number in #FF4D00 (Space Mono) */}
            <div className="card-num font-mono">{item.num}</div>

            {/* Title + Tags + Desc */}
            <div className="card-main">
              <h3 className="card-title font-archivo">{item.title}</h3>
              <p className="card-desc">{item.desc}</p>
              <div className="card-tags-row">
                {item.tags.map((tag, tIdx) => (
                  <span key={tIdx} className="pill-tag">
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            {/* Hidden Orange Arrow Icon revealed at 45deg on hover */}
            <div className="card-arrow-wrap">
              <svg
                width="36"
                height="36"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#FF4D00"
                strokeWidth="2.5"
                strokeLinecap="square"
                strokeLinejoin="miter"
                className="arrow-icon"
              >
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
