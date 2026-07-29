export default function SkewedMarquee() {
  const row1Items = [
    "PDF SUMMARIZATION",
    "•",
    "DEPARTMENT ROUTING",
    "•",
    "GROQ ACCELERATED",
    "•",
    "EMBEDDINGS STORE",
    "•",
    "AUTOMATED EMAIL DISPATCH",
    "•"
  ];

  const row2Items = [
    "ZERO LATENCY ANALYTICS",
    "•",
    "MULTI-MODEL ROUTER",
    "•",
    "RAG RETRIEVAL AGENTS",
    "•",
    "VECTOR EMBEDDING MATCH",
    "•",
    "BRUTALIST INFERENCE",
    "•"
  ];

  return (
    <section id="marquee" className="skewed-marquee-section">
      <div className="skewed-marquee-wrapper">
        {/* Row 1: Orange Archivo Black Text 10vw, Left Marquee */}
        <div className="marquee-row row-orange">
          <div className="marquee-track track-left">
            {[...row1Items, ...row1Items, ...row1Items].map((item, idx) => (
              <span key={`r1-${idx}`} className="marquee-text font-archivo">
                {item}
              </span>
            ))}
          </div>
        </div>

        {/* Row 2: White 80% opacity Archivo Black Text, Right Marquee (Reverse) */}
        <div className="marquee-row row-white">
          <div className="marquee-track track-right">
            {[...row2Items, ...row2Items, ...row2Items].map((item, idx) => (
              <span key={`r2-${idx}`} className="marquee-text font-archivo">
                {item}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
