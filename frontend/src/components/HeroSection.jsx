import RotatingScrollIndicator from './RotatingScrollIndicator';

export default function HeroSection({ onStartClick }) {
  return (
    <section id="hero" className="hero-section">
      <div className="hero-top-tag">
        <span className="hero-status-pill">
          <span className="pulse-dot"></span> GROQ INFRASTRUCTURE // LIVE
        </span>
        <span className="hero-tag-text">KINETIC BRUTALIST ENGINE v2.4</span>
      </div>

      {/* Main Massive 16vw Headline */}
      <div className="hero-headline-wrap">
        <h1 className="hero-headline">ROUTEX AI</h1>
      </div>

      {/* 2px Solid Black Border Divider */}
      <div className="hero-divider"></div>

      {/* Hero Metadata Row */}
      <div className="hero-metadata-row">
        {/* Left Label */}
        <div className="hero-meta-left">
          <span className="meta-label">BASED IN</span>
          <span className="meta-val">INFERENCE AGENTS &amp; RAG</span>
        </div>

        {/* Center Rotating Circular Scroll Indicator */}
        <div className="hero-meta-center">
          <a href="#marquee" className="scroll-link">
            <RotatingScrollIndicator text="SCROLL DOWN • ROUTEX AI • " />
          </a>
        </div>

        {/* Right Title / Role */}
        <div className="hero-meta-right">
          <span className="meta-label">ARCHITECTURE</span>
          <span className="meta-title">DOCUMENT PROCESSING &amp; AUTO-ROUTING ENGINE</span>
          <button className="hero-cta-btn" onClick={onStartClick}>
            LAUNCH SYSTEM &rarr;
          </button>
        </div>
      </div>
    </section>
  );
}
