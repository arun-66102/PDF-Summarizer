export default function RotatingScrollIndicator({ text = "SCROLL DOWN • ROUTEX AI • " }) {
  // SVG circular text path setup
  const repeatedText = `${text}${text}${text}`;

  return (
    <div className="scroll-indicator-wrap" aria-label="Scroll down indicator">
      <div className="scroll-indicator-svg-box">
        <svg viewBox="0 0 144 144" className="rotating-text-svg">
          <path
            id="scrollCirclePath"
            d="M 72, 72 m -52, 0 a 52,52 0 1,1 104,0 a 52,52 0 1,1 -104,0"
            fill="none"
          />
          <text fill="#000000" className="scroll-path-text">
            <textPath href="#scrollCirclePath" startOffset="0%">
              {repeatedText}
            </textPath>
          </text>
        </svg>
      </div>

      {/* Center Static Arrow */}
      <div className="scroll-indicator-center-icon">
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#000000"
          strokeWidth="2.5"
          strokeLinecap="square"
          strokeLinejoin="miter"
        >
          <path d="M12 5v14M19 12l-7 7-7-7" />
        </svg>
      </div>
    </div>
  );
}
