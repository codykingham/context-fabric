type CalloutType = "info" | "warning" | "tip" | "note";

interface CalloutProps {
  type?: CalloutType;
  title?: string;
  children: React.ReactNode;
}

const config: Record<
  CalloutType,
  {
    bg: string;
    border: string;
    text: string;
    icon: string;
    iconChar: string;
    codeBg: string;
    codeBorder: string;
  }
> = {
  info: {
    bg: "#EFF6FF",        // blue-50
    border: "#3B82F6",    // blue-500
    text: "#1E3A5F",      // dark blue
    icon: "#2563EB",      // blue-600
    iconChar: "i",
    codeBg: "#DBEAFE",    // blue-100
    codeBorder: "#93C5FD", // blue-300
  },
  warning: {
    bg: "#FEF3C7",        // amber-100
    border: "#F59E0B",    // amber-500
    text: "#78350F",      // amber-900
    icon: "#D97706",      // amber-600
    iconChar: "!",
    codeBg: "#FDE68A",    // amber-200
    codeBorder: "#FCD34D", // amber-300
  },
  tip: {
    bg: "#DCFCE7",        // green-100
    border: "#22C55E",    // green-500
    text: "#14532D",      // green-900
    icon: "#16A34A",      // green-600
    iconChar: "*",
    codeBg: "#BBF7D0",    // green-200
    codeBorder: "#86EFAC", // green-300
  },
  note: {
    bg: "#F5F5F4",        // stone-100
    border: "#78716C",    // stone-500
    text: "#1C1917",      // stone-900
    icon: "#57534E",      // stone-600
    iconChar: "#",
    codeBg: "#E7E5E4",    // stone-200
    codeBorder: "#A8A29E", // stone-400
  },
};

export function Callout({ type = "info", title, children }: CalloutProps) {
  const c = config[type];
  const id = `callout-${Math.random().toString(36).slice(2, 9)}`;

  return (
    <>
      {/* Override all text and code styling within this callout */}
      <style>{`
        #${id},
        #${id} p,
        #${id} li,
        #${id} span,
        #${id} strong,
        #${id} em,
        #${id} a {
          color: ${c.text} !important;
        }
        #${id} code {
          background-color: ${c.codeBg} !important;
          border: 1px solid ${c.codeBorder} !important;
          color: ${c.text} !important;
          padding: 0.125rem 0.375rem !important;
          border-radius: 0.25rem !important;
          font-size: 0.875rem !important;
        }
        #${id} a {
          text-decoration: underline !important;
          font-weight: 500 !important;
        }
      `}</style>

      <div
        id={id}
        className="rounded-lg border-l-4 p-4 my-4"
        style={{
          backgroundColor: c.bg,
          borderLeftColor: c.border,
          borderTop: `1px solid ${c.border}40`,
          borderRight: `1px solid ${c.border}40`,
          borderBottom: `1px solid ${c.border}40`,
          color: c.text,
        }}
      >
        <div className="flex gap-3">
          <span
            className="text-lg flex-shrink-0 font-mono font-bold select-none"
            style={{ color: c.icon }}
          >
            {c.iconChar}
          </span>
          <div className="text-sm [&>p]:mb-0 [&>p:last-child]:mb-0 [&>ul]:mb-0 [&>ol]:mb-0">
            {title && (
              <p className="font-semibold mb-1 text-base" style={{ color: c.text }}>
                {title}
              </p>
            )}
            <div style={{ color: c.text }}>{children}</div>
          </div>
        </div>
      </div>
    </>
  );
}
