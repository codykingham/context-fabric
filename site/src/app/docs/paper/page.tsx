export const metadata = {
  title: "Technical Paper | Context-Fabric",
  description:
    "Introduction to Context-Fabric: A graph-based corpus engine for annotated text.",
};

export default function PaperPage() {
  return (
    <div>
      <header className="mb-8">
        <h1 className="font-serif text-[2.5rem] font-semibold text-[var(--color-text)] mb-4">
          Technical Paper
        </h1>
        <p className="text-[var(--color-text-secondary)] leading-relaxed mb-6">
          A comprehensive overview of Context-Fabric&apos;s architecture,
          design principles, and performance characteristics.
        </p>
        <a
          href="/papers/intro-to-cf.pdf"
          download
          className="inline-flex items-center gap-2 px-4 py-2 bg-[var(--color-accent)] text-white rounded-lg hover:opacity-90 transition-opacity font-medium"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          Download PDF
        </a>
      </header>

      <div className="bg-[var(--color-bg-alt)] border border-[var(--color-border)] rounded-lg overflow-hidden shadow-sm">
        <iframe
          src="/papers/intro-to-cf.pdf"
          className="w-full h-[75vh]"
          title="Introduction to Context-Fabric"
        />
      </div>

      <p className="mt-6 text-sm text-[var(--color-text-secondary)]">
        For the best reading experience,{" "}
        <a
          href="/papers/intro-to-cf.pdf"
          className="text-[var(--color-accent)] hover:underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          open the PDF in a new tab
        </a>
        .
      </p>
    </div>
  );
}
