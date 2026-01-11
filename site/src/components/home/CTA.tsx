"use client";

import { useState } from "react";

export function CTA() {
  const [copied, setCopied] = useState(false);
  const command = "pip install context-fabric";

  const handleCopy = async () => {
    await navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section className="py-24 px-6 md:px-10 text-center bg-[var(--color-bg-alt)] border-t border-[var(--color-border)]">
      <h2 className="text-[2.25rem] mb-4 tracking-tight">Ready to explore?</h2>
      <p className="text-[1.0625rem] text-[var(--color-text-secondary)] mb-8 max-w-[480px] mx-auto">
        Install Context-Fabric and start querying linguistic corpora in minutes.
      </p>
      <button
        onClick={handleCopy}
        className="inline-flex items-center gap-3 bg-[var(--color-text)] text-[var(--color-bg)] px-6 py-4 rounded-lg font-mono text-[0.9375rem] hover:opacity-90 transition-opacity cursor-pointer group"
      >
        <span className="text-[var(--color-accent)]">$</span>
        {command}
        <span className="ml-2 opacity-60 group-hover:opacity-100 transition-opacity">
          {copied ? (
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          ) : (
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          )}
        </span>
      </button>
    </section>
  );
}
