"use client";

import { useState, useEffect, ReactElement } from "react";
import { CopyButton } from "./CopyButton";

interface CodeBlockProps {
  children: React.ReactNode;
  className?: string;
}

export function CodeBlock({ children, className }: CodeBlockProps) {
  const [highlightedCode, setHighlightedCode] = useState<string | null>(null);

  // Extract language from className (e.g., "language-python")
  const match = /language-(\w+)/.exec(className || "");
  const language = match ? match[1] : "text";

  // Get raw code from children
  const code =
    typeof children === "string"
      ? children
      : (children as ReactElement<{ children?: string }>)?.props?.children || "";

  useEffect(() => {
    import("shiki").then(async ({ codeToHtml }) => {
      const html = await codeToHtml(code.trim(), {
        lang: language,
        theme: "github-dark",
      });
      setHighlightedCode(html);
    });
  }, [code, language]);

  return (
    <div className="relative group my-4">
      <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        <CopyButton code={code.trim()} />
      </div>
      <div className="bg-[#1F1F1F] rounded-lg overflow-hidden">
        <div className="px-4 py-2 text-xs font-medium text-white/50 border-b border-white/10 font-mono">
          {language}
        </div>
        {highlightedCode ? (
          <div
            className="p-4 text-sm overflow-x-auto [&_pre]:!bg-transparent [&_code]:!bg-transparent"
            dangerouslySetInnerHTML={{ __html: highlightedCode }}
          />
        ) : (
          <pre className="p-4 text-sm overflow-x-auto text-white/85">
            <code>{code}</code>
          </pre>
        )}
      </div>
    </div>
  );
}
