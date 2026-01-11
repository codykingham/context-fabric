import type { MDXComponents } from "mdx/types";
import { CodeBlock } from "@/components/docs/CodeBlock";
import { Callout } from "@/components/docs/Callout";
import { CorporaTable } from "@/components/docs/CorporaTable";

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h1: ({ children }) => (
      <h1 className="font-serif text-[2.5rem] font-semibold text-[var(--color-text)] mb-6">
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className="text-xl font-semibold text-[var(--color-text)] mb-4 pb-2 border-b border-[var(--color-border)] mt-10">
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-lg font-semibold text-[var(--color-text)] mb-3 mt-8">
        {children}
      </h3>
    ),
    p: ({ children }) => (
      <p className="text-[var(--color-text-secondary)] leading-relaxed mb-4">
        {children}
      </p>
    ),
    pre: ({ children }) => {
      // Get the code element's className for language detection
      const codeElement = children as React.ReactElement<{ className?: string }>;
      const className = codeElement?.props?.className || "";
      return <CodeBlock className={className}>{children}</CodeBlock>;
    },
    code: ({ children, className }) => {
      // Inline code (no className means inline)
      if (!className) {
        return (
          <code className="bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-1.5 py-0.5 rounded font-mono text-sm border border-gray-300 dark:border-gray-600">
            {children}
          </code>
        );
      }
      // Code inside pre blocks - just pass through
      return <code className={className}>{children}</code>;
    },
    a: ({ href, children }) => (
      <a
        href={href}
        className="text-[var(--color-accent)] hover:underline"
        target={href?.startsWith("http") ? "_blank" : undefined}
        rel={href?.startsWith("http") ? "noopener noreferrer" : undefined}
      >
        {children}
      </a>
    ),
    ul: ({ children }) => (
      <ul className="list-disc list-outside ml-6 space-y-2 mb-4 text-[var(--color-text-secondary)]">
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className="list-decimal list-outside ml-6 space-y-2 mb-4 text-[var(--color-text-secondary)]">
        {children}
      </ol>
    ),
    li: ({ children }) => <li className="leading-relaxed">{children}</li>,
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-[var(--color-accent)] pl-4 italic text-[var(--color-text-secondary)] my-4">
        {children}
      </blockquote>
    ),
    table: ({ children }) => (
      <div className="overflow-x-auto my-4">
        <table className="w-full border-collapse text-sm">{children}</table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className="bg-[var(--color-bg-alt)]">{children}</thead>
    ),
    tbody: ({ children }) => <tbody>{children}</tbody>,
    tr: ({ children }) => <tr>{children}</tr>,
    th: ({ children }) => (
      <th className="border border-[var(--color-border)] px-4 py-2 text-left font-semibold text-[var(--color-text)]">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="border border-[var(--color-border)] px-4 py-2 text-[var(--color-text-secondary)]">
        {children}
      </td>
    ),
    hr: () => <hr className="border-[var(--color-border)] my-8" />,
    strong: ({ children }) => (
      <strong className="font-semibold text-[var(--color-text)]">
        {children}
      </strong>
    ),
    // Custom components available in MDX
    Callout,
    CorporaTable,
    ...components,
  };
}
