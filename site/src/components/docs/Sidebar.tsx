"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { NavItem, NavSection } from "@/types/docs";
import { clsx } from "clsx";
import { useState, useEffect } from "react";

function ChevronIcon({ isOpen, className }: { isOpen: boolean; className?: string }) {
  return (
    <svg
      className={clsx(
        "w-4 h-4 transition-transform duration-200",
        isOpen && "rotate-90",
        className
      )}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 5l7 7-7 7"
      />
    </svg>
  );
}

function NavItemComponent({
  item,
  currentPath,
  depth = 0,
}: {
  item: NavItem;
  currentPath: string;
  depth?: number;
}) {
  const isExternal = item.path.startsWith("http");
  const isActive = !isExternal && currentPath === item.path;
  const hasChildren = item.children && item.children.length > 0;
  const isParentOfActive =
    hasChildren &&
    item.children!.some((child) => currentPath.startsWith(child.path));
  const [isOpen, setIsOpen] = useState(isActive || isParentOfActive);

  const linkClasses = clsx(
    "block py-1.5 px-2 rounded text-[0.875rem] transition-colors flex-1",
    isActive
      ? "bg-[var(--color-accent)] text-white font-medium"
      : "text-[var(--color-text-secondary)] hover:text-[var(--color-text)] hover:bg-[var(--color-border)]"
  );

  const linkStyle = { paddingLeft: `${depth * 12 + 8}px` };

  return (
    <li>
      <div className="flex items-center">
        {hasChildren && (
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="mr-1 p-1 hover:bg-[var(--color-border)] rounded"
            aria-label={isOpen ? "Collapse" : "Expand"}
          >
            <ChevronIcon isOpen={isOpen ?? false} className="w-3 h-3" />
          </button>
        )}
        {isExternal ? (
          <a
            href={item.path}
            target="_blank"
            rel="noopener noreferrer"
            className={linkClasses}
            style={linkStyle}
          >
            {item.title}
            <span className="ml-1 text-[0.75rem] opacity-60">↗</span>
          </a>
        ) : (
          <Link
            href={item.path}
            className={linkClasses}
            style={linkStyle}
          >
            {item.title}
          </Link>
        )}
      </div>

      {hasChildren && isOpen && (
        <ul className="mt-1">
          {item.children!.map((child) => (
            <NavItemComponent
              key={child.path}
              item={child}
              currentPath={currentPath}
              depth={depth + 1}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function CollapsibleSection({
  section,
  currentPath,
  defaultOpen = true,
}: {
  section: NavSection;
  currentPath: string;
  defaultOpen?: boolean;
}) {
  // Check if any item in this section is active
  const hasActiveItem = section.items.some(
    (item) =>
      currentPath === item.path ||
      currentPath.startsWith(item.path + "/") ||
      (item.children?.some((child) => currentPath.startsWith(child.path)))
  );

  const [isOpen, setIsOpen] = useState(defaultOpen || hasActiveItem);

  // Keep section open if it contains the active page
  useEffect(() => {
    if (hasActiveItem) {
      setIsOpen(true);
    }
  }, [hasActiveItem]);

  return (
    <div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full group px-2 py-2 rounded hover:bg-[var(--color-border)] transition-colors"
      >
        <h2 className="text-[0.8125rem] font-bold uppercase tracking-wider text-[var(--color-text)] group-hover:text-[var(--color-accent)]">
          {section.title}
        </h2>
        <ChevronIcon
          isOpen={isOpen}
          className="text-[var(--color-text-secondary)] group-hover:text-[var(--color-accent)]"
        />
      </button>

      {isOpen && (
        <ul className="mt-1 space-y-1">
          {section.items.map((item) => (
            <NavItemComponent
              key={item.path}
              item={item}
              currentPath={currentPath}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

interface SidebarProps {
  navigation: NavSection[];
}

export function Sidebar({ navigation }: SidebarProps) {
  const pathname = usePathname();

  return (
    <nav className="w-64 flex-shrink-0 border-r border-[var(--color-border)] bg-[var(--color-bg-alt)] h-[calc(100vh-4rem)] sticky top-16 overflow-y-auto">
      <div className="p-4 space-y-4">
        {navigation.map((section, index) => (
          <CollapsibleSection
            key={section.title}
            section={section}
            currentPath={pathname}
            defaultOpen={index < 3}
          />
        ))}
      </div>
    </nav>
  );
}
