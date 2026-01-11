import { Header } from "@/components/layout";
import { Sidebar, ScrollToTop } from "@/components/docs";
import { fullNavigation } from "@/lib/docs";

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <Header />
      <ScrollToTop />
      <div className="flex min-h-screen pt-16">
        <Sidebar navigation={fullNavigation} />
        <main className="flex-1 p-8 overflow-auto bg-[var(--color-bg)]">
          <div className="max-w-4xl">{children}</div>
        </main>
      </div>
    </>
  );
}
