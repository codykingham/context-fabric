"use client";

import { useState, useMemo } from "react";

interface Corpus {
  name: string;
  language: string;
  category: string;
  period: string;
  description: string;
  repository: string;
  size?: string;
}

const corpora: Corpus[] = [
  // Hebrew
  { name: "bhsa", language: "Hebrew", category: "Biblical", period: "1000-200 BCE", description: "Biblia Hebraica Stuttgartensia Amstelodamensis", repository: "https://github.com/ETCBC/bhsa", size: "1.1 GB" },
  { name: "dss", language: "Hebrew", category: "Religious", period: "300 BCE-100 CE", description: "Dead Sea Scrolls", repository: "https://github.com/ETCBC/dss", size: "936 MB" },
  { name: "sp", language: "Hebrew", category: "Biblical", period: "516 BCE-70 CE", description: "Samaritan Pentateuch", repository: "https://github.com/DT-UCPH/sp", size: "147 MB" },
  { name: "extrabiblical", language: "Hebrew", category: "Historical", period: "200 BCE-200 CE", description: "Extra-biblical Hebrew texts", repository: "https://github.com/ETCBC/extrabiblical" },

  // Greek - Biblical
  { name: "lxx", language: "Greek", category: "Biblical", period: "300-100 BCE", description: "Septuagint (Rahlfs edition)", repository: "https://github.com/CenterBLC/LXX", size: "268 MB" },
  { name: "n1904", language: "Greek", category: "Biblical", period: "100-400 CE", description: "Nestle 1904 Greek New Testament", repository: "https://github.com/CenterBLC/N1904", size: "319 MB" },
  { name: "SBLGNT", language: "Greek", category: "Biblical", period: "100-400 CE", description: "SBL Greek New Testament", repository: "https://github.com/CenterBLC/SBLGNT" },
  { name: "nestle1904", language: "Greek", category: "Biblical", period: "100-400 CE", description: "NT from LOWFAT-XML syntax trees", repository: "https://github.com/ETCBC/nestle1904" },
  { name: "Nestle1904GBI", language: "Greek", category: "Biblical", period: "100-400 CE", description: "Nestle 1904 (tonyjurg)", repository: "https://github.com/tonyjurg/Nestle1904GBI" },
  { name: "tischendorf_tf", language: "Greek", category: "Biblical", period: "100-400 CE", description: "Tischendorf 8th Edition Greek NT", repository: "https://github.com/codykingham/tischendorf_tf", size: "34 MB" },
  { name: "bible", language: "Greek", category: "Biblical", period: "300 BCE-400 CE", description: "Greek OT, NT, and extra-biblical", repository: "https://github.com/pthu/bible" },

  // Greek - Classical/Patristic
  { name: "patristics", language: "Greek", category: "Religious", period: "100-500 CE", description: "Church Fathers", repository: "https://github.com/pthu/patristics" },
  { name: "greek_literature", language: "Greek", category: "Literary", period: "400 BCE-400 CE", description: "Perseus & Open Greek texts", repository: "https://github.com/pthu/greek_literature" },
  { name: "athenaeus", language: "Greek", category: "Literary", period: "80-170 CE", description: "Athenaeus' Deipnosophistae", repository: "https://github.com/pthu/athenaeus" },

  // Syriac
  { name: "peshitta", language: "Syriac", category: "Biblical", period: "1000 BCE-900 CE", description: "Syriac Old Testament", repository: "https://github.com/ETCBC/peshitta", size: "55 MB" },
  { name: "syrnt", language: "Syriac", category: "Biblical", period: "0-1000 CE", description: "Syriac New Testament", repository: "https://github.com/ETCBC/syrnt", size: "52 MB" },
  { name: "syriac", language: "Syriac", category: "Religious", period: "Various", description: "Syriac texts collection", repository: "https://github.com/ETCBC/syriac" },

  // Arabic
  { name: "quran", language: "Arabic", category: "Religious", period: "600-900 CE", description: "Quranic Arabic Corpus", repository: "https://github.com/q-ran/quran", size: "73 MB" },
  { name: "fusus", language: "Arabic", category: "Religious", period: "Medieval", description: "Ibn Arabi's Fusus Al Hikam", repository: "https://github.com/among/fusus" },

  // Aramaic
  { name: "nena_tf", language: "Aramaic", category: "Historical", period: "Modern", description: "North Eastern Neo-Aramaic", repository: "https://github.com/CambridgeSemiticsLab/nena_tf" },

  // Akkadian / Cuneiform
  { name: "uruk", language: "Proto-Cuneiform", category: "Historical", period: "4000-3100 BCE", description: "Archaic tablets from Uruk", repository: "https://github.com/Nino-cunei/uruk" },
  { name: "oldassyrian", language: "Akkadian", category: "Historical", period: "2000-1600 BCE", description: "Old Assyrian documents", repository: "https://github.com/Nino-cunei/oldassyrian" },
  { name: "oldbabylonian", language: "Akkadian", category: "Historical", period: "1900-1600 BCE", description: "Old Babylonian letters", repository: "https://github.com/Nino-cunei/oldbabylonian" },
  { name: "ninmed", language: "Akkadian", category: "Historical", period: "ca. 800 BCE", description: "Medical Encyclopedia from Nineveh", repository: "https://github.com/Nino-cunei/ninmed" },

  // Ugaritic
  { name: "cuc", language: "Ugaritic", category: "Historical", period: "1223-1172 BCE", description: "Copenhagen Ugaritic Corpus", repository: "https://github.com/DT-UCPH/cuc", size: "1.6 MB" },

  // Pali
  { name: "dhammapada", language: "Pali", category: "Religious", period: "300 BCE", description: "Ancient Buddhist verses", repository: "https://github.com/ETCBC/dhammapada" },

  // Latin
  { name: "translatin-manif", language: "Latin", category: "Literary", period: "Early Modern", description: "Early modern Latin drama analysis", repository: "https://github.com/HuygensING/translatin-manif" },

  // Dutch
  { name: "wp6-missieven", language: "Dutch", category: "Historical", period: "1600-1800 CE", description: "VOC General Missives", repository: "https://github.com/CLARIAH/wp6-missieven" },
  { name: "wp6-daghregisters", language: "Dutch", category: "Historical", period: "1640-1641", description: "Batavia daily records", repository: "https://github.com/CLARIAH/wp6-daghregisters" },
  { name: "wp6-ferdinandhuyck", language: "Dutch", category: "Literary", period: "1884", description: "Dutch novel by Jacob van Lennep", repository: "https://github.com/CLARIAH/wp6-ferdinandhuyck" },
  { name: "mondriaan", language: "Dutch", category: "Historical", period: "1892-1923", description: "Piet Mondriaan letters", repository: "https://github.com/annotation/mondriaan" },

  // French/Latin/Dutch
  { name: "descartes-tf", language: "French/Latin/Dutch", category: "Historical", period: "1619-1650", description: "Descartes correspondence", repository: "https://github.com/CLARIAH/descartes-tf" },

  // Italian
  { name: "suriano", language: "Italian", category: "Historical", period: "1616-1623", description: "Diplomatic correspondence", repository: "https://github.com/HuygensING/suriano" },

  // English
  { name: "mobydick", language: "English", category: "Literary", period: "1851", description: "Herman Melville novel with NLP annotations", repository: "https://github.com/annotation/mobydick" },
  { name: "banks", language: "English", category: "Literary", period: "1987", description: "Iain M. Banks' Consider Phlebas", repository: "https://github.com/annotation/banks" },

  ];

type SortKey = "name" | "language" | "category" | "period";
type SortDirection = "asc" | "desc";

export function CorporaTable() {
  const [search, setSearch] = useState("");
  const [languageFilter, setLanguageFilter] = useState<string>("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("language");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");

  // Extract unique languages and categories
  const languages = useMemo(() => {
    const langs = [...new Set(corpora.map((c) => c.language))].sort();
    return ["all", ...langs];
  }, []);

  const categories = useMemo(() => {
    const cats = [...new Set(corpora.map((c) => c.category))].sort();
    return ["all", ...cats];
  }, []);

  // Filter and sort corpora
  const filteredCorpora = useMemo(() => {
    let result = corpora.filter((corpus) => {
      const matchesSearch =
        search === "" ||
        corpus.name.toLowerCase().includes(search.toLowerCase()) ||
        corpus.description.toLowerCase().includes(search.toLowerCase()) ||
        corpus.language.toLowerCase().includes(search.toLowerCase());

      const matchesLanguage =
        languageFilter === "all" || corpus.language === languageFilter;

      const matchesCategory =
        categoryFilter === "all" || corpus.category === categoryFilter;

      return matchesSearch && matchesLanguage && matchesCategory;
    });

    result.sort((a, b) => {
      const aVal = a[sortKey].toLowerCase();
      const bVal = b[sortKey].toLowerCase();
      const cmp = aVal.localeCompare(bVal);
      return sortDirection === "asc" ? cmp : -cmp;
    });

    return result;
  }, [search, languageFilter, categoryFilter, sortKey, sortDirection]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDirection("asc");
    }
  };

  const SortIcon = ({ column }: { column: SortKey }) => {
    if (sortKey !== column) {
      return <span className="ml-1 text-gray-400">&#8597;</span>;
    }
    return (
      <span className="ml-1">
        {sortDirection === "asc" ? "\u2191" : "\u2193"}
      </span>
    );
  };

  return (
    <div className="my-8">
      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search corpora..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <div className="flex gap-4">
          <select
            value={languageFilter}
            onChange={(e) => setLanguageFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {languages.map((lang) => (
              <option key={lang} value={lang}>
                {lang === "all" ? "All Languages" : lang}
              </option>
            ))}
          </select>
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat === "all" ? "All Categories" : cat}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Results count */}
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Showing {filteredCorpora.length} of {corpora.length} corpora
      </p>

      {/* Table */}
      <div className="overflow-x-auto border border-gray-200 dark:border-gray-700 rounded-lg">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th
                onClick={() => handleSort("name")}
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Corpus <SortIcon column="name" />
              </th>
              <th
                onClick={() => handleSort("language")}
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Language <SortIcon column="language" />
              </th>
              <th
                onClick={() => handleSort("category")}
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Category <SortIcon column="category" />
              </th>
              <th
                onClick={() => handleSort("period")}
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Period <SortIcon column="period" />
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Description
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
            {filteredCorpora.map((corpus) => (
              <tr
                key={corpus.name}
                className="hover:bg-gray-50 dark:hover:bg-gray-800"
              >
                <td className="px-4 py-3 whitespace-nowrap">
                  <a
                    href={corpus.repository}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline font-mono text-sm font-semibold"
                    style={{ color: 'var(--color-accent)' }}
                  >
                    {corpus.name}
                  </a>
                  {corpus.size && (
                    <span className="ml-2 text-xs text-gray-500 dark:text-gray-500">
                      {corpus.size}
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                  {corpus.language}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                  {corpus.category}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                  {corpus.period}
                </td>
                <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                  {corpus.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {filteredCorpora.length === 0 && (
        <p className="text-center py-8 text-gray-500 dark:text-gray-400">
          No corpora match your search criteria.
        </p>
      )}
    </div>
  );
}
