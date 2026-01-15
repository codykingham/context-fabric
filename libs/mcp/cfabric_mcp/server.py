"""FastMCP server for Context-Fabric.

This module defines the MCP server that exposes corpus operations
to AI agents via the Model Context Protocol.

Transports:
- stdio (default): For Claude Desktop and local MCP clients
- sse (--sse PORT): Server-Sent Events for remote MCP clients (e.g., Cursor)
- http (--http PORT): Streamable HTTP for production deployments

Tool Set (11 tools):
Discovery:
- list_corpora: List available corpora
- describe_corpus: Corpus structure (node types, sections)
- list_features: Browse features with optional node_type filter
- describe_feature: Feature details with sample values
- get_text_formats: Text encoding samples (cached)
Search:
- search: Pattern search with return_type (results/count/statistics/passages)
- search_continue: Paginated search continuation
- search_csv: Export search results to CSV file (stdio only)
- search_syntax_guide: Search syntax docs (section-based)
Data Access:
- get_passages: Batch section lookup
- get_node_features: Batch feature lookup

Note: Usage guide is provided via MCP instructions field at connection time.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from cfabric_mcp import tools
from cfabric_mcp import resources
from cfabric_mcp.corpus_manager import corpus_manager

logger = logging.getLogger("cfabric_mcp.server")

# Instructions provided to clients at connection time via MCP protocol
INSTRUCTIONS = """Context-Fabric exposes corpora for structured analysis. \
Corpora contain hierarchical objects (e.g., words, sentences, documents) \
with arbitrary annotations. A corpus can be annotated for almost anything: \
linguistics, music, manuscripts, DNA sequences, legal documents, etc.

## Capabilities
- Search for patterns using structural templates
- Filter by any annotated feature on any object type
- Analyze feature distributions across search results
- Retrieve passages by section reference
- Explore hierarchical relationships between objects

## Workflows

### Explore corpus structure
Understand what's in the corpus before searching:
1. describe_corpus() - Get node types and section structure
2. list_features(node_types=[...]) - See features for a node type
3. describe_feature('feature_name') - Get sample values for a feature

### Understand text encoding
Learn how text is encoded before lexical searches (critical for lex=... or surface text queries):
1. get_text_formats() - See original script and transliteration samples
2. Use samples to construct accurate search patterns

### Search for patterns
Find patterns matching structural templates:
1. search_syntax_guide() - Learn search template syntax
2. search(template, return_type='count') - Check result count
3. search(template) - Get actual results (validation errors returned automatically)

### Analyze distributions
Understand feature patterns across results:
- search(template, return_type='statistics', aggregate_features=[...])

### Read passages
Retrieve text by section reference:
- get_passages([[section_ref], ...])

## Tips
- Start with describe_corpus() to understand the structure
- Use list_features(node_types=[...]) to find relevant features
- Call get_text_formats() before lexical/surface text searches
- Use search(..., return_type='count') before fetching full results
- Invalid templates return detailed error messages automatically

## Next Step
Call describe_corpus() to see node types and section structure.
"""

# Create the FastMCP server
mcp = FastMCP(
    name="Context-Fabric",
    instructions=INSTRUCTIONS,
)


# ============================================================================
# Corpus Tools
# ============================================================================


@mcp.tool()
def list_corpora() -> dict[str, Any]:
    """List all currently loaded corpora.

    Returns:
        Dictionary with 'corpora' list and 'current' corpus name.
    """
    return tools.list_loaded_corpora()


@mcp.tool()
def describe_corpus(corpus: str | None = None) -> dict[str, Any]:
    """Get corpus structure overview.

    Returns node types and section structure. Use list_features() to browse
    features, describe_feature() for details, get_text_formats() for encoding.

    Args:
        corpus: Corpus name (defaults to current corpus)

    Returns:
        Corpus overview including:
        - node_types: All node types with counts
        - sections: Section hierarchy levels
    """
    return tools.describe_corpus(corpus)


@mcp.tool()
def describe_feature(
    feature: str | list[str],
    sample_limit: int = 20,
    corpus: str | None = None,
) -> dict[str, Any]:
    """Get detailed info about one or more features.

    Returns metadata, node_types, and sample values (by frequency).
    Use list_features() first to browse available features.

    Args:
        feature: Feature name or list of names (e.g., "sp" or ["sp", "vt"])
        sample_limit: Max sample values per feature (default 20)
        corpus: Corpus name (defaults to current corpus)

    Returns:
        Feature details including:
        - name, kind, value_type, description
        - node_types: Which object types have this feature
        - unique_values: Count of unique values
        - sample_values: Top values by frequency with counts
    """
    return tools.describe_feature(feature, sample_limit, corpus)


@mcp.tool()
def list_features(
    kind: str = "all",
    node_types: list[str] | None = None,
    corpus: str | None = None,
) -> dict[str, Any]:
    """List features with optional filtering.

    Returns lightweight catalog for discovery. Use node_types to filter by
    object type. For full details with samples, use describe_feature().

    Args:
        kind: Filter by "all", "node", or "edge"
        node_types: Filter to features for these types (e.g., ["word"])
        corpus: Corpus name (defaults to current corpus)

    Returns:
        List of features with name, kind, value_type, description.
    """
    return tools.list_features(kind, node_types, corpus)


@mcp.tool()
def get_text_formats(corpus: str | None = None) -> dict[str, Any]:
    """Get text encoding samples showing original script and transliteration.

    Returns format pairs with diverse samples. Use when constructing search
    queries that need specific text encodings. Results are cached per corpus.

    Args:
        corpus: Corpus name (defaults to current corpus)

    Returns:
        Text format information with sample pairs.
    """
    return tools.get_text_formats(corpus)


# ============================================================================
# Search Tools
# ============================================================================


@mcp.tool()
def search(
    template: str,
    return_type: str = "results",
    aggregate_features: list[str] | None = None,
    group_by_section: bool = False,
    top_n: int = 50,
    limit: int = 100,
    max_override: bool = False,
    corpus: str | None = None,
) -> dict[str, Any]:
    """Search for patterns in the corpus.

    Args:
        template: Search template (use search_syntax_guide() for syntax help)
        return_type: What to return:
            - "results": Paginated node info with cursor (default)
            - "count": Just total count
            - "statistics": Feature distributions per matched node position
            - "passages": Formatted text grouped by section
        aggregate_features: For statistics: which features to aggregate
        group_by_section: For statistics: include distribution by book
        top_n: For statistics: max values per feature (default 50)
        limit: For results/passages: page size (default 100)
        max_override: Bypass limit cap. May produce large responses - use judiciously.
        corpus: Corpus name (defaults to current corpus)

    Returns:
        Search results formatted according to return_type.
        For "results", includes cursor for pagination via search_continue().
    """
    return tools.search(
        template,
        return_type=return_type,
        aggregate_features=aggregate_features,
        group_by_section=group_by_section,
        top_n=top_n,
        limit=limit,
        max_override=max_override,
        corpus=corpus,
    )


@mcp.tool()
def search_continue(
    cursor_id: str,
    offset: int = 0,
    limit: int = 100,
    max_override: bool = False,
) -> dict[str, Any]:
    """Continue a paginated search using a cursor ID.

    Use this to get additional pages of results from a previous search.
    Cursors expire after 5 minutes.

    Args:
        cursor_id: The cursor ID from a previous search
        offset: Number of results to skip
        limit: Maximum number of results to return (default 100)
        max_override: Bypass limit cap. May produce large responses - use judiciously.

    Returns:
        Next page of search results with updated cursor.
    """
    return tools.search_continue(cursor_id, offset, limit, max_override)


@mcp.tool()
def search_csv(
    template: str,
    file_path: str,
    limit: int = 10000,
    delimiter: str = ",",
    corpus: str | None = None,
) -> dict[str, Any]:
    """Export search results to a CSV file.

    Use this tool instead of search() when exporting large result sets that
    would be unwieldy to return inline. Results are written directly to a file
    rather than returned in the response.

    Note: This tool writes to the local filesystem and requires stdio transport.
    Not available over HTTP/SSE - use search() with a limit parameter instead.

    Writes delimited values with header row. Multi-node search results
    are flattened with positional prefixes (node0_*, node1_*, etc.).

    Args:
        template: Search template (use search_syntax_guide() for syntax help)
        file_path: Absolute path to write the CSV file
        limit: Maximum rows to export (default 10000)
        delimiter: Field separator (default ",", use "\\t" for TSV)
        corpus: Corpus name (defaults to current corpus)

    Returns:
        File path, total matches found, and rows written.
    """
    return tools.search_csv(template, file_path, limit, delimiter, corpus)


@mcp.tool()
def search_syntax_guide(section: str | None = None) -> dict[str, Any]:
    """Get documentation on search template syntax.

    Call without args for summary + available sections. Call with section
    name to get detailed content for that topic.

    Args:
        section: Optional section name (basics, structure, relations,
                 quantifiers, examples). None returns overview.

    Returns:
        Without section: summary + list of available sections.
        With section: detailed content for that section.
    """
    return tools.search_syntax_guide(section)


# ============================================================================
# Data Access Tools
# ============================================================================


@mcp.tool()
def get_passages(
    sections: list[list[str | int]],
    limit: int = 50,
    lang: str = "en",
    max_override: bool = False,
    corpus: str | None = None,
) -> dict[str, Any]:
    """Get passages by section references.

    Batch lookup of multiple sections at once.

    Args:
        sections: List of section references, e.g., [['Genesis', 1, 1], ['Exodus', 2, 3]]
        limit: Maximum sections to return (default 50)
        lang: ISO 639 language code for section names (e.g., 'en' for English book names)
        max_override: Bypass limit cap. May produce large responses - use judiciously.
        corpus: Corpus name (defaults to current corpus)

    Returns:
        List of passages with text and node information.
    """
    return tools.get_passages(sections, limit, lang, max_override, corpus)


@mcp.tool()
def get_node_features(
    nodes: list[int],
    features: list[str],
    corpus: str | None = None,
) -> dict[str, Any]:
    """Get feature values for a list of nodes.

    Batch lookup of feature values for multiple nodes.

    Args:
        nodes: List of node IDs
        features: List of feature names to retrieve
        corpus: Corpus name (defaults to current corpus)

    Returns:
        Feature values for each node.
    """
    return tools.get_node_features(nodes, features, corpus)


# ============================================================================
# Resources
# ============================================================================


@mcp.resource("corpus://{corpus_name}")
def corpus_resource(corpus_name: str) -> str:
    """Get corpus information.

    Returns detailed information about a loaded corpus including
    node types, features, and statistics.
    """
    return resources.get_corpus_resource(corpus_name)


@mcp.resource("corpus://{corpus_name}/node/{node_id}")
def node_resource(corpus_name: str, node_id: int) -> str:
    """Get node information.

    Returns detailed information about a specific node including
    type, text, section reference, and slot positions.
    """
    return resources.get_node_resource(corpus_name, node_id)


@mcp.resource("corpus://{corpus_name}/features")
def features_resource(corpus_name: str) -> str:
    """Get list of available features.

    Returns lists of node features and edge features available in the corpus.
    """
    return resources.get_feature_list_resource(corpus_name)


@mcp.resource("corpus://{corpus_name}/types")
def types_resource(corpus_name: str) -> str:
    """Get node type information.

    Returns information about all node types including counts and node ranges.
    """
    return resources.get_node_types_resource(corpus_name)


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    """Run the MCP server with configurable transport."""
    parser = argparse.ArgumentParser(
        description="Context-Fabric MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # stdio transport (default) - for Claude Desktop
  cfabric-mcp --corpus /path/to/corpus

  # SSE transport - for Cursor and remote MCP clients
  cfabric-mcp --corpus /path/to/corpus --sse 8000

  # Streamable HTTP transport - recommended for production
  cfabric-mcp --corpus /path/to/corpus --http 8000

  # Multiple corpora with names
  cfabric-mcp --corpus corpus1=/path/to/first --corpus corpus2=/path/to/second

  # Load only specific features
  cfabric-mcp --corpus /path/to/corpus --features "feature1 feature2 feature3"
""",
    )
    parser.add_argument(
        "--corpus",
        "-c",
        action="append",
        required=True,
        metavar="[NAME=]PATH",
        help="Corpus to load. Format: /path or name=/path. Can be specified multiple times.",
    )
    parser.add_argument(
        "--features",
        "-f",
        help="Space-separated features to load (default: all)",
    )
    parser.add_argument(
        "--sse",
        type=int,
        metavar="PORT",
        help="Run with SSE transport on specified port (for Cursor, remote clients)",
    )
    parser.add_argument(
        "--http",
        type=int,
        metavar="PORT",
        help="Run with Streamable HTTP transport on specified port (production)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to for SSE/HTTP transports (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    args = parser.parse_args()

    # Configure log level
    if args.verbose:
        logging.getLogger("cfabric_mcp").setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Determine transport mode
    if args.sse:
        transport = "sse"
        port = args.sse
    elif args.http:
        transport = "http"
        port = args.http
    else:
        transport = "stdio"
        port = None

    logger.info("Starting Context-Fabric MCP Server (transport: %s)", transport)

    # Set transport for tools that need it (e.g., search_csv)
    tools.set_transport(transport)

    # Pre-load corpora before starting server
    logger.info("Loading %d corpus/corpora...", len(args.corpus))
    for corpus_spec in args.corpus:
        if "=" in corpus_spec:
            name, path = corpus_spec.split("=", 1)
        else:
            name = None
            path = corpus_spec
        corpus_manager.load(path, name=name, features=args.features)

    loaded = corpus_manager.list_corpora()
    logger.info("Server ready with corpora: %s", ", ".join(loaded))

    # Run MCP server with selected transport
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        logger.info("SSE endpoint: http://%s:%d/sse", args.host, port)
        mcp.settings.host = args.host
        mcp.settings.port = port
        mcp.run(transport="sse")
    elif transport == "http":
        logger.info("HTTP endpoint: http://%s:%d/mcp", args.host, port)
        mcp.settings.host = args.host
        mcp.settings.port = port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
