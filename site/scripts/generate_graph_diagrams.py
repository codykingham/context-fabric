#!/usr/bin/env python3
"""
Generate graph model diagrams for Context-Fabric documentation.

Requires: graphviz (pip install graphviz)
Also requires Graphviz binaries: brew install graphviz (macOS) or apt install graphviz (Linux)

Usage:
    python generate_graph_diagrams.py

Outputs SVG files to: site/public/docs/images/
"""

from pathlib import Path

try:
    from graphviz import Digraph
except ImportError:
    print("Error: graphviz package not installed.")
    print("Install with: pip install graphviz")
    print("Also install Graphviz binaries: brew install graphviz (macOS)")
    exit(1)


OUTPUT_DIR = Path(__file__).parent.parent / "public" / "docs" / "images"

# Typography - use system fonts that match brand style
FONTS = {
    "sans": "Helvetica Neue",      # Clean sans-serif for labels
    "mono": "Menlo",               # Monospace for code-like elements
}

# Context-Fabric brand colors
COLORS = {
    # Primary accent (golden)
    "accent": "#C8A46A",
    "accent_light": "#E8D5B5",
    "accent_muted": "#F5EDE0",

    # Warm neutrals (for node hierarchy - lighter = higher level)
    "level_1": "#FAF9F7",  # Document level - lightest
    "level_2": "#F5F3F0",  # Chapter/section
    "level_3": "#EFECE8",  # Sentence
    "level_4": "#E8E5E0",  # Clause
    "level_5": "#E0DDD8",  # Phrase
    "level_6": "#D8D4CF",  # Word/slot - darkest

    # Highlighted/current node
    "highlight": "#C8A46A",
    "highlight_bg": "#FBF7F0",

    # Text and edges
    "text": "#1F1F1F",
    "text_secondary": "#5C5C5C",
    "edge": "#9C9890",
    "edge_light": "#C4C0B8",

    # For features - subtle variations
    "feature_1": "#F8F6F3",
    "feature_2": "#F5F2EE",
    "feature_3": "#F2EEE9",
    "feature_4": "#EFEBE5",
    "feature_5": "#ECE7E0",
}


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_slot_sequence():
    """Generate diagram showing slot nodes in sequence with text."""
    g = Digraph("slot_sequence", format="svg")
    g.attr(rankdir="LR", bgcolor="transparent")
    g.attr("node", shape="box", style="filled,rounded", fillcolor=COLORS["level_6"],
           fontname=FONTS["mono"], fontsize="11",
           color=COLORS["edge"], fontcolor=COLORS["text"])
    g.attr("edge", color=COLORS["edge"], arrowsize="0.7")

    # Slot nodes with words
    words = [
        ("1", "In"),
        ("2", "the"),
        ("3", "beginning"),
        ("4", "God"),
        ("5", "created"),
        ("6", "the"),
        ("7", "heavens"),
    ]

    # Create nodes
    for slot_id, word in words:
        label = f"Slot {slot_id}\\n\"{word}\""
        g.node(f"s{slot_id}", label)

    # Create sequence edges
    for i in range(len(words) - 1):
        g.edge(f"s{words[i][0]}", f"s{words[i+1][0]}")

    # Add ellipsis node
    g.node("more", "...", shape="plaintext", fontsize="14", fontcolor=COLORS["text_secondary"])
    g.edge("s7", "more", style="dashed", color=COLORS["edge_light"])

    output_path = OUTPUT_DIR / "slot-sequence"
    g.render(output_path, cleanup=True)
    print(f"Generated: {output_path}.svg")


def generate_containment_hierarchy():
    """Generate diagram showing containment hierarchy."""
    g = Digraph("containment", format="svg")
    g.attr(rankdir="TB", bgcolor="transparent", nodesep="0.4", ranksep="0.5")
    g.attr("node", shape="box", style="filled,rounded",
           fontname=FONTS["mono"], fontsize="11",
           color=COLORS["edge"], fontcolor=COLORS["text"])
    g.attr("edge", color=COLORS["edge"], arrowsize="0.7")

    # Node numbers show different ranges per type (realistic for a ~100K word corpus)
    # Each type has its own contiguous block of IDs
    g.node("doc", "Document\\n(node 1052001)", fillcolor=COLORS["level_1"])

    # Chapter level
    g.node("ch1", "Chapter\\n(node 1050003)", fillcolor=COLORS["level_2"])

    # Sentence level
    g.node("sent1", "Sentence\\n(node 920150)", fillcolor=COLORS["level_3"])

    # Clause level
    g.node("cl1", "Clause\\n(node 780042)", fillcolor=COLORS["level_4"])

    # Phrase level
    g.node("ph1", "Phrase\\n(node 503201)", fillcolor=COLORS["level_5"])
    g.node("ph2", "Phrase\\n(node 503202)", fillcolor=COLORS["level_5"])

    # Word/slot level
    with g.subgraph() as s:
        s.attr(rank="same")
        for i, word in enumerate(["In", "the", "beginning", "God", "created"], 1):
            s.node(f"w{i}", f"Slot {i}\\n\"{word}\"", fillcolor=COLORS["level_6"])

    # Containment edges
    g.edge("doc", "ch1", label="contains", fontsize="9", fontcolor=COLORS["text_secondary"])
    g.edge("ch1", "sent1")
    g.edge("sent1", "cl1")
    g.edge("cl1", "ph1")
    g.edge("cl1", "ph2")

    # Phrase to word edges
    g.edge("ph1", "w1")
    g.edge("ph1", "w2")
    g.edge("ph1", "w3")
    g.edge("ph2", "w4")
    g.edge("ph2", "w5")

    output_path = OUTPUT_DIR / "containment-hierarchy"
    g.render(output_path, cleanup=True)
    print(f"Generated: {output_path}.svg")


def generate_spanning_nodes():
    """Generate diagram showing non-slot nodes spanning slots.

    Uses HTML-like record labels for precise alignment.
    """
    g = Digraph("spanning", format="svg")
    g.attr(bgcolor="transparent", rankdir="TB", ranksep="0.4", nodesep="0.1")
    g.attr("node", shape="box", style="filled,rounded",
           fontname=FONTS["mono"], fontsize="10",
           color=COLORS["edge"], fontcolor=COLORS["text"])

    # Use a single wide node per row with proper labeling
    # This ensures consistent width alignment

    # Sentence row - ~10% wider on each side than word row
    g.node("sentence", "Sentence  —  spans slots 1-7",
           fillcolor=COLORS["level_3"], width="7.0", fixedsize="true")

    # Clause row
    g.node("clause", "Clause  —  spans slots 1-7",
           fillcolor=COLORS["level_4"], width="7.0", fixedsize="true")

    # Phrase row - use HTML table for side-by-side, ~10% wider on each side
    phrase_label = '''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="8">
    <TR>
        <TD BGCOLOR="{}" STYLE="rounded" WIDTH="216">Phrase (subject)<BR/>spans slots 1-3</TD>
        <TD BGCOLOR="{}" STYLE="rounded" WIDTH="288">Phrase (predicate)<BR/>spans slots 4-7</TD>
    </TR>
    </TABLE>
    >'''.format(COLORS["level_5"], COLORS["level_5"])
    g.node("phrases", phrase_label, shape="none", fillcolor="transparent")

    # Word row - use HTML table
    word_cells = ""
    words = ["In", "the", "beginning", "God", "created", "the", "heavens"]
    for i, word in enumerate(words, 1):
        word_cells += f'<TD BGCOLOR="{COLORS["level_6"]}" STYLE="rounded" WIDTH="60">{i}: {word}</TD>'

    word_label = f'''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">
    <TR>{word_cells}</TR>
    <TR><TD COLSPAN="7"><FONT POINT-SIZE="9" COLOR="{COLORS["text_secondary"]}">Slot Nodes (words)</FONT></TD></TR>
    </TABLE>
    >'''
    g.node("words", word_label, shape="none", fillcolor="transparent")

    # Invisible edges for vertical layout
    g.edge("sentence", "clause", style="invis")
    g.edge("clause", "phrases", style="invis")
    g.edge("phrases", "words", style="invis")

    output_path = OUTPUT_DIR / "spanning-nodes"
    g.render(output_path, cleanup=True)
    print(f"Generated: {output_path}.svg")


def generate_features_diagram():
    """Generate diagram showing features attached to a node."""
    g = Digraph("features", format="svg")
    g.attr(rankdir="LR", bgcolor="transparent")
    g.attr("node", fontname=FONTS["mono"], fontsize="11",
           color=COLORS["edge"], fontcolor=COLORS["text"])
    g.attr("edge", color=COLORS["edge"], arrowsize="0.7")

    # Central node - highlighted with accent
    g.node("word", "Word Node\\n(slot 5)", shape="box", style="filled,rounded",
           fillcolor=COLORS["highlight_bg"], color=COLORS["accent"],
           penwidth="2", width="1.5", height="0.8")

    # Feature nodes - subtle variations
    features = [
        ("pos", "pos = verb", COLORS["feature_1"]),
        ("lemma", "lemma = create", COLORS["feature_2"]),
        ("tense", "tense = past", COLORS["feature_3"]),
        ("person", "person = 3rd", COLORS["feature_4"]),
        ("number", "number = sg", COLORS["feature_5"]),
    ]

    for feat_id, label, color in features:
        g.node(feat_id, label, shape="box", style="filled,rounded", fillcolor=color)
        g.edge("word", feat_id, label="F." + feat_id, fontsize="9",
               fontcolor=COLORS["text_secondary"])

    output_path = OUTPUT_DIR / "node-features"
    g.render(output_path, cleanup=True)
    print(f"Generated: {output_path}.svg")


def generate_locality_api():
    """Generate diagram showing L API navigation."""
    g = Digraph("locality", format="svg")
    g.attr(rankdir="TB", bgcolor="transparent", nodesep="0.5", ranksep="0.6")
    g.attr("node", shape="box", style="filled,rounded",
           fontname=FONTS["mono"], fontsize="11",
           color=COLORS["edge"], fontcolor=COLORS["text"])

    # Vertical hierarchy
    g.node("sent", "Sentence", fillcolor=COLORS["level_3"])
    g.node("clause", "Clause", fillcolor=COLORS["level_4"])
    g.node("phrase", "Phrase", fillcolor=COLORS["level_5"])

    # Current node - highlighted with accent
    g.node("word", "Word\\n(current)", fillcolor=COLORS["highlight_bg"],
           color=COLORS["accent"], penwidth="2.5")

    # Siblings
    g.node("prev_word", "Previous\\nWord", fillcolor=COLORS["level_6"])
    g.node("next_word", "Next\\nWord", fillcolor=COLORS["level_6"])

    # L.u edges (up) - use accent color for API labels
    g.edge("word", "phrase", label="L.u()", color=COLORS["accent"],
           fontcolor=COLORS["accent"], fontsize="10", penwidth="1.5")
    g.edge("phrase", "clause", color=COLORS["edge_light"], style="dashed")
    g.edge("clause", "sent", color=COLORS["edge_light"], style="dashed")

    # L.d edge (down)
    g.edge("phrase", "word", label="L.d()", color=COLORS["text_secondary"],
           fontcolor=COLORS["text_secondary"], fontsize="10",
           constraint="false", style="dashed")

    # Sibling edges
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("prev_word")
        s.node("word")
        s.node("next_word")

    g.edge("prev_word", "word", label="L.n()", color=COLORS["accent"],
           fontcolor=COLORS["accent"], fontsize="10", penwidth="1.5")
    g.edge("word", "next_word", label="L.n()", color=COLORS["accent"],
           fontcolor=COLORS["accent"], fontsize="10", penwidth="1.5")
    g.edge("word", "prev_word", label="L.p()", color=COLORS["text_secondary"],
           fontcolor=COLORS["text_secondary"], fontsize="10",
           constraint="false", style="dashed")

    output_path = OUTPUT_DIR / "locality-api"
    g.render(output_path, cleanup=True)
    print(f"Generated: {output_path}.svg")


def main():
    ensure_output_dir()
    print(f"Generating diagrams to: {OUTPUT_DIR}\n")

    generate_slot_sequence()
    generate_containment_hierarchy()
    generate_spanning_nodes()
    generate_features_diagram()
    generate_locality_api()

    print(f"\nDone! Generated 5 SVG diagrams.")


if __name__ == "__main__":
    main()
