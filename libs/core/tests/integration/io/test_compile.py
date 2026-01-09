"""Integration tests for .tf to .cfm compilation."""

import pytest
import tempfile
import shutil
from pathlib import Path
from cfabric.core import Fabric
from cfabric.io.compiler import Compiler, compile_corpus


class TestCompiler:
    """Test the Compiler class."""

    @pytest.fixture
    def mini_corpus_copy(self):
        """Create a copy of mini_corpus in a temp directory."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)
            yield test_dir

    def test_compile_creates_cfm_directory(self, mini_corpus_copy):
        """Compiler creates .cfm directory structure."""
        compiler = Compiler(str(mini_corpus_copy))
        success = compiler.compile()

        assert success
        cfm_path = mini_corpus_copy / '.cfm' / '1'
        assert cfm_path.exists()
        assert (cfm_path / 'meta.json').exists()
        assert (cfm_path / 'warp').is_dir()
        assert (cfm_path / 'computed').is_dir()
        assert (cfm_path / 'features').is_dir()
        assert (cfm_path / 'edges').is_dir()

    def test_compile_creates_warp_files(self, mini_corpus_copy):
        """Compiler creates WARP feature files."""
        compiler = Compiler(str(mini_corpus_copy))
        compiler.compile()

        warp_dir = mini_corpus_copy / '.cfm' / '1' / 'warp'
        assert (warp_dir / 'otype.npy').exists()
        assert (warp_dir / 'otype_types.json').exists()
        assert (warp_dir / 'oslots_indptr.npy').exists()
        assert (warp_dir / 'oslots_data.npy').exists()

    def test_compile_creates_computed_files(self, mini_corpus_copy):
        """Compiler creates precomputed data files."""
        compiler = Compiler(str(mini_corpus_copy))
        compiler.compile()

        computed_dir = mini_corpus_copy / '.cfm' / '1' / 'computed'
        assert (computed_dir / 'rank.npy').exists()
        assert (computed_dir / 'order.npy').exists()
        assert (computed_dir / 'levels.json').exists()
        assert (computed_dir / 'levup_indptr.npy').exists()
        assert (computed_dir / 'levdown_indptr.npy').exists()


class TestCompileCorpus:
    """Test the compile_corpus convenience function."""

    @pytest.fixture
    def mini_corpus_copy(self):
        """Create a copy of mini_corpus in a temp directory."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)
            yield test_dir

    def test_compile_corpus_function(self, mini_corpus_copy):
        """compile_corpus convenience function works."""
        success = compile_corpus(str(mini_corpus_copy))

        assert success
        assert (mini_corpus_copy / '.cfm' / '1' / 'meta.json').exists()


class TestFabricCompile:
    """Test Fabric.compile() method."""

    @pytest.fixture
    def fabric_with_corpus(self):
        """Create Fabric with mini_corpus copy."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            TF = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            yield TF, test_dir

    def test_fabric_compile_method(self, fabric_with_corpus):
        """Fabric.compile() method works."""
        TF, test_dir = fabric_with_corpus
        success = TF.compile()

        assert success
        assert (test_dir / '.cfm' / '1' / 'meta.json').exists()


class TestLoadCfm:
    """Test loading from .cfm format."""

    @pytest.fixture
    def compiled_corpus(self):
        """Create and compile a mini_corpus copy."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            TF = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            TF.compile()
            yield TF, test_dir

    def test_load_from_cfm_returns_api(self, compiled_corpus):
        """load() returns an API object when loading from .cfm."""
        TF, test_dir = compiled_corpus
        api = TF.load("")

        assert api is not None
        assert hasattr(api, 'F')
        assert hasattr(api, 'E')
        assert hasattr(api, 'C')

    def test_load_cfm_otype_feature(self, compiled_corpus):
        """load() correctly loads otype feature from .cfm."""
        TF, test_dir = compiled_corpus
        api = TF.load("")

        # Slot nodes are type 'word'
        assert api.F.otype.v(1) == 'word'
        assert api.F.otype.v(5) == 'word'

        # Non-slot nodes
        assert api.F.otype.v(6) == 'phrase'
        assert api.F.otype.v(8) == 'sentence'

    def test_load_cfm_oslots_feature(self, compiled_corpus):
        """load() correctly loads oslots feature from .cfm."""
        TF, test_dir = compiled_corpus
        api = TF.load("")

        # Slot nodes return themselves
        assert 1 in api.E.oslots.s(1)

        # Non-slot nodes return their slots
        slots = api.E.oslots.s(6)
        assert len(slots) == 3  # phrase 6 has 3 words

    def test_load_cfm_node_features(self, compiled_corpus):
        """loadAll() correctly loads node features from .cfm."""
        TF, test_dir = compiled_corpus
        api = TF.loadAll()

        # String feature
        assert api.F.word.v(1) == 'hello'

        # Int feature (if present)
        if hasattr(api.F, 'number'):
            num = api.F.number.v(1)
            assert num is None or isinstance(num, int)

    def test_load_cfm_computed_data(self, compiled_corpus):
        """load() correctly loads computed data from .cfm."""
        TF, test_dir = compiled_corpus
        api = TF.load("")

        # Rank and order arrays
        assert hasattr(api.C, 'rank')
        assert hasattr(api.C, 'order')
        assert len(api.C.rank.data) > 0
        assert len(api.C.order.data) > 0


class TestCfmVsTfEquivalence:
    """Test that .cfm produces same results as .tf loading."""

    @pytest.fixture
    def both_apis(self):
        """Load corpus via both .tf and .cfm methods."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            # Load from .tf (dict-based backend)
            TF_source = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            api_tf = TF_source.load('word pos number parent')

            # Compile and load from .cfm (mmap-based backend)
            TF_cfm = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            TF_cfm.compile()
            api_cfm = TF_cfm.loadAll()

            yield api_tf, api_cfm

    def test_otype_equivalence(self, both_apis):
        """otype values match between .tf and .cfm loading."""
        api_tf, api_cfm = both_apis

        for n in range(1, 9):
            assert api_tf.F.otype.v(n) == api_cfm.F.otype.v(n)

    def test_word_feature_equivalence(self, both_apis):
        """word feature values match between .tf and .cfm loading."""
        api_tf, api_cfm = both_apis

        for n in range(1, 6):  # slots only
            assert api_tf.F.word.v(n) == api_cfm.F.word.v(n)


class TestCfmLocalityApi:
    """Test Locality API works correctly when loaded from .cfm."""

    @pytest.fixture
    def cfm_api(self):
        """Load corpus from .cfm format."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            TF = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            TF.compile()
            api = TF.load("")
            yield api

    def test_locality_u_from_slot(self, cfm_api):
        """L.u() returns embedders of a slot node."""
        result = cfm_api.L.u(1)
        assert isinstance(result, tuple)
        # Slot 1 should be embedded in phrase and sentence
        assert len(result) >= 1

    def test_locality_u_with_otype_filter(self, cfm_api):
        """L.u() with otype filter works."""
        result = cfm_api.L.u(1, otype='sentence')
        assert isinstance(result, tuple)
        for n in result:
            assert cfm_api.F.otype.v(n) == 'sentence'

    def test_locality_d_from_sentence(self, cfm_api):
        """L.d() returns embeddees of a non-slot node."""
        # Find a sentence node
        sentence_node = None
        for n in range(6, 10):
            if cfm_api.F.otype.v(n) == 'sentence':
                sentence_node = n
                break

        if sentence_node:
            result = cfm_api.L.d(sentence_node)
            assert isinstance(result, tuple)
            assert len(result) > 0

    def test_locality_d_with_otype_filter(self, cfm_api):
        """L.d() with otype filter works."""
        # Find a sentence node
        sentence_node = None
        for n in range(6, 10):
            if cfm_api.F.otype.v(n) == 'sentence':
                sentence_node = n
                break

        if sentence_node:
            result = cfm_api.L.d(sentence_node, otype='word')
            assert isinstance(result, tuple)
            for n in result:
                assert cfm_api.F.otype.v(n) == 'word'

    def test_locality_p_previous_node(self, cfm_api):
        """L.p() returns previous sibling nodes."""
        result = cfm_api.L.p(2)
        assert isinstance(result, tuple)

    def test_locality_n_next_node(self, cfm_api):
        """L.n() returns next sibling nodes."""
        result = cfm_api.L.n(1)
        assert isinstance(result, tuple)


class TestCfmEdgeFeatureApi:
    """Test Edge Feature API works correctly when loaded from .cfm."""

    @pytest.fixture
    def cfm_api(self):
        """Load corpus from .cfm format."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            TF = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            TF.compile()
            api = TF.load("")
            yield api

    def test_edge_f_forward(self, cfm_api):
        """E.parent.f() returns forward edges."""
        if hasattr(cfm_api.E, 'parent'):
            result = cfm_api.E.parent.f(1)
            assert isinstance(result, tuple)

    def test_edge_t_backward(self, cfm_api):
        """E.parent.t() returns backward edges."""
        if hasattr(cfm_api.E, 'parent'):
            # Find a node that has incoming parent edges
            for n in range(6, 10):
                result = cfm_api.E.parent.t(n)
                assert isinstance(result, tuple)


class TestCfmSearchApi:
    """Test Search API works correctly when loaded from .cfm."""

    @pytest.fixture
    def cfm_api(self):
        """Load corpus from .cfm format."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            TF = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            TF.compile()
            api = TF.loadAll()
            yield api

    def test_search_simple_type(self, cfm_api):
        """Simple type search works."""
        results = list(cfm_api.S.search('word'))
        assert len(results) > 0
        for (n,) in results:
            assert cfm_api.F.otype.v(n) == 'word'

    def test_search_with_feature_constraint(self, cfm_api):
        """Search with feature constraint works."""
        results = list(cfm_api.S.search('word word=hello'))
        assert len(results) > 0
        for (n,) in results:
            assert cfm_api.F.word.v(n) == 'hello'

    def test_search_embedding(self, cfm_api):
        """Search with embedding relation works."""
        results = list(cfm_api.S.search('''
sentence
  word
'''))
        assert len(results) > 0
        for sentence, word in results:
            assert cfm_api.F.otype.v(sentence) == 'sentence'
            assert cfm_api.F.otype.v(word) == 'word'

    def test_search_phrase_word_embedding(self, cfm_api):
        """Search phrase containing word works."""
        results = list(cfm_api.S.search('''
phrase
  word
'''))
        # Results should be phrase-word pairs
        for result in results:
            assert len(result) == 2
            phrase, word = result
            assert cfm_api.F.otype.v(phrase) == 'phrase'
            assert cfm_api.F.otype.v(word) == 'word'


class TestCfmTextApi:
    """Test Text API works correctly when loaded from .cfm."""

    @pytest.fixture
    def cfm_api(self):
        """Load corpus from .cfm format."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            TF = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            TF.compile()
            api = TF.loadAll()
            yield api

    def test_text_api_exists(self, cfm_api):
        """T (Text) API is available."""
        assert hasattr(cfm_api, 'T')

    def test_text_single_node(self, cfm_api):
        """T.text() works for a single node."""
        # Get text of word node 1
        text = cfm_api.T.text(1)
        assert isinstance(text, str)
        assert text == 'hello'

    def test_text_multiple_nodes(self, cfm_api):
        """T.text() works for multiple nodes."""
        # Get text of first 3 word nodes
        text = cfm_api.T.text([1, 2, 3])
        assert isinstance(text, str)
        assert 'hello' in text

    def test_text_non_slot_node(self, cfm_api):
        """T.text() works for non-slot nodes (phrase/sentence)."""
        # Get text of phrase node (node 6 should be a phrase)
        phrase_node = None
        for n in range(6, 10):
            if cfm_api.F.otype.v(n) == 'phrase':
                phrase_node = n
                break

        if phrase_node:
            text = cfm_api.T.text(phrase_node)
            assert isinstance(text, str)
            assert len(text) > 0


class TestCfmEdgeValuesNone:
    """Test that None values in edge features are preserved correctly.

    Edge features with @edgeValues can have entries without explicit values,
    which parse as None. Uses INT32_MIN (-2147483648) as sentinel.

    The distance.tf test fixture has edges with value 0, None, and other ints.
    """

    @pytest.fixture
    def cfm_api(self):
        """Load corpus from .cfm format with fresh compile."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            # Remove any existing .cfm to force recompile
            cfm_dir = test_dir / '.cfm'
            if cfm_dir.exists():
                shutil.rmtree(cfm_dir)

            TF = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            TF.compile()
            api = TF.loadAll()
            yield api

    def test_edge_values_none_vs_zero(self, cfm_api):
        """None and 0 values are distinguished correctly."""
        # Edge 1->2 has explicit value 0
        edges_1 = dict(cfm_api.E.distance.f(1))
        assert edges_1[2] == 0

        # Edge 2->3 has no value (should be None)
        edges_2 = dict(cfm_api.E.distance.f(2))
        assert edges_2[3] is None

    def test_edge_values_sentinel_in_metadata(self, cfm_api):
        """Sentinel value is stored in metadata for int-valued edge features."""
        meta = cfm_api.E.distance.meta
        assert 'none_sentinel' in meta
        assert meta['none_sentinel'] == -2147483648

    def test_edge_values_inverse_preserves_types(self, cfm_api):
        """Inverse lookups preserve None and 0 correctly."""
        # Edge 1->2 has value 0
        assert dict(cfm_api.E.distance.t(2))[1] == 0
        # Edge 2->3 has no value
        assert dict(cfm_api.E.distance.t(3))[2] is None


class TestCfmSectionsComputed:
    """Test that sections computed feature is saved and loaded from .cfm.

    This tests the bug where T.nodeFromSection() fails when loading from
    cached .cfm because C.sections wasn't compiled and saved.
    """

    @pytest.fixture
    def cfm_api_with_sections(self):
        """Load corpus with sections from .cfm format (two-stage: compile then reload)."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            # Remove any existing .cfm to force fresh compile
            cfm_dir = test_dir / '.cfm'
            if cfm_dir.exists():
                shutil.rmtree(cfm_dir)

            # Stage 1: Fresh compile from .tf
            TF1 = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            TF1.loadAll()  # This triggers compilation
            del TF1  # Delete the instance

            # Stage 2: Load from cached .cfm (simulating kernel restart)
            TF2 = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            api = TF2.loadAll()  # This should load from .cfm cache
            yield api

    def test_sections_computed_exists(self, cfm_api_with_sections):
        """C.sections computed feature exists when loading from .cfm."""
        assert hasattr(cfm_api_with_sections.C, 'sections')

    def test_node_from_section_works(self, cfm_api_with_sections):
        """T.nodeFromSection() works when loading from .cfm."""
        api = cfm_api_with_sections
        # Get node for section ('S1',) - the sentence
        node = api.T.nodeFromSection(('S1',))
        assert node == 8  # sentence node

    def test_node_from_section_phrase_works(self, cfm_api_with_sections):
        """T.nodeFromSection() works for phrase level when loading from .cfm."""
        api = cfm_api_with_sections
        # Get node for section ('S1', 1) - first phrase in sentence
        node = api.T.nodeFromSection(('S1', 1))
        assert node == 6  # phrase node 6


class TestCfmNodeFeatureNone:
    """Test that None values in int node features are preserved correctly.

    Uses -1 as sentinel for missing int node feature values.

    The score.tf test fixture has nodes with value 0, None, and other ints.
    """

    @pytest.fixture
    def cfm_api(self):
        """Load corpus from .cfm format with fresh compile."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            # Remove any existing .cfm to force recompile
            cfm_dir = test_dir / '.cfm'
            if cfm_dir.exists():
                shutil.rmtree(cfm_dir)

            TF = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            TF.compile()
            api = TF.loadAll()
            yield api

    def test_node_int_none_vs_zero(self, cfm_api):
        """None and 0 values are distinguished correctly."""
        # Node 2 has explicit value 0
        assert cfm_api.F.score.v(2) == 0
        # Node 3 has no value (should be None)
        assert cfm_api.F.score.v(3) is None

    def test_node_int_s_excludes_none(self, cfm_api):
        """F.score.s(0) returns nodes with 0, not nodes with None."""
        zeros = cfm_api.F.score.s(0)
        assert 2 in zeros  # has value 0
        assert 3 not in zeros  # has None, not 0


class TestCfmCompileReloadCycle:
    """Test that compile-then-reload produces consistent results.

    These tests specifically verify the cycle of:
    1. Fresh load from .tf (compiles to .cfm)
    2. Reload from .cfm cache (simulating kernel restart)

    This catches bugs like:
    - Blank line handling in feature parsing
    - Section lookup metadata not being loaded
    - Feature data offset issues
    """

    @pytest.fixture
    def fresh_and_cached_apis(self):
        """Load corpus fresh, then simulate reload from cache."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            # Remove any existing .cfm to force fresh compile
            cfm_dir = test_dir / '.cfm'
            if cfm_dir.exists():
                shutil.rmtree(cfm_dir)

            # Stage 1: Fresh load from .tf (compiles to .cfm)
            TF1 = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            api_fresh = TF1.load('word pos')

            # Stage 2: Simulate kernel restart by creating new Fabric
            # This should load from .cfm cache
            TF2 = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            api_cached = TF2.load('word pos')

            yield api_fresh, api_cached

    def test_otype_consistent_after_reload(self, fresh_and_cached_apis):
        """otype values are the same after cache reload."""
        api_fresh, api_cached = fresh_and_cached_apis

        # Check all nodes
        max_node = api_fresh.F.otype.maxNode
        for n in range(1, max_node + 1):
            fresh_type = api_fresh.F.otype.v(n)
            cached_type = api_cached.F.otype.v(n)
            assert fresh_type == cached_type, f"Node {n}: {fresh_type} != {cached_type}"

    def test_word_feature_consistent_after_reload(self, fresh_and_cached_apis):
        """String features are the same after cache reload."""
        api_fresh, api_cached = fresh_and_cached_apis

        # Check all word nodes
        for n in api_fresh.F.otype.s('word'):
            fresh_val = api_fresh.F.word.v(n)
            cached_val = api_cached.F.word.v(n)
            assert fresh_val == cached_val, f"Node {n}: {fresh_val!r} != {cached_val!r}"

    def test_locality_consistent_after_reload(self, fresh_and_cached_apis):
        """L.d() and L.u() are the same after cache reload."""
        api_fresh, api_cached = fresh_and_cached_apis

        # Check L.d() for a sentence node
        for n in api_fresh.F.otype.s('sentence'):
            fresh_down = api_fresh.L.d(n)
            cached_down = api_cached.L.d(n)
            assert fresh_down == cached_down, f"L.d({n}): {fresh_down} != {cached_down}"

            fresh_words = api_fresh.L.d(n, otype='word')
            cached_words = api_cached.L.d(n, otype='word')
            assert fresh_words == cached_words

    def test_search_consistent_after_reload(self, fresh_and_cached_apis):
        """Search results are the same after cache reload."""
        api_fresh, api_cached = fresh_and_cached_apis

        # Simple search
        fresh_results = set(api_fresh.S.search('word'))
        cached_results = set(api_cached.S.search('word'))
        assert fresh_results == cached_results


class TestCfmSectionsCycle:
    """Test that section features work after cache reload.

    Tests specifically for the bug where sections computed feature
    was not available after loading from .cfm cache.
    """

    @pytest.fixture
    def fresh_and_cached_apis_with_sections(self):
        """Load corpus with sections, fresh and cached."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            cfm_dir = test_dir / '.cfm'
            if cfm_dir.exists():
                shutil.rmtree(cfm_dir)

            # Stage 1: Fresh load
            TF1 = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            api_fresh = TF1.loadAll()

            # Stage 2: Cached load
            TF2 = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            api_cached = TF2.loadAll()

            yield api_fresh, api_cached, TF1, TF2

    def test_sections_computed_exists_after_reload(self, fresh_and_cached_apis_with_sections):
        """C.sections exists after cache reload."""
        api_fresh, api_cached, TF1, TF2 = fresh_and_cached_apis_with_sections

        if TF1.sectionsOK:
            assert hasattr(api_fresh.C, 'sections')
            assert hasattr(api_cached.C, 'sections')

    def test_section_types_consistent(self, fresh_and_cached_apis_with_sections):
        """sectionTypes are same after reload."""
        api_fresh, api_cached, TF1, TF2 = fresh_and_cached_apis_with_sections

        assert TF1.sectionTypes == TF2.sectionTypes
        assert TF1.sectionFeats == TF2.sectionFeats

    def test_node_from_section_consistent(self, fresh_and_cached_apis_with_sections):
        """T.nodeFromSection returns same results after reload."""
        api_fresh, api_cached, TF1, TF2 = fresh_and_cached_apis_with_sections

        if not TF1.sectionsOK:
            pytest.skip("Corpus has no section configuration")

        # Get a section from fresh load
        sections = api_fresh.C.sections.data
        sec1 = sections.get('sec1', {})

        # Try each section in sec1
        for sec0_node, chapter_map in list(sec1.items())[:3]:  # Check first 3
            for ch_heading, ch_node in list(chapter_map.items())[:2]:
                # Build the section tuple
                sec0_feat = TF1.sectionFeats[0]
                sec0_name = api_fresh.F.__dict__.get(sec0_feat, None)
                if sec0_name:
                    sec0_val = sec0_name.v(sec0_node)
                    if sec0_val:
                        section = (sec0_val, ch_heading)
                        fresh_node = api_fresh.T.nodeFromSection(section)
                        cached_node = api_cached.T.nodeFromSection(section)
                        assert fresh_node == cached_node, \
                            f"Section {section}: fresh={fresh_node} cached={cached_node}"


class TestCfmEdgeStringValues:
    """Test that edge features with string values work after cfm reload.

    This tests the bug where edges with @edgeValues + @valueType=str
    fail to load from .cfm cache due to object dtype can't be mmap'd.
    """

    @pytest.fixture
    def fresh_and_cached_apis(self):
        """Compile and reload to test string edge values."""
        mini_corpus = Path(__file__).parent.parent.parent / 'fixtures' / 'mini_corpus'
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / 'mini_corpus'
            shutil.copytree(mini_corpus, test_dir)

            cfm_dir = test_dir / '.cfm'
            if cfm_dir.exists():
                shutil.rmtree(cfm_dir)

            # Stage 1: Fresh compile (explicitly load relation edge feature)
            TF1 = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            api_fresh = TF1.load('relation')
            del TF1

            # Stage 2: Load from cache
            TF2 = Fabric(
                locations=[str(test_dir.parent)],
                modules=['mini_corpus'],
                silent='deep'
            )
            api_cached = TF2.loadAll()

            yield api_fresh, api_cached

    def test_string_edge_values_after_reload(self, fresh_and_cached_apis):
        """Edge string values are preserved after cache reload."""
        api_fresh, api_cached = fresh_and_cached_apis

        # Edge 1->6 should have value 'subject'
        assert dict(api_cached.E.relation.f(1))[6] == 'subject'
        # Edge 2->6 should have value 'predicate'
        assert dict(api_cached.E.relation.f(2))[6] == 'predicate'
        # Edge 3->6 should have value 'object'
        assert dict(api_cached.E.relation.f(3))[6] == 'object'

    def test_string_edge_values_consistent(self, fresh_and_cached_apis):
        """Edge string values match between fresh and cached load."""
        api_fresh, api_cached = fresh_and_cached_apis

        # Check all edges match
        for n in range(1, 6):
            fresh_edges = dict(api_fresh.E.relation.f(n))
            cached_edges = dict(api_cached.E.relation.f(n))
            assert fresh_edges == cached_edges, f"Node {n}: {fresh_edges} != {cached_edges}"
