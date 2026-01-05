"""
Compile .tf source files to .cfm mmap format.

This module provides the Compiler class that converts Text-Fabric (.tf) source
files into the Context Fabric memory-mapped (.cfm) format. The cfm format uses
numpy arrays with memory mapping for efficient multi-process access.
"""

from __future__ import annotations

import json
import numpy as np
from collections.abc import Iterable
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from cfabric.core.config import (
    CFM_VERSION,
    OTYPE,
    OSLOTS,
    OTEXT,
    WARP,
    NODE_DTYPE,
    TYPE_DTYPE,
    INDEX_DTYPE,
    MISSING_STR_INDEX,
)
from cfabric.storage.csr import CSRArray, CSRArrayWithValues
from cfabric.storage.string_pool import StringPool, IntFeatureArray
from cfabric.utils.files import dirMake, fileExists, fileOpen
from cfabric.utils.timestamp import Timestamp, SILENT_D
from cfabric.utils.helpers import setFromSpec, valueFromTf, makeInverse, makeInverseVal
import cfabric.precompute.prepare as prepare

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _check_sentinel_collision(
    values: Iterable[Any],
    sentinel: int,
    feature_name: str,
    feature_kind: str,
    tmObj: Timestamp,
) -> bool:
    """Check if data contains the sentinel value and warn if so.

    Parameters
    ----------
    values : iterable
        Values to check for sentinel collision
    sentinel : int
        The sentinel value used for None/missing
    feature_name : str
        Name of the feature (for warning message)
    feature_kind : str
        'node' or 'edge' (for warning message)
    tmObj : Timestamp
        Timestamp object for logging warnings

    Returns
    -------
    bool
        True if sentinel collision was found, False otherwise
    """
    for v in values:
        if v == sentinel:
            tmObj.warning(
                f"{feature_kind.capitalize()} feature '{feature_name}' contains "
                f"value {sentinel} which collides with the None sentinel. "
                f"This value will be incorrectly read as None after loading."
            )
            return True
    return False


class Compiler:
    """
    Compile TF source files to CF mmap format.

    The compiler reads .tf plain text feature files and converts them to
    memory-mapped numpy arrays organized in the .cfm directory structure.

    Usage
    -----
    compiler = Compiler(source_dir='/path/to/tf/files')
    success = compiler.compile(output_dir='/path/to/output/.cfm/1/')

    Parameters
    ----------
    source_dir : str
        Path to directory containing .tf source files
    tmObj : Timestamp, optional
        Timestamp object for logging messages. If None, a new one is created.
    """

    def __init__(self, source_dir: str, tmObj: Timestamp | None = None) -> None:
        self.source_dir: Path = Path(source_dir)
        self.tmObj: Timestamp = tmObj or Timestamp()
        self.info = self.tmObj.info
        self.error = self.tmObj.error
        self.warning = self.tmObj.warning

        # Will be populated during compilation
        self.max_slot: int = 0
        self.max_node: int = 0
        self.slot_type: str = ""
        self.node_types: list[str] = []
        self.type_order: list[str] = []  # Types in level order

        # Parsed feature data
        self._otype_data: tuple[tuple[str, ...], int, int, str] | None = None
        self._oslots_data: tuple[tuple[tuple[int, ...], ...], int, int] | None = None
        self._otext_meta: dict[str, str] = {}
        self._feature_meta: dict[str, dict[str, str]] = {}
        self._node_features: dict[str, dict[int, str | int]] = {}
        self._edge_features: dict[str, tuple[dict[int, Any], bool]] = {}  # (data, has_values)

        # Computed data (populated by _precompute)
        self._levels_data: list[tuple[str, float, int, int]] | None = None
        self._order_data: list[int] | None = None
        self._rank_data: list[int] | None = None
        self._levup_data: list[tuple[int, ...]] | None = None
        self._levdown_data: list[tuple[int, ...]] | None = None

    def compile(self, output_dir: str | Path | None = None) -> bool:
        """
        Compile all .tf files to .cfm format.

        Parameters
        ----------
        output_dir : str, optional
            Output directory. Defaults to {source_dir}/.cfm/{CFM_VERSION}/

        Returns
        -------
        bool
            True if compilation succeeded
        """
        if output_dir is None:
            output_dir = self.source_dir / '.cfm' / CFM_VERSION
        output_dir = Path(output_dir)

        self.info(f"Compiling {self.source_dir} to {output_dir}")

        # Create directory structure
        self._create_directories(output_dir)

        # 1. Load and compile WARP features
        self.info("Loading WARP features...")
        if not self._load_otype():
            return False
        if not self._load_oslots():
            return False
        self._load_otext()

        # Compile WARP to numpy format
        self.info("Compiling WARP features...")
        if not self._compile_otype(output_dir):
            return False
        if not self._compile_oslots(output_dir):
            return False

        # 2. Run precomputation
        self.info("Running precomputation...")
        if not self._precompute(output_dir):
            return False

        # 3. Load and compile regular features
        self.info("Loading regular features...")
        self._load_features()

        self.info("Compiling node features...")
        self._compile_node_features(output_dir)

        self.info("Compiling edge features...")
        self._compile_edge_features(output_dir)

        # 4. Write metadata
        self._write_meta(output_dir)

        self.info("Compilation complete")
        return True

    def _create_directories(self, output_dir: Path) -> None:
        """Create the .cfm directory structure."""
        dirMake(str(output_dir))
        dirMake(str(output_dir / 'warp'))
        dirMake(str(output_dir / 'computed'))
        dirMake(str(output_dir / 'features'))
        dirMake(str(output_dir / 'edges'))

    def _parse_tf_file(
        self, path: Path
    ) -> tuple[dict[str, str], dict[int, Any], bool, bool, bool]:
        """
        Parse a .tf file.

        Returns
        -------
        tuple
            (metadata, data, is_edge, edge_has_values, is_config)
        """
        metadata: dict[str, str] = {}
        data: dict[int, Any] = {}
        is_edge = False
        edge_values = False
        is_config = False

        if not fileExists(str(path)):
            self.error(f"Feature file not found: {path}")
            return metadata, data, is_edge, edge_values, is_config

        with fileOpen(str(path)) as fh:
            lines = fh.readlines()

        if not lines:
            return metadata, data, is_edge, edge_values, is_config

        i = 0
        # First line: @node/@edge/@config
        first = lines[0].strip()
        if first == '@edge':
            is_edge = True
        elif first == '@config':
            is_config = True
        elif first != '@node':
            self.error(f"{path.name}: Line 1: missing @node/@edge/@config")
            return metadata, data, is_edge, edge_values, is_config
        i = 1

        # Parse metadata
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                break
            if line == '@edgeValues':
                edge_values = True
            elif line.startswith('@'):
                if '=' in line:
                    key, val = line[1:].split('=', 1)
                    metadata[key] = val
            i += 1

        if is_config:
            return metadata, data, is_edge, edge_values, is_config

        # Parse data
        data_type = metadata.get('valueType', 'str')
        is_num = data_type == 'int'
        norm_fields = 3 if is_edge and edge_values else 2
        implicit_node = 1

        while i < len(lines):
            line = lines[i].rstrip('\n')
            i += 1
            # NOTE: Do NOT skip blank lines here! In TF format, blank lines
            # in the data section represent empty values and must still
            # increment implicit_node. The original TF parser processes every
            # line, and so must we.

            fields = line.split('\t')
            lfields = len(fields)

            nodes: set[int]
            nodes2: set[int]
            val_tf: str

            if lfields == norm_fields:
                nodes = setFromSpec(fields[0])
                if is_edge:
                    if fields[1] == '':
                        continue
                    nodes2 = setFromSpec(fields[1])
                if not is_edge or edge_values:
                    val_tf = fields[-1]
            else:
                if is_edge:
                    if edge_values:
                        if lfields == norm_fields - 1:
                            nodes = {implicit_node}
                            nodes2 = setFromSpec(fields[0])
                            val_tf = fields[-1]
                        elif lfields == norm_fields - 2:
                            nodes = {implicit_node}
                            nodes2 = setFromSpec(fields[0])
                            val_tf = ''
                        else:
                            nodes = {implicit_node}
                            val_tf = ''
                            continue
                    else:
                        if lfields == norm_fields - 1:
                            nodes = {implicit_node}
                            if fields[0] == '':
                                continue
                            nodes2 = setFromSpec(fields[0])
                        else:
                            nodes = {implicit_node}
                            continue
                else:
                    nodes = {implicit_node}
                    if lfields == 1:
                        val_tf = fields[0]
                    else:
                        val_tf = ''

            implicit_node = max(nodes) + 1

            if not is_edge or edge_values:
                value: str | int | None = (
                    int(val_tf)
                    if is_num and val_tf != ''
                    else None
                    if is_num
                    else ''
                    if val_tf == ''
                    else valueFromTf(val_tf)
                )

            if is_edge:
                for n in nodes:
                    for m in nodes2:
                        if not edge_values:
                            data.setdefault(n, set()).add(m)
                        else:
                            data.setdefault(n, {})[m] = value
            else:
                for n in nodes:
                    if value is not None:
                        data[n] = value

        return metadata, data, is_edge, edge_values, is_config

    def _load_otype(self) -> bool:
        """Load and parse the otype feature."""
        path = self.source_dir / f'{OTYPE}.tf'
        metadata, data, is_edge, edge_values, is_config = self._parse_tf_file(path)

        if not data:
            self.error(f"Failed to load {OTYPE}")
            return False

        # Transform to otype format (same as data.py)
        slot_type = data[1]
        otype_list: list[str] = []
        max_slot = 1

        for n in sorted(data):
            if data[n] == slot_type:
                max_slot = n
                continue
            otype_list.append(data[n])

        max_node = len(data)

        self._otype_data = (tuple(otype_list), max_slot, max_node, slot_type)
        self.max_slot = max_slot
        self.max_node = max_node
        self.slot_type = slot_type

        # Collect unique node types
        type_set: set[str] = {slot_type}
        for t in otype_list:
            type_set.add(t)
        self.node_types = sorted(type_set)

        self._feature_meta[OTYPE] = metadata
        return True

    def _load_oslots(self) -> bool:
        """Load and parse the oslots feature."""
        path = self.source_dir / f'{OSLOTS}.tf'
        metadata, data, is_edge, edge_values, is_config = self._parse_tf_file(path)

        if not data:
            self.error(f"Failed to load {OSLOTS}")
            return False

        # Transform to oslots format (same as data.py)
        node_list = sorted(data)
        max_slot = node_list[0] - 1
        max_node = node_list[-1]

        oslots: list[tuple[int, ...]] = []
        for n in node_list:
            oslots.append(tuple(sorted(data[n])))

        self._oslots_data = (tuple(oslots), max_slot, max_node)
        self._feature_meta[OSLOTS] = metadata
        return True

    def _load_otext(self) -> None:
        """Load the otext configuration feature."""
        path = self.source_dir / f'{OTEXT}.tf'
        if not fileExists(str(path)):
            self.warning(f"{OTEXT} not found, using defaults")
            return

        metadata, data, is_edge, edge_values, is_config = self._parse_tf_file(path)
        self._otext_meta = metadata
        self._feature_meta[OTEXT] = metadata

    def _load_features(self) -> None:
        """Load all non-WARP features from the source directory."""
        for tf_file in self.source_dir.glob('*.tf'):
            # Skip directories (e.g., .tf cache directory)
            if tf_file.is_dir():
                continue
            feature_name = tf_file.stem
            if feature_name in WARP:
                continue

            metadata, data, is_edge, edge_values, is_config = self._parse_tf_file(tf_file)

            if is_config:
                self._feature_meta[feature_name] = metadata
                continue

            self._feature_meta[feature_name] = metadata

            if is_edge:
                self._edge_features[feature_name] = (data, edge_values)
            else:
                self._node_features[feature_name] = data

    def _compile_otype(self, output_dir: Path) -> bool:
        """Compile otype to numpy format."""
        if self._otype_data is None:
            return False

        (otype_list, max_slot, max_node, slot_type) = self._otype_data

        # Build type mapping
        unique_types = sorted(set(otype_list))
        type_to_idx = {t: i for i, t in enumerate(unique_types)}

        # Create uint8 array of type indices for non-slot nodes
        num_nonslot = max_node - max_slot
        otype_arr: NDArray[np.uint8] = np.zeros(num_nonslot, dtype=TYPE_DTYPE)

        for i, t in enumerate(otype_list):
            otype_arr[i] = type_to_idx[t]

        # Save arrays
        warp_dir = output_dir / 'warp'
        np.save(str(warp_dir / 'otype.npy'), otype_arr)

        # Save type list as JSON
        with open(warp_dir / 'otype_types.json', 'w') as f:
            json.dump(unique_types, f, indent=1)

        self.type_order = unique_types
        return True

    def _compile_oslots(self, output_dir: Path) -> bool:
        """Compile oslots to CSR format."""
        if self._oslots_data is None:
            return False

        (oslots_list, max_slot, max_node) = self._oslots_data

        # Create CSR from sequences
        csr = CSRArray.from_sequences(oslots_list)

        # Save CSR arrays
        warp_dir = output_dir / 'warp'
        csr.save(str(warp_dir / 'oslots'))

        return True

    def _precompute(self, output_dir: Path) -> bool:
        """Run precomputation steps and save results."""
        if self._otype_data is None or self._oslots_data is None:
            return False

        def log_info(msg: str, tm: bool = True) -> None:
            self.info(f"  {msg}")

        def log_error(msg: str, tm: bool = True) -> None:
            self.error(f"  {msg}")

        computed_dir = output_dir / 'computed'

        # 1. Compute levels
        self.info("  Computing levels...")
        levels_data = prepare.levels(
            log_info, log_error,
            self._otype_data,
            self._oslots_data,
            self._otext_meta
        )

        # Save levels as JSON
        levels_json: list[dict[str, Any]] = [
            {'type': t, 'avgSlots': avg, 'minNode': mn, 'maxNode': mx}
            for t, avg, mn, mx in levels_data
        ]
        with open(computed_dir / 'levels.json', 'w') as f:
            json.dump(levels_json, f, indent=1)

        # 2. Compute order
        self.info("  Computing order...")
        order_data = prepare.order(
            log_info, log_error,
            self._otype_data,
            self._oslots_data,
            levels_data
        )
        order_arr: NDArray[np.uint32] = np.array(order_data, dtype=NODE_DTYPE)
        np.save(str(computed_dir / 'order.npy'), order_arr)

        # 3. Compute rank
        self.info("  Computing rank...")
        rank_data = prepare.rank(
            log_info, log_error,
            self._otype_data,
            order_data
        )
        rank_arr: NDArray[np.uint32] = np.array(rank_data, dtype=NODE_DTYPE)
        np.save(str(computed_dir / 'rank.npy'), rank_arr)

        # 4. Compute levUp
        self.info("  Computing levUp...")
        levup_data = prepare.levUp(
            log_info, log_error,
            self._otype_data,
            self._oslots_data,
            rank_data
        )
        levup_csr = CSRArray.from_sequences(levup_data)
        levup_csr.save(str(computed_dir / 'levup'))

        # 5. Compute levDown
        self.info("  Computing levDown...")
        levdown_data = prepare.levDown(
            log_info, log_error,
            self._otype_data,
            levup_data,
            rank_data
        )
        levdown_csr = CSRArray.from_sequences(levdown_data)
        levdown_csr.save(str(computed_dir / 'levdown'))

        # 6. Compute boundary
        self.info("  Computing boundary...")
        boundary_data = prepare.boundary(
            log_info, log_error,
            self._otype_data,
            self._oslots_data,
            rank_data
        )
        (first_slots, last_slots) = boundary_data

        # Save boundary as CSR arrays
        first_csr = CSRArray.from_sequences(first_slots)
        first_csr.save(str(computed_dir / 'boundary_first'))

        last_csr = CSRArray.from_sequences(last_slots)
        last_csr.save(str(computed_dir / 'boundary_last'))

        # Store computed data for potential later use
        self._levels_data = levels_data
        self._order_data = order_data
        self._rank_data = rank_data
        self._levup_data = levup_data
        self._levdown_data = levdown_data

        return True

    def _compile_node_features(self, output_dir: Path) -> None:
        """Compile all node features."""
        features_dir = output_dir / 'features'

        for feature_name, data in self._node_features.items():
            if not data:
                continue

            metadata = self._feature_meta.get(feature_name, {})
            value_type = metadata.get('valueType', 'str')

            self.info(f"  Compiling {feature_name} ({value_type})...")

            if value_type == 'int':
                self._compile_int_feature(feature_name, data, features_dir, metadata)
            else:
                self._compile_str_feature(feature_name, data, features_dir, metadata)

    def _compile_int_feature(
        self,
        feature_name: str,
        data: dict[int, int],
        output_dir: Path,
        metadata: dict[str, str],
    ) -> None:
        """Compile an integer-valued node feature."""
        # Check for sentinel collision (-1 is used for missing values)
        _check_sentinel_collision(
            data.values(), IntFeatureArray.MISSING,
            feature_name, 'node', self.tmObj
        )

        int_arr = IntFeatureArray.from_dict(data, self.max_node)
        int_arr.save(str(output_dir / f'{feature_name}.npy'))

        # Save metadata
        meta: dict[str, Any] = {
            'name': feature_name,
            'kind': 'node',
            'value_type': 'int',
            **{k: v for k, v in metadata.items() if k != 'valueType'}
        }
        with open(output_dir / f'{feature_name}_meta.json', 'w') as f:
            json.dump(meta, f, indent=1)

    def _compile_str_feature(
        self,
        feature_name: str,
        data: dict[int, str],
        output_dir: Path,
        metadata: dict[str, str],
    ) -> None:
        """Compile a string-valued node feature."""
        str_pool = StringPool.from_dict(data, self.max_node)
        str_pool.save(str(output_dir / feature_name))

        # Save metadata
        meta: dict[str, Any] = {
            'name': feature_name,
            'kind': 'node',
            'value_type': 'str',
            'unique_values': len(str_pool.strings),
            **{k: v for k, v in metadata.items() if k != 'valueType'}
        }
        with open(output_dir / f'{feature_name}_meta.json', 'w') as f:
            json.dump(meta, f, indent=1)

    def _compile_edge_features(self, output_dir: Path) -> None:
        """Compile all edge features."""
        edges_dir = output_dir / 'edges'

        for feature_name, (data, has_values) in self._edge_features.items():
            if not data:
                continue

            metadata = self._feature_meta.get(feature_name, {})

            self.info(f"  Compiling {feature_name} (edge, values={has_values})...")

            if has_values:
                self._compile_edge_with_values(
                    feature_name, data, edges_dir, metadata
                )
            else:
                self._compile_edge_no_values(
                    feature_name, data, edges_dir, metadata
                )

    def _compile_edge_no_values(
        self,
        feature_name: str,
        data: dict[int, set[int]],
        output_dir: Path,
        metadata: dict[str, str],
    ) -> None:
        """Compile an edge feature without values."""
        # Convert to CSR format
        # Create sequences for each node (sorted by target)
        sequences: list[list[int]] = []
        for n in range(1, self.max_node + 1):
            if n in data:
                sequences.append(sorted(data[n]))
            else:
                sequences.append([])

        csr = CSRArray.from_sequences(sequences)
        csr.save(str(output_dir / feature_name))

        # Compute and save inverse edges
        inverse = makeInverse(data)
        inv_sequences: list[list[int]] = []
        for n in range(1, self.max_node + 1):
            if n in inverse:
                inv_sequences.append(sorted(inverse[n]))
            else:
                inv_sequences.append([])

        inv_csr = CSRArray.from_sequences(inv_sequences)
        inv_csr.save(str(output_dir / f'{feature_name}_inv'))

        # Save metadata
        meta: dict[str, Any] = {
            'name': feature_name,
            'kind': 'edge',
            'has_values': False,
            **{k: v for k, v in metadata.items() if k not in ('valueType', 'edgeValues')}
        }
        with open(output_dir / f'{feature_name}_meta.json', 'w') as f:
            json.dump(meta, f, indent=1)

    def _compile_edge_with_values(
        self,
        feature_name: str,
        data: dict[int, dict[int, Any]],
        output_dir: Path,
        metadata: dict[str, str],
    ) -> None:
        """Compile an edge feature with values."""
        value_type = metadata.get('valueType', 'str')
        is_int = value_type == 'int'

        # Determine value dtype and sentinel for None values
        # TF allows edges with @edgeValues where some edges have no explicit
        # value - these parse as None. We use a sentinel to preserve the
        # distinction between None and actual values (like 0).
        value_dtype: str
        none_sentinel: int | None
        if is_int:
            value_dtype = 'int32'
            # INT32_MIN as sentinel - extremely unlikely to be a real value
            none_sentinel = -2147483648

            # Check for sentinel collision
            all_values = (v for row in data.values() for v in row.values())
            _check_sentinel_collision(
                all_values, none_sentinel, feature_name, 'edge', self.tmObj
            )
        else:
            value_dtype = 'object'
            # For strings, we can store None directly in object arrays
            none_sentinel = None

        # Convert to CSRArrayWithValues format
        # First, convert data to 0-indexed format expected by CSRArrayWithValues
        data_0indexed: dict[int, dict[int, Any]] = {}
        for n in range(1, self.max_node + 1):
            if n in data:
                row = data[n]
                # Convert None values to sentinel for int types
                data_0indexed[n - 1] = {
                    m: (v if v is not None else none_sentinel)
                    for m, v in row.items()
                }

        csr = CSRArrayWithValues.from_dict_of_dicts(
            data_0indexed, self.max_node, value_dtype
        )
        csr.save(str(output_dir / feature_name))

        # Compute and save inverse edges
        inverse = makeInverseVal(data)
        inv_data_0indexed: dict[int, dict[int, Any]] = {}
        for n in range(1, self.max_node + 1):
            if n in inverse:
                row = inverse[n]
                # Convert None values to sentinel for int types
                inv_data_0indexed[n - 1] = {
                    m: (v if v is not None else none_sentinel)
                    for m, v in row.items()
                }

        inv_csr = CSRArrayWithValues.from_dict_of_dicts(
            inv_data_0indexed, self.max_node, value_dtype
        )
        inv_csr.save(str(output_dir / f'{feature_name}_inv'))

        # Save metadata - include sentinel so loader can restore None values
        meta: dict[str, Any] = {
            'name': feature_name,
            'kind': 'edge',
            'has_values': True,
            'value_type': value_type,
            **{k: v for k, v in metadata.items() if k not in ('valueType', 'edgeValues')}
        }
        if is_int:
            meta['none_sentinel'] = none_sentinel
        with open(output_dir / f'{feature_name}_meta.json', 'w') as f:
            json.dump(meta, f, indent=1)

    def _write_meta(self, output_dir: Path) -> None:
        """Write corpus metadata to meta.json."""
        # Collect feature names
        node_features = list(self._node_features.keys())
        edge_features = list(self._edge_features.keys())

        meta: dict[str, Any] = {
            'cfm_version': CFM_VERSION,
            'source': str(self.source_dir.name),
            'max_slot': self.max_slot,
            'max_node': self.max_node,
            'slot_type': self.slot_type,
            'node_types': self.node_types,
            'type_order': self.type_order,
            'features': {
                'node': node_features,
                'edge': edge_features
            },
            'created': datetime.now(timezone.utc).isoformat()
        }

        # Include otext metadata if present
        if self._otext_meta:
            meta['otext'] = self._otext_meta

        with open(output_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)


def compile_corpus(source_dir: str, output_dir: str | None = None) -> bool:
    """
    Convenience function to compile a TF corpus to CFM format.

    Parameters
    ----------
    source_dir : str
        Path to directory containing .tf source files
    output_dir : str, optional
        Output directory. Defaults to {source_dir}/.cfm/{CFM_VERSION}/

    Returns
    -------
    bool
        True if compilation succeeded
    """
    compiler = Compiler(source_dir)
    return compiler.compile(output_dir)
