"""Tree-sitter parser for Python source files."""

from typing import List, Optional, Set, Tuple
import tree_sitter
import tree_sitter_python

from sahcr.models import ExtractedCodeUnit, ParameterInfo
from sahcr.parsers.base_parser import BaseParser
from sahcr.parsers.normalizer import split_identifier, tokenize_text

# Node types of structural interest inside function bodies
STRUCTURAL_NODE_TYPES: Set[str] = {
    "for_statement",
    "while_statement",
    "if_statement",
    "try_statement",
    "with_statement",
    "return_statement",
    "raise_statement",
    "yield",
    "call",
    "binary_operator",
    "dictionary",
    "list",
    "set",
    "tuple",
    "generator_expression",
    "list_comprehension",
    "dictionary_comprehension",
    "set_comprehension",
}


class PythonParser(BaseParser):
    """Extracts function boundaries, signatures, docstrings, and AST cues for Python."""

    def __init__(self):
        super().__init__("python", tree_sitter.Language(tree_sitter_python.language()))

    def extract_units_from_tree(
        self,
        tree: tree_sitter.Tree,
        source_bytes: bytes,
        file_path: str
    ) -> List[ExtractedCodeUnit]:
        units: List[ExtractedCodeUnit] = []
        self._traverse_node(tree.root_node, source_bytes, file_path, current_class=None, units=units)
        return units

    def _traverse_node(
        self,
        node: tree_sitter.Node,
        source_bytes: bytes,
        file_path: str,
        current_class: Optional[str],
        units: List[ExtractedCodeUnit]
    ):
        if node.type == "class_definition":
            # Extract class name to associate with inner methods
            name_node = node.child_by_field_name("name")
            class_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8") if name_node else None
            for child in node.children:
                self._traverse_node(child, source_bytes, file_path, current_class=class_name, units=units)
            return

        if node.type == "function_definition":
            unit = self._parse_function(node, source_bytes, file_path, current_class)
            if unit:
                units.append(unit)
            # Traverse children for nested functions
            body_node = node.child_by_field_name("body")
            if body_node:
                for child in body_node.children:
                    self._traverse_node(child, source_bytes, file_path, current_class=current_class, units=units)
            return

        for child in node.children:
            self._traverse_node(child, source_bytes, file_path, current_class=current_class, units=units)

    def _parse_function(
        self,
        node: tree_sitter.Node,
        source_bytes: bytes,
        file_path: str,
        current_class: Optional[str]
    ) -> Optional[ExtractedCodeUnit]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        func_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
        
        # Check if async
        is_async = False
        prev = node.prev_sibling
        if prev and prev.type == "async":
            is_async = True
        elif any(c.type == "async" for c in node.children if c.start_byte < name_node.start_byte):
            is_async = True

        # Extract parameters
        parameters: List[ParameterInfo] = []
        param_tokens: List[str] = []
        params_node = node.child_by_field_name("parameters")
        if params_node:
            idx = 0
            for child in params_node.children:
                if child.type in ("identifier", "typed_parameter", "default_parameter", "typed_default_parameter"):
                    param_info = self._parse_parameter(child, source_bytes, idx)
                    if param_info:
                        parameters.append(param_info)
                        param_tokens.extend(split_identifier(param_info.name))
                        if param_info.type_annotation:
                            param_tokens.extend(tokenize_text(param_info.type_annotation))
                        idx += 1

        # Extract return annotation
        return_annotation = None
        return_tokens: List[str] = []
        ret_node = node.child_by_field_name("return_type")
        if ret_node:
            return_annotation = source_bytes[ret_node.start_byte:ret_node.end_byte].decode("utf-8").strip()
            return_tokens = tokenize_text(return_annotation)

        # Extract docstring
        docstring = None
        docstring_tokens: List[str] = []
        body_node = node.child_by_field_name("body")
        if body_node and len(body_node.children) > 0:
            first_stmt = body_node.children[0]
            if first_stmt.type == "expression_statement":
                expr_child = first_stmt.children[0] if first_stmt.children else None
                if expr_child and expr_child.type == "string":
                    raw_doc = source_bytes[expr_child.start_byte:expr_child.end_byte].decode("utf-8")
                    # Strip quotes
                    docstring = raw_doc.strip("\"' \n\t")
                    docstring_tokens = tokenize_text(docstring)

        # Extract body AST structural node types
        ast_node_types = self._collect_structural_nodes(node)

        # Source snippet
        source_code = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        start_line = node.start_point.row + 1
        end_line = node.end_point.row + 1

        unit_id = f"{file_path}:{func_name}:{start_line}"

        return ExtractedCodeUnit(
            id=unit_id,
            file_path=file_path,
            language="python",
            name=func_name,
            normalized_name_tokens=split_identifier(func_name),
            parameters=parameters,
            parameter_tokens=param_tokens,
            return_annotation=return_annotation,
            return_tokens=return_tokens,
            docstring=docstring,
            docstring_tokens=docstring_tokens,
            source_code=source_code,
            start_line=start_line,
            end_line=end_line,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            ast_node_types=ast_node_types,
            is_async=is_async,
            is_method=current_class is not None,
            parent_class=current_class,
        )

    def _parse_parameter(
        self,
        node: tree_sitter.Node,
        source_bytes: bytes,
        position: int
    ) -> Optional[ParameterInfo]:
        if node.type == "identifier":
            pname = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
            return ParameterInfo(name=pname, position=position)

        if node.type == "typed_parameter":
            name_node = node.child_by_field_name("name") or node.children[0]
            pname = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
            type_node = node.child_by_field_name("type")
            ptype = source_bytes[type_node.start_byte:type_node.end_byte].decode("utf-8") if type_node else None
            return ParameterInfo(name=pname, type_annotation=ptype, position=position)

        if node.type == "default_parameter":
            name_node = node.child_by_field_name("name") or node.children[0]
            pname = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
            val_node = node.child_by_field_name("value")
            pval = source_bytes[val_node.start_byte:val_node.end_byte].decode("utf-8") if val_node else None
            return ParameterInfo(name=pname, default_value=pval, position=position)

        if node.type == "typed_default_parameter":
            name_node = node.child_by_field_name("name") or node.children[0]
            pname = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
            type_node = node.child_by_field_name("type")
            ptype = source_bytes[type_node.start_byte:type_node.end_byte].decode("utf-8") if type_node else None
            val_node = node.child_by_field_name("value")
            pval = source_bytes[val_node.start_byte:val_node.end_byte].decode("utf-8") if val_node else None
            return ParameterInfo(name=pname, type_annotation=ptype, default_value=pval, position=position)

        return None

    def _collect_structural_nodes(self, func_node: tree_sitter.Node) -> List[str]:
        found_types: Set[str] = set()

        def _walk(node: tree_sitter.Node):
            if node.type in STRUCTURAL_NODE_TYPES:
                found_types.add(node.type)
            for child in node.children:
                _walk(child)

        body = func_node.child_by_field_name("body")
        if body:
            _walk(body)
        return sorted(list(found_types))

