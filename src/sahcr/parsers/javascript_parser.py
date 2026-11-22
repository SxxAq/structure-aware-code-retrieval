"""Tree-sitter parser for JavaScript source files."""

from typing import List, Optional, Set, Tuple
import tree_sitter
import tree_sitter_javascript

from sahcr.models import ExtractedCodeUnit, ParameterInfo
from sahcr.parsers.base_parser import BaseParser
from sahcr.parsers.normalizer import split_identifier, tokenize_text

JS_STRUCTURAL_NODES: Set[str] = {
    "for_statement",
    "for_in_statement",
    "for_of_statement",
    "while_statement",
    "do_statement",
    "if_statement",
    "switch_statement",
    "try_statement",
    "return_statement",
    "throw_statement",
    "call_expression",
    "await_expression",
    "arrow_function",
    "object",
    "array",
    "template_string",
}


class JavaScriptParser(BaseParser):
    """Extracts function boundaries, signatures, JSDocs, and AST cues for JavaScript."""

    def __init__(self):
        super().__init__("javascript", tree_sitter.Language(tree_sitter_javascript.language()))

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
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            class_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8") if name_node else None
            for child in node.children:
                self._traverse_node(child, source_bytes, file_path, current_class=class_name, units=units)
            return

        # Handle function declarations, method definitions, and variable assigned arrow/functions
        if node.type in ("function_declaration", "generator_function_declaration"):
            unit = self._parse_function_declaration(node, source_bytes, file_path, current_class)
            if unit:
                units.append(unit)
        elif node.type == "method_definition":
            unit = self._parse_method_definition(node, source_bytes, file_path, current_class)
            if unit:
                units.append(unit)
        elif node.type == "variable_declarator":
            unit = self._parse_variable_function(node, source_bytes, file_path, current_class)
            if unit:
                units.append(unit)

        for child in node.children:
            self._traverse_node(child, source_bytes, file_path, current_class=current_class, units=units)

    def _parse_function_declaration(
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
        is_async = any(c.type == "async" for c in node.children)
        params_node = node.child_by_field_name("parameters")
        body_node = node.child_by_field_name("body")

        return self._build_unit(
            node=node,
            func_name=func_name,
            params_node=params_node,
            body_node=body_node,
            source_bytes=source_bytes,
            file_path=file_path,
            is_async=is_async,
            is_method=False,
            current_class=current_class
        )

    def _parse_method_definition(
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
        is_async = any(c.type == "async" for c in node.children)
        params_node = node.child_by_field_name("parameters")
        body_node = node.child_by_field_name("body")

        return self._build_unit(
            node=node,
            func_name=func_name,
            params_node=params_node,
            body_node=body_node,
            source_bytes=source_bytes,
            file_path=file_path,
            is_async=is_async,
            is_method=True,
            current_class=current_class
        )

    def _parse_variable_function(
        self,
        node: tree_sitter.Node,
        source_bytes: bytes,
        file_path: str,
        current_class: Optional[str]
    ) -> Optional[ExtractedCodeUnit]:
        name_node = node.child_by_field_name("name")
        value_node = node.child_by_field_name("value")
        if not name_node or not value_node:
            return None

        if value_node.type not in ("arrow_function", "function_expression"):
            return None

        func_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
        is_async = any(c.type == "async" for c in value_node.children)
        params_node = value_node.child_by_field_name("parameters")
        body_node = value_node.child_by_field_name("body")

        # Represent the full declarator or function as unit
        return self._build_unit(
            node=value_node,
            func_name=func_name,
            params_node=params_node,
            body_node=body_node,
            source_bytes=source_bytes,
            file_path=file_path,
            is_async=is_async,
            is_method=False,
            current_class=current_class
        )

    def _build_unit(
        self,
        node: tree_sitter.Node,
        func_name: str,
        params_node: Optional[tree_sitter.Node],
        body_node: Optional[tree_sitter.Node],
        source_bytes: bytes,
        file_path: str,
        is_async: bool,
        is_method: bool,
        current_class: Optional[str]
    ) -> ExtractedCodeUnit:
        # Extract parameters
        parameters: List[ParameterInfo] = []
        param_tokens: List[str] = []
        if params_node:
            idx = 0
            for child in params_node.children:
                if child.type in ("identifier", "formal_parameter", "rest_pattern", "assignment_pattern"):
                    pname = self._get_param_name(child, source_bytes)
                    if pname:
                        parameters.append(ParameterInfo(name=pname, position=idx))
                        param_tokens.extend(split_identifier(pname))
                        idx += 1

        # Extract leading JSDoc / comments
        docstring, docstring_tokens = self._extract_jsdoc(node, source_bytes)

        # Body AST types
        ast_node_types = self._collect_structural_nodes(body_node)

        source_code = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        start_line = node.start_point.row + 1
        end_line = node.end_point.row + 1

        unit_id = f"{file_path}:{func_name}:{start_line}"

        return ExtractedCodeUnit(
            id=unit_id,
            file_path=file_path,
            language="javascript",
            name=func_name,
            normalized_name_tokens=split_identifier(func_name),
            parameters=parameters,
            parameter_tokens=param_tokens,
            return_annotation=None,
            return_tokens=[],
            docstring=docstring,
            docstring_tokens=docstring_tokens,
            source_code=source_code,
            start_line=start_line,
            end_line=end_line,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            ast_node_types=ast_node_types,
            is_async=is_async,
            is_method=is_method,
            parent_class=current_class
        )

    def _get_param_name(self, node: tree_sitter.Node, source_bytes: bytes) -> Optional[str]:
        if node.type == "identifier":
            return source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        for child in node.children:
            if child.type == "identifier":
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
        return None

    def _extract_jsdoc(self, node: tree_sitter.Node, source_bytes: bytes) -> Tuple[Optional[str], List[str]]:
        prev = node.prev_sibling
        if prev and prev.type == "comment":
            raw_comment = source_bytes[prev.start_byte:prev.end_byte].decode("utf-8")
            clean_comment = raw_comment.strip("/* \n\t")
            return clean_comment, tokenize_text(clean_comment)
        return None, []

    def _collect_structural_nodes(self, body_node: Optional[tree_sitter.Node]) -> List[str]:
        if not body_node:
            return []
        found_types: Set[str] = set()

        def _walk(node: tree_sitter.Node):
            if node.type in JS_STRUCTURAL_NODES:
                found_types.add(node.type)
            for child in node.children:
                _walk(child)

        _walk(body_node)
        return sorted(list(found_types))
