import ast
import json
import math
import operator
import re
from collections.abc import Callable
from typing import Any
import unicodedata


class SafeMathEvaluator(ast.NodeVisitor):
    """Small AST evaluator for deterministic calculator calls."""

    MAX_ABS_RESULT = 1_000_000_000_000
    MAX_POWER_EXPONENT = 100
    MAX_FACTORIAL_INPUT = 20
    MAX_COMBINATORIC_INPUT = 1_000

    BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    ALLOWED_NAMES = {
        name: value
        for name, value in math.__dict__.items()
        if not name.startswith("__") and (callable(value) or isinstance(value, (int, float)))
    }
    ALLOWED_NAMES.update({"abs": abs, "round": round})

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, bool):
            raise ValueError("Boolean values are not allowed.")
        if isinstance(node.value, (int, float)):
            return self._ensure_safe_number(node.value)
        raise ValueError("Only numeric constants are allowed.")

    def visit_Name(self, node: ast.Name) -> object:
        if node.id in self.ALLOWED_NAMES:
            return self.ALLOWED_NAMES[node.id]
        raise NameError(f"Unauthorized name: {node.id}")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        operator_type = type(node.op)
        if operator_type not in self.BINARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator_type.__name__}")
        left_value = self.visit(node.left)
        right_value = self.visit(node.right)
        if operator_type is ast.Pow and abs(float(right_value)) > self.MAX_POWER_EXPONENT:
            raise ValueError("Exponent is too large for the safe calculator.")
        return self._ensure_safe_number(self.BINARY_OPERATORS[operator_type](left_value, right_value))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operator_type = type(node.op)
        if operator_type not in self.UNARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator_type.__name__}")
        operand = self.visit(node.operand)
        return self._ensure_safe_number(self.UNARY_OPERATORS[operator_type](operand))

    def visit_Call(self, node: ast.Call) -> float:
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed.")
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct math function calls are allowed.")

        function_name = node.func.id
        function = self.visit(node.func)
        if function not in self.ALLOWED_NAMES.values():
            raise ValueError("Only approved math functions are allowed.")

        arguments = [self.visit(argument) for argument in node.args]
        self._validate_function_call(function_name, arguments)
        return self._ensure_safe_number(function(*arguments))

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    def _validate_function_call(self, function_name: str, arguments: list[float]) -> None:
        if function_name == "pow" and len(arguments) >= 2 and abs(float(arguments[1])) > self.MAX_POWER_EXPONENT:
            raise ValueError("Exponent is too large for the safe calculator.")
        if function_name == "factorial" and arguments and arguments[0] > self.MAX_FACTORIAL_INPUT:
            raise ValueError("factorial input is too large for the safe calculator.")
        if function_name in {"comb", "perm"} and any(argument > self.MAX_COMBINATORIC_INPUT for argument in arguments):
            raise ValueError(f"{function_name} input is too large for the safe calculator.")

    def _ensure_safe_number(self, value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("Calculator result must be numeric.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Calculator result is not finite.")
        if abs(value) > self.MAX_ABS_RESULT:
            raise ValueError("Calculator result is too large.")
        return value


def calculate_math(expression: str) -> str:
    """
    Evaluate a small mathematical expression safely.
    Use this tool when precise calculation is needed.
    """
    try:
        clean_expression = expression.strip()
        if not clean_expression:
            raise ValueError("Expression is empty.")
        if len(clean_expression) > 160:
            raise ValueError("Expression is too long for the safe calculator.")

        parsed_expression = ast.parse(clean_expression, mode="eval")
        if sum(1 for _ in ast.walk(parsed_expression)) > 80:
            raise ValueError("Expression is too complex for the safe calculator.")

        result = SafeMathEvaluator().visit(parsed_expression)
        return str(result)
    except Exception as error:
        return f"[Calculator Error] {error}"

FORBIDDEN_HAN_PATTERN = r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]"

CJK_PUNCTUATION_TRANSLATION = str.maketrans({
    "，": ",",
    "。": ".",
    "？": "?",
    "！": "!",
    "：": ":",
    "；": ";",
    "（": "(",
    "）": ")",
    "［": "[",
    "］": "]",
    "｛": "{",
    "｝": "}",
    "、": ",",
    "“": "\"",
    "”": "\"",
    "‘": "'",
    "’": "'",
})

def normalize_model_punctuation(text: str) -> str:
    """모델이 흘린 전각/호환 문자와 CJK 문장부호를 일반 표기로 정리합니다."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    return normalized.translate(CJK_PUNCTUATION_TRANSLATION)

def has_forbidden_han(text: str) -> bool:
    """중국어/한자 계열 Han 문자를 감지합니다."""
    clean_text = normalize_model_punctuation(text)
    return bool(re.search(FORBIDDEN_HAN_PATTERN, clean_text))

def remove_forbidden_han(text: str) -> str:
    clean_text = normalize_model_punctuation(text)
    return re.sub(FORBIDDEN_HAN_PATTERN, "", clean_text)

def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a model-emitted JSON object with light tail repair."""
    clean_text = normalize_model_punctuation(str(text or "").strip())
    if not clean_text:
        return None

    clean_text = re.sub(r"<\|.*?\|>", "", clean_text)
    clean_text = re.sub(r"```(?:json|xml)?", "", clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.replace("```", "").strip()

    candidates = [clean_text]

    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start != -1:
        if end != -1 and end > start:
            candidates.append(clean_text[start:end + 1])
        candidates.append(_close_json_tail(clean_text[start:]))

    candidates.append(_close_json_tail(clean_text))

    for candidate in dict.fromkeys(candidate for candidate in candidates if candidate):
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            continue

    return None


def _close_json_tail(text: str) -> str:
    clean_text = re.sub(r",\s*$", "", str(text or "").strip())
    stack: list[str] = []
    in_string = False
    escaped = False

    for character in clean_text:
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character in "[{":
            stack.append(character)
        elif character in "]}":
            if stack and (
                (stack[-1] == "[" and character == "]")
                or (stack[-1] == "{" and character == "}")
            ):
                stack.pop()

    if in_string:
        clean_text += '"'

    closing_pairs = {"[": "]", "{": "}"}
    while stack:
        clean_text += closing_pairs[stack.pop()]

    return clean_text


def write_fact(notepad: Any, key: str, value: str) -> str:
    """Write a deterministic fact through the guarded tool layer."""
    clean_key = str(key or "").strip()
    clean_value = str(value or "").strip()

    if not clean_key or not clean_value:
        raise ValueError("write_fact expects key/value")

    if notepad.add_fact(clean_key, clean_value):
        return f"fact saved: {clean_key}"

    raise ValueError("invalid fact")
