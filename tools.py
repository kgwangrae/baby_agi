import ast
import math
import operator
from collections.abc import Callable


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
