import ast
import math
import operator
from collections.abc import Callable


class SafeMathEvaluator(ast.NodeVisitor):
    """Small AST evaluator for deterministic calculator calls."""

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
        if isinstance(node.value, (int, float)):
            return node.value
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
        return self.BINARY_OPERATORS[operator_type](left_value, right_value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operator_type = type(node.op)
        if operator_type not in self.UNARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator_type.__name__}")
        operand = self.visit(node.operand)
        return self.UNARY_OPERATORS[operator_type](operand)

    def visit_Call(self, node: ast.Call) -> float:
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed.")

        function = self.visit(node.func)
        if function not in self.ALLOWED_NAMES.values():
            raise ValueError("Only approved math functions are allowed.")

        arguments = [self.visit(argument) for argument in node.args]
        return function(*arguments)

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate_math(expression: str) -> str:
    """
    Evaluate a small mathematical expression safely.
    Use this tool when precise calculation is needed.
    """
    try:
        parsed_expression = ast.parse(expression, mode="eval")
        result = SafeMathEvaluator().visit(parsed_expression)
        return str(result)
    except Exception as error:
        return f"[Calculator Error] {error}"
