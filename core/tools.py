# core/tools.py


from typing import Any, Callable, TypeVar

from core.bridge import Bridge

ToolFn = Callable[
    ...,
    Any,
]  # 给 Tool 函数一个类型 ... 表示输入任意参数 Any 表示返回 Any

F = TypeVar(
    "F",
    bound=ToolFn,
)  # F 代表某个具体的工具函数类型（受 ToolFn 约束）


class ToolRegistry:
    def __init__(self) -> None:
        self.registered_tools: dict[str, tuple[dict[str, Any], ToolFn]] = {}
        # 先创建一个存 registered_tools 的字典

    def register(self, schema: dict[str, Any]) -> Callable[[F], F]:
        """
        装饰器工厂：传入 schema，返回一个真正的装饰器 deco

        Args:
            schema: dict[str, Any] => 工具函数的属性

        Returns:
            Callable[[F], F] => deco 接收什么类型的函数，就原样返回同一类型

        """

        def deco(fn: F) -> F:
            """
            decorator 函数 接受被修饰的函数 将工具函数记录到 registered_tools 字典中

            Args:
                fn: F => 传入的函数

            Returns:
                F => 返回同样类型

            """
            self.registered_tools[schema["function"]["name"]] = (schema, fn)
            return fn

        return deco


bridge = Bridge()
tools = ToolRegistry()


@tools.register(
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
)
def get_time() -> str:
    from datetime import datetime

    return datetime.now().strftime("%H:%M")


@tools.register(
    {
        "type": "function",
        "function": {
            "name": "shake_hand",
            "description": "让机器人伸出手握手",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
)
def shake_hand() -> str:
    try:
        bridge.send_arm(27)
        return "Succeed!"
    except Exception as e:
        return f"{e}"
