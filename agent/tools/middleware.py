from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompt, load_system_prompt


@wrap_tool_call()  # 装饰器，用于包装工具调用函数
def monitor_tool(request:ToolCallRequest,handler:Callable[[ToolCallRequest],ToolMessage|Command]) ->ToolMessage|Command:

    """
    监控工具调用的函数

    参数:
        request: ToolCallRequest类型，包含工具调用的请求信息
        handler: Callable类型，是一个处理函数，接受一个ToolCallRequest参数，返回ToolMessage或Command

    返回:
        ToolMessage|Command: 返回处理结果，可能是工具消息或命令
    """
    logger.info(f"[Tool Monitor]执行: {request.tool_call['name']}")
    logger.info(f"[Tool Monitor]执行: {request.tool_call['args']}")
    try:
        result = handler(request)  # 调用处理函数并返回其结果
        logger.info(f"[Tool Monitor]执行结果: 工具{request.tool_call['name']}执行成功")

        if(request.tool_call['name']=='file_context_for_report'):
            request.runtime.context['report'] = True
        return result
    except Exception as e:
        logger.error(f"[Tool Monitor]执行结果: 工具{request.tool_call['name']}执行失败，错误信息: {str(e)}")
        raise e


@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    logger.info(f"[log_before_model]即将调用模型:带有{len(state['messages'])}条消息")
    if state['messages']:
        logger.debug(f"[log_before_model]即将调用模型:{type(state['messages'][-1]).__name__}|{state['messages'][-1].content.strip()}")
    return None


# 动态提示词装饰器，用于在函数执行时动态修改或生成提示词
@dynamic_prompt
def  report_prompt_switch(request:ModelRequest):
    # 从请求的运行时上下文中获取'report'键的值，如果不存在则默认为False
    is_report = request.runtime.context.get('report',False)
    if is_report:
        return load_report_prompt()
    return  load_system_prompt()


@tool(description="无入参,无返回值,调用后触发中间件自动为报告生成的场景动态注入上下文信息,为后续提示词切换提供上下文信息")
def file_context_for_report() :
    return "file_context_for_report -----已经被调用"