"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.config import config
from loguru import logger
from app.api import chat, health, file, aiops
from app.core.milvus_client import milvus_manager
from app.services.vector_store_manager import vector_store_manager


@asynccontextmanager
# FastAPI 应用生命周期管理函数
# 该函数是一个异步上下文管理器，用于处理应用启动和关闭时的资源初始化与清理
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")
    
    # 连接 Milvus
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")
    
    # 初始化向量存储（延迟初始化，需在 Milvus 连接后执行）
    logger.info("📦 正在初始化向量存储...")
    vector_store_manager.ensure_initialized()
    logger.info("✅ 向量存储初始化成功")
    
    logger.info("=" * 60)
    
# 这是一个生成器函数的暂停点
# yield 关键字用于将函数转换为生成器
# 当执行到 yield 时，函数会暂停执行并返回值
# 下次调用生成器的 __next__() 方法时，函数会从暂停点继续执行
    yield
    
    # 关闭时执行
    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")  # 使用FastAPI的装饰器，将root函数注册为处理根路径("/")的GET请求处理函数
async def root():  # 定义一个异步函数root，用于处理根路径的请求
    """返回首页"""  # 函数的文档字符串，说明函数的功能是返回首页
    # 构建首页文件的完整路径，将static_dir和"index.html"拼接起来
    index_path = os.path.join(static_dir, "index.html")
    # 检查首页文件是否存在
    if os.path.exists(index_path):
        # 如果文件存在，则返回该文件作为响应
        return FileResponse(index_path)
    # 如果文件不存在，则返回一个包含API信息的JSON响应
    return {
        "message": f"Welcome to {config.app_name} API",  # 欢迎消息，包含应用名称
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
