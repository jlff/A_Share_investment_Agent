import os
import time
#from google import genai
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

# 设置日志记录
logger = setup_logger('api_calls')


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 验证环境变量
api_key = os.getenv("API_KEY")
model = os.getenv("API_MODEL")
url=os.getenv("API_URL")

if not api_key:
    logger.error(f"{ERROR_ICON} 未找到 GEMINI_API_KEY 环境变量")
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not model:
    model = "gemini-1.5-flash"
    logger.info(f"{WAIT_ICON} 使用默认模型: {model}")

# # 初始化 Gemini 客户端
# client = genai.Client(api_key=api_key)
# logger.info(f"{SUCCESS_ICON} Gemini 客户端初始化成功")

# 初始化 Gemini 客户端
#client = genai.Client(api_key=api_key)
client = OpenAI(
    api_key=api_key, # 在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
    base_url=url,
)
logger.info(f"{SUCCESS_ICON} KIMI客户端初始化成功")

def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    """获取聊天完成结果，包含重试逻辑"""
    try:
        if model is None:
            model = os.getenv("API_MODEL", "moonshot-v1-8k")

        logger.info(f"{WAIT_ICON} 使用模型: {model}")
        logger.info(f"消息内容: {messages}")

        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model = model,
                    messages = messages,
                    temperature = 0.3,
                )

                if completion is None:
                    logger.warning(
                        f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    return None

                # 转换响应格式
                # chat_message = ChatMessage(content=response.text)
                # chat_choice = ChatChoice(message=chat_message)
                # completion = ChatCompletion(choices=[chat_choice])

                logger.info(f"API 原始响应: {completion.choices[0].message.content}")
                logger.info(f"{SUCCESS_ICON} 成功获取响应")
                return completion.choices[0].message.content

            except Exception as e:
                logger.error(
                    f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                    return None

    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None
