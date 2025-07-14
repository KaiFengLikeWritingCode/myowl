# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# 导入标准库 sys，用于操作 Python 运行时的环境
import sys

# 将上一级目录（"../"）加入到模块搜索路径中，以便能够导入父目录下的包
# （注意：这通常是为了在脚本中能够引用项目根目录中的模块）
sys.path.append("../")

# 导入正则表达式模块，用于在文本中进行模式匹配
import re

# 导入类型提示：Optional[T] 表示返回值可能是 T 类型，也可能是 None
from typing import Optional

# 从 camel 框架中获取日志记录器工厂
from camel.logger import get_logger

# 创建一个模块级的 logger，用于记录警告、错误或调试信息
logger = get_logger(__name__)


def extract_pattern(content: str, pattern: str) -> Optional[str]:
    """
    从输入文本 content 中提取被 <pattern>...</pattern> 包裹的内容。

    Args:
        content (str): 原始文本，可能包含若干 <pattern> 标签。
        pattern (str): 标签名，比如 "analysis"、"final_answer" 等。

    Returns:
        Optional[str]:
            - 如果成功匹配到第一个 <pattern>...</pattern>，返回中间的文本（去除前后空白）。
            - 如果没有匹配到，或在提取过程中发生异常，则返回 None。
    """
    try:
        # 1. 构造正则表达式
        #    rf"...": 原始字符串 + f-string，允许插入变量并保留反斜杠
        #    <{pattern}>        开始标签，例如 <analysis>
        #    (.*?)              非贪婪模式，匹配任意字符，尽可能少地匹配
        #    </{pattern}>       结束标签
        #    re.DOTALL 让 "." 可以匹配换行符，否则只能匹配除换行外的任何字符
        _pattern = rf"<{pattern}>(.*?)</{pattern}>"

        # 2. 执行搜索，返回第一个匹配对象
        match = re.search(_pattern, content, re.DOTALL)

        # 3. 如果找到了匹配
        if match:
            # match.group(1) 对应正则中第一个括号 (.*?)
            text = match.group(1)
            # strip() 去除开头结尾的空白符，返回干净的内容
            return text.strip()
        else:
            # 匹配失败，返回 None
            return None

    except Exception as e:
        # 4. 如果在正则编译或搜索过程中抛出任何异常，
        #    记录一条警告日志，并返回 None 表示未能提取
        logger.warning(f"Error extracting answer: {e}, current content: {content}")
        return None
