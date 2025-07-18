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


# To run this file, you need to configure the DeepSeek API key
# You can obtain your API key from DeepSeek platform: https://platform.deepseek.com/api_keys
# Set it as DEEPSEEK_API_KEY="your-api-key" in your .env file or add it to your environment variables

import sys
from dotenv import load_dotenv
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camel.models import ModelFactory
from camel.toolkits import (
    ExcelToolkit,
    SearchToolkit,
    FileWriteToolkit,
    CodeExecutionToolkit,
)
from camel.types import ModelPlatformType, ModelType
# from camel.societies import RolePlaying
from owl.utils.enhanced_role_playing import OwlRolePlaying, run_society

from camel.logger import set_log_level

from owl.utils import run_society

import pathlib

set_log_level(level="DEBUG")

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))


# def construct_society(question: str) -> RolePlaying:
#     r"""Construct a society of agents based on the given question.
#
#     Args:
#         question (str): The task or question to be addressed by the society.
#
#     Returns:
#         RolePlaying: A configured society of agents ready to address the question.
#     """
#
#     # Create models for different components
#     models = {
#         "user": ModelFactory.create(
#             model_platform=ModelPlatformType.DEEPSEEK,
#             model_type=ModelType.DEEPSEEK_CHAT,
#             # model_type=ModelType.DEEPSEEK_REASONER,
#             model_config_dict={"temperature": 0},
#         ),
#         "assistant": ModelFactory.create(
#             model_platform=ModelPlatformType.DEEPSEEK,
#             model_type=ModelType.DEEPSEEK_CHAT,
#             # model_type=ModelType.DEEPSEEK_REASONER,
#             model_config_dict={"temperature": 0},
#         ),
#     }
#
#     # Configure toolkits
#     tools = [
#         # *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
#         # SearchToolkit().search_duckduckgo,
#         SearchToolkit().search_wiki,
#         # SearchToolkit().search_baidu,
#         # *ExcelToolkit().get_tools(),
#         # *FileWriteToolkit(output_dir="./").get_tools(),
#     ]
#
#     # Configure agent roles and parameters
#     user_agent_kwargs = {"model": models["user"]}
#     assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}
#
#     # Configure task parameters
#     task_kwargs = {
#         "task_prompt": question,
#         "with_task_specify": False,
#     }
#
#     # Create and return the society
#     society = RolePlaying(
#         **task_kwargs,
#         user_role_name="user",
#         user_agent_kwargs=user_agent_kwargs,
#         assistant_role_name="assistant",
#         assistant_agent_kwargs=assistant_agent_kwargs,
#         output_language="Chinese",
#     )
#
#     return society

def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            # model_type=ModelType.DEEPSEEK_REASONER,
            model_config_dict={"temperature": 0},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            # model_type=ModelType.DEEPSEEK_REASONER,
            model_config_dict={"temperature": 0},
        ),
    }

    # Configure toolkits
    tools = [
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        # SearchToolkit().search_duckduckgo,
        SearchToolkit().search_wiki,
        SearchToolkit().search_baidu,
        *ExcelToolkit().get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
    ]

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs
        # output_language="Chinese",
    )

    return society



def main():
    r"""Main function to run the OWL system with an example question."""
    # Example research question
    # default_task = "搜索OWL项目最近的新闻并生成一篇报告，最后保存到本地。"
    default_task = "If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?"

    # Override default task if command line argument is provided
    task = sys.argv[1] if len(sys.argv) > 1 else default_task

    # Construct and run the society
    society = construct_society(task)

    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    main()
