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

from typing import Dict, List, Optional, Tuple
import threading


from camel.agents import ChatAgent
from camel.responses import ChatAgentResponse
from camel.messages.base import BaseMessage
from camel.societies import RolePlaying
from camel.logger import get_logger


from copy import deepcopy

logger = get_logger(__name__)


class OwlRolePlaying(RolePlaying):
    def __init__(self, **kwargs):
        # 1. 从参数中提取“用户”与“助手”角色名称
        self.user_role_name = kwargs.get("user_role_name", "user")
        self.assistant_role_name = kwargs.get("assistant_role_name", "assistant")
        # 2. 可选的对话输出语言（如 "zh"、"en"）
        self.output_language = kwargs.get("output_language", None)
        # 3. 传递给 User/Assistant agent 的额外初始化参数
        self.user_agent_kwargs: dict = kwargs.get("user_agent_kwargs", {})
        self.assistant_agent_kwargs: dict = kwargs.get("assistant_agent_kwargs", {})
        # 4. 再次读取输出语言（与上面重复，但保持兼容性）
        self.output_language = kwargs.get("output_language", None)

        super().__init__(**kwargs)
        # 6. 按 GAIA 协议生成用户 & 助手端的 system message
        #    返回值顺序：(user_system_msg, assistant_system_msg)
        init_user_sys_msg, init_assistant_sys_msg = self._construct_gaia_sys_msgs()
        # 7. 提前声明属性类型，便于 IDE 提示和类型检查
        self.assistant_agent: ChatAgent
        self.user_agent: ChatAgent
        self.assistant_sys_msg: Optional[BaseMessage]
        self.user_sys_msg: Optional[BaseMessage]

        # self.is_reasoning_task = self._judge_if_reasoning_task(self.task_prompt)

        # if self.is_reasoning_task:
        #     logger.info("The task is judged as a reasoning or coding task. The assistant agent will use the reasoning model O3-MINI.")
        # else:
        #     logger.info("The assistant agent will use the default model.")
        # 8. 用上面准备好的 system messages 和 kwargs 来初始化两个 ChatAgent
        self._init_agents(
            init_assistant_sys_msg,
            init_user_sys_msg,
            assistant_agent_kwargs=self.assistant_agent_kwargs,
            user_agent_kwargs=self.user_agent_kwargs,
            output_language=self.output_language,
            # is_reasoning_task=self.is_reasoning_task
        )

    def _init_agents(
        self,
        init_assistant_sys_msg: BaseMessage,
        init_user_sys_msg: BaseMessage,
        assistant_agent_kwargs: Optional[Dict] = None,
        user_agent_kwargs: Optional[Dict] = None,
        output_language: Optional[str] = None,
        is_reasoning_task: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        r"""Initialize assistant and user agents with their system messages.

        Args:
            init_assistant_sys_msg (BaseMessage): Assistant agent's initial
                system message.
            init_user_sys_msg (BaseMessage): User agent's initial system
                message.
            assistant_agent_kwargs (Dict, optional): Additional arguments to
                pass to the assistant agent. (default: :obj:`None`)
            user_agent_kwargs (Dict, optional): Additional arguments to
                pass to the user agent. (default: :obj:`None`)
            output_language (str, optional): The language to be output by the
                agents. (default: :obj:`None`)
        """
        if self.model is not None:
            if assistant_agent_kwargs is None:
                assistant_agent_kwargs = {"model": self.model}
            elif "model" not in assistant_agent_kwargs:
                assistant_agent_kwargs.update(dict(model=self.model))
            if user_agent_kwargs is None:
                user_agent_kwargs = {"model": self.model}
            elif "model" not in user_agent_kwargs:
                user_agent_kwargs.update(dict(model=self.model))

        # # If the task is a reasoning task, the assistant agent should use the reasoning model O3-MINI
        # if is_reasoning_task:
        #     assistant_agent_kwargs['model'] = ModelFactory.create(
        #         model_platform=ModelPlatformType.OPENAI,
        #         model_type=ModelType.O3_MINI,
        #     )

        self.assistant_agent = ChatAgent(
            init_assistant_sys_msg,
            output_language=output_language,
            **(assistant_agent_kwargs or {}),
        )
        self.assistant_sys_msg = self.assistant_agent.system_message

        self.user_agent = ChatAgent(
            init_user_sys_msg,
            output_language=output_language,
            **(user_agent_kwargs or {}),
        )
        self.user_sys_msg = self.user_agent.system_message

    # def _judge_if_reasoning_task(self, question: str) -> bool:
    #     r"""Judge if the question is a reasoning task."""

    #     LLM = OpenAIModel(model_type=ModelType.O3_MINI)
    #     prompt = f"""
    #     Please judge whether the following question is a reasoning or coding task, which can be solved by reasoning without leveraging external resources, or is suitable for writing code to solve the task.
    #     If it is a reasoning or coding task, please return only "yes".
    #     If it is not a reasoning or coding task, please return only "no".
    #     Note:
    #     - If the question required some world knowledge to answer the question, please carefully judge it, because the model's own knowledge is often unreliable.
    #     - If it is suitable for writing codes (e.g. process excel files, write simulation codes, etc.), in most cases, it can be considered as a coding task.
    #     Question: <question>{question}</question>
    #     """
    #     messages = [{"role": "user", "content": prompt}]
    #     resp = LLM.run(messages)
    #     if 'yes' in resp.choices[0].message.content.lower():
    #         return True
    #     else:
    #         return False


    """
    构建 GAIA 协议所需的两条系统提示：
    - user_system_prompt: 用户代理发布 Instruction 的格式与规则
    - assistant_system_prompt: 助手代理生成 Solution 的格式与规则
    返回 (user_system_msg, assistant_system_msg)
    """
    def _construct_gaia_sys_msgs(self):
        user_system_prompt = f"""
===== RULES OF USER =====
Never forget you are a user and I am a assistant. Never flip roles! You will always instruct me. We share a common interest in collaborating to successfully complete a task.
I must help you to complete a difficult task.
You must instruct me based on my expertise and your needs to solve the task step by step. The format of your instruction is: `Instruction: [YOUR INSTRUCTION]`, where "Instruction" describes a sub-task or question.
You must give me one instruction at a time.
I must write a response that appropriately solves the requested instruction.
You should instruct me not ask me questions.

Please note that the task may be very complicated. Do not attempt to solve the task by single step. You must instruct me to find the answer step by step.
Here are some tips that will help you to give more valuable instructions about our task to me:
<tips>
- I have various tools to use, such as search toolkit, web browser simulation toolkit, document relevant toolkit, code execution toolkit, etc. Thus, You must think how human will solve the task step-by-step, and give me instructions just like that. For example, one may first use google search to get some initial information and the target url, then retrieve the content of the url, or do some web browser interaction to find the answer.
- Although the task is complex, the answer does exist. If you can't find the answer using the current scheme, try to re-plan and use other ways to find the answer, e.g. using other tools or methods that can achieve similar results.
- Always remind me to verify my final answer about the overall task. This work can be done by using multiple tools(e.g., screenshots, webpage analysis, etc.), or something else.
- If I have written code, please remind me to run the code and get the result.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc. 
- If the question mentions youtube video, in most cases you have to process the content of the mentioned video.
- For downloading files, you can either use the web browser simulation toolkit or write codes (for example, the github content can be downloaded via https://raw.githubusercontent.com/...).
- Flexibly write codes to solve some problems, such as excel relevant tasks.
</tips>

Now, here is the overall task: <task>{self.task_prompt}</task>. Never forget our task!

Now you must start to instruct me to solve the task step-by-step. Do not add anything else other than your instruction!
Keep giving me instructions until you think the task is completed.
When the task is completed, you must only reply with a single word <TASK_DONE>.
Never say <TASK_DONE> unless my responses have solved your task.
        """

        assistant_system_prompt = f"""
===== RULES OF ASSISTANT =====
Never forget you are a assistant and I am a user. Never flip roles! Never instruct me! You have to utilize your available tools to solve the task I assigned.
We share a common interest in collaborating to successfully complete a complex task.
You must help me to complete the task.

Here is our overall task: {self.task_prompt}. Never forget our task!

I must instruct you based on your expertise and my needs to complete the task. An instruction is typically a sub-task or question.

You must leverage your available tools, try your best to solve the problem, and explain your solutions.
Unless I say the task is completed, you should always start with:
Solution: [YOUR_SOLUTION]
[YOUR_SOLUTION] should be specific, including detailed explanations and provide preferable detailed implementations and examples and lists for task-solving.

Please note that our overall task may be very complicated. Here are some tips that may help you solve the task:
<tips>
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
- When trying to solve math problems, you can try to write python code and use sympy library to solve the problem.
- Always verify the accuracy of your final answers! Try cross-checking the answers by other ways. (e.g., screenshots, webpage analysis, etc.).  
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
- After writing codes, do not forget to run the code and get the result. If it encounters an error, try to debug it. Also, bear in mind that the code execution environment does not support interactive input.
- When a tool fails to run, or the code does not run correctly, never assume that it returns the correct result and continue to reason based on the assumption, because the assumed result cannot lead you to the correct answer. The right way is to think about the reason for the error and try again.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc. 
- For downloading files, you can either use the web browser simulation toolkit or write codes.
</tips>

        """

        user_sys_msg = BaseMessage.make_user_message(
            role_name=self.user_role_name, content=user_system_prompt
        )

        assistant_sys_msg = BaseMessage.make_assistant_message(
            role_name=self.assistant_role_name, content=assistant_system_prompt
        )

        return user_sys_msg, assistant_sys_msg

    def step(
        self, assistant_msg: BaseMessage
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        user_response = self.user_agent.step(assistant_msg)
        if user_response.terminated or user_response.msgs is None:
            return (
                ChatAgentResponse(msgs=[], terminated=False, info={}),
                ChatAgentResponse(
                    msgs=[],
                    terminated=user_response.terminated,
                    info=user_response.info,
                ),
            )
        """
        一轮对话推进：
          1. 用户代理处理上一条助手消息 → 生成 Instruction
          2. 在用户消息后拼接辅助提示或收尾提示
          3. 助手代理处理修改后的用户消息 → 生成 Solution
          4. 在助手回复后拼接下一步指令或终止提示
          5. 返回 (assistant_response, user_response)
        """

        # 如果有多个候选，调用 _reduce_message_options（通常会选第一条，或交给 Critic 来裁定）得到唯一的 BaseMessage。
        user_msg = self._reduce_message_options(user_response.msgs)
        # 根据当前用户消息里是否包含 <TASK_DONE>，在消息内容后追加：
        modified_user_msg = deepcopy(user_msg)


        '''
        以下是有关整个任务的辅助信息，可以帮助您理解当前任务的意图： <auxiliary_information> {self.task_prompt} <auxiliary_information> 
        如果有可用的工具并且您想调用它们，切勿说“我会...”，而是首先调用该工具并根据工具调用的结果进行回复，并告诉我您调用了哪个工具。
        '''
        if ("TASK_DONE" not in user_msg.content) and ("任务已完成" not in user_msg.content):
            modified_user_msg.content += f"""\n
            Here are auxiliary information about the overall task, which may help you understand the intent of the current task:
            <auxiliary_information>
            {self.task_prompt}
            </auxiliary_information>
            If there are available tools and you want to call them, never say 'I will ...', but first call the tool and reply based on tool call's result, and tell me which tool you have called.
            """

        else:
            # The task is done, and the assistant agent need to give the final answer about the original task
            # 任务完成，助理代理需要给出关于原始任务的最终答案
            modified_user_msg.content += f"""\n
            Now please make a final answer of the original task based on our conversation : <task>{self.task_prompt}</task>
            """

        # process assistant's response
        assistant_response = self.assistant_agent.step(modified_user_msg)
        if assistant_response.terminated or assistant_response.msgs is None:
            return (
                ChatAgentResponse(
                    msgs=[],
                    terminated=assistant_response.terminated,
                    info=assistant_response.info,
                ),
                ChatAgentResponse(
                    msgs=[user_msg], terminated=False, info=user_response.info
                ),
            )
        assistant_msg = self._reduce_message_options(assistant_response.msgs)

        modified_assistant_msg = deepcopy(assistant_msg)
        '''
        根据我的回答和我们当前的任务，为我提供下一个指示和输入（如果需要）：<task>{self.task_prompt}<task> 在产生最终答案之前，请检查我是否尽可能使用不同的工具包重新检查了最终答案。
        如果没有，请提醒我这样做。如果我编写了代码，请提醒我运行代码。如果您认为我们的任务已完成，请回复“TASK_DONE”以结束我们的对话。
        '''
        if ("TASK_DONE" not in user_msg.content) and ("任务已完成" not in user_msg.content):
            modified_assistant_msg.content += f"""\n
                Provide me with the next instruction and input (if needed) based on my response and our current task: <task>{self.task_prompt}</task>
                Before producing the final answer, please check whether I have rechecked the final answer using different toolkit as much as possible. If not, please remind me to do that.
                If I have written codes, remind me to run the codes.
                If you think our task is done, reply with `TASK_DONE` to end our conversation.
            """

        # return the modified messages
        return (
            ChatAgentResponse(
                msgs=[modified_assistant_msg],
                terminated=assistant_response.terminated,
                info=assistant_response.info,
            ),
            ChatAgentResponse(
                msgs=[modified_user_msg],
                terminated=user_response.terminated,
                info=user_response.info,
            ),
        )

    async def astep(
        self, assistant_msg: BaseMessage
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        user_response = await self.user_agent.astep(assistant_msg)
        if user_response.terminated or user_response.msgs is None:
            return (
                ChatAgentResponse(msgs=[], terminated=False, info={}),
                ChatAgentResponse(
                    msgs=[],
                    terminated=user_response.terminated,
                    info=user_response.info,
                ),
            )
        user_msg = self._reduce_message_options(user_response.msgs)

        modified_user_msg = deepcopy(user_msg)

        if ("TASK_DONE" not in user_msg.content) and ("任务已完成" not in user_msg.content):
            modified_user_msg.content += f"""\n
            Here are auxiliary information about the overall task, which may help you understand the intent of the current task:
            <auxiliary_information>
            {self.task_prompt}
            </auxiliary_information>
            If there are available tools and you want to call them, never say 'I will ...', but first call the tool and reply based on tool call's result, and tell me which tool you have called.
            """

        else:
            # The task is done, and the assistant agent need to give the final answer about the original task
            modified_user_msg.content += f"""\n
            Now please make a final answer of the original task based on our conversation : <task>{self.task_prompt}</task>
            """

        assistant_response = await self.assistant_agent.astep(modified_user_msg)
        if assistant_response.terminated or assistant_response.msgs is None:
            return (
                ChatAgentResponse(
                    msgs=[],
                    terminated=assistant_response.terminated,
                    info=assistant_response.info,
                ),
                ChatAgentResponse(
                    msgs=[user_msg], terminated=False, info=user_response.info
                ),
            )
        assistant_msg = self._reduce_message_options(assistant_response.msgs)

        modified_assistant_msg = deepcopy(assistant_msg)
        if ("TASK_DONE" not in user_msg.content) and ("任务已完成" not in user_msg.content):
            modified_assistant_msg.content += f"""\n
                Provide me with the next instruction and input (if needed) based on my response and our current task: <task>{self.task_prompt}</task>
                Before producing the final answer, please check whether I have rechecked the final answer using different toolkit as much as possible. If not, please remind me to do that.
                If I have written codes, remind me to run the codes.
                If you think our task is done, reply with `TASK_DONE` to end our conversation.
            """

        return (
            ChatAgentResponse(
                msgs=[assistant_msg],
                terminated=assistant_response.terminated,
                info=assistant_response.info,
            ),
            ChatAgentResponse(
                msgs=[user_msg],
                terminated=user_response.terminated,
                info=user_response.info,
            ),
        )


class OwlGAIARolePlaying(OwlRolePlaying):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(
        self, assistant_msg: BaseMessage
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        user_response = self.user_agent.step(assistant_msg)
        if user_response.terminated or user_response.msgs is None:
            return (
                ChatAgentResponse(msgs=[], terminated=False, info={}),
                ChatAgentResponse(
                    msgs=[],
                    terminated=user_response.terminated,
                    info=user_response.info,
                ),
            )
        user_msg = self._reduce_message_options(user_response.msgs)

        modified_user_msg = deepcopy(user_msg)

        if ("TASK_DONE" not in user_msg.content) and ("任务已完成" not in user_msg.content):
            modified_user_msg.content += f"""\n
            Here are auxiliary information about the overall task, which may help you understand the intent of the current task:
            <auxiliary_information>
            {self.task_prompt}
            </auxiliary_information>
            If there are available tools and you want to call them, never say 'I will ...', but first call the tool and reply based on tool call's result, and tell me which tool you have called.
            """

        else:
            # The task is done, and the assistant agent need to give the final answer about the original task
            modified_user_msg.content += f"""\n
            Now please make a final answer of the original task based on our conversation : <task>{self.task_prompt}</task>
            Please pay special attention to the format in which the answer is presented.
            You should first analyze the answer format required by the question and then output the final answer that meets the format requirements. 
            Your response should include the following content:
            - `analysis`: enclosed by <analysis> </analysis>, a detailed analysis of the reasoning result.
            - `final_answer`: enclosed by <final_answer> </final_answer>, the final answer to the question.
            Here are some hint about the final answer:
            <hint>
            Your final answer must be output exactly in the format specified by the question. It should be a number OR as few words as possible OR a comma separated list of numbers and/or strings:
            - If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
            - If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
            - If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
            </hint>
            """

        # process assistant's response
        assistant_response = self.assistant_agent.step(modified_user_msg)
        if assistant_response.terminated or assistant_response.msgs is None:
            return (
                ChatAgentResponse(
                    msgs=[],
                    terminated=assistant_response.terminated,
                    info=assistant_response.info,
                ),
                ChatAgentResponse(
                    msgs=[user_msg], terminated=False, info=user_response.info
                ),
            )
        assistant_msg = self._reduce_message_options(assistant_response.msgs)

        modified_assistant_msg = deepcopy(assistant_msg)
        if ("TASK_DONE" not in user_msg.content) and ("任务已完成" not in user_msg.content):
            modified_assistant_msg.content += f"""\n
                Provide me with the next instruction and input (if needed) based on my response and our current task: <task>{self.task_prompt}</task>
                Before producing the final answer, please check whether I have rechecked the final answer using different toolkit as much as possible. If not, please remind me to do that.
                If I have written codes, remind me to run the codes.
                If you think our task is done, reply with `TASK_DONE` to end our conversation.
            """

        # return the modified messages
        return (
            ChatAgentResponse(
                msgs=[modified_assistant_msg],
                terminated=assistant_response.terminated,
                info=assistant_response.info,
            ),
            ChatAgentResponse(
                msgs=[modified_user_msg],
                terminated=user_response.terminated,
                info=user_response.info,
            ),
        )


# def run_society(
#     society: OwlRolePlaying,
#     round_limit: int = 15,
# ) -> Tuple[str, List[dict], dict]:
#     overall_completion_token_count = 0
#     overall_prompt_token_count = 0
#
#     chat_history = []
#     # 现在请给我指示，逐步解决整个任务。如果任务需要一些特定的知识，请指导我使用工具来完成任务。
#     init_prompt = """
#     Now please give me instructions to solve over overall task step by step. If the task requires some specific knowledge, please instruct me to use tools to complete the task.
#         """
#     # 初始提示，让 User Agent 发布第一条 Instruction
#     input_msg = society.init_chat(init_prompt)
#     for _round in range(round_limit):
#         assistant_response, user_response = society.step(input_msg)
#         # Check if usage info is available before accessing it
#         if assistant_response.info.get("usage") and user_response.info.get("usage"):
#             overall_completion_token_count += assistant_response.info["usage"].get(
#                 "completion_tokens", 0
#             ) + user_response.info["usage"].get("completion_tokens", 0)
#             overall_prompt_token_count += assistant_response.info["usage"].get(
#                 "prompt_tokens", 0
#             ) + user_response.info["usage"].get("prompt_tokens", 0)
#
#         # convert tool call to dict
#         tool_call_records: List[dict] = []
#         if assistant_response.info.get("tool_calls"):
#             for tool_call in assistant_response.info["tool_calls"]:
#                 tool_call_records.append(tool_call.as_dict())
#
#         _data = {
#             "user": user_response.msg.content
#             if hasattr(user_response, "msg") and user_response.msg
#             else "",
#             "assistant": assistant_response.msg.content
#             if hasattr(assistant_response, "msg") and assistant_response.msg
#             else "",
#             "tool_calls": tool_call_records,
#         }
#
#         chat_history.append(_data)
#         logger.info(
#             f"Round #{_round} user_response:\n {user_response.msgs[0].content if user_response.msgs and len(user_response.msgs) > 0 else ''}"
#         )
#         logger.info(
#             f"Round #{_round} assistant_response:\n {assistant_response.msgs[0].content if assistant_response.msgs and len(assistant_response.msgs) > 0 else ''}"
#         )
#         # 终止条件：任一端 returned.terminated 或 user_msg 含 TASK_DONE
#         if (
#             assistant_response.terminated
#             or user_response.terminated
#             or "TASK_DONE" in user_response.msg.content
#         ):
#             break
#
#         input_msg = assistant_response.msg
#
#     answer = chat_history[-1]["assistant"]
#     token_info = {
#         "completion_token_count": overall_completion_token_count,
#         "prompt_token_count": overall_prompt_token_count,
#     }
#     # 返回最后一条 assistant 答案 + 全部 history + token 统计
#     return answer, chat_history, token_info

from typing import List, Tuple
from camel.responses import ChatAgentResponse
import logging

logger = logging.getLogger(__name__)

def run_society(
    society: OwlRolePlaying,
    round_limit: int = 15,
) -> Tuple[str, List[dict], dict]:
    """
    驱动多智能体社会 (OwlRolePlaying) 按照 Instruction–Solution 协议，逐轮协作解决用户任务。

    Args:
        society (OwlRolePlaying): 已初始化的多智能体社会实例，包含 user_agent 和 assistant_agent。
        round_limit (int): 最大协作轮数，默认 15。

    Returns:
        answer (str): 最后一轮助手的回答文本。
        chat_history (List[dict]): 每轮对话的记录列表，元素为包含 user/assistant 文本与工具调用的 dict。
        token_info (dict): 总的 token 消耗信息，包括 prompt_tokens 和 completion_tokens。
    """

    # 1. 初始化全局 Token 计数器
    overall_completion_token_count = 0  # 累计所有 Completion tokens
    overall_prompt_token_count = 0      # 累计所有 Prompt tokens

    # 2. 用于保存逐轮对话历史
    chat_history: List[dict] = []

    # 3. 构造第一条“指令请求”提示，让 User Agent 发布第一步 Instruction
    init_prompt = """
    Now please give me instructions to solve over overall task step by step. If the task requires some specific knowledge, please instruct me to use tools to complete the task.
    """
    # init_chat 会将系统消息和该提示组合，调用 user_agent 生成第一条 Instruction
    input_msg = society.init_chat(init_prompt)

    # 4. 进入协作循环，直到达到轮数上限或检测到终止条件
    for _round in range(round_limit):
        # 4.1 同步调用一轮：先 assistant → 再 user
        assistant_response, user_response = society.step(input_msg)

        # 4.2 累加本轮的 token 使用量（如果可用）
        #    info 字段通常来自 ChatAgentResponse.info["usage"]
        if assistant_response.info.get("usage") and user_response.info.get("usage"):
            overall_completion_token_count += (
                assistant_response.info["usage"].get("completion_tokens", 0)
                + user_response.info["usage"].get("completion_tokens", 0)
            )
            overall_prompt_token_count += (
                assistant_response.info["usage"].get("prompt_tokens", 0)
                + user_response.info["usage"].get("prompt_tokens", 0)
            )

        # 4.3 收集本轮所有助手发起的工具调用
        tool_call_records: List[dict] = []
        if assistant_response.info.get("tool_calls"):
            for tool_call in assistant_response.info["tool_calls"]:
                # 每个 tool_call 都是一个对象，调用 .as_dict() 转成可序列化 dict
                tool_call_records.append(tool_call.as_dict())

        # 4.4 构造本轮对话记录条目
        _data = {
            # 用户代理最新消息文本
            "user": user_response.msg.content
            if hasattr(user_response, "msg") and user_response.msg
            else "",
            # 助手代理最新回复文本
            "assistant": assistant_response.msg.content
            if hasattr(assistant_response, "msg") and assistant_response.msg
            else "",
            # 本轮工具调用列表
            "tool_calls": tool_call_records,
        }
        chat_history.append(_data)

        # 4.5 日志打印，方便后台查看每轮内容
        logger.info(
            f"Round #{_round} user_response:\n"
            f"{user_response.msgs[0].content if user_response.msgs else ''}"
        )
        logger.info(
            f"Round #{_round} assistant_response:\n"
            f"{assistant_response.msgs[0].content if assistant_response.msgs else ''}"
        )

        # 4.6 检查终止条件：
        #     - 如果任一 Smart Agent 标记 terminated=True
        #     - 或者 User Agent 的消息中包含关键词 "TASK_DONE"
        if (
            assistant_response.terminated
            or user_response.terminated
            or "TASK_DONE" in user_response.msg.content
        ):
            break

        # 4.7 为下一轮准备输入：将上一轮助手的消息传给 user_agent
        input_msg = assistant_response.msg

    # 5. 循环结束后，最后一条助手回复即为任务答案
    answer = chat_history[-1]["assistant"]

    # 6. 汇总 Token 使用信息
    token_info = {
        "completion_token_count": overall_completion_token_count,
        "prompt_token_count": overall_prompt_token_count,
    }

    # 7. 返回 (最终答案, 全部对话历史, Token 统计)
    return answer, chat_history, token_info



async def arun_society(
    society: OwlRolePlaying,
    round_limit: int = 15,
) -> Tuple[str, List[dict], dict]:
    """
    异步版的 run_society：在 asyncio 环境中逐轮驱动 OwlRolePlaying 社会协作，
    并返回最终答案、对话历史与 token 消耗统计。
    """

    # 1. 初始化 token 统计变量
    overall_completion_token_count = 0  # 累计所有轮次的 completion_tokens
    overall_prompt_token_count = 0      # 累计所有轮次的 prompt_tokens

    # 2. 用于保存每一轮的对话记录
    chat_history: List[dict] = []

    # 3. 构造初始提示，让 User Agent 发布第一条 Instruction
    init_prompt = """
    Now please give me instructions to solve over overall task step by step. If the task requires some specific knowledge, please instruct me to use tools to complete the task.
    """
    # 调用 society.init_chat()，使用系统消息+init_prompt初始化对话，并得到首条 Assistant 消息
    input_msg = society.init_chat(init_prompt)

    # 4. 进入异步循环，迭代多轮对话
    for _round in range(round_limit):
        # 4.1 异步执行一轮：User Agent + Assistant Agent
        assistant_response, user_response = await society.astep(input_msg)

        # 4.2 如果本轮两端都提供了 usage 信息，则累加到总量
        if assistant_response.info.get("usage") and user_response.info.get("usage"):
            # 注意：这里把 completion_tokens 累到 prompt 统计里，可能是笔误；示例保留原意
            overall_prompt_token_count += assistant_response.info["usage"].get(
                "completion_tokens", 0
            )
            overall_prompt_token_count += assistant_response.info["usage"].get(
                "prompt_tokens", 0
            ) + user_response.info["usage"].get("prompt_tokens", 0)

        # 4.3 提取本轮的工具调用记录（如果有）
        tool_call_records: List[dict] = []
        if assistant_response.info.get("tool_calls"):
            for tool_call in assistant_response.info["tool_calls"]:
                # 将每个 ToolCall 对象转换为 dict，方便序列化和展示
                tool_call_records.append(tool_call.as_dict())

        # 4.4 从响应中读取纯文本内容，防止属性不存在导致报错
        user_content = (
            user_response.msg.content
            if hasattr(user_response, "msg") and user_response.msg
            else ""
        )
        assistant_content = (
            assistant_response.msg.content
            if hasattr(assistant_response, "msg") and assistant_response.msg
            else ""
        )

        # 4.5 构造本轮对话数据，包括用户消息、助手消息及工具调用
        _data = {
            "user": user_content,
            "assistant": assistant_content,
            "tool_calls": tool_call_records,
        }
        chat_history.append(_data)

        # 4.6 日志打印，方便调试和审计
        logger.info(
            f"Round #{_round} user_response:\n"
            f"{user_response.msgs[0].content if user_response.msgs and user_response.msgs[0] else ''}"
        )
        logger.info(
            f"Round #{_round} assistant_response:\n"
            f"{assistant_response.msgs[0].content if assistant_response.msgs and assistant_response.msgs[0] else ''}"
        )

        # 4.7 终止条件：
        #    - 任一 Agent 返回 terminated=True
        #    - 用户回复中包含 "TASK_DONE" 或中文 "任务已完成"
        if (
            assistant_response.terminated
            or user_response.terminated
            or "TASK_DONE" in user_content
            or "任务已完成" in user_content
        ):
            # 跳出循环，不再继续下一轮
            break

        # 4.8 为下一轮准备输入：使用本轮助手的消息继续驱动 User Agent
        input_msg = assistant_response.msg

    # 5. 循环结束后，取最后一轮的助手回复作为最终答案
    answer = chat_history[-1]["assistant"]

    # 6. 组织 token 消耗统计字典
    token_info = {
        "completion_token_count": overall_completion_token_count,
        "prompt_token_count": overall_prompt_token_count,
    }

    # 7. 返回 (最终答案, 对话历史, token 消耗)
    return answer, chat_history, token_info

