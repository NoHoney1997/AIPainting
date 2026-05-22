"""
AI共创叙事画坊 - Streamlit 应用
一个结合AI对话与图像生成的互动叙事平台
"""
import streamlit as st
import os
import json
import time
import re
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any, List, Optional
from openai import OpenAI
from image_gen import generate_portrait, generate_comic_frame, extract_character_features, get_llm_client

# =============================================================================
# 配置
# =============================================================================
llm_client = get_llm_client()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ensure_dir = lambda path: os.makedirs(path, exist_ok=True)

# 确保目录存在
ensure_dir(DATA_DIR)

def get_session_dir(session_id: str = None) -> str:
    """获取会话目录路径"""
    if session_id is None:
        session_id = st.session_state.session_id
    return os.path.join(DATA_DIR, "sessions", session_id)

def get_images_dir(session_id: str = None) -> str:
    """获取会话图像目录路径"""
    return os.path.join(get_session_dir(session_id), "images")

def ensure_session_dirs(session_id: str = None):
    """确保会话相关目录存在"""
    ensure_dir(get_session_dir(session_id))
    ensure_dir(get_images_dir(session_id))

# =============================================================================
# System Prompt
# =============================================================================
SYSTEM_PROMPT = """你是AI共创叙事画坊的创作助手，引导用户完成角色创作体验。

【角色】友善、专业的创作引导者，像有经验的编剧指导。

【核心规则】
1. 所有对话围绕虚构角色。永远不询问用户本人经历或感受。
2. 永远不评价用户产出（不说"很好""有意思""不错""有趣"）。
3. 永远不引导用户在现实中"变得更好""调整心态""解决问题"。
4. 禁用词汇：心理、情绪、自我、评估、测量、同情、友善、批评、困境、帮助、改善。

【对话风格】
- 用你自己的话引导，不要机械重复
- 每次先承接用户内容，再自然引出下一个问题
- 像编剧讨论角色设定，不像问卷
- 每次只问一个问题
- 用户回答简短就接受

【当前阶段目标】
系统会告诉你当前阶段要达成什么目标。根据对话历史自然引导"""

# =============================================================================
# 阶段定义
# =============================================================================
STAGES = [
    "S0a", "S0b", "S0c", "S0d",  # 角色创建阶段
    "A1a", "A1b", "A1c", "A1d",  # 学业故事
    "A2a", "A2b", "A2c",         # 内心独白
    "A3a", "A3b", "A3c",         # 学业漫画
    "B1a", "B1b", "B1c", "B1d",  # 人际故事
    "B2a", "B2b", "B2c",         # 人际内心
    "B3a", "B3b", "B3c",         # 人际漫画
    "P4A_REWRITE", "P4A_DIFF",   # 学业视角切换
    "P4B_REWRITE", "P4B_DIFF",    # 人际视角切换
    "DEBRIEF",                    # 总结
    "DONE"                        # 完成
]

# 阶段中文名称
STAGE_NAMES = {
    "S0a": "角色命名", "S0b": "角色外貌", "S0c": "生成画像", "S0d": "确认形象",
    "A1a": "学业困扰", "A1b": "具体情境", "A1c": "角色反应", "A1d": "后续影响",
    "A2a": "内心独白", "A2b": "声音来源", "A2c": "困难认知",
    "A3a": "挑选瞬间", "A3b": "生成漫画", "A3c": "确认漫画",
    "B1a": "人际困扰", "B1b": "具体情境", "B1c": "角色反应", "B1d": "后续影响",
    "B2a": "内心独白", "B2b": "声音来源", "B2c": "困难认知",
    "B3a": "挑选瞬间", "B3b": "生成漫画", "B3c": "确认漫画",
    "P4A_REWRITE": "视角改写", "P4A_DIFF": "视角对比",
    "P4B_REWRITE": "视角改写", "P4B_DIFF": "视角对比",
    "DEBRIEF": "创作回顾", "DONE": "完成"
}

# 阶段目标
STAGE_GOALS = {
    "S0a": "引导用户为虚构大学生角色命名、设定专业和年级",
    "S0b": "引导用户描述角色的外貌特征",
    "S0c": "询问用户想要的画像风格，然后生成画像",
    "S0d": "简短确认角色形象，过渡到学业故事",
    "A1a": "引导用户描述角色在学业方面遇到的困扰事件",
    "A1b": "让故事更具体——追问场合、在场人物、具体发生了什么",
    "A1c": "追问角色当时的行为反应——做了什么或没做什么",
    "A1d": "追问事件对角色后续的影响——心情/作息/人际关系变化",
    "A2a": "引导用户写出角色内心的第一反应原话",
    "A2b": "引导思考两个方向：声音来源和普遍性",
    "A2c": "引导思考角色如何框定这个困难",
    "A3a": "回顾之前的故事素材，引导用户挑选3-5个关键瞬间",
    "A3b": "由代码逐格生成连环画，不通过LLM",
    "A3c": "简短确认，过渡到人际故事",
    "B1a": "引导用户描述角色在人际方面遇到的困扰",
    "B1b": "让故事更具体",
    "B1c": "追问角色当时的行为反应",
    "B1d": "追问事件对角色后续的影响",
    "B2a": "引导用户写出角色内心的第一反应原话",
    "B2b": "引导思考两个方向",
    "B2c": "引导思考角色如何框定这个困难",
    "B3a": "回顾之前的故事素材，引导用户挑选3-5个关键瞬间",
    "B3b": "由代码逐格生成连环画",
    "B3c": "简短确认，过渡到视角练习",
    "P4A_REWRITE": "引入编剧视角切换概念，引导用户用第一人称改写学业故事内心独白",
    "P4A_DIFF": "展示旁观视角和沉浸视角两个版本对比，引导讨论差异",
    "P4B_REWRITE": "引导用户用第一人称改写人际故事内心独白",
    "P4B_DIFF": "展示两个版本对比，引导讨论差异",
    "DEBRIEF": "引导用户回顾创作体验",
    "DONE": "感谢页"
}

# 阶段提示参考
STAGE_REFS = {
    "S0a": "今天一起创作一个角色——先认识一下TA。请构思一个虚构的大学生，TA叫什么？性别是？学什么？大几？",
    "S0b": "让{name}更具体——TA长什么样？发型、脸型、眉眼、身材、穿衣风格？整体气质偏活泼还是沉静？",
    "S0c": "好的，{name}的形象很清晰了。接下来我来生成TA的三视图画像。你想用什么风格？比如：写实、动漫、水彩、素描、油画、国风，或者你喜欢的其他风格都可以告诉我。",
    "S0d": "这就是{name}了——看起来是个有故事的年轻人。接下来我们一起来探索TA的某个故事。",
    "A1a": "大学里都会遇到各种学业状况——{name}在学业方面有没有什么让TA感到困扰或沮丧的事？可以是某次具体的事情，也可以是TA一直以来的某种处境或状态。",
    "A1b": "能再说具体一点吗？",
    "A1c": "当时{name}是什么反应？做了什么？或者有什么想做的但没做？",
    "A1d": "这件事之后对{name}有什么影响？心情、作息、和他人相处有什么变化？",
    "A2a": "故事很具体了。好的创作还要揣摩角色的内心——这件事发生时，{name}心里最先冒出来的那句话是什么？最直觉的第一反应。原汁原味写下来，包括语气和用词。",
    "A2b": "{name}的这个反应——更像TA自己的声音，还是像某个重要的人可能对TA说的话？TA会觉得只有自己才会这样，还是谁都可能遇到？",
    "A2c": "在{name}看来，这个困难说明了什么？一次偶然——还是暴露了TA一直以来的问题？TA是被这感觉困住，还是能意识到'我正在经历困难'？",
    "A3a": "刚才描述了{name}遇到的{summary}。现在用AI生成连环画。挑3到5个关键瞬间，每格描述{name}的状态和感受。不用画结局。",
    "A3b": "正在生成连环画...",
    "A3c": "学业故事的连环画完成了。接下来我们来探索{name}在人际关系方面的故事。",
    "B1a": "除了学业，大学里人际关系也很重要——{name}在人际方面有没有什么困扰？不管是某次具体事件，还是一直以来让TA不太舒服的状态或处境。",
    "B1b": "能再说具体一点吗？",
    "B1c": "当时{name}是什么反应？做了什么？或者有什么想做的但没做？",
    "B1d": "这件事之后对{name}有什么影响？心情、作息、和他人相处有什么变化？",
    "B2a": "故事很具体了。现在来揣摩角色的内心——这件事发生时，{name}心里最先冒出来的那句话是什么？",
    "B2b": "{name}的这个反应——更像TA自己的声音，还是像某个重要的人可能对TA说的话？TA会觉得只有自己才会这样，还是谁都可能遇到？",
    "B2c": "在{name}看来，这个困难说明了什么？",
    "B3a": "刚才描述了{name}遇到的{summary}。现在挑3到5个关键瞬间来生成连环画。",
    "B3b": "正在生成连环画...",
    "B3c": "人际故事的连环画也完成了。接下来我们来做一个小小的编剧练习。",
    "P4A_REWRITE": "来做编剧的进阶练习——写独白时编剧会试不同视角：旁观视角（第三人称）和沉浸视角（第一人称）。刚才写的是旁观视角，现在试沉浸视角——完全代入{name}，'我'就是TA。回到那个时刻，'我'的内心独白是什么？这是旁观版本：{quote}请用第一人称改写。",
    "P4A_DIFF": "两个版本有什么不同？为什么会有这些差异？",
    "P4B_REWRITE": "同样的练习——现在试沉浸视角，完全代入{name}，用'我'来写。这是旁观版本：{quote}请用第一人称改写。",
    "P4B_DIFF": "两个版本有什么不同？为什么会有这些差异？",
    "DEBRIEF": "创作完成了！想听听你的感受——整体体验怎么样？有没有哪个瞬间觉得这不只是在创作角色？你觉得背后想探索什么？",
    "DONE": "感谢你的参与和创作！"
}

# 追问提示
FOLLOW_UP_REFS = {
    "S0a": "能再多透露一点吗？比如性格特点或者TA有什么特别的地方？",
    "S0b": "能再描述得具体一些吗？比如常见的表情或者走路的样子？",
    "A1a": "嗯，这件事听起来确实让人困扰。还有什么细节吗？",
    "A1b": "理解。那之后呢？事情是怎么发展的？",
    "A1c": "嗯，当时应该挺不容易的。还有什么想补充的吗？",
    "A1d": "理解了。这种影响持续了多久？",
    "A2a": "嗯，这个反应很真实。还有别的念头闪过吗？",
    "A2b": "嗯，这个观察很有意思。还有什么补充的吗？",
    "A2c": "嗯，这种理解很重要。还有什么想法吗？",
    "A3a": "好，每个瞬间都很有画面感。还有想加的瞬间吗？",
    "B1a": "嗯，人际的困扰往往更复杂。还有什么想说的吗？",
    "B1b": "理解。那后来呢？",
    "B1c": "嗯，当时应该挺为难的。还有什么想说的吗？",
    "B1d": "理解了。这种影响持续了多久？",
    "B2a": "嗯。还有别的念头闪过吗？",
    "B2b": "嗯。还有什么补充的吗？",
    "B2c": "嗯。还有什么想法吗？",
    "B3a": "好。还有想加的瞬间吗？"
}

# =============================================================================
# LLM 调用函数
# =============================================================================
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """调用LLM生成回复"""
    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen-max"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"LLM调用失败: {e}")
        return ""


# =============================================================================
# 状态/事件判断函数（使用LLM）
# =============================================================================

def is_state_description(user_input: str, context: Dict[str, Any] = None) -> bool:
    """
    使用 LLM 判断用户描述的是具体事件还是日常状态

    Returns:
        True: 日常状态（总是、经常、一直、平时...）
        False: 具体事件（某一次、那天、当时...）
    """
    if not user_input or len(user_input.strip()) < 5:
        return False

    # 使用 st.session_state 缓存
    if "state_cache" not in st.session_state:
        st.session_state.state_cache = {}

    cache_key = user_input[:200]

    if cache_key in st.session_state.state_cache:
        return st.session_state.state_cache[cache_key]

    prompt = f"""判断以下用户描述是属于"具体事件"还是"日常状态"。

用户描述：{user_input}

判断标准：
- 具体事件：描述的是某一次、特定时刻发生的事情。关键词：那天、有一次、当时、那次、某天、有一天、记得有一次等。
- 日常状态：描述的是反复出现的模式、持续的状态、一般性情况。关键词：总是、经常、一直、每次、平时、日常、一般、习惯、状态、处境、常常、往往、持续等。

只返回 JSON：{{"type": "event"}} 或 {{"type": "state"}}"""

    system_prompt = "分类助手。只返回JSON，不添加其他内容。"

    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen-max"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )

        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'```json\s*|```', '', result_text)
        result = json.loads(result_text)

        is_state = result.get("type") == "state"
        st.session_state.state_cache[cache_key] = is_state
        return is_state
    except:
        # 降级到关键词判断
        state_keywords = ["总是", "经常", "一直", "每次", "平时", "日常", "习惯", "状态", "处境", "一般", "常常", "往往", "持续", "长期", "反复"]
        is_state = any(kw in user_input for kw in state_keywords)
        st.session_state.state_cache[cache_key] = is_state
        return is_state


def match_style_with_llm(user_input: str, styles: List[Dict]) -> Optional[str]:
    """
    使用 LLM 解析用户输入的风格描述，匹配到预定义的风格

    Returns:
        匹配到的风格key，如果没有匹配则返回 None
    """
    if not styles:
        return None

    style_list = "\n".join([f"- {s['name']} (key: {s['key']})" for s in styles])

    prompt = f"""根据用户的输入，匹配最合适的画像风格。

可用的风格：
{style_list}

用户输入：{user_input}

要求：
1. 如果用户明确提到某个风格名称，直接匹配
2. 如果用户用描述性语言（如"像二次元""像照片一样""像油画"），匹配最接近的风格
3. 如果无法匹配任何风格，返回 null

只返回 JSON：{{"matched_style": "风格key"}} 或 {{"matched_style": null}}"""

    system_prompt = "风格匹配助手。只返回JSON，不添加其他内容。"

    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen-max"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )

        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'```json\s*|```', '', result_text)
        result = json.loads(result_text)

        matched = result.get("matched_style")
        return matched if matched and matched != "null" else None
    except:
        # 降级到关键词匹配
        user_lower = user_input.lower()
        for style in styles:
            if style["name"] in user_input or style["key"] in user_lower:
                return style["key"]
        return None


def generate_context_aware_follow_up(stage: str, user_input: str, context: Dict[str, Any]) -> str:
    """
    使用 LLM 根据用户描述的类型（事件/状态）生成个性化的追问
    """
    name = context.get("name", st.session_state.get("character_name", "TA"))
    is_state = is_state_description(user_input, context)

    if is_state:
        follow_type = "状态"
        hint = "追问这种状态通常在什么情境下出现，或者有没有一个典型的例子"
    else:
        follow_type = "事件"
        hint = "追问具体的时间、地点、人物和细节"

    prompt = f"""根据用户的回答，生成一句自然的追问。

当前阶段：{stage}
角色名称：{name}
用户刚刚说：{user_input}

用户描述的类型：{follow_type}（{"日常状态" if is_state else "具体事件"}）

要求：
1. 先简短承接用户的内容（不要评价）
2. 自然引出下一步信息
3. {hint}
4. 一句话，不要超过30字
5. 不要使用"很好""有意思""不错"等评价词
6. 语气像编剧在讨论角色

只返回追问内容，不要其他。"""

    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen-max"),
            messages=[
                {"role": "system", "content": "你是叙事引导助手。直接返回追问，不要评价，不要加引号。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        follow_up = response.choices[0].message.content.strip()
        follow_up = follow_up.strip('"\'')
        return follow_up
    except:
        if is_state:
            return f"{name}在这种状态下，通常在什么场合会更明显？"
        else:
            return f"能再具体一点吗？当时是什么场合？"


# =============================================================================
# LLM 智能解析和追问函数
# =============================================================================

def parse_user_input_with_llm(stage: str, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """使用LLM解析用户输入，提取当前阶段需要存储的数据"""

    extraction_prompts = {
        "S0a": """从用户输入提取：name(姓名), gender(female/male), major(专业), grade(年级)。只提取明确提到的，没有就留空。
用户输入：{input}
返回JSON：{{"name": "", "gender": "", "major": "", "grade": ""}}""",

        "S0b": """提取用户描述的外貌特征，保持原意不添加。用户输入：{input}
返回JSON：{{"appearance": ""}}""",

        "A1a": """提取核心困扰。返回JSON：{{"dilemma": ""}}""",
        "A1b": """提取场合、人物、细节。返回JSON：{{"context": ""}}""",
        "A1c": """提取行为反应。返回JSON：{{"reaction": ""}}""",
        "A1d": """提取影响。返回JSON：{{"impact": ""}}""",
        "A2a": """完整保留内心独白原话。返回JSON：{{"quote": ""}}""",
        "A2b": """提取关键洞察。返回JSON：{{"reflection": ""}}""",
        "A2c": """提取对困难的认知。返回JSON：{{"framing": ""}}""",
        "B1a": """提取核心困扰。返回JSON：{{"dilemma": ""}}""",
        "B1b": """提取场合、人物、细节。返回JSON：{{"context": ""}}""",
        "B1c": """提取行为反应。返回JSON：{{"reaction": ""}}""",
        "B1d": """提取影响。返回JSON：{{"impact": ""}}""",
        "B2a": """完整保留内心独白。返回JSON：{{"quote": ""}}""",
        "B2b": """提取洞察。返回JSON：{{"reflection": ""}}""",
        "B2c": """提取认知。返回JSON：{{"framing": ""}}""",
        "DEBRIEF": """提取反馈。返回JSON：{{"response": ""}}""",
    }

    prompt_template = extraction_prompts.get(stage)
    if not prompt_template:
        return {}

    user_prompt = prompt_template.replace("{input}", user_input)

    system_prompt = """数据提取助手。只提取明确提到的内容，不添加。只返回JSON。"""

    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen-max"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )

        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'```json\s*|```', '', result_text)
        return json.loads(result_text)
    except:
        return {}

def generate_guide_message(stage: str, context: Dict[str, Any] = None) -> str:
    """根据当前阶段生成引导语"""
    context = context or {}

    # 获取变量值
    name = context.get("name", st.session_state.get("character_name", "TA"))
    summary = context.get("summary", st.session_state.get("summary", ""))
    quote = context.get("quote", st.session_state.get("quote", ""))

    # 从 STAGE_REFS 获取引导语
    ref = STAGE_REFS.get(stage, "请继续。")
    # 替换变量
    ref = ref.replace("{name}", name)
    ref = ref.replace("{summary}", summary)
    ref = ref.replace("{quote}", quote)

    return ref


def should_follow_up(stage: str, user_input: str) -> bool:
    """判断是否需要追问 - 使用LLM智能判断"""
    if stage in ["S0b", "S0c", "A3a", "A3b", "B3a", "B3b"]:
        return False

    if len(user_input.strip()) < 10:
        simple_words = ["是", "否", "是的", "不是", "对", "嗯", "哦", "好", "可以", "行"]
        if user_input.strip() in simple_words:
            return True

    return False

def generate_follow_up(stage: str, conversation_history: List[Dict]) -> str:
    """生成自然的追问"""
    last_user_msg = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    name = st.session_state.get("character_name", "TA")
    context = {"name": name, "stage": stage}

    return generate_context_aware_follow_up(stage, last_user_msg, context)

def _get_stage_content_from_history(target_stage: str) -> str:
    """从对话历史中获取指定阶段的用户输入内容"""
    messages = st.session_state.get("messages", [])

    # 定义阶段关键词映射
    stage_keywords = {
        "A2a": ["内心独白", "心里最先冒出来", "第一反应"],
        "A2b": ["声音来源", "普遍性", "只有自己"],
        "A2c": ["困难说明了", "偶然", "问题"],
        "B2a": ["内心独白", "心里最先冒出来", "第一反应"],
        "B2b": ["声音来源", "普遍性", "只有自己"],
        "B2c": ["困难说明了", "偶然", "问题"]
    }

    keywords = stage_keywords.get(target_stage, [])
    found_target = False

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # 检查是否是目标阶段的引导语
            for kw in keywords:
                if kw in content:
                    found_target = True
                    break
        elif msg.get("role") == "user" and found_target:
            # 返回用户对这个问题的回答
            return msg.get("content", "")

    return ""

# =============================================================================
# 数据存储函数
# =============================================================================
def store_data(stage: str, user_input: str):
    """使用LLM存储用户输入的数据"""

    context = {
        "character_name": st.session_state.get("character_name", ""),
        "character_gender": st.session_state.get("character_gender", "")
    }

    extracted = parse_user_input_with_llm(stage, user_input, context)
    if not extracted:
        return

    if stage == "S0a":
        if extracted.get("name"):
            st.session_state.character_name = extracted["name"]
        if extracted.get("gender"):
            st.session_state.character_gender = extracted["gender"]
        if extracted.get("major"):
            st.session_state.character_major = extracted["major"]
        if extracted.get("grade"):
            st.session_state.character_grade = extracted["grade"]

    elif stage == "S0b" and extracted.get("appearance"):
        st.session_state.character_appearance = extracted["appearance"]

    elif stage == "A1a":
        st.session_state.stage_A_material["dilemma"] = extracted.get("dilemma", user_input)
    elif stage == "A1b":
        st.session_state.stage_A_material["context"] = extracted.get("context", user_input)
    elif stage == "A1c":
        st.session_state.stage_A_material["reaction"] = extracted.get("reaction", user_input)
    elif stage == "A1d":
        st.session_state.stage_A_material["impact"] = extracted.get("impact", user_input)
    elif stage == "A2a":
        st.session_state.stage_A2a_quote = extracted.get("quote", user_input)
        st.session_state.stage_A_quote_timestamp = time.time()
    elif stage == "A2b":
        st.session_state.stage_A2b_reflection = extracted.get("reflection", user_input)
    elif stage == "A2c":
        st.session_state.stage_A2c_framing = extracted.get("framing", user_input)

    elif stage == "B1a":
        st.session_state.stage_B_material["dilemma"] = extracted.get("dilemma", user_input)
    elif stage == "B1b":
        st.session_state.stage_B_material["context"] = extracted.get("context", user_input)
    elif stage == "B1c":
        st.session_state.stage_B_material["reaction"] = extracted.get("reaction", user_input)
    elif stage == "B1d":
        st.session_state.stage_B_material["impact"] = extracted.get("impact", user_input)
    elif stage == "B2a":
        st.session_state.stage_B2a_quote = extracted.get("quote", user_input)
        st.session_state.stage_B_quote_timestamp = time.time()
    elif stage == "B2b":
        st.session_state.stage_B2b_reflection = extracted.get("reflection", user_input)
    elif stage == "B2c":
        st.session_state.stage_B2c_framing = extracted.get("framing", user_input)
    elif stage == "DEBRIEF":
        st.session_state.debrief_response = extracted.get("response", user_input)

def parse_comic_frames(user_input: str, expected_min: int = 3, expected_max: int = 5) -> List[Dict[str, str]]:
    """解析用户输入的漫画分镜描述
    
    Args:
        user_input: 用户输入的描述
        expected_min: 期望的最少分镜数
        expected_max: 期望的最多分镜数
        
    Returns:
        分镜列表，每个分镜包含 description 和 frame_number
    """
    frames = []
    lines = [l.strip() for l in user_input.strip().split("\n") if l.strip()]
    
    if not lines:
        return []
    
    # 尝试多种方式解析
    
    # 方式1：按段落分割（每个段落一格）
    for para in user_input.strip().split("\n\n"):
        para = para.strip()
        if para:
            # 提取段落中的描述（可能包含序号）
            content = para
            # 移除可能的序号前缀
            for pattern in [r"^第?\s*[\d一二三四五]+[格张个场]\s*[：:、.。]?\s*", 
                           r"^[\d]+[.、]\s*"]:
                match = re.match(pattern, content)
                if match:
                    content = content[match.end():].strip()
                    break
            if content:
                frames.append({"description": content})
    
    # 如果方式1只得到1格，尝试按句子分割
    if len(frames) == 1 and len(lines) > 1:
        frames = []
        # 方式2：每行一格（如果行数在合理范围内）
        if 2 <= len(lines) <= 8:
            for line in lines:
                content = line
                for pattern in [r"^第?\s*[\d一二三四五]+[格张个场]\s*[：:、.。]?\s*", 
                               r"^[\d]+[.、]\s*"]:
                    match = re.match(pattern, content)
                    if match:
                        content = content[match.end():].strip()
                        break
                if content:
                    frames.append({"description": content})
    
    # 如果方式2失败，尝试方式3：按句号/分号分割
    if len(frames) < 2:
        # 按句号、分号或换行符分割
        sentences = re.split(r'[。;；\n]+', user_input)
        frames = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 5:  # 过滤太短的片段
                # 移除序号
                content = sent
                for pattern in [r"^第?\s*[\d一二三四五]+[格张个场]\s*[：:、.。]?\s*", 
                               r"^[\d]+[.、]\s*"]:
                    match = re.match(pattern, content)
                    if match:
                        content = content[match.end():].strip()
                        break
                if content:
                    frames.append({"description": content})
    
    # 限制数量在合理范围内
    if len(frames) > expected_max:
        frames = frames[:expected_max]
    elif len(frames) < expected_min:
        # 如果解析出的分镜太少，合并短片段或添加空占位
        while len(frames) < expected_min:
            frames.append({"description": ""})
    
    # 重新编号
    for i, frame in enumerate(frames):
        frame["frame_number"] = i + 1
    
    return frames

def log_event(event_type: str, **data):
    """记录事件到日志"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "stage": STAGES[st.session_state.stage_index],
        **data
    }
    st.session_state.log_entries.append(entry)

def save_session():
    """保存完整会话数据"""
    session_id = st.session_state.session_id
    session_dir = get_session_dir(session_id)
    ensure_session_dirs(session_id)

    session_data = {
        "session_id": session_id,
        "start_time": st.session_state.get("start_time", ""),
        "last_updated": datetime.now().isoformat(),
        "current_stage": STAGES[st.session_state.stage_index],
        "character_name": st.session_state.get("character_name", ""),
        "character_major": st.session_state.get("character_major", ""),
        "character_grade": st.session_state.get("character_grade", ""),
        "character_appearance": st.session_state.get("character_appearance", ""),
        "character_features": st.session_state.get("character_features", {}),
        "portrait_style": st.session_state.get("portrait_style", ""),
        "portrait_path": st.session_state.get("portrait_path", ""),
        "portrait_final": st.session_state.get("portrait_final", ""),
        "portrait_versions": st.session_state.get("portrait_versions", []),
        "portrait_adjustment_count": st.session_state.get("portrait_adjustment_count", 0),
        "messages": st.session_state.get("messages", []),
        "stage_A_material": st.session_state.get("stage_A_material", {}),
        "stage_A2a_quote": st.session_state.get("stage_A2a_quote", ""),
        "stage_A2b_reflection": st.session_state.get("stage_A2b_reflection", ""),
        "stage_A2c_framing": st.session_state.get("stage_A2c_framing", ""),
        "stage_A_comic": st.session_state.get("stage_A_comic", {}),
        "stage_B_material": st.session_state.get("stage_B_material", {}),
        "stage_B2a_quote": st.session_state.get("stage_B2a_quote", ""),
        "stage_B2b_reflection": st.session_state.get("stage_B2b_reflection", ""),
        "stage_B2c_framing": st.session_state.get("stage_B2c_framing", ""),
        "stage_B_comic": st.session_state.get("stage_B_comic", {}),
        "stage4_rewrite_A": st.session_state.get("stage4_rewrite_A", ""),
        "stage4_rewrite_B": st.session_state.get("stage4_rewrite_B", ""),
        "stage4_diff_A": st.session_state.get("stage4_diff_A", ""),
        "stage4_diff_B": st.session_state.get("stage4_diff_B", ""),
        "debrief_response": st.session_state.get("debrief_response", ""),
    }

    session_file = os.path.join(session_dir, "session.json")
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存会话失败: {e}")

    log_file = os.path.join(session_dir, "log.json")
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.log_entries, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存日志失败: {e}")

def add_message(role: str, content: str):
    """添加消息并实时保存"""
    st.session_state.messages.append({
        "role": role,
        "content": content
    })
    save_session()

def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """加载指定会话的所有数据"""
    session_file = os.path.join(DATA_DIR, "sessions", session_id, "session.json")
    if os.path.exists(session_file):
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载会话失败: {e}")
    return None

def list_sessions():
    """列出所有会话"""
    sessions = []
    sessions_dir = os.path.join(DATA_DIR, "sessions")
    if os.path.exists(sessions_dir):
        for session_id in os.listdir(sessions_dir):
            session_path = os.path.join(sessions_dir, session_id)
            if os.path.isdir(session_path):
                session_data = load_session(session_id)
                if session_data:
                    sessions.append({
                        "session_id": session_id,
                        "character_name": session_data.get("character_name", "未命名"),
                        "start_time": session_data.get("start_time", ""),
                        "current_stage": session_data.get("current_stage", ""),
                        "message_count": len(session_data.get("messages", []))
                    })
    return sorted(sessions, key=lambda x: x["start_time"], reverse=True)

def load_session_to_state(session_id: str):
    """将加载的会话数据恢复到 session_state"""
    session_data = load_session(session_id)
    if not session_data:
        return

    st.session_state.session_id = session_id
    st.session_state.start_time = session_data.get("start_time", "")
    st.session_state.messages = session_data.get("messages", [])

    current_stage = session_data.get("current_stage", "S0a")
    if current_stage in STAGES:
        st.session_state.stage_index = STAGES.index(current_stage)
    else:
        st.session_state.stage_index = 0

    st.session_state.character_name = session_data.get("character_name", "")
    st.session_state.character_major = session_data.get("character_major", "")
    st.session_state.character_grade = session_data.get("character_grade", "")
    st.session_state.character_appearance = session_data.get("character_appearance", "")
    st.session_state.character_features = session_data.get("character_features", {})
    st.session_state.portrait_style = session_data.get("portrait_style", "")
    st.session_state.portrait_path = session_data.get("portrait_path", "")
    st.session_state.portrait_final = session_data.get("portrait_final", "")
    st.session_state.portrait_versions = session_data.get("portrait_versions", [])
    st.session_state.portrait_adjustment_count = session_data.get("portrait_adjustment_count", 0)

    st.session_state.stage_A_material = session_data.get("stage_A_material", {})
    st.session_state.stage_A2a_quote = session_data.get("stage_A2a_quote", "")
    st.session_state.stage_A2b_reflection = session_data.get("stage_A2b_reflection", "")
    st.session_state.stage_A2c_framing = session_data.get("stage_A2c_framing", "")
    st.session_state.stage_A_comic = session_data.get("stage_A_comic", {})

    st.session_state.stage_B_material = session_data.get("stage_B_material", {})
    st.session_state.stage_B2a_quote = session_data.get("stage_B2a_quote", "")
    st.session_state.stage_B2b_reflection = session_data.get("stage_B2b_reflection", "")
    st.session_state.stage_B2c_framing = session_data.get("stage_B2c_framing", "")
    st.session_state.stage_B_comic = session_data.get("stage_B_comic", {})

    st.session_state.stage4_rewrite_A = session_data.get("stage4_rewrite_A", "")
    st.session_state.stage4_rewrite_B = session_data.get("stage4_rewrite_B", "")
    st.session_state.stage4_diff_A = session_data.get("stage4_diff_A", "")
    st.session_state.stage4_diff_B = session_data.get("stage4_diff_B", "")

    st.session_state.follow_up_count = 0
    st.session_state.comic_mode = False
    st.session_state.comic_situation = ""
    st.session_state.comic_frame_idx = 0
    st.session_state.comic_frames_parsed = []
    st.session_state.stage4_mode = False
    st.session_state.stage4_situation = ""
    st.session_state.stage4_start_time = 0
    st.session_state.portrait_generating = False
    st.session_state.portrait_error = ""
    st.session_state.portrait_current_version = 0
    st.session_state.portrait_adjusting = False
    st.session_state.portrait_adjustment_input = ""
    st.session_state.comic_current_version = 0
    st.session_state.comic_adjusting_desc = False
    st.session_state.comic_adding_frame = False
    st.session_state.comic_desc_input = ""
    st.session_state.debrief_response = session_data.get("debrief_response", "")
    st.session_state.log_entries = []

    ensure_session_dirs()

# =============================================================================
# 会话状态初始化
# =============================================================================
def init_session_state():
    """初始化 Streamlit 会话状态"""

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
        st.session_state.start_time = datetime.now().isoformat()
        ensure_session_dirs()

    if "stage_index" not in st.session_state:
        st.session_state.stage_index = 0

    if "log_entries" not in st.session_state:
        st.session_state.log_entries = []

    if "messages" not in st.session_state:
        st.session_state.messages = []
        initial_msg = generate_guide_message("S0a")
        st.session_state.messages.append({
            "role": "assistant",
            "content": initial_msg
        })
        st.session_state.log_entries.append({
            "timestamp": datetime.now().isoformat(),
            "type": "initial",
            "content": initial_msg
        })
        save_session()

    if "character_name" not in st.session_state:
        st.session_state.character_name = ""

    if "character_major" not in st.session_state:
        st.session_state.character_major = ""

    if "character_grade" not in st.session_state:
        st.session_state.character_grade = ""

    if "character_gender" not in st.session_state:
        st.session_state.character_gender = ""

    if "character_appearance" not in st.session_state:
        st.session_state.character_appearance = ""

    if "portrait_path" not in st.session_state:
        st.session_state.portrait_path = ""

    if "character_features" not in st.session_state:
        st.session_state.character_features = {}

    if "portrait_versions" not in st.session_state:
        st.session_state.portrait_versions = []

    if "portrait_final" not in st.session_state:
        st.session_state.portrait_final = ""

    if "stage_A_material" not in st.session_state:
        st.session_state.stage_A_material = {
            "dilemma": "",
            "context": "",
            "reaction": "",
            "impact": ""
        }

    if "stage_A2a_quote" not in st.session_state:
        st.session_state.stage_A2a_quote = ""

    if "stage_A2b_reflection" not in st.session_state:
        st.session_state.stage_A2b_reflection = ""

    if "stage_A2c_framing" not in st.session_state:
        st.session_state.stage_A2c_framing = ""

    if "stage_A_quote_timestamp" not in st.session_state:
        st.session_state.stage_A_quote_timestamp = 0

    if "stage_B_material" not in st.session_state:
        st.session_state.stage_B_material = {
            "dilemma": "",
            "context": "",
            "reaction": "",
            "impact": ""
        }

    if "stage_B2a_quote" not in st.session_state:
        st.session_state.stage_B2a_quote = ""

    if "stage_B2b_reflection" not in st.session_state:
        st.session_state.stage_B2b_reflection = ""

    if "stage_B2c_framing" not in st.session_state:
        st.session_state.stage_B2c_framing = ""

    if "stage_B_quote_timestamp" not in st.session_state:
        st.session_state.stage_B_quote_timestamp = 0

    if "stage_A_comic" not in st.session_state:
        st.session_state.stage_A_comic = {
            "frames": [],
            "confirmed": False
        }

    if "stage_B_comic" not in st.session_state:
        st.session_state.stage_B_comic = {
            "frames": [],
            "confirmed": False
        }

    if "stage4_rewrite_A" not in st.session_state:
        st.session_state.stage4_rewrite_A = ""

    if "stage4_rewrite_B" not in st.session_state:
        st.session_state.stage4_rewrite_B = ""

    if "stage4_latency_A" not in st.session_state:
        st.session_state.stage4_latency_A = 0

    if "stage4_latency_B" not in st.session_state:
        st.session_state.stage4_latency_B = 0

    if "stage4_edit_seq_A" not in st.session_state:
        st.session_state.stage4_edit_seq_A = []

    if "stage4_edit_seq_B" not in st.session_state:
        st.session_state.stage4_edit_seq_B = []

    if "follow_up_count" not in st.session_state:
        st.session_state.follow_up_count = 0

    if "comic_mode" not in st.session_state:
        st.session_state.comic_mode = False

    if "comic_situation" not in st.session_state:
        st.session_state.comic_situation = ""

    if "comic_frame_idx" not in st.session_state:
        st.session_state.comic_frame_idx = 0

    if "comic_frames_parsed" not in st.session_state:
        st.session_state.comic_frames_parsed = []

    if "stage4_mode" not in st.session_state:
        st.session_state.stage4_mode = False

    if "stage4_situation" not in st.session_state:
        st.session_state.stage4_situation = ""

    if "stage4_start_time" not in st.session_state:
        st.session_state.stage4_start_time = 0

    if "portrait_generating" not in st.session_state:
        st.session_state.portrait_generating = False

    if "portrait_error" not in st.session_state:
        st.session_state.portrait_error = ""

    if "portrait_current_version" not in st.session_state:
        st.session_state.portrait_current_version = 0

    if "portrait_adjustment_count" not in st.session_state:
        st.session_state.portrait_adjustment_count = 0

    if "portrait_adjusting" not in st.session_state:
        st.session_state.portrait_adjusting = False

    if "portrait_adjust_mode" not in st.session_state:
        st.session_state.portrait_adjust_mode = False

    if "portrait_adjustment_input" not in st.session_state:
        st.session_state.portrait_adjustment_input = ""

    if "portrait_generated" not in st.session_state:
        st.session_state.portrait_generated = False

    if "comic_current_version" not in st.session_state:
        st.session_state.comic_current_version = 0

    if "comic_adjusting_desc" not in st.session_state:
        st.session_state.comic_adjusting_desc = False

    if "comic_adding_frame" not in st.session_state:
        st.session_state.comic_adding_frame = False

    if "comic_desc_input" not in st.session_state:
        st.session_state.comic_desc_input = ""

    if "comic_editing_frame" not in st.session_state:
        st.session_state.comic_editing_frame = 0

    if "comic_editing_mode" not in st.session_state:
        st.session_state.comic_editing_mode = False

    if "state_cache" not in st.session_state:
        st.session_state.state_cache = {}

    if "debrief_response" not in st.session_state:
        st.session_state.debrief_response = ""

# =============================================================================
# 渲染函数
# =============================================================================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("AI共创叙事画坊")
        st.markdown("---")

        sessions = list_sessions()
        if sessions:
            st.markdown("**历史会话**")
            session_options = {s["character_name"] or "未命名": s["session_id"] for s in sessions}
            session_names = list(session_options.keys())[:5]

            selected = st.selectbox(
                "选择会话",
                options=["当前会话"] + session_names,
                index=0,
                key="session_selector"
            )

            if selected != "当前会话":
                selected_id = session_options[selected]
                if selected_id != st.session_state.session_id:
                    if st.button("继续此会话"):
                        load_session_to_state(selected_id)
                        st.rerun()

        st.markdown("---")

        current_idx = st.session_state.stage_index
        current_stage = STAGES[min(current_idx, len(STAGES)-1)]

        stages_progress = {
            "创建角色": (0, 3),
            "学业故事": (4, 14),
            "人际故事": (15, 25),
            "编剧练习": (26, 28)
        }

        st.markdown("**进度**")
        for name, (start, end) in stages_progress.items():
            if current_idx >= start and current_idx <= end:
                icon = "⏳" if current_idx == start else "✅"
                st.markdown(f"{icon} {name}")
            elif current_idx > end:
                st.markdown(f"✅ {name}")
            else:
                st.markdown(f"⭕ {name}")

        progress = current_idx / (len(STAGES) - 1)
        st.progress(progress)

        st.markdown(f"**当前**: {STAGE_NAMES.get(current_stage, current_stage)}")

        st.markdown("---")

        if st.session_state.character_name:
            st.markdown("**角色信息**")

            if st.session_state.portrait_final and os.path.exists(st.session_state.portrait_final):
                st.image(st.session_state.portrait_final, width=120)

            st.markdown(f"**{st.session_state.character_name}**")
            if st.session_state.character_major:
                st.caption(f"{st.session_state.character_major}")

        with st.expander("查看完整故事"):
            if st.session_state.stage_A_material.get("dilemma"):
                st.markdown("**学业困扰**")
                st.text(st.session_state.stage_A_material["dilemma"][:200] + "...")

            if st.session_state.stage_B_material.get("dilemma"):
                st.markdown("**人际困扰**")
                st.text(st.session_state.stage_B_material["dilemma"][:200] + "...")

            if st.session_state.stage_A_comic.get("frames"):
                st.markdown("**学业连环画**")
                cols = st.columns(len(st.session_state.stage_A_comic["frames"]))
                for i, f in enumerate(st.session_state.stage_A_comic["frames"]):
                    if f.get("final_path") and os.path.exists(f["final_path"]):
                        with cols[i]:
                            st.image(f["final_path"], width=60)

            if st.session_state.stage_B_comic.get("frames"):
                st.markdown("**人际连环画**")
                cols = st.columns(len(st.session_state.stage_B_comic["frames"]))
                for i, f in enumerate(st.session_state.stage_B_comic["frames"]):
                    if f.get("final_path") and os.path.exists(f["final_path"]):
                        with cols[i]:
                            st.image(f["final_path"], width=60)

        st.markdown("---")
        if st.button("重新开始", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def render_messages():
    """渲染消息列表"""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

def _get_portrait_final_path() -> str:
    images_dir = get_images_dir()
    return os.path.join(images_dir, "portrait_final.png")

def _copy_to_final(src_path: str) -> str:
    final_path = _get_portrait_final_path()
    if src_path and os.path.exists(src_path):
        import shutil
        ensure_session_dirs()
        shutil.copy2(src_path, final_path)
        return final_path
    return src_path

def render_portrait_ui():
    """渲染画像生成界面（只负责生成和显示，风格已在对话中选择）"""

    if st.session_state.portrait_generating:
        with st.spinner("正在生成画像，请稍候..."):
            features = extract_character_features(st.session_state.character_appearance)
            st.session_state.character_features = features

            result = generate_portrait(
                st.session_state.character_appearance,
                st.session_state.session_id,
                style=st.session_state.portrait_style
            )

            st.session_state.portrait_generating = False

            if result["success"]:
                current_v = len(st.session_state.portrait_versions) + 1
                st.session_state.portrait_current_version = current_v

                version_record = {
                    "version": current_v,
                    "path": result["image_path"],
                    "prompt": st.session_state.character_appearance,
                    "adjustment_desc": "",
                    "generated_at": datetime.now().isoformat()
                }
                st.session_state.portrait_versions.append(version_record)
                st.session_state.portrait_path = result["image_path"]

                images_dir = get_images_dir()
                current_path = os.path.join(images_dir, f"portrait_v{current_v}.png")
                import shutil
                ensure_session_dirs()
                shutil.copy2(result["image_path"], current_path)
                st.session_state.portrait_path = current_path

                st.session_state.portrait_generated = True

                log_event("portrait_generated", version=current_v, path=current_path)
                save_session()
                st.rerun()
            else:
                st.session_state.portrait_error = result.get("error", "生成失败")
                st.error(f"画像生成失败: {result.get('error', '未知错误')}")

    if st.session_state.portrait_path and os.path.exists(st.session_state.portrait_path):
        st.markdown("---")
        st.markdown("### 角色画像")
        st.image(st.session_state.portrait_path, width=400)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ 确认画像", type="primary"):
                features = extract_character_features(st.session_state.character_appearance)
                st.session_state.character_features = features
                st.session_state.portrait_final = _copy_to_final(st.session_state.portrait_path)

                log_event("portrait_confirmed",
                          version=st.session_state.portrait_current_version,
                          final_path=st.session_state.portrait_final)

                # 直接进入 A1a 学业故事
                st.session_state.stage_index = STAGES.index("A1a")
                msg = generate_guide_message("A1a", {"name": st.session_state.character_name})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg
                })
                save_session()
                st.rerun()

        with col2:
            if st.button("✎ 调整修改"):
                # 进入调整模式
                st.session_state.portrait_adjust_mode = True
                st.rerun()

    # 调整修改模式
    if st.session_state.get("portrait_adjust_mode", False):
        st.markdown("### 请描述需要调整的地方")
        st.markdown("比如：眼睛再大一些、头发换成短发、表情更严肃等")

        adjustment_input = st.text_area(
            "调整要求",
            value=st.session_state.get("portrait_adjust_input", ""),
            placeholder="描述你需要调整的地方...",
            key="portrait_adjust_input_area"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✓ 确认调整", type="primary"):
                if adjustment_input.strip():
                    # 保存调整要求
                    st.session_state.portrait_adjust_input = adjustment_input

                    # 构建新的外貌描述 = 原描述 + 调整
                    original_appearance = st.session_state.character_appearance
                    new_appearance = f"{original_appearance}。调整要求：{adjustment_input}"
                    st.session_state.character_appearance = new_appearance

                    # 准备重新生成
                    st.session_state.portrait_adjust_mode = False
                    st.session_state.portrait_adjusting = True
                    st.session_state.portrait_generating = True
                    st.session_state.portrait_generated = False

                    msg = f"好的，我会根据你的要求调整：{adjustment_input}。正在重新生成，请稍候..."
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": msg
                    })
                    save_session()
                    st.rerun()

        with col2:
            if st.button("取消"):
                st.session_state.portrait_adjust_mode = False
                st.rerun()

        with col3:
            if st.button("重新选择风格"):
                st.session_state.portrait_adjust_mode = False
                st.session_state.portrait_style = ""
                st.session_state.portrait_generated = False
                st.session_state.portrait_path = ""
                msg = "好的，我们重新生成。你想用什么风格？比如：写实、动漫、水彩、素描、油画、国风等。"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg
                })
                st.rerun()

def _get_comic_final_path(situation: str, frame_index: int) -> str:
    images_dir = get_images_dir()
    return os.path.join(images_dir, f"comic_{situation}_{frame_index}_final.png")

def _copy_frame_to_final(src_path: str, situation: str, frame_index: int) -> str:
    final_path = _get_comic_final_path(situation, frame_index)
    if src_path and os.path.exists(src_path):
        import shutil
        ensure_session_dirs()
        shutil.copy2(src_path, final_path)
        return final_path
    return src_path

# =============================================================================
# 辅助函数：连环画生成
# =============================================================================
def _get_comic_frame_record(comic_data, frame_index):
    """获取指定分镜的记录，不存在则创建"""
    for f in comic_data["frames"]:
        if f.get("frame_index") == frame_index:
            return f
    # 创建新记录
    frame_record = {
        "frame_index": frame_index,
        "description": "",
        "versions": [],
        "current_version": 0,
        "final_version": 0,
        "final_path": "",
        "image_path": ""
    }
    comic_data["frames"].append(frame_record)
    return frame_record


def _ensure_frame_records():
    """确保所有分镜都有记录"""
    situation = st.session_state.comic_situation
    comic_data = st.session_state.stage_A_comic if situation == "A" else st.session_state.stage_B_comic
    frames_parsed = st.session_state.comic_frames_parsed
    
    for idx, frame in enumerate(frames_parsed):
        frame_num = idx + 1
        _get_comic_frame_record(comic_data, frame_num)
        # 更新描述
        for f in comic_data["frames"]:
            if f.get("frame_index") == frame_num:
                f["description"] = frame.get("description", "")


def _generate_single_frame(frame_num, desc_override=None):
    """生成单个分镜"""
    situation = st.session_state.comic_situation
    comic_data = st.session_state.stage_A_comic if situation == "A" else st.session_state.stage_B_comic
    frames_parsed = st.session_state.comic_frames_parsed
    frame_record = _get_comic_frame_record(comic_data, frame_num)
    
    # 获取描述
    if desc_override:
        description = desc_override
    else:
        frame_data = frames_parsed[frame_num - 1] if frame_num <= len(frames_parsed) else {}
        description = frame_data.get("description", "")
    
    current_v = frame_record["current_version"]
    
    result = generate_comic_frame(
        description=description,
        portrait_path=st.session_state.portrait_final,
        character_features=st.session_state.character_features,
        frame_index=frame_num,
        session_id=st.session_state.session_id,
        situation=f"stage_{situation}",
        style=st.session_state.get("comic_style", "动漫")
    )
    
    if result["success"]:
        new_v = current_v + 1
        frame_record["current_version"] = new_v
        
        images_dir = get_images_dir()
        versioned_path = os.path.join(images_dir, f"comic_{situation}_{frame_num}_v{new_v}.png")
        
        import shutil
        ensure_session_dirs()
        shutil.copy2(result["image_path"], versioned_path)
        
        frame_record["image_path"] = versioned_path
        frame_record["versions"].append({
            "v": new_v,
            "path": versioned_path,
            "prompt": description,
            "generated_at": datetime.now().isoformat()
        })
        
        log_event("comic_frame_generated",
                  situation=situation,
                  frame=frame_num,
                  version=new_v,
                  path=versioned_path)
        save_session()
        return True
    else:
        st.error(f"第{frame_num}格生成失败: {result.get('error', '未知错误')}")
        return False


def _render_comic_frame_row(frame_num, frame_record, frames_parsed, situation):
    """渲染单个分镜行（每行一格，左侧编辑描述，右侧显示图片）"""
    frame_data = frames_parsed[frame_num - 1] if frame_num <= len(frames_parsed) else {}
    description = frame_data.get("description", "")
    versions = frame_record.get("versions", [])
    has_image = frame_record.get("image_path") and os.path.exists(frame_record["image_path"])
    current_v = frame_record.get("current_version", 0)
    final_v = frame_record.get("final_version", 0)
    
    # 每行布局：左侧(编辑区) | 右侧(图片区)
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown(f"**第{frame_num}格**")
        # 文本输入框用于编辑描述
        new_desc = st.text_area(
            "描述",
            value=description,
            key=f"desc_{frame_num}",
            height=100,
            label_visibility="collapsed",
            placeholder="输入分镜描述..."
        )
        
        # 如果描述有变化，更新数据
        if new_desc != description:
            # 更新 frames_parsed
            if frame_num <= len(frames_parsed):
                frames_parsed[frame_num - 1]["description"] = new_desc
            # 更新 comic_data 中的 frame_record
            frame_record["description"] = new_desc
            save_session()
        
        # 操作按钮行
        col_gen, col_retry, col_confirm = st.columns(3)
        with col_gen:
            if st.button("🎨 生成", key=f"gen_{frame_num}", type="primary", use_container_width=True):
                with st.spinner(f"正在生成第{frame_num}格..."):
                    success = _generate_single_frame(frame_num)
                    if success:
                        st.rerun()
        with col_retry:
            if st.button("🔄 重试", key=f"regen_{frame_num}", use_container_width=True):
                with st.spinner(f"正在生成第{frame_num}格..."):
                    # 保持原图，生成新版本
                    success = _generate_single_frame(frame_num)
                    if success:
                        st.rerun()
        with col_confirm:
            if has_image and len(versions) > 0:
                if len(versions) == 1:
                    if st.button("✓ 确认", key=f"confirm_single_{frame_num}", use_container_width=True):
                        frame_record["final_version"] = current_v
                        frame_record["final_path"] = _copy_frame_to_final(
                            frame_record["image_path"], situation, frame_num
                        )
                        log_event("comic_frame_confirmed",
                                  situation=situation,
                                  frame=frame_num,
                                  version=current_v,
                                  final_path=frame_record["final_path"])
                        save_session()
                        st.rerun()
                else:
                    if st.button("✓ 确认", key=f"confirm_multi_{frame_num}", use_container_width=True):
                        st.session_state.comic_selecting_version = frame_num
                        st.rerun()
            else:
                st.button("✓ 确认", disabled=True, key=f"confirm_disabled_{frame_num}", use_container_width=True)
    
    with col2:
        # 显示版本选择界面（如果有多个版本）
        if st.session_state.get("comic_selecting_version") == frame_num and len(versions) > 0:
            st.markdown("**选择版本：**")
            cols_per_row = 2
            for i in range(0, len(versions), cols_per_row):
                v_cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(versions):
                        v = versions[i + j]
                        with v_cols[j]:
                            if os.path.exists(v["path"]):
                                is_selected = (v["v"] == current_v)
                                is_final = (v["v"] == final_v)
                                label = f"v{v['v']}"
                                if is_final:
                                    label += " ✓已确认"
                                elif is_selected:
                                    label += " (当前)"
                                
                                if st.button(label, key=f"select_v_{frame_num}_{v['v']}", use_container_width=True):
                                    frame_record["current_version"] = v["v"]
                                    frame_record["image_path"] = v["path"]
                                    st.session_state.comic_selecting_version = None
                                    save_session()
                                    st.rerun()
                                
                                st.image(v["path"], width=200)
            
            col_confirm_sel, col_cancel = st.columns(2)
            with col_confirm_sel:
                if st.button("✓ 确认选中版本", key=f"confirm_sel_{frame_num}", type="primary", use_container_width=True):
                    frame_record["final_version"] = frame_record["current_version"]
                    frame_record["final_path"] = _copy_frame_to_final(
                        frame_record["image_path"], situation, frame_num
                    )
                    log_event("comic_frame_confirmed",
                              situation=situation,
                              frame=frame_num,
                              version=frame_record["current_version"],
                              final_path=frame_record["final_path"])
                    st.session_state.comic_selecting_version = None
                    save_session()
                    st.rerun()
            with col_cancel:
                if st.button("取消", key=f"cancel_sel_{frame_num}", use_container_width=True):
                    st.session_state.comic_selecting_version = None
                    st.rerun()
        elif has_image and os.path.exists(frame_record["image_path"]):
            st.image(frame_record["image_path"], width=350)
            if len(versions) > 1:
                st.caption(f"共 {len(versions)} 个版本，点击「确认」选择")
            elif final_v > 0:
                st.caption("✓ 已确认")
        else:
            st.info("点击上方「生成」按钮开始生成")


def _render_comic_complete_view():
    """渲染连环画完成视图"""
    situation = st.session_state.comic_situation
    comic_data = st.session_state.stage_A_comic if situation == "A" else st.session_state.stage_B_comic
    
    st.markdown("#### 预览")
    frames = comic_data.get("frames", [])
    cols_count = min(len(frames), 3)
    
    if cols_count > 0:
        cols = st.columns(cols_count)
        for i, frame in enumerate(frames):
            with cols[i % cols_count]:
                final_path = frame.get("final_path") or frame.get("image_path")
                if final_path and os.path.exists(final_path):
                    st.image(final_path, width=250)
                    st.caption(f"第{i+1}格")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    target_stage = "B1a" if situation == "A" else "P4A_REWRITE"
    next_stage_name = STAGE_NAMES.get(target_stage, "下一阶段")
    
    with col1:
        if st.button(f"✓ 确认，进入{next_stage_name}", type="primary"):
            comic_data["confirmed"] = True
            log_event("comic_complete",
                      situation=situation,
                      frame_count=len(comic_data["frames"]))
            
            if situation == "A":
                st.session_state.stage_index = STAGES.index("A3c")
                msg = generate_guide_message("A3c", {"name": st.session_state.character_name})
            else:
                st.session_state.stage_index = STAGES.index("P4A_REWRITE")
                msg = generate_guide_message("P4A_REWRITE", {
                    "name": st.session_state.character_name,
                    "quote": st.session_state.stage_A2a_quote
                })
                st.session_state.stage4_mode = True
                st.session_state.stage4_situation = "A"
                st.session_state.stage4_start_time = time.time()
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg
            })
            save_session()
            st.rerun()
    
    with col2:
        if st.button("🔄 重新生成"):
            if situation == "A":
                st.session_state.stage_A_comic = {"frames": [], "confirmed": False}
            else:
                st.session_state.stage_B_comic = {"frames": [], "confirmed": False}
            st.session_state.comic_frame_idx = 0
            _ensure_frame_records()
            save_session()
            st.rerun()


def _render_comic_editing_mode():
    """渲染连环画编辑模式"""
    situation = st.session_state.comic_situation
    comic_data = st.session_state.stage_A_comic if situation == "A" else st.session_state.stage_B_comic
    frames_parsed = st.session_state.comic_frames_parsed
    
    editing_frame = st.session_state.get("comic_editing_frame", 0)
    
    if editing_frame > 0:
        # 编辑单个分镜描述
        frame_record = _get_comic_frame_record(comic_data, editing_frame)
        frame_data = frames_parsed[editing_frame - 1] if editing_frame <= len(frames_parsed) else {}
        
        st.markdown(f"#### 编辑第 {editing_frame} 格")
        
        new_desc = st.text_area(
            "分镜描述",
            value=frame_data.get("description", ""),
            key="edit_frame_desc",
            height=100,
            placeholder="描述这一格的画面..."
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✓ 确认修改", type="primary"):
                # 更新描述
                if editing_frame <= len(frames_parsed):
                    frames_parsed[editing_frame - 1]["description"] = new_desc
                frame_record["description"] = new_desc
                # 标记需要重新生成
                frame_record["image_path"] = ""
                frame_record["current_version"] = 0
                st.session_state.comic_editing_frame = 0
                save_session()
                st.rerun()
        
        with col2:
            if st.button("取消"):
                st.session_state.comic_editing_frame = 0
                st.rerun()
        
        with col3:
            if st.button("🗑️ 删除此格"):
                # 删除分镜
                frames_parsed.pop(editing_frame - 1)
                comic_data["frames"] = [f for f in comic_data["frames"] if f.get("frame_index") != editing_frame]
                # 重新编号
                for i, fp in enumerate(frames_parsed):
                    fp["frame_number"] = i + 1
                for f in comic_data["frames"]:
                    f["frame_index"] = f.get("frame_index", 0)
                    for fp in frames_parsed:
                        if f.get("description") == fp.get("description"):
                            f["frame_index"] = fp["frame_number"]
                            break
                # 简化：直接重建 frames
                comic_data["frames"] = []
                for i, fp in enumerate(frames_parsed):
                    frame_record = {
                        "frame_index": i + 1,
                        "description": fp.get("description", ""),
                        "versions": [],
                        "current_version": 0,
                        "final_version": 0,
                        "final_path": "",
                        "image_path": ""
                    }
                    comic_data["frames"].append(frame_record)
                
                st.session_state.comic_editing_frame = 0
                save_session()
                st.rerun()
    else:
        # 显示所有分镜列表供编辑
        st.markdown("#### 编辑分镜")
        st.caption("点击「修改」按钮编辑单个分镜的描述")
        
        for i, frame in enumerate(frames_parsed):
            frame_num = i + 1
            frame_record = next((f for f in comic_data["frames"] if f.get("frame_index") == frame_num), None)
            has_image = frame_record and frame_record.get("image_path") and os.path.exists(frame_record["image_path"])
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**第{frame_num}格**: {frame.get('description', '')[:50]}{'...' if len(frame.get('description', '')) > 50 else ''}")
            with col2:
                if st.button("✎ 修改", key=f"edit_list_{frame_num}"):
                    st.session_state.comic_editing_frame = frame_num
                    st.rerun()
            with col3:
                if frame_record and has_image:
                    st.markdown("✅")
                else:
                    st.markdown("⏳")
        
        st.markdown("---")
        if st.button("← 返回生成界面"):
            st.rerun()


def _render_comic_generation_view():
    """渲染连环画生成主界面"""
    situation = st.session_state.comic_situation
    comic_data = st.session_state.stage_A_comic if situation == "A" else st.session_state.stage_B_comic
    frames_parsed = st.session_state.comic_frames_parsed
    total_frames = len(frames_parsed)
    
    # 确保所有分镜都有记录
    _ensure_frame_records()
    
    # 每行1个分镜
    for frame_num in range(1, total_frames + 1):
        frame_record = next(
            (f for f in comic_data["frames"] if f.get("frame_index") == frame_num),
            None
        )
        if frame_record:
            _render_comic_frame_row(frame_num, frame_record, frames_parsed, situation)
        st.markdown("---")
    
    # 统计完成情况
    confirmed_count = len([f for f in comic_data.get("frames", []) if f.get("final_path")])
    generated_count = len([f for f in comic_data.get("frames", []) if f.get("image_path")])
    
    st.markdown(f"**进度：{confirmed_count}/{total_frames} 格已确认，{generated_count}/{total_frames} 格已生成**")
    
    # 底部工具栏
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ 增加一格", use_container_width=True):
            new_frame = {
                "description": "",
                "frame_number": total_frames + 1
            }
            frames_parsed.append(new_frame)
            
            frame_record = {
                "frame_index": total_frames + 1,
                "description": "",
                "versions": [],
                "current_version": 0,
                "final_version": 0,
                "final_path": "",
                "image_path": ""
            }
            comic_data["frames"].append(frame_record)
            save_session()
            st.rerun()
    
    with col2:
        if total_frames > 1:
            if st.button("➖ 删除最后一格", use_container_width=True):
                frames_parsed.pop()
                comic_data["frames"] = [
                    f for f in comic_data["frames"] 
                    if f.get("frame_index") != total_frames
                ]
                for i, f in enumerate(comic_data["frames"]):
                    f["frame_index"] = i + 1
                save_session()
                st.rerun()
        else:
            st.button("➖ 删除最后一格", disabled=True, use_container_width=True)
    
    with col3:
        if st.button("✓ 完成连环画", type="primary", use_container_width=True):
            # 确认所有已生成的图片
            for f in comic_data.get("frames", []):
                if f.get("image_path") and not f.get("final_path"):
                    f["final_version"] = f.get("current_version", 1)
                    f["final_path"] = f.get("image_path", "")
            
            comic_data["confirmed"] = True
            log_event("comic_complete",
                      situation=situation,
                      frame_count=len(comic_data["frames"]))
            
            if situation == "A":
                st.session_state.stage_index = STAGES.index("A3c")
                msg = generate_guide_message("A3c", {"name": st.session_state.character_name})
            else:
                st.session_state.stage_index = STAGES.index("P4A_REWRITE")
                msg = generate_guide_message("P4A_REWRITE", {
                    "name": st.session_state.character_name,
                    "quote": st.session_state.stage_A2a_quote
                })
                st.session_state.stage4_mode = True
                st.session_state.stage4_situation = "A"
                st.session_state.stage4_start_time = time.time()
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg
            })
            save_session()
            st.rerun()


def render_comic_ui():
    """渲染连环画生成界面"""
    situation = st.session_state.comic_situation
    comic_data = st.session_state.stage_A_comic if situation == "A" else st.session_state.stage_B_comic
    frames_parsed = st.session_state.comic_frames_parsed
    total_frames = len(frames_parsed)
    
    # 显示标题
    story_type = "学业" if situation == "A" else "人际"
    st.markdown(f"### 🎬 {story_type}故事连环画生成")
    
    if total_frames == 0:
        st.warning("没有分镜数据，请返回重新选择关键瞬间")
        if st.button("← 返回"):
            if situation == "A":
                st.session_state.stage_index = STAGES.index("A3a")
            else:
                st.session_state.stage_index = STAGES.index("B3a")
            st.rerun()
        return
    
    # 正常生成视图
    _render_comic_generation_view()

def render_rewrite_ui():
    """渲染视角改写界面"""
    current_stage = STAGES[st.session_state.stage_index]
    if current_stage == "P4A_REWRITE":
        situation = "A"
    else:
        situation = "B"

    if st.session_state.stage4_start_time == 0:
        st.session_state.stage4_start_time = time.time()

    # 获取对应的连环画数据
    if situation == "A":
        comic_data = st.session_state.stage_A_comic
        story_type = "学业"
    else:
        comic_data = st.session_state.stage_B_comic
        story_type = "人际"

    if situation == "A":
        material = st.session_state.stage_A_material
        original_quote = st.session_state.stage_A2a_quote or material.get("dilemma") or material.get("context") or "请描述这个关键时刻"
        target_rewrite = "stage4_rewrite_A"
        latency_key = "stage4_latency_A"
    else:
        material = st.session_state.stage_B_material
        original_quote = st.session_state.stage_B2a_quote or material.get("dilemma") or material.get("context") or "请描述这个关键时刻"
        target_rewrite = "stage4_rewrite_B"
        latency_key = "stage4_latency_B"

    st.markdown(f"### 编剧练习：{story_type}故事视角转换")

    # 显示连环画引导回忆
    st.markdown(f"请回忆刚才创作的 **{story_type}故事连环画** 中的关键时刻：")
    
    frames = comic_data.get("frames", [])
    if frames:
        # 显示连环画预览
        cols = st.columns(min(len(frames), 5))
        for i, frame in enumerate(frames):
            with cols[i % len(cols)]:
                final_path = frame.get("final_path") or frame.get("image_path")
                if final_path and os.path.exists(final_path):
                    st.image(final_path, width=150)
                    st.caption(f"第{i+1}格")

    st.markdown("---")

    # 直接进入沉浸视角练习
    st.markdown("**沉浸视角（第一人称）**")
    st.markdown("现在请你完全代入角色，用'我'来写TA的内心独白。回到那个时刻，'我'的内心会说什么？")

    current_text = st.session_state.get(target_rewrite, "")
    rewrite_text = st.text_area(
        "用第一人称改写",
        value=current_text,
        height=200,
        key=f"stage4_{situation}_textarea",
        placeholder="在这里写下角色的内心独白..."
    )

    st.session_state[target_rewrite] = rewrite_text

    if st.button("提交", type="primary"):
        latency_ms = (time.time() - st.session_state.stage4_start_time) * 1000
        st.session_state[latency_key] = latency_ms

        log_event("p4_rewrite_submit", situation=situation,
                  latency_ms=latency_ms, content_length=len(rewrite_text))

        if situation == "A":
            st.session_state.stage_index = STAGES.index("P4A_DIFF")
        else:
            st.session_state.stage_index = STAGES.index("P4B_DIFF")

        next_stage = STAGES[st.session_state.stage_index]
        msg = generate_diff_guide(original_quote, rewrite_text, situation)
        st.session_state.messages.append({
            "role": "assistant",
            "content": msg
        })
        st.rerun()

def generate_diff_guide(original_quote: str, rewrite_quote: str, situation: str) -> str:
    story_type = "学业" if situation == "A" else "人际"

    user_prompt = f"""这是{story_type}故事的视角改写对比：

旁观视角：「{original_quote}」

沉浸视角（第一人称）：「{rewrite_quote}」

请生成一段简短的引导语，帮助用户观察和讨论这两个版本的差异。"""

    return call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7, max_tokens=300)

def render_diff_ui():
    """渲染视角对比界面"""
    current_stage = STAGES[st.session_state.stage_index]
    if current_stage == "P4A_DIFF":
        situation = "A"
    else:
        situation = "B"

    if situation == "A":
        material = st.session_state.stage_A_material
        # 优先使用存储的数据，回退到对话历史
        original_quote = st.session_state.stage_A2a_quote or material.get("dilemma") or material.get("context") or ""
        reflection = st.session_state.stage_A2b_reflection or material.get("reaction") or ""
        framing = st.session_state.stage_A2c_framing or material.get("impact") or ""
        # 如果都为空，从对话历史中获取 A2a 阶段的内容
        if not original_quote:
            original_quote = _get_stage_content_from_history("A2a")
        if not reflection:
            reflection = _get_stage_content_from_history("A2b")
        if not framing:
            framing = _get_stage_content_from_history("A2c")
        rewrite_quote = st.session_state.stage4_rewrite_A
        diff_comment_key = "stage4_diff_A"
        story_type = "学业"
    else:
        material = st.session_state.stage_B_material
        original_quote = st.session_state.stage_B2a_quote or material.get("dilemma") or material.get("context") or ""
        reflection = st.session_state.stage_B2b_reflection or material.get("reaction") or ""
        framing = st.session_state.stage_B2c_framing or material.get("impact") or ""
        if not original_quote:
            original_quote = _get_stage_content_from_history("B2a")
        if not reflection:
            reflection = _get_stage_content_from_history("B2b")
        if not framing:
            framing = _get_stage_content_from_history("B2c")
        rewrite_quote = st.session_state.stage4_rewrite_B
        diff_comment_key = "stage4_diff_B"
        story_type = "人际"

    st.markdown(f"### 两个版本的对比：{story_type}故事")

    # 显示旁观视角的三个维度
    st.markdown("**旁观视角（第三人称）**")

    st.markdown(f"**内心独白**：{original_quote}")

    if reflection:
        st.markdown(f"**声音来源与普遍性**：{reflection}")

    if framing:
        st.markdown(f"**对困难的认知**：{framing}")

    st.markdown("---")

    # 显示沉浸视角
    st.markdown("**沉浸视角（第一人称）**")
    st.markdown(f"> {rewrite_quote}")

    st.markdown("---")
    st.markdown("**你觉得两个版本有什么不同？为什么会有这些差异？**")

    current_diff = st.session_state.get(diff_comment_key, "")
    diff_comment = st.text_area(
        "你的观察和思考",
        value=current_diff,
        height=100,
        key=f"diff_comment_{situation}",
        placeholder="分享你的观察..."
    )

    st.session_state[diff_comment_key] = diff_comment

    if st.button("提交并继续", type="primary"):
        log_event("p4_diff_submit", situation=situation,
                  diff_length=len(diff_comment))

        if situation == "A":
            st.session_state.stage_index = STAGES.index("P4B_REWRITE")
            st.session_state.stage4_start_time = time.time()
            msg = generate_guide_message("P4B_REWRITE", {
                "name": st.session_state.character_name,
                "quote": st.session_state.stage_B2a_quote
            })
        else:
            st.session_state.stage_index = STAGES.index("DEBRIEF")
            msg = generate_guide_message("DEBRIEF")

        st.session_state.messages.append({
            "role": "assistant",
            "content": msg
        })
        st.rerun()

def render_debrief_ui():
    """渲染总结界面"""
    st.markdown("### 创作回顾")

    st.markdown("""
    创作完成了！想听听你的感受：
    
    1. **整体体验** - 这次创作过程中，整体感觉怎么样？
    2. **创作感受** - 有没有哪个瞬间觉得这不只是在创作角色？
    3. **背后意图** - 你觉得这个体验背后想探索什么？
    """)

    response = st.text_area(
        "你的回应",
        height=150,
        key="debrief_response_input",
        placeholder="分享你的感受和想法..."
    )

    if st.button("完成创作", type="primary"):
        st.session_state.debrief_response = response

        log_event("debrief_complete", response_length=len(response))

        save_session()

        st.session_state.stage_index = STAGES.index("DONE")
        st.rerun()

def render_done_ui():
    """渲染完成界面"""
    st.markdown("## 感谢你的参与和创作！")

    st.markdown("""
    你成功创作了一个虚构角色的故事，并亲身体验了编剧视角转换的练习。
    
    在旁观视角和沉浸视角之间切换，是创作者常用的技巧。
    它帮助我们理解角色的内在世界，同时保持一定的距离来审视故事。
    
    ---
    
    **你的角色**: {name}
    
    **学业故事**: {a_summary}
    
    **人际故事**: {b_summary}
    """.format(
        name=st.session_state.character_name,
        a_summary=st.session_state.stage_A_material.get("dilemma", "未填写")[:100] + "...",
        b_summary=st.session_state.stage_B_material.get("dilemma", "未填写")[:100] + "..."
    ))

    if st.session_state.portrait_final and os.path.exists(st.session_state.portrait_final):
        st.image(st.session_state.portrait_final, width=300, caption="最终角色画像")

    st.markdown("---")
    st.markdown("感谢你的时间和创意！如有需要可以重新开始。")

def handle_auto_transition(target_stage: str):
    """处理自动过渡"""
    st.session_state.stage_index = STAGES.index(target_stage)
    msg = generate_guide_message(target_stage, {"name": st.session_state.character_name})
    st.session_state.messages.append({
        "role": "assistant",
        "content": msg
    })
    st.rerun()

def handle_user_input(user_input: str):
    """处理用户输入"""
    current_stage = STAGES[st.session_state.stage_index]

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    save_session()

    log_event("user_input", content_length=len(user_input))

    store_data(current_stage, user_input)

    # S0c 阶段：解析风格并生成画像
    if current_stage == "S0c":
        from image_gen import get_available_styles
        styles = get_available_styles()

        matched_style = match_style_with_llm(user_input, styles)

        if matched_style:
            st.session_state.portrait_style = matched_style
            style_name = next((s["name"] for s in styles if s["key"] == matched_style), matched_style)
            confirm_msg = f"好的，使用{style_name}风格生成画像，请稍候..."
        else:
            default_style = styles[0]["key"] if styles else "写实"
            st.session_state.portrait_style = default_style
            style_name = styles[0]["name"] if styles else "写实"
            confirm_msg = f"收到，我将使用「{style_name}」风格为您生成画像，请稍候..."

        st.session_state.messages.append({
            "role": "assistant",
            "content": confirm_msg
        })

        st.session_state.portrait_generating = True
        save_session()
        st.rerun()
        return

    should_follow = should_follow_up(current_stage, user_input)
    follow_ups = st.session_state.follow_up_count < 2

    if should_follow and follow_ups:
        follow_up_msg = generate_follow_up(
            current_stage,
            st.session_state.messages[-4:]
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": follow_up_msg
        })
        log_event("follow_up", follow_up_count=st.session_state.follow_up_count)
        st.session_state.follow_up_count += 1
        st.rerun()
    else:
        st.session_state.follow_up_count = 0

        next_idx = min(st.session_state.stage_index + 1, len(STAGES) - 1)
        log_event("stage_advance", from_stage=current_stage, to_stage=STAGES[next_idx])

        if current_stage == "S0b":
            st.session_state.stage_index = STAGES.index("S0c")
            msg = generate_guide_message("S0c", {"name": st.session_state.character_name})
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg
            })

        else:
            st.session_state.stage_index = min(st.session_state.stage_index + 1, len(STAGES) - 1)

            next_stage = STAGES[st.session_state.stage_index]
            next_msg = generate_guide_message(
                next_stage,
                {
                    "name": st.session_state.character_name,
                    "summary": st.session_state.stage_A_material.get("dilemma", "")[:50] if "A" in next_stage else st.session_state.stage_B_material.get("dilemma", "")[:50],
                    "quote": st.session_state.stage_A2a_quote if "A" in next_stage and "REWRITE" in next_stage else st.session_state.stage_B2a_quote
                }
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": next_msg
            })

        st.rerun()

# =============================================================================
# 主函数
# =============================================================================
def main():
    """主函数"""
    st.set_page_config(
        page_title="AI共创叙事画坊",
        page_icon=":art:",
        layout="wide"
    )

    init_session_state()
    render_sidebar()

    st.title("AI共创叙事画坊")
    st.markdown("---")

    current_stage = STAGES[min(st.session_state.stage_index, len(STAGES)-1)]

    render_messages()

    # 根据阶段渲染对应界面
    show_chat = True

    if current_stage == "S0c":
        if st.session_state.get("portrait_generated", False):
            render_portrait_ui()
            show_chat = False
        elif st.session_state.get("portrait_generating", False):
            render_portrait_ui()
            show_chat = False
        else:
            st.info("🎨 请告诉我你想要的画像风格，例如：写实、动漫、水彩、素描、油画、国风等")

    elif current_stage in ["A3a", "B3a"]:
        # 分镜描述输入界面 - 直接进入逐格生成页面
        situation = "A" if current_stage == "A3a" else "B"
        story_type = "学业" if situation == "A" else "人际"
        
        st.markdown(f"### 🎬 {story_type}故事连环画生成")
        st.markdown("请为每个分镜输入描述，然后点击「生成」按钮")
        
        # 检查是否需要重新初始化（根据 situation 区分）
        current_comic = st.session_state.stage_A_comic if situation == "A" else st.session_state.stage_B_comic
        need_init = (
            st.session_state.comic_frames_parsed is None or 
            len(st.session_state.comic_frames_parsed) == 0 or
            st.session_state.comic_situation != situation
        )
        
        if need_init:
            # 默认3个空分镜
            default_frames = [
                {"description": "", "frame_number": 1},
                {"description": "", "frame_number": 2},
                {"description": "", "frame_number": 3}
            ]
            st.session_state.comic_frames_parsed = default_frames
            
            # 初始化 comic_data
            comic_data = {"frames": [], "confirmed": False}
            for i, fp in enumerate(default_frames):
                frame_record = {
                    "frame_index": i + 1,
                    "description": "",
                    "versions": [],
                    "current_version": 0,
                    "final_version": 0,
                    "final_path": "",
                    "image_path": ""
                }
                comic_data["frames"].append(frame_record)
            
            if situation == "A":
                st.session_state.stage_A_comic = comic_data
            else:
                st.session_state.stage_B_comic = comic_data
        
        st.session_state.comic_situation = situation
        st.session_state.stage_index = STAGES.index(f"{situation}3b")
        save_session()
        st.rerun()

    elif current_stage in ["A3b", "B3b"]:
        render_comic_ui()
        show_chat = False

    elif current_stage in ["P4A_REWRITE", "P4B_REWRITE"]:
        render_rewrite_ui()
        show_chat = False

    elif current_stage in ["P4A_DIFF", "P4B_DIFF"]:
        render_diff_ui()
        show_chat = False

    elif current_stage == "DEBRIEF":
        render_debrief_ui()
        show_chat = False

    elif current_stage == "DONE":
        render_done_ui()
        show_chat = False

    elif current_stage in ["S0d", "A3c", "B3c"]:
        if st.button("继续", type="primary"):
            if current_stage == "S0d":
                handle_auto_transition("A1a")
            elif current_stage == "A3c":
                handle_auto_transition("B1a")
            elif current_stage == "B3c":
                handle_auto_transition("P4A_REWRITE")
        show_chat = False

    if show_chat:
        if prompt := st.chat_input("请输入你的回应..."):
            handle_user_input(prompt)


if __name__ == "__main__":
    main()