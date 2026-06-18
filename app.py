"""
AI共创叙事画坊 - Streamlit 应用
一个结合AI对话与图像生成的互动叙事平台
"""
import io
import zipfile
import streamlit as st
import os
import json
import time
import re
import shutil
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any, List, Optional
from openai import OpenAI
from image_gen import generate_portrait, generate_comic_frame, extract_character_features, get_llm_client

st.set_page_config(page_title="AI共创叙事画坊", page_icon=":art:", layout="centered")

# =============================================================================
# 响应式样式（电脑 / 手机自适应）
# =============================================================================
MOBILE_STYLE = """
<style>
/* 收紧内容区，避免手机端顶栏占屏过多 */
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 960px; margin: 0 auto;}
@media (max-width: 640px) {
  .block-container {padding-left: .75rem; padding-right: .75rem;}
  h1 {font-size: 1.5rem !important;}
  h2 {font-size: 1.25rem !important;}
  h3 {font-size: 1.1rem !important;}
}

/* 图片自适应 */
.stImage, [data-testid="stImage"] {max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto;}
img {max-width: 100%; height: auto;}

/* 按钮在手机上更容易点击 */
button, [data-testid="stButton"] button {min-height: 42px;}
</style>
"""
st.markdown(MOBILE_STYLE, unsafe_allow_html=True)

# =============================================================================
# 配置
# =============================================================================
llm_client = get_llm_client()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ensure_dir = lambda path: os.makedirs(path, exist_ok=True)

# 确保目录存在
ensure_dir(DATA_DIR)

def build_session_folder_name(start_time: Optional[str] = None) -> str:
    """根据会话开始时间生成目录名"""
    raw_time = start_time or datetime.now().isoformat()
    try:
        dt = datetime.fromisoformat(raw_time)
    except ValueError:
        dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def get_session_storage_name(session_id: Optional[str] = None) -> str:
    """获取当前会话实际使用的存储目录名"""
    if session_id is None:
        session_id = st.session_state.session_id

    session_dirs = st.session_state.get("session_dirs", {})
    if session_id in session_dirs:
        return session_dirs[session_id]

    if "start_time" in st.session_state:
        folder_name = build_session_folder_name(st.session_state.get("start_time"))
        session_dirs[session_id] = folder_name
        st.session_state.session_dirs = session_dirs
        return folder_name

    return session_id


def get_session_dir(session_id: str = None) -> str:
    """获取会话目录路径"""
    storage_name = get_session_storage_name(session_id)
    return os.path.join(DATA_DIR, "sessions", storage_name)

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
# 阶段定义（引导语与展示名称；故事流程由情境内目标驱动）
# =============================================================================

# 阶段中文名称
STAGE_NAMES = {
    "S0a": "角色命名", "S0b": "角色外貌", "S0c": "生成画像", "S0d": "确认形象",
    "A1a": "困扰收集", "A1b": "具体情境", "A1c": "角色反应", "A1d": "后续影响",
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
    "S0a": "引导用户为虚构人物命名、确定年龄和性别，并说明上学或工作的具体信息",
    "S0b": "引导用户描述角色的外貌特征",
    "S0c": "询问用户想要的画像风格，然后生成画像",
    "S0d": "简短确认角色形象，过渡到后续故事",
    "A1a": "引导用户描述角色在工作或学业方面遇到的困扰事件",
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
    "P4A_REWRITE": "引入编剧视角切换概念，引导用户用第一人称改写{work_or_study}故事内心独白",
    "P4A_DIFF": "展示旁观视角和沉浸视角两个版本对比，引导讨论差异",
    "P4B_REWRITE": "引导用户用第一人称改写人际故事内心独白",
    "P4B_DIFF": "展示两个版本对比，引导讨论差异",
    "DEBRIEF": "引导用户回顾创作体验",
    "DONE": "感谢页"
}

# 阶段提示参考
STAGE_REFS = {
    "S0a": "今天一起创作一个角色——先认识一下TA。请构思一个虚构的人物，TA的姓名是？性别是？年龄多大了？现在是在上学还是工作？",
    "S0b": "让{name}的形象更具体——TA长什么样？发型、脸型、眉眼、身材、穿衣风格？整体气质偏活泼还是沉静？",
    "S0c": "好的，{name}的形象很清晰了。接下来我来生成TA的画像。你想用什么风格？比如：写实、动漫、水彩、素描、油画、国风，或者你喜欢的其他风格都可以告诉我。",
    "S0d": "这就是{name}了——看起来是个有故事的人。接下来我们一起来探索TA的某些故事。",
    "A1a": "每个人都会遇到各种状况——{name}在{work_or_study}方面有没有什么让TA感到困扰或沮丧的事？可以是某次具体的事情，也可以是TA一直以来的某种处境或状态。",
    "A1b": "能再说具体一点吗？",
    "A1c": "当时{name}是什么反应？做了什么？或者有什么想做的但没做？",
    "A1d": "这件事之后对{name}有什么影响？心情、作息、和他人相处有什么变化？",
    "A2a": "故事很具体了。好的创作还要揣摩角色的内心——这件事发生时，{name}心里最先冒出来的那句话是什么？最直觉的第一反应。原汁原味写下来，包括语气和用词。",
    "A2b": "{name}的这个反应——更像TA自己的声音，还是像某个重要的人可能对TA说的话？TA会觉得只有自己才会这样，还是谁都可能遇到？",
    "A2c": "在{name}看来，这个困难说明了什么？一次偶然——还是暴露了TA一直以来的问题？TA是被这感觉困住，还是能意识到'我正在经历困难'？",
    "A3a": "刚才描述了{name}的故事。现在用AI生成连环画。挑3个及以上的关键瞬间，每格描述具体的场景、{name}的状态和感受。",
    "A3b": "正在生成连环画...",
    "A3c": "{work_or_study}故事的连环画完成了。接下来我们来探索{name}在人际关系方面的故事。",
    "B1a": "除了{work_or_study}，日常中人际关系也很重要——{name}在人际方面有没有什么困扰？不管是某次具体事件，还是一直以来让TA不太舒服的状态或处境。",
    "B1b": "能再说具体一点吗？",
    "B1c": "当时{name}是什么反应？做了什么？或者有什么想做的但没做？",
    "B1d": "这件事之后对{name}有什么影响？心情、作息、和他人相处有什么变化？",
    "B2a": "故事很具体了。现在来揣摩角色的内心——这件事发生时，{name}心里最先冒出来的那句话是什么？",
    "B2b": "{name}的这个反应——更像TA自己的声音，还是像某个重要的人可能对TA说的话？TA会觉得只有自己才会这样，还是谁都可能遇到？",
    "B2c": "在{name}看来，这个困难说明了什么？一次偶然——还是暴露了TA一直以来的问题？TA是被这感觉困住，还是能意识到'我正在经历困难'？",
    "B3a": "刚才描述了{name}的故事。现在挑3到5个关键瞬间来生成连环画，每格描述具体的场景、{name}的状态和感受。",
    "B3b": "正在生成连环画...",
    "B3c": "人际故事的连环画也完成了。接下来我们来做一个小小的编剧练习。",
    "P4A_REWRITE": "来做编剧的进阶练习——写独白时编剧会试不同视角：旁观视角（第三人称）和沉浸视角（第一人称）。刚才写的是旁观视角，现在试沉浸视角——完全代入{name}，假如你就是{name}。",
    "P4B_REWRITE": "同样的练习——现在试沉浸视角，完全代入{name}，用'我'来写。这是旁观版本：{quote}，假如你就是{name}，请代入改写。",
    "P4B_DIFF": "两个版本有什么不同？为什么会有这些差异？",
    "DEBRIEF": "创作完成了！想听听你的感受——整体体验怎么样？有没有哪个瞬间觉得这不只是在创作角色？你觉得背后想探索什么？",
    "DONE": "感谢你的参与和创作！"
}

# =============================================================================
# 情境内目标定义（学业 / 人际）
# =============================================================================

CONTEXT_ACADEMIC = "academic"
CONTEXT_SOCIAL = "social"

ACADEMIC_TARGETS = [
    "story_dilemma", "story_context", "story_reaction", "story_impact",
    "self_kindness", "common_humanity", "mindfulness",
    "comic_frames", "comic_confirmed",
]

SOCIAL_TARGETS = list(ACADEMIC_TARGETS)

TARGET_NAMES = {
    "story_dilemma": "核心困扰",
    "story_context": "具体情境",
    "story_reaction": "行为反应",
    "story_impact": "后续影响",
    "self_kindness": "内心第一反应",
    "common_humanity": "声音来源/普遍性",
    "mindfulness": "困难框架",
    "comic_frames": "连环画分镜",
    "comic_confirmed": "连环画确认",
}

TARGET_DESCRIPTIONS = {
    "story_dilemma": "描述角色遇到的{work_or_study}/人际困扰",
    "story_context": "描述具体的时间、地点、人物和细节",
    "story_reaction": "描述角色的行为反应",
    "story_impact": "描述事件对角色的后续影响",
    "self_kindness": "写出角色内心的第一反应原话",
    "common_humanity": "探讨声音来源和普遍性",
    "mindfulness": "探讨角色如何框定这个困难",
    "comic_frames": "挑选3-5个关键瞬间生成连环画",
    "comic_confirmed": "确认连环画生成",
}

TARGET_SUFFICIENCY_CRITERIA = {
    "S0a": "用户明确提供了角色姓名、年龄、性别，并说明上学或工作；若上学还需学段/年级，大学则需要进一步说明专业，若工作还需职业和工龄",
    "S0b": "用户描述了至少三处具体外貌或气质特征，而非仅一个词。",
    "story_dilemma": "用户说明了角色具体的困扰、事件或持续处境，有实质内容而非空泛表述。",
    "story_context": "用户补充了具体场合、在场人物或发生了什么，能让场景成形。",
    "story_reaction": "用户描述了角色当时的行为、反应或想做什么却没做，而非仅情绪标签。",
    "story_impact": "用户说明了事件之后对心情、作息或与他人相处等方面的变化。",
    "self_kindness": "用户给出了角色内心第一反应的原话或贴近原话的转述，带语气、用词自然；需体现内心声音，不能只有「很难过」「很焦虑」等概括。",
    "common_humanity": "用户对声音来源（更像自己还是像重要他人）或普遍性（是否觉得只有自己会这样）至少有一点具体判断。",
    "mindfulness": "用户对困难的意义有立场：偶然/长期、被困住还是能意识到正在经历困难等，而非空泛的「不知道」。",
    "comic_frames": "用户提供了至少3个可画成画面的瞬间描述，含人物状态或感受。",
    "comic_confirmed": "用户明确表示确认或满意。",
}

TARGET_COLLECT_PROMPTS = {
    "S0a": "虚构角色的年龄、姓名、性别、上学/工作状态，以及对应学段/学历或职业与工龄，大学需要专业，可含一点性格或特点。",
    "S0b": "角色的外貌：发型、脸型、眉眼、身材、穿衣风格、常见表情或整体气质。",
    "story_dilemma": "角色遇到了什么{work_or_study}/人际困扰？是什么事件或处境？",
    "story_context": "具体场合、在场人物、当时发生了什么细节？",
    "story_reaction": "角色当时做了什么、没做什么，或强烈想做什么？",
    "story_impact": "这件事之后对角色心情、作息或与他人相处有什么变化？",
    "self_kindness": "角色内心第一反应的原话是什么？最直觉的那个声音，保留语气和用词。",
    "common_humanity": "这个反应是角色自己的声音还是像别人会说的话？角色会觉得只有自己才会这样吗？",
    "mindfulness": "这个困难是偶然还是一直以来的问题？角色是被感觉困住，还是能意识到自己在经历困难？",
    "comic_frames": "挑3到5个关键瞬间，每格描述角色在画面中的状态和感受。",
    "comic_confirmed": "确认连环画是否满意、是否继续下一步。",
}


def get_target_prompt(target_id: str) -> str:
    """返回每个目标需要收集的信息特征描述"""
    return TARGET_COLLECT_PROMPTS.get(
        target_id,
        TARGET_DESCRIPTIONS.get(target_id, "请补充与该目标相关的具体信息。"),
    )


def get_target_sufficiency_criteria(target_id: str) -> str:
    """返回该目标充分回答的判定标准"""
    return TARGET_SUFFICIENCY_CRITERIA.get(
        target_id,
        "用户回答与目标相关且包含可记录的具体信息，而非敷衍或空泛。",
    )


TARGET_TO_STAGE_MAPPING = {
    CONTEXT_ACADEMIC: {
        "story_dilemma": "A1a", "story_context": "A1b", "story_reaction": "A1c",
        "story_impact": "A1d", "self_kindness": "A2a", "common_humanity": "A2b",
        "mindfulness": "A2c", "comic_frames": "A3a", "comic_confirmed": "A3c",
    },
    CONTEXT_SOCIAL: {
        "story_dilemma": "B1a", "story_context": "B1b", "story_reaction": "B1c",
        "story_impact": "B1d", "self_kindness": "B2a", "common_humanity": "B2b",
        "mindfulness": "B2c", "comic_frames": "B3a", "comic_confirmed": "B3c",
    },
}

TARGET_GUIDES = {
    CONTEXT_ACADEMIC: {
        "story_dilemma": STAGE_REFS["A1a"],
        "story_context": STAGE_REFS["A1b"],
        "story_reaction": STAGE_REFS["A1c"],
        "story_impact": STAGE_REFS["A1d"],
        "self_kindness": STAGE_REFS["A2a"],
        "common_humanity": STAGE_REFS["A2b"],
        "mindfulness": STAGE_REFS["A2c"],
        "comic_frames": STAGE_REFS["A3a"],
        "comic_confirmed": STAGE_REFS["A3c"],
    },
    CONTEXT_SOCIAL: {
        "story_dilemma": STAGE_REFS["B1a"],
        "story_context": STAGE_REFS["B1b"],
        "story_reaction": STAGE_REFS["B1c"],
        "story_impact": STAGE_REFS["B1d"],
        "self_kindness": STAGE_REFS["B2a"],
        "common_humanity": STAGE_REFS["B2b"],
        "mindfulness": STAGE_REFS["B2c"],
        "comic_frames": STAGE_REFS["B3a"],
        "comic_confirmed": STAGE_REFS["B3c"],
    },
}

# 线性流程阶段（角色创建、视角练习、总结；不含学业/人际故事子阶段）
LINEAR_PHASES = [
    "S0a", "S0b", "S0c", "S0d",
    "P4A_REWRITE", "P4A_DIFF", "P4B_REWRITE", "P4B_DIFF",
    "DEBRIEF", "DONE",
]


def _empty_collected_targets() -> Dict[str, Dict]:
    return {CONTEXT_ACADEMIC: {}, CONTEXT_SOCIAL: {}}


def get_context_targets(context: str) -> List[str]:
    if context == CONTEXT_ACADEMIC:
        return ACADEMIC_TARGETS
    if context == CONTEXT_SOCIAL:
        return SOCIAL_TARGETS
    return []


def get_current_incomplete_targets(context: str = None) -> List[str]:
    if context is None:
        context = st.session_state.get("current_context", CONTEXT_ACADEMIC)
    collected = st.session_state.collected_targets.get(context, {})
    return [t for t in get_context_targets(context) if not collected.get(t, {}).get("met")]


def get_next_target(context: str = None) -> Optional[str]:
    incomplete = get_current_incomplete_targets(context)
    return incomplete[0] if incomplete else None


def is_context_complete(context: str = None) -> bool:
    return len(get_current_incomplete_targets(context)) == 0


def mark_target_met(target_id: str, context: str = None, data: Any = None):
    if context is None:
        context = st.session_state.get("current_context", CONTEXT_ACADEMIC)
    if context not in st.session_state.collected_targets:
        st.session_state.collected_targets[context] = {}
    st.session_state.collected_targets[context][target_id] = {
        "met": True, "data": data, "timestamp": datetime.now().isoformat()
    }


def is_in_story_flow() -> bool:
    """是否处于学业/人际目标驱动对话中"""
    return st.session_state.get("story_flow_active", False)


def _get_work_or_study_term() -> str:
    """根据角色状态返回'学业'或'工作'"""
    status = st.session_state.get("character_status", "")
    if status == "study":
        return "学业"
    if status == "work":
        return "工作"
    return "学业"


def get_work_or_study_label(label_type: str = "default") -> str:
    """
    获取动态的学业/工作相关标签。
    
    Args:
        label_type: 标签类型
            - "default": 返回"学业"或"工作"
            - "story": 返回"学业故事"或"工作故事"
            - "comic": 返回"学业连环画"或"工作连环画"
            - "trouble": 返回"学业困扰"或"工作困扰"
            - "complete": 返回"学业故事...已完成"或"工作故事...已完成"
    """
    term = _get_work_or_study_term()
    
    labels = {
        "default": term,
        "story": f"{term}故事",
        "comic": f"{term}连环画",
        "trouble": f"{term}困扰",
        "complete": f"{term}故事的连环画完成了",
    }
    
    return labels.get(label_type, term)


def generate_target_guide(target_id: str, context: str = None) -> str:
    if context is None:
        context = st.session_state.get("current_context", CONTEXT_ACADEMIC)
    template = TARGET_GUIDES.get(context, {}).get(target_id, "请继续。")
    name = st.session_state.get("character_name", "TA")
    if context == CONTEXT_ACADEMIC:
        summary = (st.session_state.stage_A_material.get("dilemma") or "")[:50]
    else:
        summary = (st.session_state.stage_B_material.get("dilemma") or "")[:50]
    text = template.replace("{name}", name).replace("{summary}", summary)
    term = _get_work_or_study_term()
    text = text.replace("{work_or_study}", term)
    return text.replace("学业", term)


def switch_to_social_context() -> str:
    """切换到人际故事上下文，返回引导语"""
    st.session_state.current_context = CONTEXT_SOCIAL
    st.session_state.current_target = "story_dilemma"
    # 从 TARGET_GUIDES 获取模板并替换所有占位符
    template = TARGET_GUIDES[CONTEXT_SOCIAL]["story_dilemma"]
    name = st.session_state.get("character_name", "TA")
    work_or_study = _get_work_or_study_term()
    return template.replace("{name}", name).replace("{work_or_study}", work_or_study)


def evaluate_user_input(
    user_input: str,
    context: str = None,
    conversation_history: List[Dict] = None,
) -> Dict[str, Any]:
    if context is None:
        context = st.session_state.get("current_context", CONTEXT_ACADEMIC)
    conversation_history = conversation_history or []
    current_target = st.session_state.get("current_target") or get_next_target(context)
    if not current_target:
        return {"identified_targets": [], "is_sufficient": True, "follow_up_needed": False, "extracted_data": {}}

    history_summary = ""
    for msg in conversation_history[-6:]:
        role = "用户" if msg.get("role") == "user" else "助手"
        history_summary += f"{role}：{msg.get('content', '')[:120]}\n"

    ctx_label = f"{_get_work_or_study_term()}故事" if context == CONTEXT_ACADEMIC else "人际故事"
    collect_hint = get_target_prompt(current_target)
    sufficiency_rule = get_target_sufficiency_criteria(current_target)

    system_prompt = f"""你是创作对话评估助手。根据「充分回答标准」判断用户是否已满足当前目标，只返回JSON。

【当前情境】{ctx_label}
【当前目标ID】{current_target}
【目标名称】{TARGET_NAMES.get(current_target, current_target)}
【需收集的信息特征】{collect_hint}

【充分回答标准（is_sufficient=true 须同时满足）】
{sufficiency_rule}

【追问标准（follow_up_needed=true）】
当 is_sufficient 为 false，且用户回答过短、空泛、跑题或缺少上述核心特征时设为 true。

【提取】将用户输入中与当前目标相关的实质内容写入 extracted_data。"""

    user_prompt = f"""对话摘要：
{history_summary}

用户最新输入：{user_input}

只返回JSON：
{{"identified_targets":["{current_target}"], "is_sufficient":true/false, "follow_up_needed":true/false, "extracted_data":{{"{current_target}":"提取的内容"}}}}"""

    try:
        model = os.getenv("LLM_MODEL", "qwen-max")
        print(f"\n{'='*60}")
        print(f"[模型调用] extract_target_data (数据提取)")
        print(f"  模型: {model}")
        print(f"  用途: 提取对话目标数据")
        print(f"{'='*60}")
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        print(f"[模型调用] extract_target_data 完成")
        result_text = re.sub(r"```json\s*|```", "", response.choices[0].message.content.strip())
        result = json.loads(result_text)
        is_sufficient = bool(result.get("is_sufficient", False))
        follow_up_needed = bool(result.get("follow_up_needed", False))
        if not is_sufficient and not follow_up_needed:
            follow_up_needed = True
        return {
            "identified_targets": result.get("identified_targets", [current_target]),
            "is_sufficient": is_sufficient,
            "follow_up_needed": follow_up_needed,
            "extracted_data": result.get("extracted_data", {current_target: user_input}),
        }
    except Exception:
        stripped = user_input.strip()
        trivial = stripped in ["是", "否", "嗯", "哦", "好", "行", "对", "可以"]
        is_sufficient = len(stripped) >= 8 and not trivial
        if current_target == "self_kindness":
            is_sufficient = is_sufficient and (
                "「" in user_input or "」" in user_input or '"' in user_input
                or "我" in user_input or len(stripped) >= 12
            )
        return {
            "identified_targets": [current_target],
            "is_sufficient": is_sufficient,
            "follow_up_needed": not is_sufficient,
            "extracted_data": {current_target: user_input},
        }


def store_target_data(target_id: str, context: str, data: Any):
    """写入 collected_targets 并同步到既有 stage_A/B_material 等字段"""
    mark_target_met(target_id, context, data)
    stage = TARGET_TO_STAGE_MAPPING.get(context, {}).get(target_id)
    if stage and data:
        store_data(stage, data if isinstance(data, str) else str(data))


def _init_comic_frames():
    situation = "A" if st.session_state.current_context == CONTEXT_ACADEMIC else "B"
    saved_comic = st.session_state.stage_A_comic if situation == "A" else st.session_state.stage_B_comic
    existing_frames = saved_comic.get("frames", []) if isinstance(saved_comic, dict) else []

    def _frame_to_parsed(frame, idx):
        return {
            "description": frame.get("description", ""),
            "story_text": frame.get("story_text", frame.get("description", "")),
            "visual_description": frame.get("visual_description", frame.get("description", "")),
            "frame_number": frame.get("frame_index", idx + 1),
        }

    if existing_frames:
        st.session_state.comic_frames_parsed = [_frame_to_parsed(frame, idx) for idx, frame in enumerate(existing_frames)]
        st.session_state.comic_situation = situation
        st.session_state.comic_style = st.session_state.get("portrait_style", "动漫")
        st.session_state.comic_story_generated = any(frame.get("story_text") for frame in existing_frames)
        st.session_state.comic_story_approved = saved_comic.get("confirmed", False)
        st.session_state.comic_story_data = [
            {
                "story_text": frame.get("story_text", frame.get("description", "")),
                "visual_description": frame.get("visual_description", frame.get("description", "")),
            }
            for frame in existing_frames
        ]
        return

    default_frames = [
        {
            "description": "",
            "story_text": "",
            "visual_description": "",
            "frame_number": 1,
        },
        {
            "description": "",
            "story_text": "",
            "visual_description": "",
            "frame_number": 2,
        },
        {
            "description": "",
            "story_text": "",
            "visual_description": "",
            "frame_number": 3,
        },
    ]
    st.session_state.comic_frames_parsed = default_frames
    comic_data = {"frames": [], "confirmed": False}
    for i, _ in enumerate(default_frames):
        comic_data["frames"].append({
            "frame_index": i + 1,
            "description": "",
            "story_text": "",
            "visual_description": "",
            "versions": [],
            "current_version": 0,
            "final_version": 0,
            "final_path": "",
            "image_path": "",
        })
    if situation == "A":
        st.session_state.stage_A_comic = comic_data
    else:
        st.session_state.stage_B_comic = comic_data
    st.session_state.comic_situation = situation
    st.session_state.comic_style = st.session_state.get("portrait_style", "动漫")
    st.session_state.comic_story_generated = False
    st.session_state.comic_story_approved = False
    st.session_state.comic_story_data = []


def legacy_stage_to_context_target(stage: str):
    """旧版 stage 标识 -> (context, target)"""
    for ctx, mapping in TARGET_TO_STAGE_MAPPING.items():
        for target, stg in mapping.items():
            if stg == stage:
                return ctx, target
    return None, None


def get_app_phase() -> str:
    return st.session_state.get("app_phase", "S0a")


def _on_comic_flow_complete(situation: str):
    """连环画确认完成后的情境切换"""
    context = CONTEXT_ACADEMIC if situation == "A" else CONTEXT_SOCIAL
    mark_target_met("comic_confirmed", context, {"confirmed": True})
    if situation == "A":
        msg = switch_to_social_context()
    else:
        st.session_state.story_flow_active = False
        st.session_state.app_phase = "P4A_REWRITE"
        msg = generate_guide_message("P4A_REWRITE", {
            "name": st.session_state.character_name,
            "quote": st.session_state.stage_A2a_quote,
        })
        st.session_state.stage4_mode = True
        st.session_state.stage4_situation = "A"
        st.session_state.stage4_start_time = time.time()
    st.session_state.messages.append({"role": "assistant", "content": msg})
    save_session()
    st.rerun()


# =============================================================================
# LLM 调用函数
# =============================================================================
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """调用LLM生成回复"""
    try:
        model = os.getenv("LLM_MODEL", "qwen-max")
        print(f"\n{'='*60}")
        print(f"[模型调用] call_llm (通用对话)")
        print(f"  模型: {model}")
        print(f"  用途: 生成回复")
        print(f"{'='*60}")
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        print(f"[模型调用] call_llm 完成")
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
        model = os.getenv("LLM_MODEL", "qwen-max")
        print(f"\n{'='*60}")
        print(f"[模型调用] is_state_description (状态/事件判断)")
        print(f"  模型: {model}")
        print(f"  用途: 判断用户描述是日常状态还是具体事件")
        print(f"{'='*60}")
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200,
        )
        print(f"[模型调用] is_state_description 完成")

        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'```json\s*|```', '', result_text)
        result = json.loads(result_text)

        is_state = result.get("type") == "state"
        st.session_state.state_cache[cache_key] = is_state
        return is_state
    except Exception:
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
        model = os.getenv("LLM_MODEL", "qwen-max")
        print(f"\n{'='*60}")
        print(f"[模型调用] match_style_with_llm (风格匹配)")
        print(f"  模型: {model}")
        print(f"  用途: 匹配用户描述到预定义风格")
        print(f"{'='*60}")
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        print(f"[模型调用] match_style_with_llm 完成")

        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'```json\s*|```', '', result_text)
        result = json.loads(result_text)

        matched = result.get("matched_style")
        return matched if matched and matched != "null" else None
    except Exception:
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
        model = os.getenv("LLM_MODEL", "qwen-max")
        print(f"\n{'='*60}")
        print(f"[模型调用] generate_context_aware_follow_up (个性化追问)")
        print(f"  模型: {model}")
        print(f"  用途: 根据描述类型生成个性化追问")
        print(f"{'='*60}")
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是叙事引导助手。直接返回追问，不要评价，不要加引号。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        print(f"[模型调用] generate_context_aware_follow_up 完成")
        follow_up = response.choices[0].message.content.strip()
        follow_up = follow_up.strip('"\'')
        return follow_up
    except Exception:
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
        "S0a": """从用户输入提取：name(姓名), age(年龄), gender(female/male), status(study/work), education_level(学段/学历), occupation(职业), work_years(工龄)。只提取明确提到的，没有就留空。
用户输入：{input}
返回JSON：{{"name": "", "age": "", "gender": "", "status": "", "education_level": "", "occupation": "", "work_years": ""}}""",

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
        model = os.getenv("LLM_MODEL", "qwen-max")
        print(f"\n{'='*60}")
        print(f"[模型调用] parse_user_input_with_llm (用户输入解析)")
        print(f"  模型: {model}")
        print(f"  用途: 解析用户输入，提取阶段数据")
        print(f"{'='*60}")
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        print(f"[模型调用] parse_user_input_with_llm 完成")

        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'```json\s*|```', '', result_text)
        return json.loads(result_text)
    except Exception:
        # 降级处理：尝试直接使用原始输入
        return {"_raw": user_input}

def _get_work_or_study_term() -> str:
    status = st.session_state.get("character_status", "")
    if status == "work":
        return "工作"
    return "学业"


def generate_guide_message(stage: str, context: Dict[str, Any] = None, add_transition: bool = False) -> str:
    """根据当前阶段生成引导语

    Args:
        stage: 阶段标识
        context: 上下文变量，用于替换引导语中的占位符
        add_transition: 是否添加过渡句（仅对 S0a 阶段生效）
    """
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
    ref = ref.replace("{work_or_study}", _get_work_or_study_term())

    # S0a 阶段且 add_transition=True 时添加过渡句
    if add_transition and stage == "S0a":
        transition = "好，那我们开始吧。"
        ref = transition + ref

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


def evaluate_s0_sufficiency(stage: str, user_input: str, history: List[Dict]) -> Dict[str, Any]:
    """评估 S0a/S0b 阶段用户输入的充分性，使用 LLM 判断"""

    history_summary = ""
    for msg in history[-6:]:
        role = "用户" if msg.get("role") == "user" else "助手"
        history_summary += f"{role}：{msg.get('content', '')[:120]}\n"

    if stage == "S0a":
        system_prompt = """你是创作对话评估助手。判断用户是否为角色提供了充分的设定信息。

【当前目标】引导用户为虚构人物命名、确定年龄和性别，并说明现在是在上学还是工作。

【充分回答标准（须全部满足才能 is_sufficient=true）】
1. 角色有明确姓名（不等于"他/她/TA"）
2. 用户说明了年龄（如具体年龄数字，或"二十多岁""中年"等年龄段描述）
3. 用户说明了性别（male/female/男/女等）
4. 用户说明了当前状态：上学或工作
5. 若状态为上学，需包含学段或年级；若状态为工作，需包含职业和大致工龄

【追问标准】
当上述任一条件缺失时，follow_up_needed=true，并生成一句自然追问语补足缺失项。
不要重复用户已提供的信息。
追问要像编剧在确认角色档案，不像是问卷。

【提取】
从用户输入中提取：name, age, gender, status, education_level, occupation, work_years。"""

        user_prompt = f"""对话摘要：
{history_summary}

用户最新输入：{user_input}

只返回JSON：
{{"is_sufficient": true/false, "follow_up_needed": true/false, "follow_up_message": "追问语（仅当 is_sufficient=false 时）", "extracted_data": {{"name":"", "age":"", "gender":"", "status":"", "education_level":"", "occupation":"", "work_years":""}}}}"""

    elif stage == "S0b":
        system_prompt = """你是创作对话评估助手。判断用户是否提供了足够的外貌描述。

【当前目标】引导用户描述角色的外貌特征

【充分回答标准（须同时满足才能 is_sufficient=true）】
1. 描述了至少3处不同的具体外貌或气质特征（如：发型、脸型、眉眼、身材、穿衣风格、气质等）
2. 不能只是单个词（如"短发""普通""正常"等过于笼统的单特征）

【追问标准】
当特征数量不足3处、或描述过于笼统时，follow_up_needed=true，并生成一句自然追问语请求补充。
追问要像编剧在脑海里构思角色形象，自然地多问一两个方面。

【提取】
将用户描述的完整外貌信息提取为 appearance 字段。"""

        user_prompt = f"""对话摘要：
{history_summary}

用户最新输入：{user_input}

只返回JSON：
{{"is_sufficient": true/false, "follow_up_needed": true/false, "follow_up_message": "追问语（仅当 is_sufficient=false 时）", "extracted_data": {{"appearance": "完整外貌描述"}}}}"""

    else:
        return {"is_sufficient": True, "follow_up_needed": False, "extracted_data": {}}

    try:
        model = os.getenv("LLM_MODEL", "qwen-max")
        print(f"\n{'='*60}")
        print(f"[模型调用] evaluate_s0_sufficiency (S0阶段充分性评估)")
        print(f"  模型: {model}")
        print(f"  用途: 评估用户输入是否充分")
        print(f"{'='*60}")
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        print(f"[模型调用] evaluate_s0_sufficiency 完成")
        result_text = re.sub(r"```json\s*|```", "", response.choices[0].message.content.strip())
        result = json.loads(result_text)
        is_sufficient = bool(result.get("is_sufficient", False))
        follow_up_needed = bool(result.get("follow_up_needed", not is_sufficient))
        if not is_sufficient and not follow_up_needed:
            follow_up_needed = True
        return {
            "is_sufficient": is_sufficient,
            "follow_up_needed": follow_up_needed,
            "follow_up_message": result.get("follow_up_message", ""),
            "extracted_data": result.get("extracted_data", {}),
        }
    except Exception:
        # 降级：S0a 需要姓名、性别、状态，以及学段/学历 或 职业+工龄
        stripped = user_input.strip()
        if stage == "S0a":
            name = st.session_state.get("character_name", "")
            age = st.session_state.get("character_age", "")
            gender = st.session_state.get("character_gender", "")
            status = st.session_state.get("character_status", "")
            education_level = st.session_state.get("character_education_level", "")
            occupation = st.session_state.get("character_occupation", "")
            work_years = st.session_state.get("character_work_years", "")
            base_ok = bool(name and age and gender and status)
            study_ok = bool(education_level)
            work_ok = bool(occupation and work_years)
            is_sufficient = base_ok and (study_ok or work_ok)
        else:
            is_sufficient = len(stripped) >= 8
        return {
            "is_sufficient": is_sufficient,
            "follow_up_needed": not is_sufficient,
            "extracted_data": {},
        }


def generate_follow_up(
    target_id: str,
    user_input: str,
    conversation_history: list,
    context: str,
) -> str:
    """基于 LLM 生成上下文追问（不使用模板）"""
    name = st.session_state.get("character_name", "TA")
    collect_hint = get_target_prompt(target_id)

    if context in (CONTEXT_ACADEMIC, CONTEXT_SOCIAL):
        ctx_label = f"{_get_work_or_study_term()}故事" if context == CONTEXT_ACADEMIC else "人际故事"
    else:
        ctx_label = "角色创建"

    history_lines = []
    for msg in (conversation_history or [])[-6:]:
        role = "用户" if msg.get("role") == "user" else "助手"
        history_lines.append(f"{role}：{msg.get('content', '')[:150]}")
    history_text = "\n".join(history_lines) if history_lines else "（暂无）"

    system_prompt = """你是AI共创叙事画坊的创作助手，像编剧一样引导用户完善虚构角色设定。
规则：不评价用户（禁用「很好」「有意思」「不错」等）；不询问用户本人经历；围绕虚构角色讨论。
只输出一句追问，不要引号、不要解释。"""

    user_prompt = f"""情境：{ctx_label}
角色名：{name}
当前要收集的目标：{TARGET_NAMES.get(target_id, target_id)}
该目标需要的信息：{collect_hint}

最近对话：
{history_text}

用户刚才说：{user_input}

请生成一句追问：
1. 先简短承接用户刚才说的内容（不评价）
2. 自然引出上述目标还缺的核心信息
3. 语气像编剧讨论角色，不像问卷
4. 一句话，不超过35字

只返回追问内容。"""

    try:
        model = os.getenv("LLM_MODEL", "qwen-max")
        print(f"\n{'='*60}")
        print(f"[模型调用] generate_follow_up (上下文追问生成)")
        print(f"  模型: {model}")
        print(f"  用途: 生成上下文感知的追问")
        print(f"{'='*60}")
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=120,
        )
        print(f"[模型调用] generate_follow_up 完成")
        follow_up = response.choices[0].message.content.strip().strip('"\'')
        if follow_up and len(follow_up) <= 80:
            return follow_up
    except Exception:
        pass

    # 降级方案：使用模板生成简单追问
    fallback_messages = {
        "story_dilemma": "能再具体说说吗？",
        "story_context": "能再说具体一点吗？",
        "story_reaction": "当时角色有什么反应？",
        "story_impact": "后来有什么变化吗？",
        "self_kindness": "心里第一个冒出来的念头是什么？",
        "common_humanity": "这是TA自己的想法还是像别人的声音？",
        "mindfulness": "这个困难意味着什么？",
        "comic_frames": "挑几个关键的瞬间来画吧",
    }
    return fallback_messages.get(target_id, "能再具体一点吗？")

def _get_stage_content_from_history(target_stage: str) -> str:
    """从对话历史中获取指定阶段的用户输入内容"""
    messages = st.session_state.get("messages", [])

    # 利用对话顺序和关键词区分 A/B 系列
    # A系列先出现，B系列后出现
    a_series_done = False
    found_count = 0

    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")

            # 检测 A2a/B2a：内心独白问题
            if target_stage in ["A2a", "B2a"]:
                # A2a: 包含"好的创作"+"揣摩"+"第一反应"
                # B2a: 包含"来揣摩"+"心里"+"最先冒出"
                if not a_series_done:
                    if "好的创作" in content and "揣摩" in content and "第一反应" in content:
                        a_series_done = True
                        if target_stage == "A2a":
                            found_count = 1
                else:
                    if "来揣摩" in content and "心里" in content and ("最先冒出" in content or "第一反应" in content):
                        if target_stage == "B2a":
                            found_count = 1

            # 检测 A2b/B2b：声音来源问题
            elif target_stage in ["A2b", "B2b"]:
                if "声音来源" in content:
                    # A系列的声音来源在B系列之前
                    if target_stage == "A2b" and not a_series_done:
                        found_count = 1
                    elif target_stage == "B2b" and a_series_done:
                        found_count = 1

            # 检测 A2c/B2c：困难认知问题
            elif target_stage in ["A2c", "B2c"]:
                if "困难说明了" in content:
                    if target_stage == "A2c" and not a_series_done:
                        found_count = 1
                    elif target_stage == "B2c" and a_series_done:
                        found_count = 1

        elif msg.get("role") == "user" and found_count == 1:
            # 找到目标阶段的下一个用户回复
            found_count = 2
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

    # 如果 LLM 解析失败，使用原始输入作为回退
    if not extracted or all(v == "" for v in extracted.values()):
        extracted = {"_raw": user_input}

    if stage == "S0a":
        if extracted.get("name"):
            st.session_state.character_name = extracted["name"]
        if extracted.get("age"):
            st.session_state.character_age = extracted["age"]
        if extracted.get("gender"):
            st.session_state.character_gender = extracted["gender"]
        if extracted.get("status"):
            st.session_state.character_status = extracted["status"]
        if extracted.get("education_level"):
            st.session_state.character_education_level = extracted["education_level"]
        if extracted.get("occupation"):
            st.session_state.character_occupation = extracted["occupation"]
        if extracted.get("work_years"):
            st.session_state.character_work_years = extracted["work_years"]

    elif stage == "S0b" and extracted.get("appearance"):
        st.session_state.character_appearance = extracted["appearance"]

    elif stage == "A1a":
        st.session_state.stage_A_material["dilemma"] = extracted.get("dilemma") or extracted.get("_raw") or user_input
    elif stage == "A1b":
        st.session_state.stage_A_material["context"] = extracted.get("context") or extracted.get("_raw") or user_input
    elif stage == "A1c":
        st.session_state.stage_A_material["reaction"] = extracted.get("reaction") or extracted.get("_raw") or user_input
    elif stage == "A1d":
        st.session_state.stage_A_material["impact"] = extracted.get("impact") or extracted.get("_raw") or user_input
    elif stage == "A2a":
        st.session_state.stage_A2a_quote = extracted.get("quote") or extracted.get("_raw") or user_input
        st.session_state.stage_A_quote_timestamp = time.time()
    elif stage == "A2b":
        st.session_state.stage_A2b_reflection = extracted.get("reflection") or extracted.get("_raw") or user_input
    elif stage == "A2c":
        st.session_state.stage_A2c_framing = extracted.get("framing") or extracted.get("_raw") or user_input

    elif stage == "B1a":
        st.session_state.stage_B_material["dilemma"] = extracted.get("dilemma") or extracted.get("_raw") or user_input
    elif stage == "B1b":
        st.session_state.stage_B_material["context"] = extracted.get("context") or extracted.get("_raw") or user_input
    elif stage == "B1c":
        st.session_state.stage_B_material["reaction"] = extracted.get("reaction") or extracted.get("_raw") or user_input
    elif stage == "B1d":
        st.session_state.stage_B_material["impact"] = extracted.get("impact") or extracted.get("_raw") or user_input
    elif stage == "B2a":
        st.session_state.stage_B2a_quote = extracted.get("quote") or extracted.get("_raw") or user_input
        st.session_state.stage_B_quote_timestamp = time.time()
    elif stage == "B2b":
        st.session_state.stage_B2b_reflection = extracted.get("reflection") or extracted.get("_raw") or user_input
    elif stage == "B2c":
        st.session_state.stage_B2c_framing = extracted.get("framing") or extracted.get("_raw") or user_input
    elif stage == "DEBRIEF":
        st.session_state.debrief_response = extracted.get("response") or extracted.get("_raw") or user_input

def log_event(event_type: str, **data):
    """记录事件到日志"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "app_phase": get_app_phase(),
        **data
    }
    if is_in_story_flow():
        entry["context"] = st.session_state.get("current_context")
        entry["target"] = st.session_state.get("current_target")
    st.session_state.log_entries.append(entry)

def cleanup_old_session_folders(session_id: str, current_storage_name: str):
    """清理同名会话的旧文件夹（UUID命名的）"""
    sessions_dir = os.path.join(DATA_DIR, "sessions")
    if not os.path.exists(sessions_dir):
        return
    
    # 遍历所有文件夹
    for folder_name in os.listdir(sessions_dir):
        folder_path = os.path.join(sessions_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # 跳过当前使用的文件夹
        if folder_name == current_storage_name:
            continue
        
        # 检查这个文件夹是否属于同一个会话
        session_file = os.path.join(folder_path, "session.json")
        if os.path.exists(session_file):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 如果 session_id 匹配，说明是旧文件夹
                if data.get("session_id") == session_id:
                    # 删除旧文件夹
                    shutil.rmtree(folder_path)
                    print(f"清理旧会话文件夹: {folder_name}")
            except Exception:
                pass


def save_session():
    """保存完整会话数据"""
    session_id = st.session_state.session_id
    storage_name = get_session_storage_name(session_id)
    session_dir = get_session_dir(session_id)
    
    # 清理旧的同名会话文件夹
    cleanup_old_session_folders(session_id, storage_name)
    
    ensure_session_dirs(session_id)

    session_data = {
        "session_id": session_id,
        "storage_name": get_session_storage_name(session_id),
        "start_time": st.session_state.get("start_time", ""),
        "last_updated": datetime.now().isoformat(),
        "app_phase": get_app_phase(),
        "current_stage": get_app_phase(),
        "story_flow_active": st.session_state.get("story_flow_active", False),
        "current_context": st.session_state.get("current_context", CONTEXT_ACADEMIC),
        "collected_targets": st.session_state.get("collected_targets", _empty_collected_targets()),
        "current_target": st.session_state.get("current_target", ""),
        "character_name": st.session_state.get("character_name", ""),
        "character_age": st.session_state.get("character_age", ""),
        "character_status": st.session_state.get("character_status", ""),
        "character_education_level": st.session_state.get("character_education_level", ""),
        "character_occupation": st.session_state.get("character_occupation", ""),
        "character_work_years": st.session_state.get("character_work_years", ""),
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
        "comic_style": st.session_state.get("comic_style", st.session_state.get("portrait_style", "动漫")),
        "comic_story_generated": st.session_state.get("comic_story_generated", False),
        "comic_story_approved": st.session_state.get("comic_story_approved", False),
        "comic_story_data": st.session_state.get("comic_story_data", []),
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
    sessions_dir = os.path.join(DATA_DIR, "sessions")
    candidate_names = [session_id]

    if os.path.exists(sessions_dir):
        for folder_name in os.listdir(sessions_dir):
            session_file = os.path.join(sessions_dir, folder_name, "session.json")
            if not os.path.exists(session_file):
                continue
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                if session_data.get("session_id") == session_id or folder_name == session_id:
                    session_data.setdefault("storage_name", folder_name)
                    return session_data
            except Exception as e:
                print(f"加载会话失败: {e}")

    for candidate_name in candidate_names:
        session_file = os.path.join(sessions_dir, candidate_name, "session.json")
        if os.path.exists(session_file):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                session_data.setdefault("storage_name", candidate_name)
                return session_data
            except Exception as e:
                print(f"加载会话失败: {e}")
    return None

def list_sessions():
    """列出所有会话"""
    sessions = []
    sessions_dir = os.path.join(DATA_DIR, "sessions")
    if os.path.exists(sessions_dir):
        for folder_name in os.listdir(sessions_dir):
            session_path = os.path.join(sessions_dir, folder_name)
            if os.path.isdir(session_path):
                session_data = load_session(folder_name)
                if session_data:
                    sessions.append({
                        "session_id": session_data.get("session_id", folder_name),
                        "storage_name": session_data.get("storage_name", folder_name),
                        "character_name": session_data.get("character_name", "未命名"),
                        "start_time": session_data.get("start_time", ""),
                        "current_stage": session_data.get("current_stage", ""),
                        "message_count": len(session_data.get("messages", []))
                    })
    return sorted(sessions, key=lambda x: x["start_time"], reverse=True)


def migrate_old_session(session_data: Dict[str, Any]) -> Dict[str, Dict]:
    """
    迁移旧版本会话数据到新的 collected_targets 结构
    
    检测旧版本数据，根据阶段数据推断当前情境和已收集的目标，
    将推断结果填充到 collected_targets 中。
    """
    collected = _empty_collected_targets()
    
    # 检查是否已有 collected_targets（无需迁移）
    if "collected_targets" in session_data:
        return session_data["collected_targets"]
    
    # 确定当前情境
    current_stage = session_data.get("current_stage", session_data.get("app_phase", "S0a"))
    current_context = session_data.get("current_context", CONTEXT_ACADEMIC)
    
    # 从阶段推断情境
    ctx, _ = legacy_stage_to_context_target(current_stage)
    if ctx:
        current_context = ctx
    
    # 迁移学业故事数据 (A阶段)
    _migrate_context_targets(collected, session_data, CONTEXT_ACADEMIC)
    
    # 迁移人际故事数据 (B阶段)
    _migrate_context_targets(collected, session_data, CONTEXT_SOCIAL)
    
    # 迁移内心独白等关键数据（基于当前情境）
    _migrate_inner_voice(collected, session_data, current_context)
    
    return collected


def _migrate_context_targets(collected: Dict[str, Dict], session_data: Dict[str, Any], context: str):
    """迁移特定情境的目标数据"""
    mapping = TARGET_TO_STAGE_MAPPING.get(context, {})
    if context not in collected:
        collected[context] = {}
    
    # 获取该情境的 material 数据
    material_key = f"stage_{context[0].upper()}_material"
    material = session_data.get(material_key, {})
    
    for target_id in mapping:
        # 从 material 迁移
        if material.get(target_id):
            collected[context][target_id] = {
                "data": material[target_id],
                "timestamp": session_data.get("start_time", datetime.now().isoformat())
            }
        
        # 从 stage_*_material 迁移（新版格式）
        stage_key = f"stage_{mapping[target_id]}_material"
        if stage_key in session_data:
            mat = session_data.get(stage_key, {})
            if mat.get(target_id):
                collected[context][target_id] = {
                    "data": mat[target_id],
                    "timestamp": session_data.get("start_time", datetime.now().isoformat())
                }


def _migrate_inner_voice(collected: Dict[str, Dict], session_data: Dict[str, Any], context: str):
    """迁移内心独白等数据"""
    if context == CONTEXT_ACADEMIC:
        targets_map = {
            "self_kindness": "stage_A2a_quote",
            "common_humanity": "stage_A2b_reflection",
            "mindfulness": "stage_A2c_framing",
        }
    else:
        targets_map = {
            "self_kindness": "stage_B2a_quote",
            "common_humanity": "stage_B2b_reflection",
            "mindfulness": "stage_B2c_framing",
        }
    
    if context not in collected:
        collected[context] = {}
    
    timestamp = session_data.get("start_time", datetime.now().isoformat())
    for target_id, field_key in targets_map.items():
        if field_key in session_data and session_data[field_key]:
            if target_id not in collected[context]:
                collected[context][target_id] = {
                    "data": session_data[field_key],
                    "timestamp": timestamp
                }


def load_session_to_state(session_id: str):
    """将加载的会话数据恢复到 session_state"""
    session_data = load_session(session_id)
    if not session_data:
        return

    actual_session_id = session_data.get("session_id", session_id)
    storage_name = session_data.get("storage_name", session_id)
    st.session_state.session_id = actual_session_id
    st.session_state.start_time = session_data.get("start_time", "")
    st.session_state.messages = session_data.get("messages", [])
    st.session_state.session_dirs = {
        **st.session_state.get("session_dirs", {}),
        actual_session_id: storage_name,
    }

    current_stage = session_data.get("current_stage", "S0a")
    st.session_state.app_phase = session_data.get("app_phase", current_stage)
    st.session_state.story_flow_active = session_data.get("story_flow_active", False)
    st.session_state.current_context = session_data.get("current_context", CONTEXT_ACADEMIC)
    
    # 迁移旧版本数据或使用现有数据
    if "collected_targets" in session_data:
        st.session_state.collected_targets = session_data.get("collected_targets", _empty_collected_targets())
    else:
        # 旧版本数据，需要迁移
        st.session_state.collected_targets = migrate_old_session(session_data)
    
    st.session_state.current_target = session_data.get("current_target", "story_dilemma")

    ctx, tgt = legacy_stage_to_context_target(current_stage)
    if ctx and not session_data.get("story_flow_active"):
        st.session_state.story_flow_active = True
        st.session_state.current_context = ctx
        if tgt:
            st.session_state.current_target = tgt
    if st.session_state.app_phase not in LINEAR_PHASES and not st.session_state.story_flow_active:
        st.session_state.app_phase = "S0a"

    st.session_state.character_name = session_data.get("character_name", "")
    st.session_state.character_age = session_data.get("character_age", "")
    st.session_state.character_status = session_data.get("character_status", "")
    st.session_state.character_education_level = session_data.get("character_education_level", "")
    st.session_state.character_occupation = session_data.get("character_occupation", "")
    st.session_state.character_work_years = session_data.get("character_work_years", "")
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
    st.session_state.comic_style = session_data.get("comic_style", st.session_state.portrait_style or "动漫")
    st.session_state.comic_story_generated = session_data.get("comic_story_generated", False)
    st.session_state.comic_story_approved = session_data.get("comic_story_approved", False)
    st.session_state.comic_story_data = session_data.get("comic_story_data", [])

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
    # 根据 current_context 决定恢复哪个 situation 的 comic_frames_parsed
    # CONTEXT_ACADEMIC = "academic", CONTEXT_SOCIAL = "social"
    ctx = st.session_state.get("current_context", CONTEXT_ACADEMIC)
    comic_data = st.session_state.stage_A_comic if ctx == CONTEXT_ACADEMIC else st.session_state.stage_B_comic
    if comic_data and comic_data.get("frames"):
        st.session_state.comic_frames_parsed = [
            {
                "description": frame.get("description", ""),
                "story_text": frame.get("story_text", frame.get("description", "")),
                "visual_description": frame.get("visual_description", frame.get("description", "")),
                "frame_number": frame.get("frame_index", idx + 1),
            }
            for idx, frame in enumerate(comic_data.get("frames", []))
        ]
    else:
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
    st.session_state.welcome_done = True
    st.session_state.log_entries = []

    ensure_session_dirs()

# =============================================================================
# 会话状态初始化
# =============================================================================
def init_session_state():
    """初始化 Streamlit 会话状态"""

    if "session_dirs" not in st.session_state:
        st.session_state.session_dirs = {}

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
        st.session_state.start_time = datetime.now().isoformat()
        st.session_state.session_dirs[st.session_state.session_id] = build_session_folder_name(st.session_state.start_time)
        ensure_session_dirs()

    if "app_phase" not in st.session_state:
        st.session_state.app_phase = "S0a"

    if "story_flow_active" not in st.session_state:
        st.session_state.story_flow_active = False

    if "current_context" not in st.session_state:
        st.session_state.current_context = CONTEXT_ACADEMIC

    if "collected_targets" not in st.session_state:
        st.session_state.collected_targets = _empty_collected_targets()

    if "current_target" not in st.session_state:
        st.session_state.current_target = "story_dilemma"

    if "welcome_done" not in st.session_state:
        st.session_state.welcome_done = False

    if "log_entries" not in st.session_state:
        st.session_state.log_entries = []

    if "messages" not in st.session_state:
        st.session_state.messages = []
        if st.session_state.welcome_done:
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

    if "character_age" not in st.session_state:
        st.session_state.character_age = ""

    if "character_major" not in st.session_state:
        st.session_state.character_major = ""

    if "character_grade" not in st.session_state:
        st.session_state.character_grade = ""

    if "character_gender" not in st.session_state:
        st.session_state.character_gender = ""

    if "character_status" not in st.session_state:
        st.session_state.character_status = ""

    if "character_education_level" not in st.session_state:
        st.session_state.character_education_level = ""

    if "character_occupation" not in st.session_state:
        st.session_state.character_occupation = ""

    if "character_work_years" not in st.session_state:
        st.session_state.character_work_years = ""

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

    if "comic_story_generated" not in st.session_state:
        st.session_state.comic_story_generated = False

    if "comic_story_approved" not in st.session_state:
        st.session_state.comic_story_approved = False

    if "comic_story_data" not in st.session_state:
        st.session_state.comic_story_data = []

    if "state_cache" not in st.session_state:
        st.session_state.state_cache = {}

    if "debrief_response" not in st.session_state:
        st.session_state.debrief_response = ""

    if "welcome_response" not in st.session_state:
        st.session_state.welcome_response = ""

    if "follow_up_count" not in st.session_state:
        st.session_state.follow_up_count = 0

def _count_met_targets(context: str, target_ids: List[str]) -> int:
    collected = st.session_state.collected_targets.get(context, {})
    return sum(1 for tid in target_ids if collected.get(tid, {}).get("met"))


def _is_perspective_practice_complete() -> bool:
    return bool(
        st.session_state.get("stage4_rewrite_A")
        and st.session_state.get("stage4_rewrite_B")
        and st.session_state.get("stage4_diff_A")
        and st.session_state.get("stage4_diff_B")
    )


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
            session_options = {
                f"{s['character_name'] or '未命名'} · {s['storage_name']}": s["session_id"]
                for s in sessions
            }
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

        if not st.session_state.get("welcome_done", False):
            st.markdown("**欢迎**")
            st.caption("完成欢迎对话后，这里会显示整体创作进度。")
        else:
            academic_targets = ACADEMIC_TARGETS
            social_targets = SOCIAL_TARGETS
            academic_completed = _count_met_targets(CONTEXT_ACADEMIC, academic_targets)
            social_completed = _count_met_targets(CONTEXT_SOCIAL, social_targets)
            perspective_done = _is_perspective_practice_complete()
            total_targets = len(academic_targets) + len(social_targets) + 1
            completed_count = academic_completed + social_completed + (1 if perspective_done else 0)
            overall_pct = completed_count / total_targets if total_targets else 0

            # 基于 app_phase 状态机判断当前阶段
            app_phase = get_app_phase()
            
            # 判断学业故事是否完成（连环画已确认）
            academic_done = _count_met_targets(CONTEXT_ACADEMIC, ["comic_confirmed"]) > 0 or bool(st.session_state.stage_A_comic.get("confirmed"))
            # 判断人际故事是否完成（连环画已确认）
            social_done = _count_met_targets(CONTEXT_SOCIAL, ["comic_confirmed"]) > 0 or bool(st.session_state.stage_B_comic.get("confirmed"))
            perspective_done = _is_perspective_practice_complete()
            
            # 基于状态机阶段判断当前显示
            st.markdown("**当前阶段**")
            if app_phase in ["S0a", "S0b", "S0c", "S0d"]:
                st.markdown("⏳ 角色创建中")
            elif app_phase in ["P4A_REWRITE", "P4A_DIFF", "P4B_REWRITE", "P4B_DIFF"]:
                st.markdown("⏳ 视角练习中")
            elif app_phase == "DEBRIEF":
                st.markdown("⏳ 创作回顾中")
            elif app_phase == "DONE":
                st.markdown("✅ 全部完成")
            elif academic_done and social_done:
                st.markdown("⏳ 视角练习中")
            elif academic_done:
                st.markdown("⏳ 人际故事中")
            else:
                st.markdown(f"⏳ {_get_work_or_study_term()}故事中")

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
                st.markdown(f"**{_get_work_or_study_term()}困扰**")
                st.text(st.session_state.stage_A_material["dilemma"][:200] + "...")

            if st.session_state.stage_B_material.get("dilemma"):
                st.markdown("**人际困扰**")
                st.text(st.session_state.stage_B_material["dilemma"][:200] + "...")

            if st.session_state.stage_A_comic.get("frames"):
                st.markdown(f"**{_get_work_or_study_term()}连环画**")
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

WELCOME_MESSAGE = """欢迎来到 AI 共创叙事画坊。

这是一个一起创作虚构角色故事的地方。你来做编剧，我来当助手。

我们会一起创造一个角色，探索 TA 可能遇到的故事，还会把关键瞬间画成连环画。

整个过程大概 25-30 分钟，所有创作都是匿名的。

准备好了吗？输入「准备好了」或者随便说点什么开始。"""

WELCOME_QUESTION_HINT = "就是一起创作一个虚构的角色，编一个TA的故事。完全匿名，不用有压力。准备好了随时告诉我~"

def render_welcome():
    """渲染欢迎界面（独立于聊天消息之外）"""
    st.markdown("### 👋 欢迎")

    welcome_container = st.container()
    with welcome_container:
        st.markdown(WELCOME_MESSAGE)

    st.markdown("---")

    welcome_input = st.text_input(
        "你的回应",
        value="",
        placeholder="准备好了吗？随便说点什么吧...",
        key="welcome_input_field"
    )

    if st.button("发送", type="primary", use_container_width=True):
        if welcome_input:
            user_input = welcome_input.strip()

            # 先检查是否包含拒绝词
            if _is_welcome_refuse_input(user_input):
                st.info("好的，没关系。如果想体验随时回来~")
                st.stop()

            # 判断用户输入类型
            if _is_welcome_ready_input(user_input):
                # 用户表示准备好，进入角色创建阶段
                st.session_state.welcome_done = True
                st.session_state.welcome_response = user_input

                # 添加 S0a 引导语到 messages（带过渡句）
                initial_msg = generate_guide_message("S0a", add_transition=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": initial_msg
                })

                # 不存储到 log_entries，不触发 store_data
                save_session()
                st.rerun()

            elif _is_welcome_question_input(user_input):
                # 用户提问，显示友好解答
                st.info(WELCOME_QUESTION_HINT)

            else:
                # 其他输入，视为准备就绪
                st.session_state.welcome_done = True
                st.session_state.welcome_response = user_input

                # 添加 S0a 引导语到 messages（带过渡句）
                initial_msg = generate_guide_message("S0a", add_transition=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": initial_msg
                })

                save_session()
                st.rerun()
        else:
            st.warning("请输入一些内容")


def _is_welcome_ready_input(user_input: str) -> bool:
    """判断用户输入是否表示准备好了"""
    ready_keywords = ["好", "行", "开始", "准备", "继续", "是", "嗯", "ok", "go", "好的", "可以", "没问题", "Lets", "lets", "start", "yes", "yep", "yeah"]

    # 检查是否包含准备好的关键词
    for keyword in ready_keywords:
        if keyword in user_input:
            return True

    # 检查是否是非疑问句且长度>5
    if len(user_input) > 5 and "？" not in user_input and "?" not in user_input:
        question_words = ["什么", "怎么", "为什么", "如何", "是不是", "能不能", "会不会", "有没有", "谁", "哪", "何时", "多少"]
        has_question_word = any(qw in user_input for qw in question_words)
        if not has_question_word:
            return True

    return False


def _is_welcome_question_input(user_input: str) -> bool:
    """判断用户输入是否包含疑问"""
    question_marks = ["？", "?"]
    question_words = ["什么", "怎么", "为什么", "如何", "是不是", "能不能", "会不会", "有没有", "谁", "哪", "何时", "多少"]

    has_question_mark = any(qm in user_input for qm in question_marks)
    has_question_word = any(qw in user_input for qw in question_words)

    return has_question_mark or has_question_word


def _is_welcome_refuse_input(user_input: str) -> bool:
    """判断用户输入是否表示拒绝"""
    refuse_keywords = ["不", "算了", "退出", "不要", "不了", "算了", "再见", "bye"]

    for keyword in refuse_keywords:
        if keyword in user_input:
            return True

    return False

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
            features = extract_character_features(
                st.session_state.character_appearance,
                age=st.session_state.get("character_age", "")
            )
            st.session_state.character_features = features

            result = generate_portrait(
                st.session_state.character_appearance,
                st.session_state.session_id,
                style=st.session_state.portrait_style,
                gender=st.session_state.get("character_gender", "unknown"),
                status=st.session_state.get("character_status", "study"),
                age=st.session_state.get("character_age", "")
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
                features = extract_character_features(
                    st.session_state.character_appearance,
                    age=st.session_state.get("character_age", "")
                )
                st.session_state.character_features = features
                st.session_state.portrait_final = _copy_to_final(st.session_state.portrait_path)

                log_event("portrait_confirmed",
                          version=st.session_state.portrait_current_version,
                          final_path=st.session_state.portrait_final)

                st.session_state.app_phase = "S0d"
                st.session_state.story_flow_active = True
                st.session_state.current_context = CONTEXT_ACADEMIC
                st.session_state.current_target = "story_dilemma"
                st.session_state.collected_targets = _empty_collected_targets()
                msg = generate_target_guide("story_dilemma", CONTEXT_ACADEMIC)
                st.session_state.messages.append({"role": "assistant", "content": msg})
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
        "story_text": "",
        "visual_description": "",
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
        # 同步各层字段
        for f in comic_data["frames"]:
            if f.get("frame_index") == frame_num:
                f["description"] = frame.get("description", "")
                f["story_text"] = frame.get("story_text", "")
                f["visual_description"] = frame.get("visual_description", frame.get("description", ""))


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
        description = frame_data.get("visual_description") or frame_data.get("description", "")
    
    current_v = frame_record["current_version"]
    
    result = generate_comic_frame(
        description=description,
        portrait_path=st.session_state.portrait_final,
        character_features=st.session_state.character_features,
        character_features_text=st.session_state.get("character_appearance", ""),
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
    """渲染单个分镜行（每行一格，左侧编辑文本，右侧显示图片）"""
    frame_data = frames_parsed[frame_num - 1] if frame_num <= len(frames_parsed) else {}
    story_text = frame_data.get("story_text", "")
    visual_description = frame_data.get("visual_description", "")
    versions = frame_record.get("versions", [])
    has_image = frame_record.get("image_path") and os.path.exists(frame_record["image_path"])
    current_v = frame_record.get("current_version", 0)
    final_v = frame_record.get("final_version", 0)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown(f"**第{frame_num}格**")
        story_input = st.text_area(
            "故事文案",
            value=story_text,
            key=f"story_{frame_num}",
            height=80,
            placeholder="这一格讲了什么，保留叙事和语气...",
        )
        visual_input = st.text_area(
            "画面描述",
            value=visual_description,
            key=f"visual_{frame_num}",
            height=100,
            placeholder="请描述画面中的人物、动作、表情、环境等（空白将自动用分镜描述）...",
        )

        if story_input != story_text or visual_input != visual_description:
            if frame_num <= len(frames_parsed):
                frames_parsed[frame_num - 1]["story_text"] = story_input
                frames_parsed[frame_num - 1]["visual_description"] = visual_input
            frame_record["story_text"] = story_input
            frame_record["visual_description"] = visual_input
            save_session()

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
                    success = _generate_single_frame(frame_num)
                    if success:
                        st.rerun()
        with col_confirm:
            current_desc = st.session_state.get(f"visual_{frame_num}", visual_input)
            if has_image and len(versions) > 0:
                if len(versions) == 1:
                    if current_desc and len(current_desc.strip()) > 0:
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
                        st.button("✓ 确认", disabled=True, use_container_width=True)
                        st.caption("请先填写画面描述")
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
                # 学业连环画完成，切换到人际故事
                msg = switch_to_social_context()
            else:
                # 人际连环画完成，进入视角练习
                st.session_state.story_flow_active = False
                st.session_state.app_phase = "P4A_REWRITE"
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
                comic_data["frames"] = []
                for i, fp in enumerate(frames_parsed):
                    frame_record = {
                        "frame_index": i + 1,
                        "description": fp.get("description", ""),
                        "story_text": fp.get("story_text", ""),
                        "visual_description": fp.get("visual_description", fp.get("description", "")),
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
                "story_text": "",
                "visual_description": "",
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
        if total_frames > 3:
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
            st.caption("每个情境至少保留3格")
    
    with col3:
        can_finish = total_frames >= 3
        if st.button("✓ 完成连环画", type="primary", use_container_width=True, disabled=not can_finish):
            # 确认所有已生成的图片
            for f in comic_data.get("frames", []):
                if f.get("image_path") and not f.get("final_path"):
                    f["final_version"] = f.get("current_version", 1)
                    f["final_path"] = f.get("image_path", "")

            comic_data["confirmed"] = True
            log_event("comic_complete",
                      situation=situation,
                      frame_count=len(comic_data["frames"]),
                      story_approved=st.session_state.get("comic_story_approved", False))

            if situation == "A":
                # 学业连环画完成，切换到人际故事
                msg = switch_to_social_context()
                st.session_state.messages.append({"role": "assistant", "content": msg})
                save_session()
                st.rerun()
            else:
                # 人际连环画完成，进入视角练习
                st.session_state.story_flow_active = False
                st.session_state.app_phase = "P4A_REWRITE"
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
        if not can_finish:
            st.caption("每个情境至少需要3格连环画")



def _build_comic_story_prompt(situation, material, frames_parsed):
    context = f"{_get_work_or_study_term()}" if situation == "A" else "人际"
    material_summary = "\n".join(
        f"- {key}: {value}" for key, value in material.items() if value
    ) or "请根据分镜描述补全"

    frame_summary = "\n".join(
        f"{idx + 1}. {frame.get('description', '')}" for idx, frame in enumerate(frames_parsed[:5])
    )
    return f"""请为{context}故事生成{len(frames_parsed[:5])}格连环画的叙事文字。
只输出JSON数组，每项只包含 story_text 字段，不要其他内容。

故事材料：
{material_summary}

分镜描述（供参考）：
{frame_summary}

请严格按照以下JSON格式返回：
[
  {{"story_text": "..."}},
  ...
]"""


def generate_comic_story(situation, material, frames_parsed):
    prompt = _build_comic_story_prompt(situation, material, frames_parsed)
    model = os.getenv("LLM_MODEL", "qwen-max")
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "输出严格JSON，不要任何说明。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=2000,
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"```json|```", "", content)
    story_data = json.loads(content)
    if not isinstance(story_data, list):
        raise ValueError("comic story data must be a list")
    return story_data


def _render_comic_story_display(situation):
    """显示故事文案编辑提示（实际编辑在分镜行中进行）"""
    story_type = f"{_get_work_or_study_term()}" if situation == "A" else "人际"
    st.markdown(f"#### ✍️ {story_type}故事文案编辑")


def render_comic_ui():
    """渲染连环画生成界面"""
    situation = st.session_state.comic_situation
    frames_parsed = st.session_state.comic_frames_parsed

    # 确保已初始化（首次进入或新会话）
    if not frames_parsed:
        _init_comic_frames()
        frames_parsed = st.session_state.comic_frames_parsed

    total_frames = len(frames_parsed)

    story_type = f"{_get_work_or_study_term()}" if situation == "A" else "人际"
    st.markdown(f"### 🎬 {story_type}故事连环画生成")

    if total_frames == 0:
        st.warning("没有分镜数据，请返回重新选择关键瞬间")
        if st.button("← 返回"):
            if situation == "A":
                st.session_state.app_phase = "A3a"
            else:
                st.session_state.app_phase = "B3a"
            st.rerun()
        return

    _ensure_frame_records()

    # 自动生成故事文案（首次进入时）
    if not st.session_state.get("comic_story_generated", False):
        with st.spinner("正在根据前面的描述生成故事文本..."):
            try:
                material = st.session_state.stage_A_material if situation == "A" else st.session_state.stage_B_material
                story_data = generate_comic_story(situation, material, frames_parsed)
                if story_data:
                    comic_data = st.session_state.stage_A_comic if situation == "A" else st.session_state.stage_B_comic
                    for idx, frame in enumerate(frames_parsed):
                        if idx < len(story_data):
                            frame["story_text"] = story_data[idx].get("story_text", frame.get("story_text", ""))
                        record = _get_comic_frame_record(comic_data, idx + 1)
                        record["story_text"] = frame.get("story_text", "")
                    st.session_state.comic_story_data = [
                        {"story_text": frame.get("story_text", "")}
                        for frame in frames_parsed
                    ]
                    st.session_state.comic_story_generated = True
                    st.session_state.comic_story_approved = True
                    log_event("comic_story_generated",
                              situation=situation,
                              frame_count=len(frames_parsed))
                    save_session()
                    st.rerun()
                    return
            except Exception as e:
                st.error(f"生成故事文案失败: {e}")
                st.session_state.comic_story_generated = False

    # 显示提示信息
    if st.session_state.get("comic_story_generated", False):
        st.info("已根据前面的描述生成故事文本，你可以根据需要修改，并填写画面描述后生成画面。")

    # 只显示文案编辑和画面生成（不再显示生成按钮）
    _render_comic_story_display(situation)

    st.markdown("---")
    st.markdown("#### 🖼️ 画面生成与确认")
    _render_comic_generation_view()

def render_rewrite_ui():
    """渲染视角改写界面"""
    current_stage = get_app_phase()
    if current_stage == "P4A_REWRITE":
        situation = "A"
    else:
        situation = "B"

    if st.session_state.stage4_start_time == 0:
        st.session_state.stage4_start_time = time.time()

    # 获取对应的连环画数据
    if situation == "A":
        comic_data = st.session_state.stage_A_comic
        story_type = f"{_get_work_or_study_term()}"
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
    st.markdown("回到那个时刻，请用「我」来写：")

    # 三个引导问题
    st.markdown("""
**1.** 事情发生时，'我'心里最先冒出的那句话是什么？

**2.** 这个反应是'我'自己的声音，还是像某个重要的人会说的话？只有'我'会这样，还是谁都可能遇到？

**3.** 这个困难是偶然，还是一直以来的问题？'我'会被困住吗？
""")

    current_text = st.session_state.get(target_rewrite, "")
    rewrite_text = st.text_area(
        "在这里写下你的回答...",
        value=current_text,
        height=200,
        key=f"stage4_{situation}_textarea",
        placeholder=""
    )

    st.session_state[target_rewrite] = rewrite_text

    if st.button("提交", type="primary"):
        latency_ms = (time.time() - st.session_state.stage4_start_time) * 1000
        st.session_state[latency_key] = latency_ms

        log_event("p4_rewrite_submit", situation=situation,
                  latency_ms=latency_ms, content_length=len(rewrite_text))

        if situation == "A":
            st.session_state.app_phase = "P4A_DIFF"
        else:
            st.session_state.app_phase = "P4B_DIFF"

        msg = generate_diff_guide(original_quote, rewrite_text, situation)
        st.session_state.messages.append({
            "role": "assistant",
            "content": msg
        })
        st.rerun()

def generate_diff_guide(original_quote: str, rewrite_quote: str, situation: str) -> str:
    story_type = f"{_get_work_or_study_term()}" if situation == "A" else "人际"

    user_prompt = f"""这是{story_type}故事的视角改写对比：

旁观视角：「{original_quote}」

沉浸视角（第一人称）：「{rewrite_quote}」

请生成一段简短的引导语，帮助用户观察和讨论这两个版本的差异。"""

    return call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7, max_tokens=800)

def render_diff_ui():
    """渲染视角对比界面"""
    current_stage = get_app_phase()
    if current_stage == "P4A_DIFF":
        situation = "A"
    else:
        situation = "B"

    if situation == "A":
        material = st.session_state.stage_A_material
        # 优先使用存储的数据
        original_quote = st.session_state.stage_A2a_quote or material.get("dilemma") or material.get("context") or ""
        reflection = st.session_state.stage_A2b_reflection or material.get("reaction") or ""
        framing = st.session_state.stage_A2c_framing or material.get("impact") or ""
        rewrite_quote = st.session_state.stage4_rewrite_A
        diff_comment_key = "stage4_diff_A"
        story_type = f"{_get_work_or_study_term()}"
        stage_prefix = "A"
    else:
        material = st.session_state.stage_B_material
        # 优先使用存储的数据，确保读取正确的故事数据
        original_quote = st.session_state.stage_B2a_quote or material.get("dilemma") or material.get("context") or ""
        reflection = st.session_state.stage_B2b_reflection or material.get("reaction") or ""
        framing = st.session_state.stage_B2c_framing or material.get("impact") or ""
        rewrite_quote = st.session_state.stage4_rewrite_B
        diff_comment_key = "stage4_diff_B"
        story_type = "人际"
        stage_prefix = "B"

    # 从对话历史中获取内容（使用situation区分）
    if not original_quote:
        original_quote = _get_stage_content_from_history(f"{stage_prefix}2a")
    if not reflection:
        reflection = _get_stage_content_from_history(f"{stage_prefix}2b")
    if not framing:
        framing = _get_stage_content_from_history(f"{stage_prefix}2c")

    st.markdown(f"### 两个版本的对比：{story_type}故事")

    # 显示旁观视角的三个维度（用原问题的概括作为标题）
    st.markdown("**旁观视角（第三人称）**")

    if original_quote:
        st.markdown(f"**那一刻，TA在想什么**：{original_quote}")

    if reflection:
        st.markdown(f"**TA怎么看这件事**：{reflection}")

    if framing:
        st.markdown(f"**这说明什么**：{framing}")

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
            st.session_state.app_phase = "P4B_REWRITE"
            st.session_state.stage4_start_time = time.time()
            msg = generate_guide_message("P4B_REWRITE", {
                "name": st.session_state.character_name,
                "quote": st.session_state.stage_B2a_quote
            })
        else:
            st.session_state.app_phase = "DEBRIEF"
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

        st.session_state.app_phase = "DONE"
        st.rerun()

def render_done_ui():
    """渲染完成界面"""
    st.markdown("## 感谢你的参与和创作！")

    st.markdown("""
    你成功创作了一个虚构角色的故事，并亲身体验了编剧视角转换的练习。

    在旁观视角和沉浸视角之间切换，是创作者常用的技巧。
    它帮助我们理解角色的内在世界，同时保持一定的距离来审视故事。
    """)

    # 显示角色画像
    st.markdown("---")
    st.markdown("### 角色画像")

    if st.session_state.portrait_final and os.path.exists(st.session_state.portrait_final):
        st.image(st.session_state.portrait_final, width=300, caption=f"{st.session_state.character_name}")
    else:
        st.info("尚未生成角色画像")

    # 显示两个故事的连环画
    st.markdown("---")
    st.markdown("### 故事连环画")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{_get_work_or_study_term()}故事**")
        frames_a = st.session_state.stage_A_comic.get("frames", [])
        if frames_a:
            for i, frame in enumerate(frames_a):
                final_path = frame.get("final_path") or frame.get("image_path")
                if final_path and os.path.exists(final_path):
                    st.image(final_path, width=280, caption=f"第{i+1}格")
        else:
            st.info(f"尚未生成{_get_work_or_study_term()}故事连环画")

    with col2:
        st.markdown("**人际故事**")
        frames_b = st.session_state.stage_B_comic.get("frames", [])
        if frames_b:
            for i, frame in enumerate(frames_b):
                final_path = frame.get("final_path") or frame.get("image_path")
                if final_path and os.path.exists(final_path):
                    st.image(final_path, width=280, caption=f"第{i+1}格")
        else:
            st.info("尚未生成人际故事连环画")

    # 保存创作到本地
    st.markdown("---")
    st.markdown("### 💾 保存你的创作")
    st.markdown("点击下方按钮，可以将本次创作的完整数据（对话记录 + 生成图片）下载到本地。")

    if st.button("打包下载本次创作", type="primary"):
        session_id = st.session_state.get("session_id", "")
        storage_name = st.session_state.get("storage_name", session_id)
        session_dir = os.path.join(DATA_DIR, "sessions", storage_name)

        if not os.path.exists(session_dir):
            st.error("未找到会话数据，无法打包。")
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # 写入 session.json
                session_file = os.path.join(session_dir, "session.json")
                if os.path.exists(session_file):
                    zf.write(session_file, arcname="session.json")

                # 写入 log.json
                log_file = os.path.join(session_dir, "log.json")
                if os.path.exists(log_file):
                    zf.write(log_file, arcname="log.json")

                # 写入所有图片
                images_dir = os.path.join(session_dir, "images")
                if os.path.exists(images_dir):
                    for img_name in os.listdir(images_dir):
                        img_path = os.path.join(images_dir, img_name)
                        if os.path.isfile(img_path):
                            zf.write(img_path, arcname=os.path.join("images", img_name))

            zip_buffer.seek(0)
            st.download_button(
                label="确认下载 ZIP 文件",
                data=zip_buffer.getvalue(),
                file_name=f"AIPainting_{storage_name}.zip",
                mime="application/zip",
            )
            st.success("打包完成，点击上方按钮即可下载。")

    st.markdown("---")
    st.markdown("感谢你的时间和创意！如有需要可以重新开始。")

def handle_auto_transition(target_stage: str):
    """处理自动过渡（兼容旧按钮）"""
    st.session_state.app_phase = target_stage
    msg = generate_guide_message(target_stage, {"name": st.session_state.character_name})
    st.session_state.messages.append({
        "role": "assistant",
        "content": msg
    })
    st.rerun()

def handle_user_input(user_input: str):
    """处理用户输入（角色创建用线性阶段，学业/人际用目标驱动）"""
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_session()

    app_phase = get_app_phase()

    # ---------- 角色创建线性流程 ----------
    if not is_in_story_flow():
        log_event("user_input", content_length=len(user_input), app_phase=app_phase)
        store_data(app_phase, user_input)

        if app_phase == "S0c":
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
            st.session_state.messages.append({"role": "assistant", "content": confirm_msg})
            st.session_state.portrait_generating = True
            save_session()
            st.rerun()
            return

        # S0a/S0b 使用 LLM 评估充分性，只有满足标准才推进
        if app_phase in ("S0a", "S0b"):
            evaluation = evaluate_s0_sufficiency(app_phase, user_input, st.session_state.messages[-6:])
            is_sufficient = evaluation.get("is_sufficient", False)
            follow_up_needed = evaluation.get("follow_up_needed", True)
            extracted = evaluation.get("extracted_data", {})

            # 同步提取的数据到 session_state
            if extracted.get("name"):
                st.session_state.character_name = extracted["name"]
            if extracted.get("gender"):
                st.session_state.character_gender = extracted["gender"]
            if extracted.get("major"):
                st.session_state.character_major = extracted["major"]
            if extracted.get("grade"):
                st.session_state.character_grade = extracted["grade"]
            if extracted.get("appearance"):
                st.session_state.character_appearance = extracted["appearance"]

            if not is_sufficient or follow_up_needed:
                if st.session_state.follow_up_count < 2:
                    follow_up_msg = evaluation.get("follow_up_message", "")
                    if not follow_up_msg:
                        follow_up_msg = generate_follow_up(app_phase, user_input, st.session_state.messages[-4:], "character")
                    st.session_state.messages.append({"role": "assistant", "content": follow_up_msg})
                    st.session_state.follow_up_count += 1
                    log_event("s0_insufficient_follow_up", app_phase=app_phase, is_sufficient=is_sufficient)
                    save_session()
                    st.rerun()
                    return

            st.session_state.follow_up_count = 0
            if app_phase == "S0a":
                st.session_state.app_phase = "S0b"
                msg = generate_guide_message("S0b", {"name": st.session_state.character_name})
            elif app_phase == "S0b":
                st.session_state.app_phase = "S0c"
                msg = generate_guide_message("S0c", {"name": st.session_state.character_name})
            else:
                msg = None

            if msg:
                st.session_state.messages.append({"role": "assistant", "content": msg})
                log_event("phase_advance", from_phase=app_phase, to_phase=st.session_state.app_phase)
            save_session()
            st.rerun()
            return

        # 其他线性阶段（直接推进）
        st.session_state.follow_up_count = 0
        if app_phase == "S0c":
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
            st.session_state.messages.append({"role": "assistant", "content": confirm_msg})
            st.session_state.portrait_generating = True
            save_session()
            st.rerun()
            return

        # S0d 直接推进
        st.session_state.app_phase = "S0d"
        msg = generate_guide_message("S0d", {"name": st.session_state.character_name})
        st.session_state.messages.append({"role": "assistant", "content": msg})
        log_event("phase_advance", from_phase=app_phase, to_phase=st.session_state.app_phase)
        save_session()
        st.rerun()
        return

    # ---------- 学业 / 人际目标驱动 ----------
    context = st.session_state.current_context
    current_target = st.session_state.current_target or get_next_target(context)
    if not current_target:
        st.rerun()
        return

    st.session_state.current_target = current_target

    # comic_frames 目标：直接进入连环画分镜输入界面（不等待用户聊天输入）
    if current_target == "comic_frames":
        # 初始化空的分镜数据
        situation = "A" if context == CONTEXT_ACADEMIC else "B"
        st.session_state.comic_situation = situation

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

        save_session()
        st.rerun()
        return

    evaluation = evaluate_user_input(user_input, context, st.session_state.messages[-6:])
    is_sufficient = evaluation.get("is_sufficient", False)
    follow_up_needed = evaluation.get("follow_up_needed", False)
    extracted_data = evaluation.get("extracted_data", {})

    for tid, data in extracted_data.items():
        store_target_data(tid, context, data)

    log_event("user_input", context=context, target=current_target, content_length=len(user_input))

    if follow_up_needed and not is_sufficient:
        if st.session_state.follow_up_count < 2:
            follow_up_msg = generate_follow_up(
                current_target,
                user_input,
                st.session_state.messages[-6:],
                context,
            )
            st.session_state.messages.append({"role": "assistant", "content": follow_up_msg})
            st.session_state.follow_up_count += 1
            save_session()
            st.rerun()
            return

    st.session_state.follow_up_count = 0
    extracted = extracted_data.get(current_target, user_input)
    store_target_data(current_target, context, extracted)

    # comic_frames 目标：用户描述完关键瞬间后直接进入连环画生成
    if current_target == "comic_frames":
        _init_comic_frames()
        st.session_state.current_target = "comic_frames"
        # 解析用户输入的分镜描述
        frames_parsed = st.session_state.comic_frames_parsed
        # 直接进入连环画生成界面（不添加聊天消息）
        save_session()
        st.rerun()
        return

    next_target = get_next_target(context)
    if next_target:
        if next_target in ("comic_frames", "comic_confirmed"):
            # 先显示连环画引导语
            guide_msg = generate_target_guide("comic_frames", context)
            st.session_state.messages.append({"role": "assistant", "content": guide_msg})
            _init_comic_frames()
            st.session_state.current_target = "comic_frames"
            log_event("target_completed", context=context, target=current_target, next_target=next_target)
            save_session()
            st.rerun()
            return
        st.session_state.current_target = next_target
        guide_msg = generate_target_guide(next_target, context)
        st.session_state.messages.append({"role": "assistant", "content": guide_msg})
        log_event("target_completed", context=context, target=current_target, next_target=next_target)
        save_session()
        st.rerun()
        return

    # 当前情境全部目标已完成
    if context == CONTEXT_ACADEMIC:
        transition_msg = switch_to_social_context()
        st.session_state.messages.append({"role": "assistant", "content": transition_msg})
        log_event("context_completed", context=CONTEXT_ACADEMIC)
    else:
        st.session_state.story_flow_active = False
        st.session_state.app_phase = "P4A_REWRITE"
        msg = generate_guide_message("P4A_REWRITE", {
            "name": st.session_state.character_name,
            "quote": st.session_state.stage_A2a_quote,
        })
        st.session_state.stage4_mode = True
        st.session_state.stage4_situation = "A"
        st.session_state.stage4_start_time = time.time()
        st.session_state.messages.append({"role": "assistant", "content": msg})
        log_event("context_completed", context=CONTEXT_SOCIAL)
    save_session()
    st.rerun()

# =============================================================================
# 主函数
# =============================================================================
def main():
    """主函数"""
    st.set_page_config(
        page_title="AI共创叙事画坊",
        page_icon=":art:",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # 移动端响应式样式
    st.markdown(
        """
        <style>
        /* 限制内容区宽度并居中，桌面端不会太宽 */
        .block-container {
          max-width: 900px;
          margin-left: auto;
          margin-right: auto;
        }

        /* 手机端收紧边距和标题大小 */
        @media (max-width: 640px) {
          .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            padding-top: 0.75rem;
          }
          h1 { font-size: 1.5rem !important; }
          h2 { font-size: 1.25rem !important; }
          h3 { font-size: 1.05rem !important; }
        }

        /* 图片自适应，避免横向溢出 */
        .stImage,
        [data-testid="stImage"] {
          max-width: 100%;
          height: auto;
          display: block;
          margin-left: auto;
          margin-right: auto;
        }
        img {
          max-width: 100% !important;
          height: auto !important;
        }

        /* 按钮更容易触控 */
        button,
        [data-testid="stButton"] button {
          min-height: 42px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()
    render_sidebar()

    st.title("AI共创叙事画坊")
    st.markdown("---")

    app_phase = get_app_phase()
    current_stage = app_phase
    current_context = st.session_state.get("current_context", CONTEXT_ACADEMIC)
    current_target = st.session_state.get("current_target", "")

    # 检查并显示欢迎界面
    if not st.session_state.welcome_done:
        render_welcome()
        st.stop()

    show_chat = True

    # S0a, S0b 阶段：显示消息 + 聊天输入框
    if app_phase in ["S0a", "S0b"]:
        render_messages()

    # S0c 阶段：风格选择和画像生成
    if app_phase == "S0c":
        if st.session_state.get("portrait_generated", False):
            render_portrait_ui()
            show_chat = False
        elif st.session_state.get("portrait_generating", False):
            # 先显示聊天记录，再显示生成中的提示
            render_messages()
            render_portrait_ui()
            show_chat = False
        else:
            # 还没开始生成画像时，显示消息并提示选择风格
            render_messages()
            st.info("🎨 请告诉我你想要的画像风格，例如：写实、动漫、水彩、素描、油画、国风等")
            show_chat = True

    # S0d 阶段：按时间顺序显示所有内容（消息 + 画像 + 后续消息）
    elif app_phase == "S0d" and is_in_story_flow():
        messages = st.session_state.messages

        # 如果当前目标是连环画，显示聊天记录后再进入连环画界面
        if current_target == "comic_frames":
            situation = "A" if current_context == CONTEXT_ACADEMIC else "B"
            need_init = (
                not st.session_state.comic_frames_parsed
                or len(st.session_state.comic_frames_parsed) == 0
                or st.session_state.comic_situation != situation
            )
            if need_init:
                _init_comic_frames()
            st.session_state.comic_situation = situation
            # 先显示聊天记录
            render_messages()
            # 再显示连环画界面
            render_comic_ui()
            show_chat = False
        else:
            # 找到特定消息的索引
            portrait_msg_idx = -1
            guide_msg_idx = -1

            for i, msg in enumerate(messages):
                content = msg.get("content", "")
                if msg["role"] == "assistant" and "请稍候" in content:
                    portrait_msg_idx = i
                if msg["role"] == "assistant" and f"{_get_work_or_study_term()}状况" in content:
                    guide_msg_idx = i

            # 按顺序显示所有内容
            for i, msg in enumerate(messages):
                # 在画像消息位置，先显示消息，再显示已确认的画像
                if i == portrait_msg_idx:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                    # 只显示已确认的画像，不显示调整按钮
                    if st.session_state.portrait_final and os.path.exists(st.session_state.portrait_final):
                        st.markdown("---")
                        st.markdown("### 角色画像")
                        st.image(st.session_state.portrait_final, width=400)
                # 在引导消息位置，显示消息
                elif i == guide_msg_idx:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                # 显示其他消息
                else:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            show_chat = True

    elif is_in_story_flow() and current_target == "comic_frames":
        situation = "A" if current_context == CONTEXT_ACADEMIC else "B"
        need_init = (
            not st.session_state.comic_frames_parsed
            or len(st.session_state.comic_frames_parsed) == 0
            or st.session_state.comic_situation != situation
        )
        if need_init:
            _init_comic_frames()
        st.session_state.comic_situation = situation
        render_messages()
        render_comic_ui()
        show_chat = False

    elif is_in_story_flow() and current_target == "comic_confirmed":
        st.markdown(
            f"### 🎬 {_get_work_or_study_term()}故事连环画"
        )
        render_messages()
        _render_comic_complete_view()
        show_chat = False

    elif app_phase in ["P4A_REWRITE", "P4B_REWRITE"]:
        render_messages()
        render_rewrite_ui()
        show_chat = False

    elif current_stage in ["P4A_DIFF", "P4B_DIFF"]:
        render_messages()
        render_diff_ui()
        show_chat = False

    elif current_stage == "DEBRIEF":
        render_messages()
        render_debrief_ui()
        show_chat = False

    elif current_stage == "DONE":
        render_messages()
        render_done_ui()
        show_chat = False

    if show_chat:
        if prompt := st.chat_input("请输入你的回应..."):
            handle_user_input(prompt)


if __name__ == "__main__":
    main()