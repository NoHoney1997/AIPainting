"""
AI共创叙事画坊 - 图像生成模块
使用通义万相 API 生成画像和漫画分镜
"""
import os
import base64
import time
import json
import requests
import re
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("未设置 DASHSCOPE_API_KEY，请在 .env 文件中配置")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("未设置 DEEPSEEK_API_KEY，请在 .env 文件中配置")

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

# =============================================================================
# 图像生成模型配置
# =============================================================================
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "qwen-image-2.0-pro-2026-03-03")

IMAGE_GEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

IMAGE_GEN_CONFIG = {
    "size": "1024*1024",
    "n": 1,
    "prompt_extend": True,
    "watermark": False
}

llm_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=LLM_BASE_URL
)

# 图像风格选项
IMAGE_STYLES = {
    "写实": {
        "name": "写实风格",
        "description": "真实自然的人物形象，温暖色调",
        "prompt_suffix": "写实摄影风格，高清画质，温暖柔和的光线，简洁背景"
    },
    "动漫": {
        "name": "动漫风格",
        "description": "二次元动漫人物，色彩鲜明",
        "prompt_suffix": "动漫风格，Anime style，明亮的眼睛，清晰的线条感"
    },
    "水彩": {
        "name": "水彩风格",
        "description": "艺术水彩画效果，柔和优雅",
        "prompt_suffix": "水彩画风格，watercolor painting，柔和的色彩过渡"
    },
    "素描": {
        "name": "素描风格",
        "description": "铅笔素描效果，简约有力",
        "prompt_suffix": "铅笔素描风格，pencil sketch，黑白或灰色调"
    },
    "油画": {
        "name": "油画风格",
        "description": "古典油画效果，厚重质感",
        "prompt_suffix": "油画风格，oil painting，丰富的色彩层次"
    },
    "国风": {
        "name": "国风风格",
        "description": "中国传统水墨画效果",
        "prompt_suffix": "中国水墨画风格，Chinese ink painting，淡淡的水墨晕染"
    }
}

def get_available_styles() -> List[Dict[str, str]]:
    """获取可用的图像风格列表"""
    return [
        {"key": key, "name": info["name"], "description": info["description"]}
        for key, info in IMAGE_STYLES.items()
    ]

def get_llm_client():
    """获取LLM客户端实例"""
    return llm_client

def get_image_gen_model() -> str:
    """获取当前使用的图像生成模型名称"""
    return IMAGE_GEN_MODEL

def set_image_gen_model(model_name: str):
    """动态设置图像生成模型"""
    global IMAGE_GEN_MODEL
    IMAGE_GEN_MODEL = model_name

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def extract_character_features(description: str, gender: str = "unknown") -> Dict[str, Any]:
    """使用LLM提取角色特征"""
    prompt = f"""从以下角色外貌描述中提取视觉特征。
**重要：只提取描述中明确提到的内容，不要添加未提及的特征。**

描述：{description}
性别：{gender}

返回JSON，只包含明确提到的字段：
{{
    "gender": "{gender}",
    "hair": "仅当提到发型/发色时填写",
    "eyes": "仅当提到眼睛特征（如戴眼镜）时填写",
    "face_shape": "仅当提到脸型时填写",
    "clothing_style": "仅当提到穿着时填写",
    "distinctive_features": ["仅当提到独特特征时添加"]
}}

没有提到的字段设为空字符串。只返回JSON。"""

    try:
        print(f"\n{'='*60}")
        print(f"[模型调用] extract_character_features")
        print(f"  模型: {LLM_MODEL}")
        print(f"  用途: 提取角色视觉特征")
        print(f"{'='*60}")
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "数据提取助手。只提取明确提到的内容，绝不添加。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        print(f"[模型调用] extract_character_features 完成")
        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'```json\s*|```', '', result_text)
        features = json.loads(result_text)

        if "distinctive_features" not in features:
            features["distinctive_features"] = []

        return features
    except:
        return {"gender": gender, "hair": "", "eyes": "", "face_shape": "", "clothing_style": "",
                "distinctive_features": []}

def generate_portrait(description: str, session_id: str, style: str = "写实", gender: str = "unknown", status: str = "study") -> Dict[str, Any]:
    """生成角色画像"""
    style_info = IMAGE_STYLES.get(style, IMAGE_STYLES["写实"])

    session_images_dir = os.path.join(DATA_DIR, "sessions", session_id, "images")
    ensure_dir(session_images_dir)

    timestamp = int(time.time())
    output_path = os.path.join(session_images_dir, f"portrait_{style}_{timestamp}.png")

    gender_word = "女性" if gender == "female" else "男性" if gender == "male" else "人物"
    role_word = "大学生" if status == "study" else "职场人士"

    prompt = f"""生成一张{gender_word}{role_word}的肖像画。

{style_info["prompt_suffix"]}

外貌特征：
{description}

要求：全身像，不凸显身材，人物居中，背景简洁"""

    for attempt in range(3):
        try:
            image_url = _call_wanx_t2i(prompt)
            saved_path = _download_and_save(image_url, output_path)
            return {"image_path": saved_path, "success": True, "style": style}
        except Exception as e:
            if attempt == 2:
                return {"image_path": "", "success": False, "error": str(e)}
            time.sleep(5)

    return {"image_path": "", "success": False, "error": "未知错误"}

def _summarize_character_features(features: Dict[str, Any]) -> str:
    """将角色特征字典汇总为一段自然语言描述"""
    parts = []
    for key, value in features.items():
        if value and value not in (None, "", [], {}):
            parts.append(f"{key}: {value}")
    return "；".join(parts) if parts else ""


def generate_comic_frame(
    description: str,
    portrait_path: Optional[str],
    character_features: Dict[str, Any],
    frame_index: int,
    session_id: str,
    situation: str,
    style: str = "动漫",
    character_features_text: Optional[str] = None,
) -> Dict[str, Any]:
    """生成漫画分镜"""
    style_info = IMAGE_STYLES.get(style, IMAGE_STYLES["动漫"])

    session_images_dir = os.path.join(DATA_DIR, "sessions", session_id, "images")
    ensure_dir(session_images_dir)

    timestamp = int(time.time())
    output_path = os.path.join(session_images_dir, f"comic_{situation}_{frame_index}_{style}_v{timestamp}.png")

    appearance_text = character_features_text
    if not appearance_text and character_features:
        appearance_text = _summarize_character_features(character_features)
    prompt = f"""生成一幅漫画风格的单幅场景画面。

风格：{style_info["prompt_suffix"]}

角色外貌参考：
{appearance_text}

场景描述：
{description}

重要要求：
- 单幅漫画画面，不是三视图
- 角色形象与参考图片中的形象一致，注意参考的是参考图中的形象，不是人物动作
- 画面构图完整，有背景环境"""

    has_reference = False

    for attempt in range(3):
        try:
            if portrait_path and os.path.exists(portrait_path):
                # 使用参考图生成
                image_url = _call_wanx_t2i_with_ref(prompt, portrait_path)
                has_reference = True
            else:
                # 没有参考图时直接生成
                image_url = _call_wanx_t2i(prompt)
                has_reference = False

            saved_path = _download_and_save(image_url, output_path)
            return {
                "image_path": saved_path,
                "has_reference": has_reference,
                "success": True,
                "style": style
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                time.sleep(30)
            elif attempt == 2:
                return {
                    "image_path": "",
                    "has_reference": False,
                    "success": False,
                    "error": f"分镜生成失败: {error_msg}"
                }
            time.sleep(5)

    return {
        "image_path": "",
        "has_reference": False,
        "success": False,
        "error": "未知错误"
    }

def _call_wanx_t2i(prompt: str) -> str:
    """调用通义万相文生图 API"""
    content = [{"text": prompt}]

    payload = {
        "model": IMAGE_GEN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        },
        "parameters": IMAGE_GEN_CONFIG.copy()
    }

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"\n{'='*60}")
    print(f"[模型调用] _call_wanx_t2i (文生图，无参考图)")
    print(f"  模型: {IMAGE_GEN_MODEL}")
    print(f"  用途: 生成角色画像")
    print(f"  Prompt长度: {len(prompt)} 字符")
    print(f"{'='*60}")

    response = requests.post(
        IMAGE_GEN_API_URL,
        headers=headers,
        json=payload,
        timeout=180
    )

    print(f"[模型调用] _call_wanx_t2i 响应状态: {response.status_code}")

    if response.status_code != 200:
        print(f"[模型调用] _call_wanx_t2i 错误响应: {response.text[:800]}")
        raise Exception(f"API error: {response.status_code} - {response.text}")

    result = response.json()
    print(f"[模型调用] _call_wanx_t2i 解析结果: {json.dumps(result, ensure_ascii=False)[:500]}")

    return _parse_api_response(result)

def _call_wanx_t2i_with_ref(prompt: str, ref_image_path: str) -> str:
    """调用通义万相文生图 API（带参考图）"""
    with open(ref_image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    content = [
        {"image": f"data:image/png;base64,{img_base64}"},
        {"text": prompt}
    ]

    payload = {
        "model": IMAGE_GEN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        },
        "parameters": IMAGE_GEN_CONFIG.copy()
    }

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"\n{'='*60}")
    print(f"[模型调用] _call_wanx_t2i_with_ref (文生图，带参考图)")
    print(f"  模型: {IMAGE_GEN_MODEL}")
    print(f"  用途: 生成漫画分镜（参考角色画像）")
    print(f"  参考图: {ref_image_path}")
    print(f"  Prompt长度: {len(prompt)} 字符")
    print(f"{'='*60}")

    response = requests.post(
        IMAGE_GEN_API_URL,
        headers=headers,
        json=payload,
        timeout=180
    )

    print(f"[模型调用] _call_wanx_t2i_with_ref 响应状态: {response.status_code}")

    if response.status_code != 200:
        print(f"[模型调用] _call_wanx_t2i_with_ref 错误响应: {response.text[:800]}")
        raise Exception(f"API error: {response.status_code} - {response.text}")

    result = response.json()
    print(f"[模型调用] _call_wanx_t2i_with_ref 解析结果: {json.dumps(result, ensure_ascii=False)[:500]}")

    return _parse_api_response(result)

def _parse_api_response(result: Dict[str, Any]) -> str:
    """解析通义万相 API 返回结果"""
    if "output" not in result:
        raise Exception(f"Unexpected response format: missing 'output': {result}")

    output = result["output"]

    task_status = output.get("task_status")
    if task_status == "FAILED":
        error_msg = output.get("message", "Unknown error")
        raise Exception(f"Task failed: {error_msg}")

    if "choices" in output and len(output["choices"]) > 0:
        message = output["choices"][0].get("message", {})
        content = message.get("content", [])

        if isinstance(content, list) and len(content) > 0:
            for item in content:
                if "image" in item:
                    return item["image"]
                if "image_url" in item:
                    return item["image_url"]

    if "results" in output and len(output["results"]) > 0:
        result_item = output["results"][0]
        if "url" in result_item:
            return result_item["url"]
        if "image" in result_item:
            return result_item["image"]

    raise Exception(f"Unexpected response format: cannot find image in {result}")

def _download_and_save(url: str, output_path: str) -> str:
    """下载图片并保存"""
    if url.startswith("data:"):
        match = re.search(r'data:image/\w+;base64,(.+)', url)
        if match:
            img_data = base64.b64decode(match.group(1))
            with open(output_path, "wb") as f:
                f.write(img_data)
            return output_path
        raise Exception("Invalid base64 format")

    print(f"[DEBUG] Downloading image from: {url[:100]}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"[DEBUG] Image saved to: {output_path}")
    return output_path