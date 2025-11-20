# -*- coding: utf-8 -*-
SYSTEM_PROMPT = """你是一个贴近事实的知识助手。严格基于“提供的资料片段”回答。
要求：
1) 先给出直接答案；2) 必要时简要说明依据；3) 列出引用编号。
若资料不足，请明确说明“资料不足”，不要编造。"""

def build_user_prompt(question: str, contexts: list[dict]) -> str:
    """
    Prompt 三段式：系统指令 + 资料片段（带编号和来源） + 用户问题 + 输出格式
    """
    ctx_lines = []
    for i, c in enumerate(contexts, 1):
        ctx_lines.append(f"[{i}] ({c['path']}#{c['chunk_id']})\n{c['text']}\n")
    return f"""{SYSTEM_PROMPT}

资料：
{''.join(ctx_lines)}

问题：{question}

回答格式：
- 直接答案
- 关键依据（可选）
- 引用：[编号,...]
"""
