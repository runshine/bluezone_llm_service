#!/usr/bin/env python3
"""
GLM4 工具解析器修复脚本 (vLLM >= 0.10.0)
用法: python apply_glm4_fix_final.py
"""

import os
import re
import sys
from pathlib import Path


PATCH_MARKER = '# GLM4_FIX_APPLIED'


def find_vllm_serving():
    """查找 serving.py"""
    try:
        import vllm
        base = Path(vllm.__file__).parent
        path = base / "entrypoints" / "openai" / "chat_completion" / "serving.py"
        if path.exists():
            return str(path)
    except:
        pass

    # 常见路径
    for prefix in ["/usr/local/lib", "/opt/conda/lib", "/usr/lib"]:
        for ver in ["3.8", "3.9", "3.10", "3.11", "3.12"]:
            path = Path(f"{prefix}/python{ver}/site-packages/vllm/entrypoints/openai/chat_completion/serving.py")
            if path.exists():
                return str(path)

    # 当前目录
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        for sub in ["vllm", "vllm-src"]:
            path = parent / sub / "vllm" / "entrypoints" / "openai" / "chat_completion" / "serving.py"
            if path.exists():
                return str(path)

    return None


def apply_fix(filepath):
    """应用修复 - 使用正则表达式替换"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已修复
    if PATCH_MARKER in content or '"Glm4" in type' in content:
        print("[OK] 文件已包含 GLM4 修复")
        return True

    # 查找目标代码块的起始位置 (基于第 1194 行的注释)
    # 使用正则来匹配这个代码块，更灵活
    pattern = r'''(
                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON\.
                            # Tool parsers \(e\.g\. Qwen3Coder\) store
                            # arguments as a JSON string in
                            # prev_tool_call_arr\. Calling json\.dumps\(\)
                            # on an already-serialized string would
                            # double-serialize it \(e\.g\. '[{]"k":1[}]' becomes
                            # '"[{]\\"k\\"[:]1[}]"'\), which then causes the
                            # replace\(\) below to fail and append the
                            # entire double-serialized string as a
                            # spurious final delta\.
                            args = tool_parser\.prev_tool_call_arr\[index\]\.get\(
                                "arguments", \{\}
                            \)
                            if isinstance\(args, str\):
                                expected_call = args
                            else:
                                expected_call = json\.dumps\(args, ensure_ascii=False\)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser\.streamed_args_for_tool\[index\]
                            if latest_delta_len > 0:
                                actual_call = actual_call\[:-latest_delta_len\]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call\.replace\(actual_call, "", 1\)
                            # set that as a delta message
                            delta_message = self\._create_remaining_args_delta\(
                                delta_message, remaining_call, index
                            \)
    )'''

    # 简化：直接查找关键特征并替换整个区域
    if 'autocompletes' not in content:
        print("[ERROR] 找不到目标代码 (没有 'autocompletes' 标记)")
        return False

    # 定位代码块边界
    lines = content.split('\n')
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if start_idx is None and 'get the expected call based on partial JSON' in line:
            start_idx = i
        if start_idx is not None and 'delta_message = self._create_remaining_args_delta' in line:
            # 找到这一行的闭合括号
            for j in range(i, min(i+5, len(lines))):
                if lines[j].strip() == ')':
                    end_idx = j
                    break
            if end_idx is None:
                end_idx = i + 2
            break

    if start_idx is None or end_idx is None:
        print(f"[ERROR] 无法定位代码块边界 (start={start_idx}, end={end_idx})")
        return False

    print(f"[INFO] 找到代码块: 行 {start_idx+1} 到 {end_idx+1}")

    # 构建新代码
    new_code_lines = [
        PATCH_MARKER,
        '                            # GLM 4 parsers with MTP on won\'t work properly if',
        '                            # falling back to the original autocomplete logic.',
        '                            if "Glm4" in type(tool_parser).__name__:',
        '                                actual_call = tool_parser.streamed_args_for_tool[index]',
        '                                remaining_call = ""',
        '                                if latest_delta_len > 0:',
        '                                    remaining_call = actual_call[-latest_delta_len:]',
        '                            else:',
        '                                # get the expected call based on partial JSON',
        '                                # parsing which "autocompletes" the JSON.',
        '                                # Tool parsers (e.g. Qwen3Coder) store',
        '                                # arguments as a JSON string in',
        '                                # prev_tool_call_arr. Calling json.dumps()',
        '                                # on an already-serialized string would',
        '                                # double-serialize it (e.g. \'{"k":1}\' becomes',
        '                                # \'"{\\"k\\":1}"\'), which then causes the',
        '                                # replace() below to fail and append the',
        '                                # entire double-serialized string as a',
        '                                # spurious final delta.',
        '                                args = tool_parser.prev_tool_call_arr[index].get(',
        '                                    "arguments", {}',
        '                                )',
        '                                if isinstance(args, str):',
        '                                    expected_call = args',
        '                                else:',
        '                                    expected_call = json.dumps(args, ensure_ascii=False)',
        '',
        '                                # get what we\'ve streamed so far for arguments',
        '                                # for the current tool',
        '                                actual_call = tool_parser.streamed_args_for_tool[index]',
        '                                if latest_delta_len > 0:',
        '                                    actual_call = actual_call[:-latest_delta_len]',
        '',
        '                                # check to see if there\'s anything left to stream',
        '                                remaining_call = expected_call.replace(actual_call, "", 1)',
        '                            # set that as a delta message',
        '                            delta_message = self._create_remaining_args_delta(',
        '                                delta_message, remaining_call, index',
        '                            )',
    ]

    # 保留原始缩进
    original_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())

    # 构建新内容
    new_lines = lines[:start_idx] + new_code_lines + lines[end_idx+1:]
    new_content = '\n'.join(new_lines)

    # 备份
    backup = filepath + ".backup"
    if not os.path.exists(backup):
        with open(backup, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] 已备份: {backup}")

    # 写入
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True


def main():
    print("=" * 60)
    print("GLM4 工具解析器修复脚本")
    print("=" * 60)

    filepath = find_vllm_serving()

    if not filepath:
        print("[ERROR] 找不到 serving.py")
        print("请确保 vLLM 已安装")
        sys.exit(1)

    print(f"\n目标: {filepath}")

    if apply_fix(filepath):
        print("\n[OK] 修复完成!")
        print("GLM4 模型现在可以正常工作了。")
    else:
        print("\n[ERROR] 修复失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
