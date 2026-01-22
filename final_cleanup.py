#!/usr/bin/env python3
"""
最终清理脚本 - 彻底修复所有问题
"""

import re
from pathlib import Path


def final_cleanup(file_path: Path) -> bool:
    """最终清理单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        original_lines = lines.copy()
        cleaned_lines = []
        prev_line = ""
        skip_next_empty = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 跳过重复的类/函数定义（完全相同的行）
            if line == prev_line and (
                stripped.startswith('class ') or
                stripped.startswith('def ') or
                stripped.startswith('@')
            ):
                continue

            # 跳过空的或只有引号的文档字符串行
            if re.match(r'^\s*"""*\s*$', line) or re.match(r"^\s*'''*\s*$", line):
                skip_next_empty = True
                continue

            # 跳过连续的超过2个空行
            if stripped == '' and prev_line.strip() == '':
                if i + 1 < len(lines) and lines[i + 1].strip() == '':
                    continue

            cleaned_lines.append(line)
            prev_line = line

        # 写回文件
        if cleaned_lines != original_lines:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            return True
        return False

    except Exception as e:
        print(f"❌ 错误: {file_path}: {e}")
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("最终清理")
    print("=" * 80)

    base_dir = Path('/home/user/diffusers')
    src_dir = base_dir / 'src' / 'diffusers'

    total = 0
    cleaned = 0

    exclude_patterns = ['__pycache__', '.git']

    for py_file in src_dir.rglob('*.py'):
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue

        total += 1
        if final_cleanup(py_file):
            cleaned += 1
            print(f"✓ 清理: {py_file.relative_to(src_dir)}")

    print(f"\n{'='*80}")
    print(f"总文件数: {total}")
    print(f"已清理: {cleaned}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
