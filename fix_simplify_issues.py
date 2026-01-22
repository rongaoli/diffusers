#!/usr/bin/env python3
"""
修复简化过程中产生的问题
- 移除重复的类定义
- 清理空的文档字符串
- 修复格式问题
"""

import re
from pathlib import Path


def fix_file(file_path: Path) -> bool:
    """修复单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 1. 移除空的文档字符串（只有引号的）
        content = re.sub(r'^\s*"""+"[\s]*"""', '', content, flags=re.MULTILINE)
        content = re.sub(r"^\s*'''+'[\s]*'''", '', content, flags=re.MULTILINE)

        # 2. 移除重复的类定义（紧挨着的相同类定义）
        content = re.sub(
            r'(class\s+(\w+)[^:]*:\s*\n)\s*\1+',
            r'\1',
            content
        )

        # 3. 移除重复的函数定义
        content = re.sub(
            r'(def\s+(\w+)[^:]*:\s*\n)\s*\1+',
            r'\1',
            content
        )

        # 4. 移除多余的空行（3个以上）
        content = re.sub(r'\n\n\n\n+', '\n\n\n', content)

        # 5. 修复文档字符串格式（移除多余的引号）
        content = re.sub(r'""""{2,}', '"""', content)
        content = re.sub(r"''''{2,}", "'''", content)

        # 6. 移除孤立的 r""" 标记
        content = re.sub(r'^\s*r"""[\s]*$', '', content, flags=re.MULTILINE)

        # 写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False

    except Exception as e:
        print(f"❌ 错误: {file_path}: {e}")
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("修复简化问题")
    print("=" * 80)

    base_dir = Path('/home/user/diffusers')
    src_dir = base_dir / 'src' / 'diffusers'

    total = 0
    fixed = 0

    exclude_patterns = ['__pycache__', '.git']

    for py_file in src_dir.rglob('*.py'):
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue

        total += 1
        if fix_file(py_file):
            fixed += 1
            print(f"✓ 修复: {py_file.relative_to(src_dir)}")

    print(f"\n{'='*80}")
    print(f"总文件数: {total}")
    print(f"已修复: {fixed}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
