#!/usr/bin/env python3
"""
深度代码简化脚本 - 更激进的简化策略
将冗长的文档字符串替换为简洁的中文注释
"""

import re
import os
from pathlib import Path
from typing import List, Tuple
import ast


class DeepCodeSimplifier:
    """深度代码简化器 - 更激进的简化"""

    def __init__(self):
        pass

    def simplify_docstring_aggressive(self, content: str) -> str:
        """
        激进地简化文档字符串
        - 类文档字符串：保留1-2行核心说明
        - 方法文档字符串：保留1行核心说明
        - 参数说明：移除详细的 Args/Parameters 部分，改用行内注释
        """

        def simplify_class_docstring(match):
            """简化类的文档字符串"""
            indent = match.group(1)
            docstring = match.group(2)

            # 提取第一句话
            lines = [l.strip() for l in docstring.split('\n') if l.strip()]
            if not lines:
                return f'{indent}"""核心类"""'

            first_line = lines[0]

            # 如果第一行太长，截取
            if len(first_line) > 100:
                first_line = first_line[:97] + "..."

            return f'{indent}"""{first_line}"""'

        def simplify_function_docstring(match):
            """简化函数的文档字符串"""
            indent = match.group(1)
            docstring = match.group(2)

            # 提取第一句话
            lines = [l.strip() for l in docstring.split('\n') if l.strip()]
            if not lines:
                return f'{indent}"""核心功能"""'

            first_line = lines[0]

            # 特殊处理：如果包含 "call function"，说明是 __call__ 方法
            if "call function" in first_line.lower():
                return f'{indent}"""主调用函数"""'

            # 如果第一行太长，截取
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."

            return f'{indent}"""{first_line}"""'

        # 1. 简化类文档字符串（通常在 class 定义后）
        content = re.sub(
            r'(class \w+.*?:\s+)(r?""")(.*?)(""")',
            lambda m: m.group(1) + simplify_class_docstring(m),
            content,
            flags=re.DOTALL
        )

        # 2. 简化函数/方法文档字符串
        content = re.sub(
            r'(\n\s+)(r?""")(.*?)(""")',
            lambda m: m.group(1) + simplify_function_docstring(m),
            content,
            flags=re.DOTALL
        )

        return content

    def add_inline_comments(self, content: str) -> str:
        """
        为关键参数添加行内中文注释
        例如: def forward(self, x, timestep, condition):
        改为: def forward(self, x, timestep, condition):  # x:输入, timestep:时间步, condition:条件
        """
        # 这个功能比较复杂，暂时跳过
        return content

    def remove_examples_section(self, content: str) -> str:
        """移除冗长的 Examples 部分"""

        def replace_examples(match):
            """替换示例块"""
            return '\n        使用示例见文档\n'

        # 匹配 Examples: 开始的整个块（包括代码示例）
        content = re.sub(
            r'\n\s+Examples?:\s*\n\s+```[\s\S]*?```',
            replace_examples,
            content,
            flags=re.IGNORECASE
        )

        return content

    def remove_args_section(self, content: str) -> str:
        """移除冗长的 Args/Parameters/Returns 部分"""

        # 匹配 Args:, Parameters:, Returns: 等部分
        patterns = [
            r'\n\s+(Args|Arguments|Parameters):\s*\n(\s+\w+.*?\n)*',
            r'\n\s+Returns:\s*\n(\s+.*?\n)*',
            r'\n\s+Raises:\s*\n(\s+.*?\n)*',
        ]

        for pattern in patterns:
            content = re.sub(pattern, '\n', content)

        return content

    def simplify_type_annotations(self, content: str) -> str:
        """简化类型注解"""

        # Optional[Union[A, List[A]]] -> Optional[A]
        content = re.sub(
            r'Optional\[Union\[([\w.]+),\s*List\[\1\]\]\]',
            r'Optional[\1]',
            content
        )

        # Union[A, B, C] -> Union[A, B, C] (保持不变，但可以考虑简化)
        # 这里暂时不做过多简化，避免破坏类型检查

        return content

    def remove_license_and_headers(self, content: str) -> str:
        """移除许可证头和版权信息"""
        lines = content.split('\n')
        result_lines = []
        skip_until_import = False
        found_code = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 跳过开头的注释块（许可证等）
            if i < 50 and not found_code:
                # 如果是注释或空行，并且还没遇到代码
                if stripped.startswith('#') or stripped == '':
                    # 检查是否是许可证相关
                    if any(kw in line for kw in ['Copyright', 'License', 'Licensed', 'Apache', 'MIT']):
                        skip_until_import = True
                        continue
                    if skip_until_import:
                        continue
                # 如果遇到 import 或 from，说明代码开始了
                if stripped.startswith('import ') or stripped.startswith('from '):
                    found_code = True
                    skip_until_import = False

            result_lines.append(line)

        return '\n'.join(result_lines)

    def simplify_file(self, file_path: Path) -> Tuple[bool, str]:
        """深度简化单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 应用深度简化
            content = self.remove_license_and_headers(content)
            content = self.remove_examples_section(content)
            content = self.remove_args_section(content)
            content = self.simplify_docstring_aggressive(content)
            content = self.simplify_type_annotations(content)

            # 移除多余空行
            content = re.sub(r'\n\n\n+', '\n\n', content)

            # 确保文件以换行符结尾
            if content and not content.endswith('\n'):
                content += '\n'

            # 写入文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, ""
            else:
                return False, ""

        except Exception as e:
            return False, str(e)

    def process_directory(self, directory: Path, pattern: str = "*.py") -> dict:
        """处理目录中的所有文件"""
        stats = {
            'total': 0,
            'modified': 0,
            'skipped': 0,
            'errors': 0,
        }

        exclude_patterns = ['__pycache__', '.git', 'test', 'example']

        for py_file in directory.rglob(pattern):
            # 跳过排除的目录
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            stats['total'] += 1
            modified, error = self.simplify_file(py_file)

            if error:
                stats['errors'] += 1
                print(f"❌ {py_file.relative_to(directory)}: {error}")
            elif modified:
                stats['modified'] += 1
                print(f"✓ {py_file.relative_to(directory)}")
            else:
                stats['skipped'] += 1

        return stats


def main():
    """主函数"""
    print("=" * 80)
    print("深度代码简化工具")
    print("=" * 80)
    print()

    base_dir = Path('/home/user/diffusers')
    src_dir = base_dir / 'src' / 'diffusers'

    if not src_dir.exists():
        print(f"❌ 目录不存在: {src_dir}")
        return

    print(f"📁 目标目录: {src_dir}")
    print()

    simplifier = DeepCodeSimplifier()

    # 处理所有 Python 文件
    print("开始深度简化所有文件...")
    stats = simplifier.process_directory(src_dir)

    print(f"\n{'='*80}")
    print("统计信息")
    print(f"{'='*80}")
    print(f"总文件数: {stats['total']}")
    print(f"已修改: {stats['modified']}")
    print(f"未修改: {stats['skipped']}")
    print(f"错误: {stats['errors']}")
    print()


if __name__ == '__main__':
    main()
