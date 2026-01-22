#!/usr/bin/env python3
"""
代码简化脚本 - 自动简化 diffusers 代码库中的所有 Python 文件
根据用户提供的示例进行简化，保留核心逻辑，添加简洁的中文注释
"""

import re
import os
from pathlib import Path
from typing import List, Tuple


class CodeSimplifier:
    """代码简化器"""

    def __init__(self):
        # 许可证相关的关键词
        self.license_keywords = [
            'Copyright', 'LICENSE', 'License', 'Licensed',
            'Apache', 'MIT', 'BSD', 'GPL', 'Attribution',
            'permission', 'Permission', 'warranty', 'Warranty',
            'WITHOUT WARRANTIES', 'AS IS', 'THE SOFTWARE'
        ]

    def remove_license_header(self, content: str) -> str:
        """移除文件开头的许可证头"""
        lines = content.split('\n')
        result_lines = []
        in_license = False
        license_block_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 检测许可证块的开始
            if i < 50 and any(kw in line for kw in self.license_keywords):
                if not in_license:
                    in_license = True
                    license_block_start = i
                continue

            # 如果在许可证块中，跳过注释行
            if in_license:
                if stripped.startswith('#') or stripped == '':
                    continue
                else:
                    in_license = False

            result_lines.append(line)

        return '\n'.join(result_lines)

    def simplify_docstring(self, content: str) -> str:
        """简化文档字符串 - 将冗长的英文文档字符串替换为简洁的中文注释"""

        # 匹配三引号文档字符串
        def replace_docstring(match):
            docstring = match.group(1)
            lines = docstring.strip().split('\n')

            # 如果文档字符串很短（<3行），保留
            if len([l for l in lines if l.strip()]) < 3:
                return match.group(0)

            # 提取第一行作为简要说明
            first_line = lines[0].strip() if lines else ""

            # 如果第一行为空，找第一个非空行
            if not first_line:
                for line in lines:
                    if line.strip():
                        first_line = line.strip()
                        break

            # 简化为单行注释（保留原有缩进）
            indent = match.group(0).split('"""')[0]
            if first_line:
                return f'{indent}"""{first_line}"""'
            else:
                return f'{indent}"""核心功能"""'

        # 替换三引号文档字符串
        content = re.sub(r'(\s+)"""(.*?)"""', replace_docstring, content, flags=re.DOTALL)
        content = re.sub(r"(\s+)'''(.*?)'''", replace_docstring, content, flags=re.DOTALL)

        return content

    def simplify_type_hints(self, content: str) -> str:
        """简化类型注解 - 移除过于复杂的类型注解"""

        # 移除 Optional[...] 的部分复杂嵌套
        # Optional[Union[torch.Generator, List[torch.Generator]]] -> Optional[torch.Generator]
        content = re.sub(
            r'Optional\[Union\[([\w.]+),\s*List\[\1\]\]\]',
            r'Optional[\1]',
            content
        )

        return content

    def remove_verbose_comments(self, content: str) -> str:
        """移除过于冗长的单行注释"""
        lines = content.split('\n')
        result_lines = []

        for line in lines:
            # 如果是超过100个字符的纯注释行，可能需要简化
            if line.strip().startswith('#') and len(line) > 100:
                # 保留重要的 TODO, FIXME, NOTE 等
                if any(kw in line for kw in ['TODO', 'FIXME', 'NOTE', 'IMPORTANT', 'WARNING']):
                    result_lines.append(line)
                else:
                    # 简化为前50个字符
                    simplified = line[:50].rstrip() + '...'
                    result_lines.append(simplified)
            else:
                result_lines.append(line)

        return '\n'.join(result_lines)

    def remove_blank_lines(self, content: str) -> str:
        """移除多余的空行 - 最多保留2个连续空行"""
        # 将3个或更多连续空行替换为2个
        content = re.sub(r'\n\n\n+', '\n\n', content)
        return content

    def simplify_examples(self, content: str) -> str:
        """简化示例代码块 - 移除过长的 Examples 部分"""

        def replace_example(match):
            example_block = match.group(0)

            # 如果示例太长（>1000字符），简化
            if len(example_block) > 1000:
                indent = match.group(1) if match.lastindex >= 1 else ''
                return f'{indent}"""\n{indent}使用示例见文档\n{indent}"""'

            return example_block

        # 匹配 Examples: 开始的块
        content = re.sub(
            r'(\s+)Examples?:\s*\n\s+```.*?```',
            replace_example,
            content,
            flags=re.DOTALL | re.IGNORECASE
        )

        return content

    def simplify_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        简化单个文件

        返回: (是否修改, 错误信息)
        """
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # 备份原始内容
            backup_content = original_content

            # 应用简化规则
            content = original_content

            # 1. 移除许可证头
            content = self.remove_license_header(content)

            # 2. 简化文档字符串
            content = self.simplify_docstring(content)

            # 3. 简化类型注解
            content = self.simplify_type_hints(content)

            # 4. 移除冗长注释
            content = self.remove_verbose_comments(content)

            # 5. 简化示例
            content = self.simplify_examples(content)

            # 6. 移除多余空行
            content = self.remove_blank_lines(content)

            # 确保文件以换行符结尾
            if content and not content.endswith('\n'):
                content += '\n'

            # 如果内容有变化，写入文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, ""
            else:
                return False, ""

        except Exception as e:
            return False, str(e)

    def simplify_directory(self, directory: Path, exclude_patterns: List[str] = None) -> dict:
        """
        简化目录中的所有 Python 文件

        返回统计信息
        """
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'tests', 'examples']

        stats = {
            'total': 0,
            'modified': 0,
            'skipped': 0,
            'errors': 0,
            'error_files': []
        }

        # 遍历所有 Python 文件
        for py_file in directory.rglob('*.py'):
            # 跳过排除的目录
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            stats['total'] += 1

            # 简化文件
            modified, error = self.simplify_file(py_file)

            if error:
                stats['errors'] += 1
                stats['error_files'].append((str(py_file), error))
                print(f"❌ 错误: {py_file.relative_to(directory)}: {error}")
            elif modified:
                stats['modified'] += 1
                print(f"✓ 已简化: {py_file.relative_to(directory)}")
            else:
                stats['skipped'] += 1

        return stats


def main():
    """主函数"""
    print("=" * 80)
    print("代码简化工具 - Diffusers 代码库")
    print("=" * 80)
    print()

    # 获取 diffusers 源代码目录
    base_dir = Path('/home/user/diffusers')
    src_dir = base_dir / 'src' / 'diffusers'

    if not src_dir.exists():
        print(f"❌ 错误: 找不到目录 {src_dir}")
        return

    print(f"📁 目标目录: {src_dir}")
    print()

    # 创建简化器
    simplifier = CodeSimplifier()

    # 处理各个模块
    modules = [
        ('models', 'models'),
        ('schedulers', 'schedulers'),
        ('pipelines', 'pipelines'),
        ('modular_pipelines', 'modular_pipelines'),
        ('utils', 'utils'),
        ('loaders', 'loaders'),
        ('quantizers', 'quantizers'),
        ('guiders', 'guiders'),
        ('hooks', 'hooks'),
        ('commands', 'commands'),
    ]

    total_stats = {
        'total': 0,
        'modified': 0,
        'skipped': 0,
        'errors': 0,
        'error_files': []
    }

    # 处理每个模块
    for module_name, module_path in modules:
        module_dir = src_dir / module_path
        if not module_dir.exists():
            print(f"⚠️  跳过 {module_name}: 目录不存在")
            continue

        print(f"\n{'='*60}")
        print(f"处理模块: {module_name}")
        print(f"{'='*60}")

        stats = simplifier.simplify_directory(module_dir)

        # 累计统计
        for key in ['total', 'modified', 'skipped', 'errors']:
            total_stats[key] += stats[key]
        total_stats['error_files'].extend(stats['error_files'])

        print(f"\n模块统计:")
        print(f"  总文件数: {stats['total']}")
        print(f"  已修改: {stats['modified']}")
        print(f"  未修改: {stats['skipped']}")
        print(f"  错误: {stats['errors']}")

    # 处理顶层文件
    print(f"\n{'='*60}")
    print(f"处理顶层文件")
    print(f"{'='*60}")

    for py_file in src_dir.glob('*.py'):
        total_stats['total'] += 1
        modified, error = simplifier.simplify_file(py_file)

        if error:
            total_stats['errors'] += 1
            total_stats['error_files'].append((str(py_file), error))
            print(f"❌ 错误: {py_file.name}: {error}")
        elif modified:
            total_stats['modified'] += 1
            print(f"✓ 已简化: {py_file.name}")
        else:
            total_stats['skipped'] += 1

    # 打印总体统计
    print(f"\n{'='*80}")
    print("总体统计")
    print(f"{'='*80}")
    print(f"总文件数: {total_stats['total']}")
    print(f"已修改: {total_stats['modified']}")
    print(f"未修改: {total_stats['skipped']}")
    print(f"错误: {total_stats['errors']}")

    if total_stats['error_files']:
        print(f"\n错误文件列表:")
        for file_path, error in total_stats['error_files'][:10]:  # 只显示前10个
            print(f"  - {file_path}: {error}")
        if len(total_stats['error_files']) > 10:
            print(f"  ... 还有 {len(total_stats['error_files']) - 10} 个错误")

    print(f"\n{'='*80}")
    print("✓ 简化完成!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
