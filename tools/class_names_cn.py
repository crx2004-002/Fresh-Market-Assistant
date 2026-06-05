"""
水果类别中英文名称映射
统一管理所有脚本使用的类别名称
支持39类水果（dataset_merged数据集）
"""

# 水果类别中英文名称映射（39类）
# key 需与数据集目录名完全一致
FRUIT_NAMES_CN = {
    'apple': '苹果',
    'apricot': '杏',
    'avocado': '牛油果',
    'banana': '香蕉',
    'bayberry': '杨梅',
    'blueberry': '蓝莓',
    'cantaloupe': '哈密瓜',
    'carambola': '杨桃',
    'cherry': '樱桃',
    'coconut': '椰子',
    'cranberry': '蔓越莓',
    'dragonfruit': '火龙果',
    'durian': '榴莲',
    'fig': '无花果',
    'grape': '葡萄',
    'grapefruit': '柚子',
    'guava': '番石榴',
    'hawthorn': '山楂',
    'jackfruit': '菠萝蜜',
    'kiwi fruit': '猕猴桃',
    'kumquat': '金桔',
    'lemon': '柠檬',
    'longan': '龙眼',
    'loquat': '枇杷',
    'lychee': '荔枝',
    'mandarine': '柑橘',
    'mango': '芒果',
    'mulberry': '桑葚',
    'orange': '橙子',
    'peach': '桃子',
    'pear': '梨',
    'persimmon': '柿子',
    'pineapple': '菠萝',
    'plumcot': '李杏',
    'pomegranate': '石榴',
    'pomelo': '柚子',
    'strawberry': '草莓',
    'tomato': '番茄',
    'watermelon': '西瓜',
}

# 别名映射（兼容旧代码中的 'kiwi' 等写法）
FRUIT_NAME_ALIASES = {
    'kiwi': 'kiwi fruit',
}

# 水果类别 Emoji 映射（用于界面显示）
FRUIT_EMOJI = {
    'apple': '🍎',
    'apricot': '🍑',
    'avocado': '🥑',
    'banana': '🍌',
    'bayberry': '🫐',
    'blueberry': '🫐',
    'cantaloupe': '🍈',
    'carambola': '⭐',
    'cherry': '🍒',
    'coconut': '🥥',
    'cranberry': '🔴',
    'dragonfruit': '🐉',
    'durian': '🟤',
    'fig': '🟣',
    'grape': '🍇',
    'grapefruit': '🍊',
    'guava': '🟢',
    'hawthorn': '🔴',
    'jackfruit': '🍈',
    'kiwi fruit': '🥝',
    'kumquat': '🍊',
    'lemon': '🍋',
    'longan': '⚪',
    'loquat': '🟡',
    'lychee': '🔴',
    'mandarine': '🍊',
    'mango': '🥭',
    'mulberry': '🟣',
    'orange': '🍊',
    'peach': '🍑',
    'pear': '梨',
    'persimmon': '🟠',
    'pineapple': '🍍',
    'plumcot': '🍑',
    'pomegranate': '🔴',
    'pomelo': '🍈',
    'strawberry': '🍓',
    'tomato': '🍅',
    'watermelon': '🍉',
}

def get_cn_name(en_name):
    """获取水果中文名称，支持别名"""
    # 先尝试别名映射
    resolved = FRUIT_NAME_ALIASES.get(en_name, en_name)
    return FRUIT_NAMES_CN.get(resolved, en_name)

def get_emoji(en_name):
    """获取水果Emoji，支持别名"""
    resolved = FRUIT_NAME_ALIASES.get(en_name, en_name)
    return FRUIT_EMOJI.get(resolved, '❓')

def get_full_name(en_name):
    """获取完整名称 (emoji + 中文名)"""
    emoji = get_emoji(en_name)
    cn_name = get_cn_name(en_name)
    return f"{emoji} {cn_name}"

# 测试
if __name__ == "__main__":
    print("水果类别中英文名称映射 (39类):")
    print("-" * 50)
    for en_name, cn_name in FRUIT_NAMES_CN.items():
        emoji = FRUIT_EMOJI.get(en_name, '')
        print(f"{emoji} {cn_name:8s} ({en_name})")
