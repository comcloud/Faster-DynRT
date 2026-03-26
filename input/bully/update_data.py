import pandas as pd

def filter_text_file(excel_file, text_file, output_file):
    # 读取Excel文件，获取第一列的图片名（去掉扩展名）
    excel_data = pd.read_excel(excel_file, header=None)
    image_names_in_excel = set(excel_data.iloc[:, 0].apply(lambda x: x.split('.')[0]))  # 获取图片前缀

    # 读取文本文件
    with open(text_file, 'r') as f:
        lines = f.readlines()

    # 过滤文本文件中的行，检查前缀是否在Excel数据中
    filtered_lines = []
    for line in lines:
        # 假设每行的第一个元素是图片前缀
        line_prefix = eval(line)[0].split('.')[0]  # 取每行的第一个词作为前缀
        if line_prefix in image_names_in_excel:
            filtered_lines.append(line)

    # 将过滤后的内容写入新的文本文件
    with open(output_file, 'w') as f:
        f.writelines(filtered_lines)

# 使用示例
excel_file = '/Users/rayss/Public/读研经历/论文/dataset/bully/bully_clean.xlsx'  # Excel文件路径
text_file = 'data.txt'  # 原始文本文件路径
output_file = 'filtered_image_data.txt'  # 输出的过滤后的文本文件路径

filter_text_file(excel_file, text_file, output_file)
