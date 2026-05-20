import json

# 读取 json 文件
with open("/home/wjh/Wan2.2/evaluation_results/results_2026-05-05-18:36:00_eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取所有 aes_score
scores = [item["aes_score"] for item in data.values()]

# 计算平均分
average_score = sum(scores) / len(scores)

print("平均分:", average_score)