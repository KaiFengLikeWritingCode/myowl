import matplotlib.pyplot as plt

# 数据
data = {
    "stars": 1500,
    "forks": 300
}

# 创建柱状图
plt.bar(data.keys(), data.values(), color=['blue', 'green'])

# 添加标题和标签
plt.title("GitHub Stats for camel-ai/camel")
plt.xlabel("Metrics")
plt.ylabel("Count")

# 显示图表
plt.show()

# 保存图表到本地文件
plt.savefig("github_stats.png")
print("图表已保存为 'github_stats.png'")