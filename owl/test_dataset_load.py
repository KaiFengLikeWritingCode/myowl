from datasets import load_dataset

ds = load_dataset("google/frames-benchmark", cache_dir="./")
print(ds)  # 查看数据集结构
print(ds["test"][0])  # 查看第一条测试数据
print(ds["test"])

test_data = ds["test"]

# # 转换为 DataFrame
# df = test_data.to_pandas()
#
# # 导出完整数据
# df.to_excel("frames_benchmark_full.xlsx", index=False)
#
# # 导出筛选字段（可选）
# df[["Prompt", "Answer"]].to_excel("prompts_and_answers.xlsx", index=False)
#
# print("导出完成！")



