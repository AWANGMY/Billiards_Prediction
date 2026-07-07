# 指令
请你根据最新代码以及复现结果（/work/gw17/w17001/Data/code/Billiards_Prediction/Doc/blformer_reproduction_check_20260707.md），撰写实验报告，放在/work/gw17/w17001/Data/code/Billiards_Prediction/Doc/Final_report中。



# 报告要求
- Adding new fundamental block with new mechanisms or architecture is preferred and need to be justified.
- Must explain all preprocessing steps and justify every transformation applied. Examples include normalization, feature extraction, augmentation, balancing, sequence preparation, and train/validation/test split.
- Must explain the architecture, training process, loss function, optimizer, and evaluation metrics.

- The report must include project objective, methodology, dataset description, preprocessing, model architecture, results, analysis, and conclusion.
- Discuss limitations, errors, or failure cases observed during experimentation.
- Consider the report as a research paper. As much as details that can fit in 4 one columns page (minimum font size 12) that justify each step with reference or math.

确保你的story telling：
- Novel architecture ideas
- Strong experimentation
- Excellent visualization and analysis

# 补充
- /work/gw17/w17001/Data/code/Billiards_Prediction/Doc中的其他Doc是基于旧代码的结果，仅供参考。
- 使用latex撰写，你可能需要配置合适的环境。
- 请添加必要的图、表，按照顶会论文标准。
- 为了必要的数据，你可以额外运行实验。GPU使用方法在/work/gw17/w17001/Data/code/Billiards_Prediction/Doc/BLFormer_exploration_report.md中。
- 必要时你可以写代码，但确保你添加的所有代码都集中在一个文件夹下。

# 修改建议
- 请对齐AI论文的写作标准（背景、动机、方法、对比实验、消融实验、结果分析、丰富的图表，不只是平铺直叙实验结果，更要进行包装），完成报告后，请用一个独立的agent，按照AI论文审稿标准进行审核，直到没有明显大问题。
- 目前包含了很多随意的名词，比如"paper40 split", "hybrid d64 + ord", "joint d80 clsmean marg0.5"等，这明显不符合论文标准。另外，路径、日期也不应该出现。
- 不记录GPU型号。
