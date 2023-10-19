from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef


# 两个变量的数据
x = [1, 0, 1, 1, 0]
y = [0, 1, 1, 1, 0]
matthews_corr = float(matthews_corrcoef(x, y))
print(matthews_corr)
print(pearsonr(x, y))
print(spearmanr(x,y)[0])
# # 计算 Pearson 相关系数和 p-value
# corr, p_value = pearsonr(x, y)
#
# # 打印结果
# print("Pearson correlation coefficient:", corr)
# print("p-value:", p_value)