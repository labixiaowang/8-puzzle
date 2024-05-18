# # 实验一
# --------------问题一----------------
print("helloworld!")
# ----------问题二&&问题四----------------
print("请输入一个年份")
year=int(input())
# 判断闰年&&调用函数
def judge(year):
    """
    year:输入的年份
    return True or False
    """
    if year%400==0 or year%4==0 and year%100!=0:
        return True
    else:
        return False


for i in range(2020,2051,1):   # 使用循环判断一段时间内的闰年
    if judge(i):
        print("{}是闰年".format(i))
#------------问题三----------------
# 使用列表
x = input("请输入一组数组并使用逗号隔开:").split(',')
i = 0
sum = 0
while i<len(x):
    sum+=len(x[i])
print("输入数据的和:".format(sum))
print("输入数据的均值:".format(sum/len(x)))
# 使用字典
dic = {}
s = input()
for i in s:
    if i in dic:
        dic[i] = dic[i]+1
    else:
        dic[i] = 1
print(dic)