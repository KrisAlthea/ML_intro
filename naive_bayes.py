from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# 扩展的训练数据
train_texts = [
    "亲爱的用户，您有一条未读消息。",
    "最新活动，仅限今日！快来参加。",
    "我们一起去公园散步吧。",
    "特价优惠，立刻购买。",
    "请问您周末有空吗？",
    "信用卡账单逾期，请立即处理。",
    "这个天气真适合出去玩。",
    "恭喜您获得我们的专属优惠券。",
    "今晚一起去看电影怎么样？",
    "抓紧最后的折扣机会！",
    "新款手机大促销，不要错过。",
    "周末烧烤聚会，期待你的参与。",
    "你的包裹已发货，请注意查收。",
    "独家优惠等你来拿。",
    "这个周末有免费的瑜伽课程。",
    "请确认您的银行账户信息。",
    "我们这周末组织了一个小聚会。",
    "只限今天，赶紧来抢购。",
    "感谢您对我们产品的支持。",
    "恭喜发财，大吉大利。"
]
train_labels = [
    'Non-Spam', 'Spam', 'Non-Spam', 'Spam', 'Non-Spam',
    'Spam', 'Non-Spam', 'Spam', 'Non-Spam', 'Spam',
    'Spam', 'Non-Spam', 'Non-Spam', 'Spam', 'Non-Spam',
    'Spam', 'Non-Spam', 'Spam', 'Non-Spam', 'Spam'
]


# 扩展的测试数据
test_texts = [
    "快来参加我们的夏日促销活动。",
    "这周末我打算去爬山，你要一起吗？",
    "您的账户存在安全风险，请立即更改密码。",
    "我们这个周末组织了一次聚餐，欢迎大家参加。",
    "限时优惠，最后三天！",
    "今晚的月色真美，一起去散步吧。",
    "恭喜您中奖了，请点击此链接领奖。",
    "想不想加入我们的读书会？",
    "不要错过今天的特别折扣哦。",
    "你好，明天有时间帮我一下吗？",
    "独家优惠，仅限今日！",
    "明天晚上有一场音乐会，我有两张票。",
    "警告：您的邮箱即将超出存储限制。",
    "周末一起去海边放松一下怎么样？",
    "抢购通知：新品上市，数量有限！"
]
test_labels = [
    'Spam', 'Non-Spam', 'Spam', 'Non-Spam', 'Spam',
    'Non-Spam', 'Spam', 'Non-Spam', 'Spam', 'Non-Spam',
    'Spam', 'Non-Spam', 'Spam', 'Non-Spam', 'Spam'
]


# 创建一个文本处理和朴素贝叶斯分类的管道
# model = make_pipeline(CountVectorizer(), MultinomialNB())
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# model = make_pipeline(TfidfVectorizer(), BernoulliNB())
model = make_pipeline(CountVectorizer(), BernoulliNB())

# 训练模型
model.fit(train_texts, train_labels)

# 预测新邮件的类别
predicted_labels = model.predict(test_texts)

# 打印结果
print("\n分类报告:\n", classification_report(test_labels, predicted_labels, zero_division=0))

# 如果需要计算准确率
accuracy = accuracy_score(test_labels, predicted_labels)
print("准确率:", accuracy)
