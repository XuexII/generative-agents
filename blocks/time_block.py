from datetime import datetime, timedelta
from typing import Dict

weekday_map = {1: "周一", 2: "周二", 3: "周三",
               4: "周四", 5: "周五", 6: "周六", 7: "周日"}


def get_date_info(date: datetime) -> Dict:
    _, week, weekday = date.isocalendar()

    date_dict = {"year": date.year, "month": date.month, "day": date.day,
                 "hour": date.hour, "minute": date.minute, "second": date.second,
                 "week": week, "weekday": weekday_map.get(weekday, "周一")}
    date_dict = {k: v if v is not None else 0 for k, v in date_dict.items()}
    return date_dict


class MyDateTime:

    def __init__(self, date_string, date_format="%Y-%m-%d %H:%M:%S"):
        self.date = datetime.strptime(date_string, date_format)
        self.format = date_format

    def update(self, second):
        new_date = self.date.timestamp() + second
        self.date = datetime.fromtimestamp(new_date)

    def __str__(self):
        return self.date.strftime(self.format)

    def __sub__(self, other):
        """返回秒"""
        if not isinstance(other, MyDateTime):
            return TypeError(f"需要的数据类型为：MyDateTime，传入的数据类型为：{type(other)}")
        return self.date.timestamp() - other.date.timestamp()

    def past(self, days: int):
        """查询过去几天的时间"""
        date = self.date - timedelta(days=days)
        return get_date_info(date)

    def next(self, days: int):
        """查询未来几天的时间"""
        date = self.date + timedelta(days=days)
        return get_date_info(date)

    def dict(self, *args, **kwargs):
        args = locals()  # 获取参数
        exclude = set()  # 不包含的字段
        date_dict = get_date_info(self.date)
        return date_dict
