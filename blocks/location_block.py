from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Optional


class MyLocation(BaseModel):
    name: str  # 地名
    desc: str = ""  # 地点描述
    owner: Optional[str] = None  # 所有者
    coordinate: Tuple = ()  # 坐标
    bbox: Tuple = ()  # 边界框
    lay: List = []  # 陈设
    guest: List = []  # 来客

    def __str__(self):
        return self.desc


loc = MyLocation(name="家")


class MyMap:
    def __init__(self):
        self.locations: Dict[MyLocation] = {"家": loc}  # 所有地点


my_map = MyMap()


def get_loc(name) -> MyLocation:
    return my_map.locations.get(name)
