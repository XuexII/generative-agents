from pydantic import BaseModel
from typing import Any, List, Dict


class MyLocation(BaseModel):
    name: str  # 地名
    desc: str = ""  # 地点描述
    coordinate: Any  # 坐标
    bbox: Any  # 边界框
    lay: Any  # 陈设
    owner: Any  # 所有者
    guest: List  # 来客

    def __str__(self):
        return self.desc


class MyMap(BaseModel):
    locations: Dict[MyLocation] = {}  # 所有地点


my_map = MyMap()


def get_loc(name) -> MyLocation:
    return my_map.locations.get(name)
