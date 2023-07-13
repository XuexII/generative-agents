from pydantic import BaseModel, Extra, Field, root_validator
from typing import Any, List, Dict, Tuple, Optional


class MyLocation(BaseModel):
    name: str  # 地名
    desc: str = ""  # 地点描述
    owner: Optional[str] = None  # 所有者
    coordinate: Tuple = ()  # 坐标
    bbox: Tuple = ()  # 边界框
    lay: List = []  # 陈设
    guest: List = []  # 来客
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def __str__(self):
        return self.name

    def get_surroundings(self):
        surroundings = []
        for loc in self.lay:
            surroundings.append(str(loc))
        return ",".join(surroundings)

    def is_leaf(self) -> bool:
        if len(self.lay) > 1:
            return False
        return True



loc = MyLocation(name="家")


class MyMap:
    def __init__(self):
        self.locations: Dict[MyLocation] = {"家": loc}  # 所有地点


my_map = MyMap()


def get_loc(name) -> MyLocation:
    """获取地点"""
    return my_map.locations.get(name)

