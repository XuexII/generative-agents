from typing import List, Dict, Tuple, Optional

from pydantic import BaseModel, Extra


class MyLocation(BaseModel):
    name: str  # 地名
    desc: str = ""  # 地点描述
    owner: Optional[str] = None  # 所有者
    layout: List = []  # 陈设
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


class MyMap:
    def __init__(self):
        self.locations: Dict[MyLocation] = {}  # 所有地点

    def get_loc(self, name: str) -> MyLocation:
        """
        根据名称获取地点
        """
        pass


my_map = MyMap()


def get_loc(name) -> MyLocation:
    """获取地点"""
    return my_map.locations.get(name)

map = {
    "张三家": {
        "layout": {
            "张三的卧室": {
                "layout": {}
            },
            "厨房": {
                "layout": {}
            },
            "张小西的卧室": {
                "layout": {}
            },
            "卫生间": {
                "layout": {}
            }
        }
    },
    "李四家": {
        "layout": {
            "李四的卧室": {
                "layout": {}
            },
            "厨房": {
                "layout": {}
            },
            "卫生间": {
                "layout": {}
            }
        }
    },
}

