from typing import List, Dict, Tuple, Optional, Any

from pydantic import BaseModel, Extra
from queue import Queue

type_map = {"1": "空间", "2": "物品"}


class Res(BaseModel):
    name: str  # 名称
    desc: str = ""  # 描述
    status: str = ""  # 状态

    def __str__(self):
        return f"{self.name}({self.desc})"


class MyLocation(BaseModel):
    name: str  # 地名
    type: str  # 地点类型
    desc: Optional[str] = None  # 点的描述
    owner: Any = None  # 父节点
    coord: Optional[Tuple] = None  # 坐标
    layout: List[Res] = []  # 陈设的物品
    subspace: List = []  # 子节点
    guest: Dict = {}  # 来客

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def get_surroundings(self):
        """
        获取周围的情况
        """
        surroundings = []
        for loc in self.subspace:
            surroundings.append(loc.name)
        return ",".join(surroundings)

    def viewed(self, obj):
        """
        能被看到的内容, 陈设和访客，当前仅实现访客
        """
        guest_list = [agt for agt in self.guest.values() if agt != obj]
        return guest_list

    def is_leaf(self) -> bool:
        """
        判断是否是叶子节点
        """

        if len(self.subspace) > 1:
            return False
        return True

    def accept(self, obj):
        """
        接收成员
        """
        if obj.id not in self.guest:
            self.guest[obj.id] = obj

    def remove(self, obj):
        """
        移除成员
        """
        self.guest.pop(obj.id, None)

    def get_guest(self):
        """
        获取所有访客
        """
        return list(self.guest.values())

    def add_subspace(self, obj):
        """
        添加子空间
        """
        self.subspace.append(obj)

    def add_layout(self, obj):
        """
        添加陈设
        """
        self.layout.append(obj)

    def get_all_path(self):
        path = [self.name]
        owner: MyLocation = self.owner
        while isinstance(owner, MyLocation):
            path.append(owner.name)
            owner = owner.owner

        return ":".join(path[::-1])

    def __str__(self):
        if self.desc:
            return f"{self.name}({self.desc})"
        return self.name

    def __hash__(self):
        return hash(f"{self.name}-{self.type}")

    def __eq__(self, other):
        if not isinstance(other, MyLocation):
            return False
        if self.name != other.name:
            return False
        if self.type != other.type:
            return False
        return True


class MyMap:
    def __init__(self):
        self.locations: Dict[str: MyLocation] = {}  # 所有地点

    def init_map(self, map_dict):
        """
        初始化地图
        """
        # TODO 对于重名的怎么处理
        q = Queue()
        q.put((None, map_dict))

        while not q.empty():
            owner, locs = q.get()

            for loc in locs:
                name = loc["name"]
                subspace = loc.pop("subspace", [])
                loc["layout"] = [Res(**i) for i in loc.get("layout", [])]
                loc["owner"] = owner
                loc_obj = MyLocation(**loc)
                if isinstance(owner, MyLocation):
                    owner.add_subspace(loc_obj)

                self.locations[name] = loc_obj
                if subspace:
                    q.put((loc_obj, subspace))

    def add_guest(self, agent_list: List):
        for agent in agent_list:
            loc_obj = self.get_loc(agent.loc)
            if loc_obj is not None:
                loc_obj.accept(agent)

    def get_loc(self, name: str, default=None) -> MyLocation:
        """
        根据名称获取地点  TODO 修改为通过坐标获取地点
        """
        return self.locations.get(name, default)


my_map = MyMap()

maps = [
    {
        "name": "张三家",
        "type": 1,
        "pos": None,
        "layout": [
            {"name": "沙发"}
        ],
        "subspace": [
            {
                "name": "张三的卧室",
                "type": "1",
                "pos": None,
                "subspace": []
            },
            {
                "name": "张三家的厨房",
                "type": "1",
                "pos": None,
                "subspace": []
            },
            {
                "name": "张小西的卧室",
                "type": "1",
                "pos": None,
                "subspace": []
            },
            {
                "name": "张三家的卫生间",
                "type": "1",
                "pos": None,
                "subspace": []
            }
        ]
    },
    {
        "name": "李四家",
        "type": "1",
        "pos": None,
        "subspace": [
            {
                "name": "李四的卧室",
                "type": "1",
                "pos": None,
                "subspace": []
            },
            {
                "name": "李四家的厨房",
                "type": "1",
                "pos": None,
                "subspace": []
            },
            {
                "name": "李四家的卫生间",
                "type": "1",
                "pos": None,
                "subspace": []
            }
        ]
    },
    {
        "name": "CBD",
        "type": "1",
        "pos": None,
        "subspace": [
            {
                "name": "一层办公室",
                "type": "1",
                "pos": None,
                "subspace": []
            },
            {
                "name": "二层办公室",
                "type": "1",
                "pos": None,
                "subspace": []
            },
            {
                "name": "三层办公室",
                "type": "1",
                "pos": None,
                "subspace": []
            }
        ]
    },

]
