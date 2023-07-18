from pydantic import BaseModel, Extra

class MyThing(BaseModel):
    name: str  # 名称
    desc: str = ""  # 描述
    status: str = ""  # 状态

    def __str__(self):
        return f"{self.name}({self.desc})"