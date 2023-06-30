from queue import Queue
from typing import List, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from blocks.time_block import MyDateTime


class Plan(BaseModel):
    start: str  # 开始时间 08:00
    end: str  # 结束时间 08:05
    task: str  # 要执行的任务

    def __str__(self):
        return f"[{self.start}-{self.end}]: {self.task}"

    def __setattr__(self, key, value):
        super(Plan, self).__setattr__(key, value)

    def __getattribute__(self, item):
        return super(Plan, self).__getattribute__(item)

    def get_info(self):
        def _to_timedelta(time_str):
            hours, minutes = map(int, time_str.split(':'))
            td = timedelta(hours=hours, minutes=minutes)
            return td

        start = _to_timedelta(self.start)
        end = _to_timedelta(self.end)
        return start, end, self.task


class PlanQueue(Queue):

    def batch_put(self, plans: List[Plan]):
        for plan in plans:
            self.put(plan)

    def __str__(self):
        plans_str = [str(plan) for plan in self.queue]
        return "\n".join(plans_str)

    def get(self, block=True, timeout=None):
        plan = super(PlanQueue, self).get(block, timeout)
        return plan
