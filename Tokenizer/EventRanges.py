from dataclasses import dataclass
import math


@dataclass
class EventRange:
    eventType: str
    eventStartRange: float
    eventEndRange: float
    eventRangeStep: float = 1.0

    @property
    def numberOfTokens(self) -> int:
        return int((self.eventEndRange - self.eventStartRange) / self.eventRangeStep) + 1

    # def encode(self, value):
    #     assert (self.eventStartRange <= value <= self.eventEndRange)
    #

def generateTimeRange(maxTimeSteps, timeStepsPerSecond):
    return EventRange("time", 0, maxTimeSteps / timeStepsPerSecond, 1 / timeStepsPerSecond)


@dataclass
class Event:
    eventName: str
    subEvents: list[EventRange]

    @property
    def numberOfTokens(self):
        return math.prod(i.numberOfTokens for i in self.subEvents)
