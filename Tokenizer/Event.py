from dataclasses import dataclass
import math
from lerp import *
import numpy as np


@dataclass
class EventVariable:
    eventType: str
    eventStartRange: float
    eventEndRange: float
    eventRangeStep: float = 1.0
    roundAtEnd: bool = True

    def __post_init__(self):
        self._numberOfTokens = int((self.eventEndRange - self.eventStartRange) / self.eventRangeStep) + 1

    @property
    def numberOfTokens(self) -> int:
        return self._numberOfTokens

    def encode(self, value):
        assert (self.eventStartRange <= value <= self.eventEndRange)
        return round(remap(self.eventStartRange, self.eventEndRange, 0, self.numberOfTokens - 1, value))

    def decode(self, value):
        # TODO: search tree can be used here to decode
        assert 0 <= value <= self.numberOfTokens - 1
        new_value = remap(0, self.numberOfTokens - 1, self.eventStartRange, self.eventEndRange, value)
        return round(new_value) if self.roundAtEnd else new_value


@dataclass
class Event:
    eventName: str
    subEvents: list[EventVariable]

    def __post_init__(self):
        reversedEvents = self.subEvents[::-1]
        currentSize = reversedEvents[0].numberOfTokens
        weights = [1, currentSize]
        for i in range(1, len(reversedEvents)):
            currentSize *= reversedEvents[i].numberOfTokens
            weights.append(currentSize)
        self._weights = weights[::-1][1:]
        self._numberOfTokens = math.prod(i.numberOfTokens for i in self.subEvents)

    @property
    def numberOfTokens(self):
        return self._numberOfTokens

    def encode(self, data: list):
        if self.numberOfTokens == 1:
            return 0
        assert len(self.subEvents) == len(data)
        value = 0
        for i in range(0, len(self.subEvents)):
            value += self.subEvents[i].encode(data[i]) * self._weights[i]
        return value

    def decode(self, value) -> list:
        if self.numberOfTokens == 1:
            return [True]
        assert 0 <= value < self.numberOfTokens
        data = []
        remaining = value
        for i in range(0, len(self.subEvents)):
            temp = math.floor(remaining / self._weights[i])
            data.append(self.subEvents[i].decode(temp))
            remaining -= temp * self._weights[i]
        return data


def generateTimeRange(numOfSeconds, timeStepsPerSecond):
    return EventVariable("time", 0, numOfSeconds, 1 / timeStepsPerSecond,False)


if __name__ == '__main__':
    thing = EventVariable("test", -2.5, 2.5, 0.5, False)
    test_range = np.linspace(-2.5, 2.5, 11)
    print(test_range)
    encoded = [thing.encode(i) for i in test_range]
    print(encoded)
    # TODO: fix this decode issue
    print([thing.decode(i) for i in encoded])

    thing = EventVariable("tap", 0, 1)
    test_range = np.linspace(0, 1, 2)
    print(test_range)
    encoded = [thing.encode(i) for i in test_range]
    print(encoded)
    print([thing.decode(i) for i in encoded])

    thing = EventVariable("fret", 0, 25)
    test_range = np.linspace(0, 25, thing.numberOfTokens)
    print(test_range)
    encoded = [thing.encode(i) for i in test_range]
    print(encoded)
    print([thing.decode(i) for i in encoded])

    thing = Event(
        "node",
        [
            EventVariable("fret", 0, 25),
            EventVariable("string", 0, 5),
            EventVariable("test", -2.5, 2.5, 0.5, False),
            EventVariable("test2", 0, 1),
            EventVariable("tap", 0, 1)
        ]
    )
    print(thing.encode([25, 5, 2.0, 1, 1]))
    print(thing.decode(6859))

    thing = Event(
        "time",
        [
            generateTimeRange(2, 100)
        ]
    )
    print(thing.numberOfTokens)
    print(thing.encode([0.1]))
    print(thing.encode([0.4]))
    print(thing.encode([0.6]))
    print(thing.encode([0.7]))
    print(thing.encode([1.7]))
    print(thing.encode([1.95]))

    thing = Event(
        "EOS",
        [
            EventVariable("EOS", 0, 0)
        ]
    )
    print(thing.numberOfTokens)
    print(thing.encode([0]))
    print(thing.decode(0))
