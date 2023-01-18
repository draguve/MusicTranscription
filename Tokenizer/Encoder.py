import Event
from dataclasses import dataclass
from bisect import bisect_right


# encoding and tie section inspired by https://github.com/magenta/mt3

@dataclass
class Encoder:
    events: list[Event.Event]

    def __post_init__(self):
        self._locations = []
        self._nameToLocation = {}
        self._handlers = {}
        current = 0
        for event in self.events:
            self._handlers[event.eventName] = event
            self._locations.append(current)
            self._nameToLocation[event.eventName] = current
            current += event.numberOfTokens
        self._numberOfTokens = current

    def encode(self, eventName, data):
        assert eventName in self._handlers
        eventOffset = self._nameToLocation[eventName]
        eventIndex = self._handlers[eventName].encode(data)
        return eventOffset + eventIndex

    def decode(self, value):
        handlerIndex = bisect_right(dataEncoder._locations, value) - 1
        handler = self.events[handlerIndex]
        eventOffset = self._locations[handlerIndex]
        return handler.eventName, handler.decode(value - eventOffset)

    @property
    def numberOfTokens(self):
        return self._numberOfTokens


if __name__ == '__main__':
    noteStartHandler = Event.Event(
        "noteStart",
        [
            Event.EventVariable("string", 0, 5),
            Event.EventVariable("fret", 0, 25),
            Event.EventVariable("palm_mute", 0, 1),
            Event.EventVariable("hammer_on", 0, 1),
            # EventRanges.EventRange("hopo", 0, 1), hopo is hammer on pull off
            Event.EventVariable("tap", 0, 1)
        ]
    )

    # Slides are notes that start somewhere but end somewhere else?
    noteEndHandler = Event.Event(
        "noteEnd",
        [
            Event.EventVariable("string", 0, 5),
            Event.EventVariable("fret", 0, 25),
            Event.EventVariable("pull_off", 0, 1),
            Event.EventVariable("unpitched_slide", 0, 1)
        ]
    )

    bendHandler = Event.Event(
        "bend",
        [
            Event.EventVariable("string", 1, 6),
            Event.EventVariable("semi-tone", -2.5, 2.5, 0.5, False),
            Event.EventVariable("tap", 0, 1)
        ]
    )

    dataEncoder = Encoder([
        noteStartHandler,
        noteEndHandler,
        bendHandler
    ])

    encoded = []
    encoded.append(dataEncoder.encode("bend", [2, 0, 0]))
    encoded.append(dataEncoder.encode("noteStart", [0, 23, 1, 0, 0]))
    print(encoded)
    for code in encoded:
        print(dataEncoder.decode(code))

# TODO: need to create an end tie section, End of Sequence Event and time event
