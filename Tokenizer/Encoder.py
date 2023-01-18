import Event

# encoding and tie section inspired by https://github.com/magenta/mt3

noteStart = EventRanges.Event(
    "notes",
    [
        EventRanges.EventVariable("string", 0, 5),
        EventRanges.EventVariable("fret", 0, 25),
        EventRanges.EventVariable("palm_mute", 0, 1),
        EventRanges.EventVariable("hammer_on", 0, 1),
        # EventRanges.EventRange("hopo", 0, 1), hopo is hammer on pull off
        EventRanges.EventVariable("tap", 0, 1)
    ]
)

# Slides are notes that start somewhere but end somewhere else?
noteEnd = EventRanges.Event(
    "notes",
    [
        EventRanges.EventVariable("string", 0, 5),
        EventRanges.EventVariable("fret", 0, 25),
        EventRanges.EventVariable("pull_off", 0, 1),
        EventRanges.EventVariable("unpitched_slide", 0, 1)
    ]
)

bend = EventRanges.Event(
    "bend",
    [
        EventRanges.EventVariable("string", 1, 6),
        EventRanges.EventVariable("semi-tone", -2.5, 2.5, 0.5, False),
        EventRanges.EventVariable("tap", 0, 1)
    ]
)
