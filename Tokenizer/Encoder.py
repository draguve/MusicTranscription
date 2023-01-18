import EventRanges

# encoding and tie section inspired by https://github.com/magenta/mt3

noteStart = EventRanges.Event(
    "notes",
    [
        EventRanges.EventRange("string", 0, 5),
        EventRanges.EventRange("fret", 0, 25),
        EventRanges.EventRange("palm_mute", 0, 1),
        EventRanges.EventRange("hammer_on", 0, 1),
        # EventRanges.EventRange("hopo", 0, 1), hopo is hammer on pull off
        EventRanges.EventRange("tap", 0, 1)
    ]
)

# Slides are notes that start somewhere but end somewhere else?
noteEnd = EventRanges.Event(
    "notes",
    [
        EventRanges.EventRange("string", 0, 5),
        EventRanges.EventRange("fret", 0, 25),
        EventRanges.EventRange("pull_off", 0, 1),
        EventRanges.EventRange("unpitched_slide",0,1)
    ]
)

bend = EventRanges.Event(
    "bend",
    [
        EventRanges.EventRange("string", 1, 6),
        EventRanges.EventRange("semi-tone", -2.5, 2.5, 0.5),
        EventRanges.EventRange("tap", 0, 1)
    ]
)
