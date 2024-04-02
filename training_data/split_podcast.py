#!/usr/local/bin/python3

# For training data, we'll be using an hour-long interview that the
# assemblymember gave on the Max Politics podcast. It's available on
# soundcloud at 
# https://soundcloud.com/gotham-gazette-max-murphy/assemblymember-alex-bores-on-ai-opportunity-court-reform-housing-policy-more.

from pydub import AudioSegment

input_file = (
    "Assemblymember Alex Bores on AI Opportunity, "
    + "Court Reform, Housing Policy, & More.mp3"
)

# convert take the mp3 from soundcloud...
sound = AudioSegment.from_mp3(input_file)

for timeRange_ms in enumerate(
    [ # timestamps for slices are in milliseconds -- I got them
      # by listening to the podcast and noting when Alex starts
      # and stops talking.
      # (start time, stop time), with time as (min*60 + sec) * 1000
      ((4  * 60 + 22) * 1000, (4  * 60 + 28) * 1000),
      ((5  * 60 + 11) * 1000, (5  * 60 + 45) * 1000),
      ((6  * 60 + 14) * 1000, (6  * 60 + 45) * 1000),
      ((7  * 60 +  9) * 1000, (8  * 60 + 14) * 1000),
      ((10 * 60 + 13) * 1000, (11 * 60 + 46) * 1000),
      ((12 * 60 + 23) * 1000, (13 * 60 +  6) * 1000),
      ((15 * 60 +  9) * 1000, (16 * 60 +  6) * 1000),
      ((18 * 60 +  7) * 1000, (19 * 60 + 21) * 1000),
      ((19 * 60 + 25) * 1000, (19 * 60 + 40) * 1000),
      ((20 * 60 + 17) * 1000, (21 * 60 + 23) * 1000),
      ((21 * 60 + 53) * 1000, (23 * 60 + 58) * 1000),
      ((24 * 60 + 27) * 1000, (25 * 60 + 25) * 1000),
      ((25 * 60 + 51) * 1000, (26 * 60 + 45) * 1000),
      ((27 * 60 + 12) * 1000, (27 * 60 + 25) * 1000),
      ((29 * 60 + 42) * 1000, (32 * 60 + 18) * 1000),
      ((32 * 60 + 57) * 1000, (33 * 60 + 22) * 1000),
      ((33 * 60 + 51) * 1000, (35 * 60 + 24) * 1000),
      ((36 * 60 + 27) * 1000, (39 * 60 + 52) * 1000),
      ((39 * 60 + 57) * 1000, (40 * 60 + 14) * 1000),
      ((41 * 60 + 59) * 1000, (47 * 60 + 13) * 1000),
      ((48 * 60 +  4) * 1000, (48 * 60 + 44) * 1000),
      ((50 * 60 + 13) * 1000, (51 * 60 + 50) * 1000),
      ((52 * 60 + 20) * 1000, (53 * 60 + 44) * 1000),
      ]
    ):
    sound[timeRange_ms[1][0] : timeRange_ms[1][1]].export(
        "AlexBoresVoice" + str(timeRange_ms[0]) + ".wav", format="wav"
        )