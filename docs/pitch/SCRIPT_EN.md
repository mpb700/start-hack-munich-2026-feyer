# Pitch script (English, 3 minutes)

357 words. With the three seconds of silence at the start: about 2:36 at a
normal pace (140 words per minute), about 2:54 at a slow pace (125).
Times are calculated from the word count, not rehearsed with a stopwatch.

| Block | Window | Words | at 125 wpm | at 140 wpm |
| --- | --- | ---: | ---: | ---: |
| Hook (plus 3 s silence) | 0:00–0:20 | 32 | 15 s | 14 s |
| Detection | 0:20–0:45 | 52 | 25 s | 22 s |
| Goal | 0:45–1:05 | 39 | 19 s | 17 s |
| Problem | 1:05–1:35 | 59 | 28 s | 25 s |
| Our contribution | 1:35–2:15 | 93 | 45 s | 40 s |
| Outlook | 2:15–2:50 | 67 | 32 s | 29 s |
| Close | 2:50–3:00 | 15 | 7 s | 6 s |
| **Total with silence** | 3:00 | **357** | **2:54** | **2:36** |

At the slow pace "Our contribution" runs 5 s over its window and the whole
pitch keeps only 6 s of buffer. See "If time gets tight".

## Open decisions

1. **The speedup figure** (bold sentence in "Our contribution"). The final
   number is not fixed yet. The current sentence and two replacements are
   listed under "Speedup sentence: variants".
2. **The sentence about Feyer** ("AI that invents new machines for the
   physical world"). It is based on two third-party company profiles, not on
   Feyer's own words. Someone who has talked to the Feyer team should confirm it.
3. **191 or 193.** The animation counts to 191, the challenge README says 193
   design variables. The spoken "almost two hundred" fits both; the counter in
   the video should show the right one.

## Hook (0:00–0:20), 32 words

*[Three seconds of silence while the black holes circle.]*

More than a billion years ago, two black holes circled each other, faster and
faster, until they merged. The collision shook space itself. That tremor has
been travelling towards us ever since.

## Detection (0:20–0:45), 52 words

A century ago, Einstein predicted gravitational waves. He believed nobody
would ever measure them. In 2015, physicists did. Their detector sends laser
light down two arms, each four kilometres long. When the wave passed, the arms
changed length by a thousandth of the width of a proton. That is the signal.

## Goal (0:45–1:05), 39 words

The further away a collision is, the fainter it arrives. A more sensitive
detector reaches deeper into space, in every direction. Double the reach, and
you do not get twice as many events. You get eight times as many.

## Problem (1:05–1:35), 59 words

But how do you build a more sensitive detector? You turn knobs: the length of
the arms, the power of the laser, every mirror. Almost two hundred in all, and
each one affects the others. So a computer improves the design step by step:
simulate, see which way to turn each knob, turn them all a little. Then again.

## Our contribution (1:35–2:15), 93 words

This search is Feyer's work: AI that invents new machines for the physical
world. The heart of every step is a derivative: for every knob, does a small
turn make the detector better or worse? That is the calculation we made
faster. The trick: the simulation has already done most of the work the
derivative needs. We reuse it instead of calculating it again. Now all those
answers together cost a quarter of one simulation.
**The result: a search that took two days now takes five minutes.**
More steps, a more sensitive detector.

## Outlook (2:15–2:50), 67 words

Zoom into the laser beam. It is single particles of light, arriving at random,
like raindrops on a roof. That randomness is quantum physics, and today it
sets the limit for these detectors. So in one machine, the two great theories
of physics meet. Einstein's gravity is the signal. The quantum world is the
noise. A better design turns the noise down, and the wave comes through.

## Close (2:50–3:00), 15 words

We made every step faster. What the search finds is still out there. Thank you.

## Speedup sentence: variants

Swap only the bold sentence. Numbers from `docs/pitch/PITCH.md` on
`origin/main`, as of 22 Nov, 02:39 (commit `e770ba2`).

| Variant | Sentence | Basis |
| --- | --- | --- |
| Two days to five minutes (in the script) | The result: a search that took two days now takes five minutes. | 10,000 steps: 2.0 days against 4.7 minutes. Calculated from the measured time per step (17.7 s against 28 ms), not timed as one run. |
| Factor per step | The result: every step is more than five hundred times faster. | Measured: about 631 times per further step. "More than five hundred" also holds for the earlier figure of 514. |
| Scored first run | The result: the scored run is thirty-five times faster. | Measured cold against cold: 1.25 s against 44.2 s; about 34 times on the same machine. Needs a word of explanation for a lay audience. |

Also from the technical pitch: "a quarter of one simulation" is the gradient
at 26 % of a forward run.

## If time gets tight

1. Cut "We reuse it instead of calculating it again." (Our contribution), saves about 4 s.
2. Cut "Then again." (Problem), saves about 1 s.
3. Cut "like raindrops on a roof" (Outlook), saves about 2 s.

With all three the pitch is at about 2:47 at the slow pace.

Do not cut the silence at the start or the two short sentences
"Einstein's gravity is the signal. The quantum world is the noise."

## Numbers the audience should keep

A thousandth of the width of a proton, eight times as many events, two days,
five minutes. The others (a billion years, a century, 2015, four kilometres,
almost two hundred, a quarter) set the scene and need not be remembered.
