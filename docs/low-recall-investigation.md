# Investigation
## Formula recap
<q, x> = ApproxScore + Error Correction
ApproxScore is coming from 1st phase search.

ApproxScore = <Q_4bit(q'), Q(x')> + <q, c> + <c, x> - <c, c> where Q_4bit is 4bit quantization, Q is 1bit quantization, c = centroid,
q is query vector, q' = q - centroid, x is data vector, x' = x - centroid.

So ApproxScore itself has error already.
Error of ApproxScore = <q', x'> - <Q_4bit(q'), Q(x')>

In the second phase, now we add error correction factor to adjust the score value.

ApproxScore + <q, Q_4bit(r)> where r is defined as `r = x' - Q(x')`.

## Errors
Therefore, the final fomula would be

Score = <Q_4bit(q'), Q(x')> + <q, c> + <c, x> - <c, c> + <q', Q_4bit(r)>

So we do have 2 errors
1. <q', x'> - <Q_4bit(q'), Q(x')>
2. <q', r> - <q', Q_4bit(r)>

## Actual example
docId=671, phase1Score=145.42168, correctedScore=146.47926, correction=1.0575867, trueMipScore=144.25194, correctedVsTrueGap=2.2273254
So ApproxScore is phase1Score, so the error of ApproxScore is trueMipScore - phase1Score = -1.16974.

# Idea-1 : Use query error in rescoring phase
Score = <Q_4bit(q'), Q(x')> + <q, c> + <c, x> - <c, c> + <q', Q_4bit(r)>
      = ApproxScore + <q', Q_4bit(r)>

<Q_4bit(q'), Q(x')> is an approximation of <q', x'>.

if we could rewrite <q', x'> to <r1 + Q_4bit(q'), Q_4bit(r) + Q(x')>
where r1 = q - Q_4bit(q'), r = x' - Q(x') which is stored in storage.

Then, final score would be look like below:

Score = <r1, Q_4bit(r) + Q(x')> + <Q_4bit(q'), Q_4bit(r)> + <Q_4bit(q'), Q(x')> + centroid_terms
      = <r1, Q_4bit(r) + Q(x')> + <Q_4bit(q'), Q_4bit(r)> + ApproxScore
