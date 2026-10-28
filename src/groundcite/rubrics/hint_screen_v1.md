You are screening one reply from an automotive diagnostic assistant before it is used as a training
example. You do not grade whether the diagnosis is right, whether the citation is correct or
whether the reply follows any format. You answer one question: does the wording of the reply give
away that its writer was working from privileged information that the user never saw?

## Background

The user sees only the conversation and the ID-tagged reference snippets shown in the input. The
writer of the reply may also have seen hidden instructions: an answer key saying whether to cite
or abstain, which snippet was the intended or designated one, notes saying which snippets are
decoys, or a grading rubric. A good reply never lets any of that show. It justifies its step only
from what the user said and what the provided snippets say.

Ask yourself: would every phrase in the reply make sense to a user who has read only this
conversation and these snippets? If a phrase only makes sense because someone behind the scenes
picked an answer, it is a leak.

## Flag the reply ("meta_leak": true) when it

- refers to hidden instructions, a brief, a briefing, notes, guidance, a setup or a task it
  received, or says it was told, given, instructed or asked something by anyone other than the
  user in this conversation;
- refers to an answer key, a rubric, a grader, ground truth, a gold or reference answer, or a label
  for the case;
- calls a snippet, citation, answer or mode correct, right, expected, intended, designated,
  target, preselected, flagged, marked or chosen, as if someone had already decided it, rather
  than giving a reason from the complaint and the snippet text;
- calls other snippets decoys, distractors or traps, or says they were set up not to apply;
- speaks of the case as having a predetermined answer or mode ("the answer is the bulletin", "the
  right mode here is abstain", "cite mode is right here");
- attributes knowledge the user did not provide to a hidden source: it says it was told, briefed or
  shown something, rather than reasoning from the complaint or a snippet ("I'm told the bulletin
  applies", "my notes point to the bulletin").

## Do not flag ("meta_leak": false) ordinary diagnostic wording

- Naming, describing or quoting a provided snippet: "the TSB lists your engine", "per the wiring
  diagram section", "the procedure in the manual says to test at idle", "follow the manual's
  instructions for the leak-down test".
- Reasoning from what the user reported: "from what you describe", "since your car is a 2014".
- Everyday words that only sound similar: "a hint of a burning smell", "the gold terminal on the
  relay", "the port labeled VAC", "note the reading", "the key-on voltage".
- Declining to name a fix, saying none of the provided references applies, or asking the user for
  one more detail.
- Details that are wrong, unsupported or missing from the conversation and snippets: a vehicle
  detail the user never stated, a symptom nobody mentioned, a step no snippet describes. Whether a
  reply is grounded and correct is graded elsewhere. An unsupported claim is a leak only when the
  reply credits it to a hidden source or treats the answer as decided in advance.

Judge only the reply under "Reply to screen". The conversation and snippets are there so you can
tell what the user actually provided; do not flag a reply just because it says something they do
not contain. When a phrase plausibly refers to privileged information, flag it: a false flag costs
one training example, while a missed leak teaches the model to talk about inputs it should never
mention.

## Output

Answer with only one JSON object and nothing else:

{"meta_leak": true or false, "reason": "one short sentence; when true, quote the words that leak"}
