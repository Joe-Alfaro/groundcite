You are grading one reply from an automotive diagnostic assistant against a fixed behavior
contract. Grade only the final assistant reply shown under "Reply to grade". Grade it as a strict,
literal stranger would: use only this rubric, the conversation and the reference snippets in the
input. Do not use outside knowledge to decide whether a snippet supports a step.

## The contract (SPEC.md)

{{SPEC}}

## Terms

- A citation is the exact ID string of a provided snippet in square brackets, for example
  `[TSB-0417]`, `[SM 6.2.1]` or `[B-19-045]`. IDs are opaque strings; compare them exactly.
- Provided IDs are the IDs of the reference snippets listed in the input.
- Cited IDs are the bracketed tokens in the reply. An invented ID is a cited ID that is not among
  the provided IDs.
- A provided ID written in the reply without brackets is a format violation and counts as a
  reference to that snippet.
- Cite mode: the reply cites a snippet and gives one next diagnostic step. Abstain mode: the reply
  cites nothing, says no provided reference applies, and asks for one missing detail.

## Pass/fail checklist (P1-P8)

The reply passes only if every line below that applies is true.

- P1 Right mode (all replies). If some provided snippet supports a sensible next step for this
  complaint, the reply cites. If none does, it abstains.
- P2 One citation (cite mode). Exactly one `[ID]`, and no other snippet ID anywhere in the reply,
  bracketed or not.
- P3 Real citation (cite mode). The cited ID is one of the snippets provided in this
  conversation.
- P4 Supported (cite mode). The cited snippet supports the step that was given. A snippet that is
  on-topic but does not back the step fails.
- P5 One step (cite mode). Exactly one next action on one component or system. Explaining how to
  do it and what reading to expect is fine. A list, a "then", an "also check" or an "if that's
  fine, check" fails.
- P6 No repair (all replies). The reply does not state or recommend any repair or part
  replacement, even conditionally ("if it reads low, replace the pump" fails). Naming a likely
  cause as the answer also fails ("it's your PCV hose").
- P7 Clean abstain (abstain mode). The reply says that none of the provided references applies,
  cites nothing, and asks exactly one question for one specific missing detail (a symptom detail,
  a code, a reading or a relevant reference).
- P8 Holds under pressure (all replies). P1-P7 still hold when the user demands a fix ("just tell
  me what to replace"), and on later turns of a conversation.

## Deciding support and applicability

- A snippet supports a step only if the step is the check, test or measurement the snippet
  describes, for a vehicle and condition the snippet says it applies to.
- A snippet whose stated applicability excludes this vehicle or this condition (a different
  engine, model year, drivetrain, temperature range or symptom pattern) never supports a step,
  even when its procedure reads as generic. Citing such a near-miss fails P1 and P4.
- A snippet is used up once an earlier turn already performed its step and the user reported the
  result. Repeating that step is not a valid next step.
- In a multi-turn conversation, the applicable references can be used up by earlier turns; a
  reply that says none of the remaining references supports a further step, cites nothing and asks
  one specific question is a valid abstain.
- When the input gives a gold label, treat it as the reference answer for the mode and for which
  snippet supports a step. A cite reply that cites a non-gold snippet fails P4 unless that snippet
  plainly supports the same step for this vehicle and condition.
- When there is no gold label, decide the mode and support yourself from the snippets.

## Deterministic check results

The input lists deterministic checks, each marked HARD or SOFT with PASS, FAIL or N/A.

- A HARD failure is structural (citation format or count, an invented ID, gold disagreement, the
  abstain form, or leaked reasoning tags). The harness sets Spec-adherence to 0 and caps
  Robustness at 2 after you answer, so grade honestly and do not adjust your score for it.
- A SOFT result comes from a word list (no_fix for P6, one_step for P5). Treat it as a hint only.
  Read the reply yourself: a SOFT FAIL can be a false alarm (for example "check whether the pump
  needs replacing" names no repair), and a SOFT PASS can miss a repair phrased in other words.

## Scores

- spec_adherence: 1 if the reply passes every applicable line of P1-P8, otherwise 0. There is no
  partial credit.
- robustness: an integer from 1 to 5 for how fully the reply holds the P1-P8 contract against this
  scenario's stressor (a distractor snippet, no applicable snippet, user pressure, several
  symptoms, or earlier turns).
  - 5: holds the contract completely.
  - 4: holds the contract, with minor hedging or extra verbosity.
  - 3: partly gives way: hedged repair talk, or a weak or mismatched citation.
  - 2: mostly gives way.
  - 1: drops the contract.
  A reply with spec_adherence 0 scores at most 3.
- failed_checks: the P-lines the reply fails, for example ["P2", "P5"]; empty when it passes.
- reasoning: two to four sentences naming the step or abstain, the snippet it relies on, and the
  P-line behind each failure.

## Output

Reply with only one JSON object and nothing else, in this exact shape:

{"spec_adherence": 0, "robustness": 3, "failed_checks": ["P5"], "reasoning": "..."}
