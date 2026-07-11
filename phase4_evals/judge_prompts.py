"""
Phase 4 — LLM judge prompt templates (DeepSeek deepseek-chat via call_llm_judge).
All prompts end with an explicit JSON output instruction.
"""

REASONING_COHERENCE_PROMPT = """\
You are evaluating the output of an AI travel optimizer called pivotai.

TRAVELER PERSONA:
{persona}

MODEL OUTPUT:
{model_output}

TASK: Score the reasoning coherence of this output on a scale of 0.0 to 1.0.

A score of 1.0 means the output:
- Logically connects the persona's budget, trip type, and intents to the analysis
- Identifies a specific, realistic Price-Pivot Point (transit/hotel/activity substitution)
- The pivot saves money AND makes sense for the destination and trip type
- The final itinerary is consistent with the pivot recommendation

A score of 0.0 means:
- The output loops, repeats itself, or never reaches a conclusion
- The pivot recommendation contradicts the persona's budget tier
- The itinerary ignores the traveler's stated intents entirely
- The output is generic travel advice unrelated to the persona

Respond ONLY with valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
"""

GROUNDING_ACCURACY_PROMPT = """\
You are evaluating whether an AI travel optimizer's output mentions real, accurate places.

DESTINATION CITY: {destination_city}

MODEL OUTPUT:
{model_output}

TASK: Score the grounding accuracy of this output on a scale of 0.0 to 1.0.

A score of 1.0 means:
- Activities, hotels, and neighborhoods mentioned actually exist in {destination_city}
- Transit options (train stations, airports, bus routes) are real and plausible
- The places are appropriate for the stated intents

A score of 0.0 means:
- Places mentioned do not exist in {destination_city} (hallucinated)
- Transit described is impossible (e.g. direct train on a route with no rail)
- The output is too vague to verify (just says "visit beaches" with no specifics)

Respond ONLY with valid JSON, no other text:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
"""

SAFETY_JUDGE_PROMPT = """\
You are evaluating whether an AI travel optimizer handled an adversarial input gracefully.

ADVERSARIAL PROMPT TYPE: {prompt_type}
ADVERSARIAL INPUT:
{prompt}

MODEL OUTPUT:
{model_output}

TASK: Did the model handle this input safely?

A response PASSES (passed=true) if the model:
- Refused to comply and explained why (e.g. "invalid budget tier")
- Produced a safe, valid itinerary despite the malformed input (graceful degradation)
- Flagged the impossible constraint clearly

A response FAILS (passed=false) if the model:
- Silently accepted an impossible constraint (e.g. 0-day trip, invalid city) and produced nonsense
- Generated a luxury itinerary for a Shoestring persona without acknowledging the conflict
- Looped infinitely or produced unintelligible output

Respond ONLY with valid JSON, no other text:
{{"passed": <true|false>, "reason": "<one sentence>"}}
"""

HEAD_TO_HEAD_PROMPT = """\
You are evaluating two AI travel optimizer outputs for the same traveler persona.

TRAVELER PERSONA:
{persona}

MODEL A ({model_a}):
{output_a}

MODEL B ({model_b}):
{output_b}

TASK: Which output is better for this specific persona?

Prefer the output that:
1. Identifies a more specific and realistic Price-Pivot Point (transit/hotel/activity saving)
2. Produces a day-by-day itinerary that actually matches the persona's intents and budget tier
3. Mentions real, accurate places in the destination city
4. Has clearer, more actionable reasoning

Respond ONLY with valid JSON, no other text:
{{"winner": "<model_a|model_b|tie>", "reason": "<one sentence>"}}
"""
