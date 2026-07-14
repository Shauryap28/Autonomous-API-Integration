"""
selector — derive the target endpoint section from the user's goal.

Closes a real gap: TARGET_ENDPOINT used to be a constant hand-typed after eyeballing
discover.py. Now the system genuinely takes "docs URL + goal" and works out the rest,
with the human as a CHECK rather than a configurator.

How it works:
  • input  = the goal + the list of section names (free — chunking already produced them)
  • one small LLM call: section NAMES only, not the docs. Cheap.
  • output = either a denial ("no endpoint here can do that") or a ranked top-3.
  • the human confirms (or overrides), unless settings.CONFIRM_ENDPOINT is False.

Denying an impossible goal EARLY saves a codegen call and up to MAX_RETRIES sandbox runs.
"""
from pydantic import BaseModel, Field

from backend.config import settings
from backend.llm import complete_json


class Candidate(BaseModel):
    section: str = Field(description="must be EXACTLY one of the provided section names")
    confidence: str = Field(default="medium", description="high | medium | low")
    why: str = Field(default="", description="one short line: why this section fits the goal")


class SelectionResult(BaseModel):
    can_fulfill: bool = Field(description="can ANY of the given sections satisfy the goal?")
    reason: str = Field(default="", description="if can_fulfill is false, explain why")
    candidates: list[Candidate] = Field(
        default_factory=list, description="top 3, best first; empty if can_fulfill is false"
    )


_INSTRUCTION = """You are matching a user's goal to the right section of an API's documentation.

GOAL: {goal}

AVAILABLE DOCUMENTATION SECTIONS (these are the ONLY valid choices):
{sections}

TASK:
- Decide whether ANY of these sections documents an endpoint that could satisfy the goal.
- If none can (e.g. the goal needs an endpoint this API does not document, or requires
  writing/deleting data when only read endpoints exist), set can_fulfill=false and explain
  briefly in `reason`. Do NOT invent a section.
- If one or more can, set can_fulfill=true and return the TOP 3 (or fewer), best first.
  Each `section` MUST be copied EXACTLY from the list above.
- Give each candidate a confidence (high/medium/low) and a one-line reason.
"""


def select_endpoint(goal, section_names):
    """Ask the LLM which section(s) could satisfy the goal. Returns SelectionResult."""
    listing = "\n".join(f"- {name}" for name in section_names)
    prompt = _INSTRUCTION.format(goal=goal, sections=listing)
    return complete_json(prompt, SelectionResult, settings.SELECT_MAX_OUTPUT_TOKENS)


def confirm_endpoint(result, section_names):
    """Show the ranked candidates and let the human accept, choose, or override.

    Returns the chosen section name, or None if the goal cannot be fulfilled / the
    user aborts. Honours settings.CONFIRM_ENDPOINT (False = auto-accept the top pick).
    """
    if not result.can_fulfill or not result.candidates:
        print("\n[selector] No endpoint in these docs can satisfy that goal.")
        print(f"           Reason: {result.reason}")
        print("\n           Available sections:")
        for name in section_names[:15]:
            print(f"             - {name}")
        return None

    top = result.candidates[0].section

    if not settings.CONFIRM_ENDPOINT:
        print(f"[selector] auto-selected: {top}  ({result.candidates[0].why})")
        return top

    print("\n[selector] Candidate endpoints for your goal:")
    for i, c in enumerate(result.candidates, 1):
        print(f"  [{i}] {c.section}   (confidence: {c.confidence})")
        if c.why:
            print(f"      {c.why}")

    choice = input(
        f"\nPress Enter to accept [1], or type 1-{len(result.candidates)}, "
        "or paste an exact section name: "
    ).strip()

    if not choice:
        return top
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(result.candidates):
            return result.candidates[idx - 1].section
        print(f"[selector] {idx} is out of range; using [1].")
        return top
    if choice in section_names:
        return choice

    print(f"[selector] '{choice}' is not a known section; using [1].")
    return top