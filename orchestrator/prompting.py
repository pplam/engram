"""Prompt rendering (§2.9).

Prompts are content-pinned data written for a model, not Python format strings: the
judge prompt legitimately contains JSON braces. `str.format` would treat those as
syntax and raise, so placeholders are substituted literally and every other brace
is left exactly as the pinned file has it.
"""


class PromptError(Exception):
    """A prompt is missing a placeholder that the stage needs to fill."""


def render(template: str, **values: str) -> str:
    """Substitute `{name}` placeholders literally, leaving all other braces untouched."""
    rendered = template
    for name, value in values.items():
        placeholder = "{" + name + "}"
        if placeholder not in rendered:
            raise PromptError(f"prompt has no {placeholder} placeholder to fill")
        rendered = rendered.replace(placeholder, value)
    return rendered
