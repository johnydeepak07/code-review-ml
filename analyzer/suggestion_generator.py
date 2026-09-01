# analyzer/suggestion_generator.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads your .env file and loads OPENAI_API_KEY into os.environ
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


def _build_issues_list(features: dict, complexity: dict) -> list:
    """
    Translate raw metric values into human-readable issue descriptions.
    These become the grounding context for the LLM prompt.
    Only adds an issue if the metric actually exceeds the threshold.
    """
    issues = []

    if complexity['cyclomatic_complexity'] > 10:
        issues.append(
            f"Cyclomatic complexity is {complexity['cyclomatic_complexity']:.1f} "
            f"(target: 5 or below). This means there are "
            f"{int(complexity['cyclomatic_complexity'])} independent paths through "
            f"the code, making it very hard to test fully."
        )

    if features['max_nesting_depth'] > 3:
        issues.append(
            f"Nesting depth is {features['max_nesting_depth']} levels deep "
            f"(target: 3 or below). Deep nesting usually signals that logic "
            f"should be split into helper functions or refactored with early returns."
        )

    if features['naming_entropy'] < 0.4:
        issues.append(
            f"Naming entropy is {features['naming_entropy']:.2f} (target: above 0.6). "
            f"Many variable names appear to be single characters or very short. "
            f"Descriptive names make code self-documenting."
        )

    if not features['has_docstrings'] and features['num_functions'] > 0:
        issues.append(
            f"No docstrings found across {features['num_functions']} function(s). "
            f"Public functions should document their parameters and return values."
        )

    if features['num_magic_numbers'] > 3:
        issues.append(
            f"{features['num_magic_numbers']} magic numbers found. "
            f"Extract numeric literals to named constants at the top of the file."
        )

    if features['avg_function_length'] > 30:
        issues.append(
            f"Average function length is {features['avg_function_length']:.0f} lines "
            f"(target: 20 or below). Long functions typically do too many things."
        )

    return issues


def generate_suggestions(
    code: str,
    features: dict,
    complexity: dict,
    readability_score: float
) -> str:
    """
    Generate targeted refactoring suggestions grounded in AST metrics.
    Only calls the OpenAI API when real issues are identified.
    If no issues, returns a success message with zero API calls made.
    """
    issues = _build_issues_list(features, complexity)

    # No issues found — skip API call entirely, return immediately
    if not issues:
        score_pct = int(readability_score * 100)
        return (
            f"Code quality looks good (readability score: {score_pct}%). "
            f"No critical structural issues found."
        )

    # Build the grounded prompt — LLM only sees issues we already identified
    issues_text = '\n'.join(f'- {issue}' for issue in issues)
    code_preview = code[:600] + ('...' if len(code) > 600 else '')

    prompt = f"""You are a senior software engineer doing a code review.

Static analysis found these specific issues (do NOT invent new issues beyond these):
{issues_text}

Code snippet:
```python
{code_preview}
```

Write exactly 3 refactoring suggestions.
Rules:
- Each suggestion must directly reference one of the issues listed above
- Be specific — name the exact metric and why it matters
- Do NOT rewrite the code for them
- Do NOT give generic advice not grounded in the analysis
- Keep each suggestion to 2 sentences maximum
- Format as a numbered list"""

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=300,
        temperature=0.2
    )

    return response.choices[0].message.content