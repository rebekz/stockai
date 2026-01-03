"""Briefing Module.

Morning and evening briefings for the 15-minute daily workflow.
Designed for passive investors who want quick actionable insights.
"""

from stockai.briefing.daily import (
    generate_morning_briefing,
    generate_evening_briefing,
    MorningBriefing,
    EveningBriefing,
)
from stockai.briefing.weekly import (
    generate_weekly_review,
    WeeklyReview,
)

__all__ = [
    "generate_morning_briefing",
    "generate_evening_briefing",
    "MorningBriefing",
    "EveningBriefing",
    "generate_weekly_review",
    "WeeklyReview",
]
