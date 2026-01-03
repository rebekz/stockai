"""Quiz Module for Learning Assessment."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Question:
    """A quiz question."""
    text: str
    options: list[str]
    correct_index: int
    explanation: str = ""

    def check_answer(self, answer_index: int) -> bool:
        """Check if answer is correct."""
        return answer_index == self.correct_index

    @property
    def correct_answer(self) -> str:
        """Get the correct answer text."""
        return self.options[self.correct_index]


@dataclass
class QuizResult:
    """Result of a quiz attempt."""
    total_questions: int
    correct_answers: int
    wrong_answers: list[tuple[Question, int]]  # (question, user_answer)

    @property
    def score(self) -> float:
        """Score as percentage."""
        return (self.correct_answers / self.total_questions) * 100 if self.total_questions > 0 else 0

    @property
    def passed(self) -> bool:
        """Check if passed (>= 70%)."""
        return self.score >= 70


class Quiz:
    """Interactive quiz for lesson assessment."""

    def __init__(self, lesson_id: str, questions: list[Question]):
        self.lesson_id = lesson_id
        self.questions = questions
        self.current_index = 0
        self.answers: list[int | None] = [None] * len(questions)

    @property
    def current_question(self) -> Question | None:
        """Get current question."""
        if 0 <= self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def answer(self, answer_index: int) -> bool:
        """Submit answer for current question.

        Returns:
            True if correct
        """
        if self.current_question is None:
            return False

        self.answers[self.current_index] = answer_index
        return self.current_question.check_answer(answer_index)

    def next(self) -> bool:
        """Move to next question.

        Returns:
            True if there is a next question
        """
        if self.current_index < len(self.questions) - 1:
            self.current_index += 1
            return True
        return False

    def previous(self) -> bool:
        """Move to previous question.

        Returns:
            True if there is a previous question
        """
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def get_result(self) -> QuizResult:
        """Get quiz result."""
        correct = 0
        wrong = []

        for i, (question, answer) in enumerate(zip(self.questions, self.answers)):
            if answer is not None and question.check_answer(answer):
                correct += 1
            elif answer is not None:
                wrong.append((question, answer))

        return QuizResult(
            total_questions=len(self.questions),
            correct_answers=correct,
            wrong_answers=wrong,
        )

    @property
    def is_complete(self) -> bool:
        """Check if all questions answered."""
        return all(a is not None for a in self.answers)

    @property
    def progress(self) -> tuple[int, int]:
        """Get progress (answered, total)."""
        answered = sum(1 for a in self.answers if a is not None)
        return answered, len(self.questions)


def create_quiz_from_lesson(lesson_data: dict[str, Any]) -> Quiz | None:
    """Create quiz from lesson data.

    Args:
        lesson_data: Lesson dict with quiz_questions field

    Returns:
        Quiz if questions exist, None otherwise
    """
    questions_data = lesson_data.get("quiz_questions", [])
    if not questions_data:
        return None

    questions = [
        Question(
            text=q["question"],
            options=q["options"],
            correct_index=q["correct"],
            explanation=q.get("explanation", ""),
        )
        for q in questions_data
    ]

    return Quiz(lesson_id=lesson_data.get("id", "unknown"), questions=questions)
