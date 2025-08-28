import unittest
from core.environment import AdaptiveAssessmentEnv
from data.questions import QUESTIONS

class TestAdaptiveAssessmentEnv(unittest.TestCase):

    def setUp(self):
        track = "web"
        self.env = AdaptiveAssessmentEnv(QUESTIONS, track)

    def test_initial_state(self):
        self.assertEqual(self.env.student_ability, 0.5)
        self.assertTrue(len(self.env.levels) > 0)

    def test_get_question(self):
        q = self.env.get_question(self.env.current_level)
        self.assertIsNotNone(q)
        self.assertIn("text", q)
        self.assertIn("correct_answer", q)

    def test_submit_answer_correct(self):
        q = self.env.get_question(self.env.current_level)
        reward, done = self.env.submit_answer(q, q['correct_answer'])
        self.assertEqual(reward, 1)
        self.assertFalse(done)

    def test_submit_answer_incorrect(self):
        q = self.env.get_question(self.env.current_level)
        wrong_answer = next(opt for opt in q['options'] if opt != q['correct_answer'])
        reward, done = self.env.submit_answer(q, wrong_answer)
        self.assertEqual(reward, -1)
        self.assertFalse(done)

    def test_done_after_10_questions(self):
        q = self.env.get_question(self.env.current_level)
        for _ in range(10):
            self.env.submit_answer(q, q['correct_answer'])
        self.assertEqual(len(self.env.question_history), 10)
        self.assertTrue(self.env.submit_answer(q, q['correct_answer'])[1])

if __name__ == "__main__":
    unittest.main()
