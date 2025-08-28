import unittest
import random
from core.agent import RLAssessmentAgent
from core.environment import AdaptiveAssessmentEnv
from data.questions import QUESTIONS

class TestRLAssessmentAgent(unittest.TestCase):

    def setUp(self):
        track = "web"
        self.env = AdaptiveAssessmentEnv(QUESTIONS, track)
        self.agent = RLAssessmentAgent(self.env)

    def test_q_table_initialized(self):
        self.assertIn(self.env.current_level, self.agent.q_table)
        self.assertIn('up', self.agent.q_table[self.env.current_level])

    def test_choose_action_returns_valid(self):
        action = self.agent.choose_action(self.env.current_level)
        self.assertIn(action, ['up', 'stay', 'down'])

    def test_update_q_table(self):
        old_value = self.agent.q_table[self.env.current_level]['stay']
        self.agent.update_q_table(
            state=self.env.current_level,
            action='stay',
            reward=1,
            next_state=self.env.current_level
        )
        new_value = self.agent.q_table[self.env.current_level]['stay']
        self.assertNotEqual(old_value, new_value)

    def test_adjust_difficulty_up(self):
        initial_level = self.env.current_level
        self.agent.adjust_difficulty('up')
        self.assertGreaterEqual(self.env.current_level, initial_level)

    def test_adjust_difficulty_down(self):
        self.env.current_level = self.env.levels[-1]
        self.agent.adjust_difficulty('down')
        self.assertLessEqual(self.env.current_level, self.env.levels[-1])

if __name__ == "__main__":
    unittest.main()
