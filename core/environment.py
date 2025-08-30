# enhanced_environment.py
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from data.questions import get_adaptive_question, generate_interview_questions, _question_manager

class AdaptiveAssessmentEnv:
    def __init__(self, max_questions=None, track: str = "web"):
        """
        Initialize the enhanced adaptive assessment environment
        
        Args:
            questions: Legacy questions (for backward compatibility)
            track: Technology track to focus on
        """
        self.track = track.lower()
        self.generator = _question_manager.generator
        
        # Validate track
        available_tracks = self.generator.get_available_tracks()
        if self.track not in available_tracks:
            print(f"Warning: Track '{self.track}' not found. Using 'web' instead.")
            print(f"Available tracks: {', '.join(available_tracks)}")
            self.track = "web"
        
        # Environment state
        self.levels = [1, 2, 3]  # Easy, Medium, Hard
        self.current_level = 2   # Start at medium
        self.student_ability = 0.5  # Initial ability estimate
        
        # Question management
        self.question_history = []
        self.used_questions = set()
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        self.total_questions_asked = 0
        self.max_questions = 10  # Default max questions per session
        
        # Performance tracking
        self.performance_history = []
        self.level_changes = []
        self.confidence_score = 0.5
        
        # Adaptive parameters
        self.ability_update_rate = 0.1
        self.confidence_threshold = 0.8
        self.min_questions_per_level = 2

    def reset(self, track: str = None):
        """Reset the environment for a new assessment"""
        if track:
            self.track = track.lower()
        
        self.current_level = 2
        self.student_ability = 0.5
        self.question_history = []
        self.used_questions = set()
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        self.total_questions_asked = 0
        self.performance_history = []
        self.level_changes = []
        self.confidence_score = 0.5
    
    def get_question(self, level: int = None) -> Optional[Dict]:
        """
        Get an adaptive question for the current or specified level
        
        Args:
            level: Specific level to get question from (optional)
            
        Returns:
            Question dictionary or None if no questions available
        """
        target_level = level if level is not None else self.current_level
        
        # Get question while avoiding duplicates
        question = get_adaptive_question(
            track=self.track,
            level=target_level,
            used_questions=list(self.used_questions)
        )
        
        if question:
            self.used_questions.add(question['text'])
            self.total_questions_asked += 1
            
            # Add metadata
            question['level'] = target_level
            question['question_number'] = self.total_questions_asked
            question['student_ability_at_time'] = self.student_ability
            
            return question
        
        # If no questions available at current level, try adjacent levels
        for alternative_level in [target_level - 1, target_level + 1]:
            if 1 <= alternative_level <= 3:
                question = get_adaptive_question(
                    track=self.track,
                    level=alternative_level,
                    used_questions=list(self.used_questions)
                )
                if question:
                    self.used_questions.add(question['text'])
                    self.total_questions_asked += 1
                    question['level'] = alternative_level
                    question['question_number'] = self.total_questions_asked
                    question['student_ability_at_time'] = self.student_ability
                    return question
        
        # Last resort: generate a custom question
        try:
            custom_question = self.generator.generate_custom_question(
                track=self.track,
                level=target_level,
                topic=f"{self.track} fundamentals"
            )
            if custom_question:
                self.used_questions.add(custom_question['text'])
                self.total_questions_asked += 1
                custom_question['level'] = target_level
                custom_question['question_number'] = self.total_questions_asked
                custom_question['student_ability_at_time'] = self.student_ability
                return custom_question
        except Exception as e:
            print(f"Failed to generate custom question: {e}")
        
        return None

    def submit_answer(self, question: Dict, answer: str) -> Tuple[float, bool]:
        """
        Process student's answer and update environment state
        
        Args:
            question: The question that was answered
            answer: Student's selected answer
            
        Returns:
            Tuple of (reward, is_done)
        """
        is_correct = question['correct_answer'] == answer
        
        # Calculate base reward
        base_reward = 1.0 if is_correct else -1.0
        
        # Level-based reward adjustment
        level_multiplier = question.get('level', self.current_level) / 2.0
        if is_correct:
            reward = base_reward * (1.0 + level_multiplier * 0.5)
        else:
            reward = base_reward * (1.0 + (3 - question.get('level', self.current_level)) * 0.2)
        
        # Update consecutive counters
        if is_correct:
            self.consecutive_correct += 1
            self.consecutive_incorrect = 0
        else:
            self.consecutive_incorrect += 1
            self.consecutive_correct = 0
        
        # Update student ability using IRT-inspired model
        self._update_student_ability(question, is_correct)
        
        # Update confidence score
        self._update_confidence_score(is_correct)
        
        # Record question history
        question_record = {
            'question': question,
            'answer': answer,
            'is_correct': is_correct,
            'level': question.get('level', self.current_level),
            'reward': reward,
            'student_ability_after': self.student_ability,
            'confidence_score': self.confidence_score,
            'timestamp': self.total_questions_asked
        }
        
        self.question_history.append(question_record)
        self.performance_history.append({
            'question_number': self.total_questions_asked,
            'is_correct': is_correct,
            'level': question.get('level', self.current_level),
            'ability': self.student_ability,
            'confidence': self.confidence_score
        })
        
        # Check if assessment is complete
        is_done = self._check_completion()
        
        return reward, is_done
    
    def _update_student_ability(self, question: Dict, is_correct: bool):
        """Update student ability using adaptive algorithm"""
        question_level = question.get('level', self.current_level)
        question_difficulty = question_level / 3.0  # Normalize to 0-1
        
        # Expected probability of correct answer
        expected_prob = self._sigmoid(self.student_ability - question_difficulty)
        
        # Actual result (1 if correct, 0 if incorrect)
        actual_result = 1.0 if is_correct else 0.0
        
        # Update ability based on difference between expected and actual
        error = actual_result - expected_prob
        
        # Adaptive learning rate based on confidence
        learning_rate = self.ability_update_rate * (2.0 - self.confidence_score)
        
        # Update ability
        self.student_ability += learning_rate * error
        
        # Clamp ability to valid range
        self.student_ability = max(0.0, min(1.0, self.student_ability))
    
    def _update_confidence_score(self, is_correct: bool):
        """Update confidence in ability estimate"""
        # Increase confidence with more questions
        base_confidence_increase = 0.05
        
        # Adjust based on consistency
        if len(self.performance_history) >= 3:
            recent_performance = [p['is_correct'] for p in self.performance_history[-3:]]
            consistency = len(set(recent_performance)) == 1  # All same result
            
            if consistency:
                self.confidence_score += base_confidence_increase * 2
            else:
                self.confidence_score += base_confidence_increase
        else:
            self.confidence_score += base_confidence_increase
        
        # Clamp confidence
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for probability calculations"""
        return 1.0 / (1.0 + np.exp(-x * 5))  # Scale factor for steeper curve
    
    def _check_completion(self) -> bool:
        """Check if the assessment should be completed"""
        # Minimum questions reached
        if self.total_questions_asked < 5:
            return False
        
        # Maximum questions reached
        if self.total_questions_asked >= self.max_questions:
            return True
        
        # High confidence in ability estimate
        if (self.confidence_score >= self.confidence_threshold and 
            self.total_questions_asked >= 7):
            return True
        
        # Consistent performance at appropriate level
        if len(self.performance_history) >= 6:
            recent_performance = self.performance_history[-4:]
            same_level_questions = [p for p in recent_performance 
                                  if abs(p['level'] - self._get_target_level()) <= 1]
            
            if (len(same_level_questions) >= 3 and 
                all(p['is_correct'] == same_level_questions[0]['is_correct'] 
                    for p in same_level_questions)):
                return True
        
        return False
    
    def _get_target_level(self) -> int:
        """Get the target difficulty level based on student ability"""
        if self.student_ability < 0.33:
            return 1  # Easy
        elif self.student_ability < 0.67:
            return 2  # Medium
        else:
            return 3  # Hard
    
    def adjust_difficulty(self, action: str = "auto"):
        """
        Adjust difficulty level based on action or automatically
        
        Args:
            action: 'up', 'down', 'stay', or 'auto' for automatic adjustment
        """
        old_level = self.current_level
        
        if action == "auto":
            # Automatic difficulty adjustment
            target_level = self._get_target_level()
            
            # Gradual adjustment to avoid large jumps
            if target_level > self.current_level:
                if self.consecutive_correct >= 2:
                    self.current_level = min(3, self.current_level + 1)
            elif target_level < self.current_level:
                if self.consecutive_incorrect >= 2:
                    self.current_level = max(1, self.current_level - 1)
        
        elif action == "up":
            if self.current_level < 3:
                self.current_level += 1
        elif action == "down":
            if self.current_level > 1:
                self.current_level -= 1
        # "stay" or invalid action: no change
        
        if old_level != self.current_level:
            self.level_changes.append({
                'from_level': old_level,
                'to_level': self.current_level,
                'question_number': self.total_questions_asked,
                'reason': action,
                'student_ability': self.student_ability
            })
    
    def get_assessment_summary(self) -> Dict:
        """Get comprehensive assessment summary"""
        if not self.question_history:
            return {"error": "No questions answered yet"}
        
        correct_answers = sum(1 for q in self.question_history if q['is_correct'])
        total_questions = len(self.question_history)
        
        # Performance by level
        level_performance = {}
        for level in [1, 2, 3]:
            level_questions = [q for q in self.question_history if q['level'] == level]
            if level_questions:
                level_correct = sum(1 for q in level_questions if q['is_correct'])
                level_performance[level] = {
                    'questions': len(level_questions),
                    'correct': level_correct,
                    'accuracy': level_correct / len(level_questions)
                }
        
        # Ability progression
        ability_progression = [p['ability'] for p in self.performance_history]
        
        return {
            'track': self.track,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'final_score': correct_answers / total_questions if total_questions > 0 else 0,
            'final_ability': self.student_ability,
            'confidence_score': self.confidence_score,
            'level_performance': level_performance,
            'level_changes': len(self.level_changes),
            'ability_progression': ability_progression,
            'consecutive_correct': self.consecutive_correct,
            'consecutive_incorrect': self.consecutive_incorrect,
            'recommended_level': self._get_target_level()
        }
    
    def get_next_question_recommendation(self) -> Dict:
        """Get recommendation for next question difficulty"""
        target_level = self._get_target_level()
        
        # Consider recent performance
        if len(self.performance_history) >= 3:
            recent_accuracy = sum(p['is_correct'] for p in self.performance_history[-3:]) / 3
            
            if recent_accuracy >= 0.8:
                suggested_level = min(3, target_level + 1)
                reason = "Strong recent performance"
            elif recent_accuracy <= 0.3:
                suggested_level = max(1, target_level - 1)
                reason = "Struggling with current level"
            else:
                suggested_level = target_level
                reason = "Appropriate challenge level"
        else:
            suggested_level = target_level
            reason = "Based on estimated ability"
        
        return {
            'suggested_level': suggested_level,
            'current_level': self.current_level,
            'target_level': target_level,
            'reason': reason,
            'confidence': self.confidence_score
        }

    def export_session_data(self, filename: str = None) -> str:
        """Export complete session data for analysis"""
        import json
        import time
        
        if not filename:
            timestamp = int(time.time())
            filename = f"assessment_session_{self.track}_{timestamp}.json"
        
        session_data = {
            'metadata': {
                'track': self.track,
                'timestamp': timestamp,
                'total_questions': self.total_questions_asked,
                'session_complete': self._check_completion()
            },
            'summary': self.get_assessment_summary(),
            'question_history': self.question_history,
            'performance_history': self.performance_history,
            'level_changes': self.level_changes
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)
        
        return filename
