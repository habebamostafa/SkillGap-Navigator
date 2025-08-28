# enhanced_agent.py
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

class RLAssessmentAgent:
    def __init__(self, env, learning_rate: float = 0.1, discount_factor: float = 0.9):
        """
        Enhanced RL agent for adaptive assessment
        
        Args:
            env: The adaptive assessment environment
            learning_rate: Learning rate for Q-table updates
            discount_factor: Discount factor for future rewards
        """
        self.env = env
        self.alpha = learning_rate
        self.gamma = discount_factor
        
        # Q-table for state-action values
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Initialize Q-values for each level and action
        for level in [1, 2, 3]:
            for action in ['up', 'stay', 'down', 'auto']:
                self.q_table[level][action] = 0.0
        
        # Exploration parameters
        self.epsilon = 0.3  # Initial exploration rate
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05
        
        # Performance tracking
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        
        # Advanced features
        self.performance_window = deque(maxlen=5)  # Recent performance tracking
        self.adaptation_threshold = 0.7  # Threshold for adaptation decisions
        self.stability_counter = 0  # Track how long at current level
        
    def get_state(self) -> Dict:
        """
        Get current state representation
        
        Returns:
            State dictionary with relevant features
        """
        return {
            'current_level': self.env.current_level,
            'student_ability': self.env.student_ability,
            'consecutive_correct': self.env.consecutive_correct,
            'consecutive_incorrect': self.env.consecutive_incorrect,
            'confidence_score': self.env.confidence_score,
            'total_questions': self.env.total_questions_asked,
            'recent_performance': self._get_recent_performance()
        }
    
    def _get_recent_performance(self) -> float:
        """Get recent performance score (last 3 questions)"""
        if len(self.env.performance_history) < 3:
            return 0.5  # Neutral if insufficient data
        
        recent = self.env.performance_history[-3:]
        return sum(p['is_correct'] for p in recent) / len(recent)
    
    def choose_action(self, state: Dict = None) -> str:
        """
        Choose action using enhanced strategy combining Q-learning and heuristics
        
        Args:
            state: Current state (if None, will get from environment)
            
        Returns:
            Action string: 'up', 'down', 'stay', or 'auto'
        """
        if state is None:
            state = self.get_state()
        
        current_level = state['current_level']
        student_ability = state['student_ability']
        consecutive_correct = state['consecutive_correct']
        consecutive_incorrect = state['consecutive_incorrect']
        confidence = state['confidence_score']
        recent_performance = state['recent_performance']
        
        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            action = self._explore_action(state)
        else:
            action = self._exploit_action(state)
        
        # Record action
        self.action_history.append({
            'action': action,
            'state': state.copy(),
            'timestamp': self.env.total_questions_asked
        })
        
        return action
    
    def _explore_action(self, state: Dict) -> str:
        """Exploration strategy with informed randomness"""
        current_level = state['current_level']
        student_ability = state['student_ability']
        
        # Weight exploration based on uncertainty
        uncertainty = 1.0 - state['confidence_score']
        
        if uncertainty > 0.5:
            # High uncertainty: more random exploration
            actions = ['up', 'down', 'stay', 'auto']
            weights = [0.25, 0.25, 0.25, 0.25]
        else:
            # Lower uncertainty: biased exploration
            if student_ability > 0.7:
                actions = ['up', 'stay', 'auto']
                weights = [0.4, 0.3, 0.3]
            elif student_ability < 0.3:
                actions = ['down', 'stay', 'auto']
                weights = [0.4, 0.3, 0.3]
            else:
                actions = ['up', 'down', 'stay', 'auto']
                weights = [0.3, 0.3, 0.2, 0.2]
        
        return np.random.choice(actions, p=weights)
    
    def _exploit_action(self, state: Dict) -> str:
        """Exploitation strategy using Q-values and heuristics"""
        current_level = state['current_level']
        
        # Get Q-values for current state
        q_values = self.q_table[current_level]
        
        # Combine Q-values with heuristic scoring
        action_scores = {}
        for action in ['up', 'down', 'stay', 'auto']:
            q_score = q_values[action]
            heuristic_score = self._calculate_heuristic_score(action, state)
            
            # Weighted combination
            action_scores[action] = 0.6 * q_score + 0.4 * heuristic_score
        
        # Choose action with highest combined score
        best_action = max(action_scores, key=action_scores.get)
        return best_action
    
    def _calculate_heuristic_score(self, action: str, state: Dict) -> float:
        """Calculate heuristic score for an action given the current state"""
        current_level = state['current_level']
        student_ability = state['student_ability']
        consecutive_correct = state['consecutive_correct']
        consecutive_incorrect = state['consecutive_incorrect']
        confidence = state['confidence_score']
        recent_performance = state['recent_performance']
        
        score = 0.0
        
        if action == 'up':
            # Reward going up if student is performing well
            if consecutive_correct >= 2:
                score += 0.8
            if recent_performance >= 0.7:
                score += 0.6
            if student_ability > 0.6 and current_level < 3:
                score += 0.5
            if current_level == 1 and student_ability > 0.4:
                score += 0.4
            
            # Penalize going up if struggling
            if consecutive_incorrect >= 2:
                score -= 0.8
            if recent_performance < 0.4:
                score -= 0.6
            if current_level == 3:
                score -= 0.3  # Already at max level
                
        elif action == 'down':
            # Reward going down if student is struggling
            if consecutive_incorrect >= 2:
                score += 0.8
            if recent_performance <= 0.3:
                score += 0.6
            if student_ability < 0.4 and current_level > 1:
                score += 0.5
            if current_level == 3 and student_ability < 0.6:
                score += 0.4
            
            # Penalize going down if performing well
            if consecutive_correct >= 2:
                score -= 0.8
            if recent_performance > 0.6:
                score -= 0.6
            if current_level == 1:
                score -= 0.3  # Already at min level
                
        elif action == 'stay':
            # Reward staying if performance is appropriate for level
            if 0.4 <= recent_performance <= 0.7:
                score += 0.7
            if confidence > 0.6:
                score += 0.3
            
            # Slight penalty for staying too long
            if self.stability_counter > 3:
                score -= 0.2
                
        elif action == 'auto':
            # Auto is generally safe but less precise
            score += 0.3
            if confidence < 0.5:
                score += 0.4  # More valuable when uncertain
        
        return score
    
    def update_q_table(self, state: Dict, action: str, reward: float, next_state: Dict):
        """
        Update Q-table using Q-learning algorithm
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state after action
        """
        current_level = state['current_level']
        next_level = next_state['current_level']
        
        # Current Q-value
        current_q = self.q_table[current_level][action]
        
        # Best Q-value for next state
        next_q_values = self.q_table[next_level]
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[current_level][action] = new_q
        
        # Record reward
        self.reward_history.append(reward)
    
    def adjust_difficulty(self, action: str):
        """
        Execute difficulty adjustment action
        
        Args:
            action: Action to execute ('up', 'down', 'stay', 'auto')
        """
        previous_level = self.env.current_level
        
        if action == 'auto':
            # Let environment handle automatic adjustment
            self.env.adjust_difficulty('auto')
        else:
            # Manual adjustment
            self.env.adjust_difficulty(action)
        
        # Update stability counter
        if self.env.current_level == previous_level:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_performance_metrics(self) -> Dict:
        """Get agent performance metrics"""
        if not self.reward_history:
            return {"error": "No performance data available"}
        
        recent_rewards = self.reward_history[-10:] if len(self.reward_history) >= 10 else self.reward_history
        
        # Calculate action distribution
        action_counts = defaultdict(int)
        for action_record in self.action_history:
            action_counts[action_record['action']] += 1
        
        total_actions = len(self.action_history)
        action_distribution = {
            action: count / total_actions 
            for action, count in action_counts.items()
        } if total_actions > 0 else {}
        
        return {
            'total_actions': len(self.action_history),
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'recent_average_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'current_epsilon': self.epsilon,
            'action_distribution': action_distribution,
            'q_table_size': sum(len(actions) for actions in self.q_table.values()),
            'stability_counter': self.stability_counter
        }
    
    def get_q_table_summary(self) -> Dict:
        """Get summary of Q-table values"""
        summary = {}
        for level, actions in self.q_table.items():
            summary[f'level_{level}'] = dict(actions)
        return summary
    
    def recommend_next_action(self, state: Dict = None) -> Dict:
        """
        Get recommendation for next action with explanation
        
        Args:
            state: Current state (optional)
            
        Returns:
            Dictionary with recommendation and reasoning
        """
        if state is None:
            state = self.get_state()
        
        # Get action scores
        current_level = state['current_level']
        action_scores = {}
        
        for action in ['up', 'down', 'stay', 'auto']:
            q_score = self.q_table[current_level][action]
            heuristic_score = self._calculate_heuristic_score(action, state)
            combined_score = 0.6 * q_score + 0.4 * heuristic_score
            
            action_scores[action] = {
                'q_score': q_score,
                'heuristic_score': heuristic_score,
                'combined_score': combined_score
            }
        
        # Find best action
        best_action = max(action_scores, key=lambda x: action_scores[x]['combined_score'])
        
        # Generate explanation
        explanation = self._explain_recommendation(best_action, state, action_scores)
        
        return {
            'recommended_action': best_action,
            'confidence': state['confidence_score'],
            'explanation': explanation,
            'action_scores': action_scores,
            'current_state': state
        }
    
    def _explain_recommendation(self, action: str, state: Dict, scores: Dict) -> str:
        """Generate human-readable explanation for action recommendation"""
        consecutive_correct = state['consecutive_correct']
        consecutive_incorrect = state['consecutive_incorrect']
        recent_performance = state['recent_performance']
        student_ability = state['student_ability']
        current_level = state['current_level']
        
        explanations = {
            'up': [],
            'down': [],
            'stay': [],
            'auto': []
        }
        
        if consecutive_correct >= 2:
            explanations['up'].append(f"Student answered {consecutive_correct} questions correctly in a row")
        
        if consecutive_incorrect >= 2:
            explanations['down'].append(f"Student struggled with {consecutive_incorrect} consecutive questions")
        
        if 0.4 <= recent_performance <= 0.7:
            explanations['stay'].append(f"Recent performance ({recent_performance:.1%}) suggests appropriate difficulty")
        
        if recent_performance > 0.7:
            explanations['up'].append(f"Strong recent performance ({recent_performance:.1%})")
        
        if recent_performance < 0.3:
            explanations['down'].append(f"Poor recent performance ({recent_performance:.1%})")
        
        if student_ability > 0.7 and current_level < 3:
            explanations['up'].append(f"High estimated ability ({student_ability:.1%}) suggests readiness for harder questions")
        
        if student_ability < 0.3 and current_level > 1:
            explanations['down'].append(f"Low estimated ability ({student_ability:.1%}) suggests need for easier questions")
        
        if state['confidence_score'] < 0.5:
            explanations['auto'].append("Low confidence in assessment suggests automatic adjustment")
        
        # Return explanation for recommended action
        if explanations[action]:
            return '; '.join(explanations[action])
        else:
            return f"Based on Q-learning algorithm (score: {scores[action]['combined_score']:.2f})"
    
    def save_model(self, filename: str = None):
        """Save the trained Q-table and agent parameters"""
        import pickle
        import time
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"rl_agent_model_{timestamp}.pkl"
        
        model_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'action_history': self.action_history,
            'reward_history': self.reward_history,
            'performance_metrics': self.get_performance_metrics()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        return filename
    
    def load_model(self, filename: str):
        """Load a previously trained model"""
        import pickle
        
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore Q-table
            self.q_table = defaultdict(lambda: defaultdict(float))
            for level, actions in model_data['q_table'].items():
                for action, value in actions.items():
                    self.q_table[level][action] = value
            
            # Restore parameters
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.alpha = model_data.get('alpha', self.alpha)
            self.gamma = model_data.get('gamma', self.gamma)
            
            # Restore history if available
            self.action_history = model_data.get('action_history', [])
            self.reward_history = model_data.get('reward_history', [])
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def reset_agent(self):
        """Reset agent state for new assessment session"""
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        self.performance_window.clear()
        self.stability_counter = 0
        
        # Reset exploration rate
        self.epsilon = 0.3
        
        # Optionally reset Q-table (uncomment if needed)
        # self.q_table = defaultdict(lambda: defaultdict(float))


# Additional utility classes

class AdaptiveStrategy:
    """Strategy pattern for different adaptation approaches"""
    
    @staticmethod
    def conservative_strategy(state: Dict) -> str:
        """Conservative adaptation - small changes"""
        if state['consecutive_correct'] >= 3:
            return 'up'
        elif state['consecutive_incorrect'] >= 3:
            return 'down'
        else:
            return 'stay'
    
    @staticmethod
    def aggressive_strategy(state: Dict) -> str:
        """Aggressive adaptation - quick changes"""
        if state['consecutive_correct'] >= 2:
            return 'up'
        elif state['consecutive_incorrect'] >= 2:
            return 'down'
        else:
            return 'stay'
    
    @staticmethod
    def ability_based_strategy(state: Dict) -> str:
        """Adaptation based on estimated ability"""
        ability = state['student_ability']
        current_level = state['current_level']
        
        target_level = 1 if ability < 0.33 else (2 if ability < 0.67 else 3)
        
        if target_level > current_level:
            return 'up'
        elif target_level < current_level:
            return 'down'
        else:
            return 'stay'

