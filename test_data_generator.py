import numpy as np
from typing import List, Tuple

class TestDataGenerator:
    """
    A utility class for generating test data for cheating detection systems.
    Generates model answers and simulated student answers, including potential cheaters.
    """
    
    def __init__(self, num_students: int = 20, num_cheaters: int = 2, 
                 num_questions: int = 30, num_choices: int = 4,
                 student_mean_acc: float = 0.7, cheater_mean_acc: float = 0.9):
        self.num_students = num_students
        self.num_cheaters = num_cheaters
        self.num_questions = num_questions
        self.num_choices = num_choices
        self.student_mean_acc = student_mean_acc
        self.cheater_mean_acc = cheater_mean_acc
        
    def generate_student_answer(self, model: np.ndarray, accuracy: float = 0.6) -> np.ndarray:
        """
        Generates a single student's answer sheet based on the model answers.
        """
        answer = np.zeros_like(model)
        for i in range(self.num_questions):
            if np.random.rand() < accuracy:
                correct_choice = np.argmax(model[i])
            else:
                # Wrong answer, choose randomly excluding the correct one
                choices = list(set(range(self.num_choices)) - {np.argmax(model[i])})
                correct_choice = np.random.choice(choices)
            answer[i, correct_choice] = 1
        return answer
    
    def generate_model_answers(self) -> np.ndarray:
        """Generate the model (correct) answer matrix."""
        model = np.zeros((self.num_questions, self.num_choices), dtype=int)
        for i in range(self.num_questions):
            model[i, i % self.num_choices] = 1
        return model
    
    def simulate_answers(self) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Simulates answers for all students (honest and cheaters).
        """
        model = self.generate_model_answers()
        
        # Generate student accuracies
        student_accuracies = np.clip(
            np.random.normal(self.student_mean_acc, 0.1, size=self.num_students), 
            0.3, 0.9
        )
        
        # Generate regular student answers
        students = [
            self.generate_student_answer(model, accuracy=acc) 
            for acc in student_accuracies
        ]
        
        # Create cheaters with higher mean accuracy
        if self.num_cheaters > 0:
            cheater_accuracy = np.clip(
                np.random.normal(self.cheater_mean_acc, 0.05), 
                0.65, 0.85
            )
            cheater_base = self.generate_student_answer(model, accuracy=cheater_accuracy)
            
            for _ in range(self.num_cheaters):
                cheater = cheater_base.copy()
                for j in range(self.num_questions):
                    if np.random.rand() < 0.1:
                        cheater[j] = np.zeros(self.num_choices)
                        cheater[j, np.random.randint(0, self.num_choices)] = 1
                students.append(cheater)
        
        scores = np.array([np.sum(np.all(student == model, axis=1)) for student in students])
        return model, students, scores