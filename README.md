# Edvanced
smarter teaching, smarter learning
![EDVANCED](https://github.com/user-attachments/assets/fbfd7a1a-2be1-41cf-9560-449dae002ee8)
# Bubble Sheet Analyzer

A robust, transparent, and explainable cheating detection system for multiple-choice exams using scanned bubble sheets.

## Features

- **Automated Bubble Sheet Processing:** Extracts student answers and IDs from scanned answer sheets using OpenCV, ArUco markers, and OCR.
- **Scoring:** Compares student responses to a model answer sheet and computes scores.
- **Cheating Detection:** Identifies suspiciously similar answer patterns using statistical analysis and clustering (DBSCAN, PCA).
- **Interactive Visualizations:** Displays score distributions, suspicious pairs, and 3D cluster plots for in-depth analysis.
- **User-Friendly Web App:** Built with Streamlit for easy data upload, analysis, and transparent reporting.

## How It Works

1. **Upload** a model answer sheet and multiple student answer sheets (images), you can find test images in the answers directory.
2. **Process** the sheets to extract answers and student IDs.
3. **Analyze** results:
   - View score distributions and individual results.
   - Detect and review statistically significant suspicious answer pairs.
   - Explore clusters of similar answer patterns in 3D.
4. **Export** results as Excel files for further reporting.



## Transparency & Explainability

- All detection methods are based on interpretable statistics and clustering.
- Visualizations and significance tests make results easy to understand and justify.

## Example
```
#simulated test data
from test_data_generator import TestDataGenerator

generator = TestDataGenerator(
    num_students=50,
    num_cheaters=2,
    num_questions=40,
    num_choices=4,
    student_mean_acc=0.7,
    cheater_mean_acc=0.7)

model, students, scores = generator.simulate_answers()

print(f"Model answers shape: {model.shape}")
print(f"Number of students: {len(students)}")
print(f"Scores: {scores}")
```
![image](https://github.com/user-attachments/assets/25a968bc-f2f6-47c4-9cf0-1dd0fd80b9ec)
![image](https://github.com/user-attachments/assets/b0b2e539-a733-414b-9ebd-4306cc241d6b)


## cheaters anomaly detecting

### What This Tool Does

- Detects suspicious pairs by counting identical wrong answers.
- Calculates the statistical significance of that agreement under the assumption of independent answering.
- Uses clustering (DBSCAN + PCA) to detect outlier behaviors visually and numerically.


###  When Should You Use It?
Use this tool when the following assumptions are likely true:

 - Most students are honest , cheaters are the exception, not the norm.
 - Cheaters are greedy , they tend to copy more than 90% of each other's answers.
 - The test isn't too easy , if nearly everyone scores above 95%, it becomes impossible to distinguish cheaters from genuinely good students.
### what to expect
cheating detection is more of a needle in a haystack problem more than it's an anomaly detection problem, and cheating is a very serious accusation, 
so this tool prioritizes minimizing false positives over catching every possible cheater. and in all my simulations both of the methods never agreed on students unless it was the simulated cheater, however, real life is messier and the idealistic assumtions and independence not always holds, people collaborate, guess similarly, or get lucky. So take results with caution, and interpret patterns rather than accuse based on single outcomes.


### Statistical Test
- For each top suspicious pair, we calculate the probability of observing k identical wrong answers by chance, under the null hypothesis that all students answer independently.
- If this probability (the p-value) is less than a chosen threshold (e.g. α = 0.001), we flag it as statistically significant.

### Clustering & Visualization
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) on the students' answer vectors using Hamming distance.
- PCA (Principal Component Analysis) to reduce to 3D and visualize clusters/outliers.
You’ll see who's way off the center of mass. maybe a cheater, maybe by mere chance... maybe just weird. Either way, it's worth looking into.



