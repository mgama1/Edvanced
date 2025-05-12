# Edvanced
smarter teaching, smarter learning
![EDVANCED](https://github.com/user-attachments/assets/fbfd7a1a-2be1-41cf-9560-449dae002ee8)



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



