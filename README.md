# Julia_PSO_Binary_classification
Real money label 1, fake money label 2. Inspired by bird, using PSO algorithm to do binary classification

Datasets can be downloaded from my other project named from Julia_XXX_XXX

Based on the idea from PSO, I follow the following steps to complete this project:

1. Generate 100 models and select the one with the highest accuracy as gbest
2. Randomly initialize the array v[i]
3. Formula: v[j]=w*v[j]+c1*Math.random()*(pbest[j]-x[j])+c2*Math.random()*(gbest-x[j ])
Use this formula to change the weight of each model：# weight = weight + v[j] (up to the maximum value)
If it is greater than vmax, take VMAX.
4. Cross-boundary judgment [-1,1], if it is greater than the maximum value, take the maximum boundary value. If it is less than the minimum value, the minimum boundary value is taken.
5. After each generation, redefine gbest until the accuracy reaches 95%

v[i] is the initial moving direction, gbest is the global optimal solution, pbest is the current optimal solution

![image](https://user-images.githubusercontent.com/100655843/182497771-e884baf3-adf3-4bc0-b7d3-06868ff55872.png)
