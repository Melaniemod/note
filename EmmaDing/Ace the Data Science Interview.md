[TOC]


## Make Your Data Science Resume Stand Out 2023

Select the relevant blue points, and then adding details to those bully points to best suit the job description. Selecting the relevant points, and adding details to these points will make the resume suit the job description better.



Another thing we can do is, we can also simplify the technical skills section. These are the technical skills in the original resume. What we can do is, we can select the skills that are most relevant to the job description, and we can organize the skills in a way that is aligned with the requirement of the position.






## Top 5 Statistics Concepts in Data Science Interviews: P-value, Confidence Interval, Power, Errors


### Explaining to a Technical Audience

When you explain a concept to a technical audience, you can structure your answer in **four steps**:

1. **Context of use**

   * Start by explaining *where or when* the terminology is used.
   * Example: “This term is often used in statistical hypothesis testing or experiment design.”

2. **Definition**

   * Provide a clear, concise definition.
   * Avoid jargon-heavy or “Wikipedia-style” phrasing.

3. **Interpretation of values**

   * If the concept is represented by numbers, explain what a **larger** or **smaller** value means.
   * Example: “A higher power means the test is more likely to detect a true effect.”

4. **Practical relevance (optional)**

   * Describe how the concept is applied in practice.
   * Example: “That’s why statistical power is crucial in A/B testing to determine the right sample size.”

---

### Explaining to a Non-Technical Audience

For a non-technical audience, the approach needs to be **simpler and more intuitive**:

1. **Use plain language**

   * Avoid introducing new technical terms (e.g., don’t mention ‘null hypothesis’ or ‘alternative hypothesis’).

2. **Examples and analogies**

   * Use real-world examples to make the concept relatable.
   * Example: Explaining **power** using a medical test analogy (detecting whether a person is sick or not).

3. **Focus on intuition, not formulas**

   * Explain what the concept *means in practice*, not the math behind it.
   * Example: “Power tells us how good a test is at finding something when it really exists.”


top 5 统计问题



statistical power is used in a binary hypothesis test. It is the probability that a test correctly rejects the null hypothesis when the alternative hypothesis is true. To put it in another way, statistical power is a likelihood that a test will detect an effect when effect is present. The higher the statistical power, the better the test is. It is commonly used in experiment design to calculate the minimum sample size required, so that one can reasonably detect an effect.

The next terminology is type 1 error. Type 1 error, also known as false positive, is used to categorize errors in a binary hypothesis test. It occurs when we mistakenly reject a true null hypothesis. It means that we conclude our findings are significant, when in fact they have occurred by chance. The larger the value, the less reliable a test is, meaning that we want to minimize the type 1 error of a test. Type 1 error is commonly used in A/B testing to show that we observe differences between two groups, but in reality there's no difference.

The third one is type 2 error. Type 2 error, also known as false negative, is used to categorize errors in binary hypothesis test. Type 2 error refers to false negative. It occurs when we fail to reject a null hypothesis which is in fact false. Basically, we conclude there is not a significant effect, when actually there really is. The larger the value, the less reliable the test results, meaning we want to minimize the type 2 error of the test. It is commonly used in A/B testing to show that we don't observe differences between two groups, but in reality there is a difference.




The first scenario is that the person is indeed infected by the virus, and the test result shows us the same. That is the power of a test. Basically, it is the chance that the test result tells us a person is infected when he truly is.

The second scenario is that the person is not infected, but the test result shows he is. That is a type 1 error. This can be really bad, because the person may take some medical treatment that is completely unnecessary.

The third scenario is that a person is indeed infected, but the test result tells us he's not. This is a type 2 error. It is also very bad, because the person may miss the best timing to get treatment that he really needs.





Confidence interval is used when we want to get an idea of what variable a sample result might be. The confidence interval is for the true value, but we never know what the true value is. The purpose of having samples and observations is to estimate the true value.

The confidence interval is a range of numbers. It tells us how often it would contain the true value, and the probability of it covering the true value is a confidence level. A commonly used value is 95 percent. The wider the interval, the more uncertain we are about the sample result. So, the more confidence we want to be and the less data we have, the wider we make the confidence interval to be enough confident of capturing the true value. In short, the higher the level of confidence, the wider the interval, and the less the sample, the wider the interval.

Okay, that's how we can explain confidence interval during an interview. I want to highlight a common misconception. It considers that the confidence interval answers this question: what is the probability that the true value lies within a certain threshold?

Well, this is not what confidence interval is answering, because the misconception assumes the true value is a variable, and the confidence interval is deterministic. The correct understanding is just the opposite. The true value is determined by nature but is unknown to us. It will not change at all. The things that can change are the boundaries of the confidence intervals, which are estimated from the samples and the level of confidence we set. Basically, for a specific confidence interval, the true value is either a hundred percent within it or not. The 95 percent refers to after the 95 percent confidence intervals computed from many samples, how likely it would cover the true value.








The next terminology is p-value. Similarly, let's explain it to a technical audience first. P-value is commonly used in hypothesis testing to connect the dots between observation and conclusion. It is a conditional probability. It measures the probability of getting testing results at least as extreme as observed results, given that the null hypothesis is true. A low p-value indicates less support for the null hypothesis. In practice, we often choose 0.05 as a cutoff value. P-value less than 0.05 denotes strong evidence against the null hypothesis, which means the null hypothesis can be rejected. And a value larger than 0.05 denotes weak evidence against the null hypothesis, which means the null hypothesis cannot be rejected.

It is commonly used in A/B testing when we have a treatment and a control group, and we want to test whether a metric is different in those two groups. Suppose we have done the experiment and obtained the measurements from the two groups. The smaller the p-value, the more we are convinced there is a difference between the two.

I have just shared with you how to describe p-value during an interview. I want to point out one common mistake people make when interpreting p-value. Very often, we have observations and we would like to prove there is a difference between two groups. The mistake people make is to define p-value as: given the observation, the probability that there is at least such a difference between the two groups.

In other words, they believe p-value captures the probability that the null hypothesis is true, given the data observed. It may sound reasonable at first, but it's almost the opposite of the true meaning of p-value, which is that, given the null hypothesis is true, the probability of obtaining differences at least as large as the data we observed. Now you understand why the misconception of p-value is wrong.

Let's try to explain p-value in layman's terms. We could reuse the example when we explained confidence interval, and that is: we want to get the average height of men in the U.S. We randomly select 30 people and get the measurement of their heights. But now the question is, we want to know if the average value is the same as a fixed value, say 175 centimeters. The p-value connects the dots between what data we observe and what conclusion we could draw.

It tells us that, assume the true value, i.e., the average height, is 175 centimeters, how likely we observe the data. A very small p-value, let's say less than 0.05, means that, assume the true average height is 175, the chance that we observe the data is very low, or the data we observe is very extreme. So, we believe the true value should not be 175 centimeters.






