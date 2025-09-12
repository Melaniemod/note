[TOC]



## Ace A/B Testing Interview Question: A Data-driven Approach for Data Scientists


### Top A/B Testing Interview Questions
Now let's dive in. Here's a diagram showing the distribution of top A/B testing interview questions.
I created this diagram after analyzing over 350 interview questions from 46 companies.

Let's look at top five A/B testing interview questions:

1. **The design question** is the top A/B testing interview question. It accounts for over 30% of interview questions. It means one out of three questions is the design question.

   * The design question is a broad question. It asks you to design a test to evaluate the effectiveness of a new feature or the success of a new product.
   * Example: *How do you design a test to evaluate the effectiveness of a referral program?*

2. **Sample size and test duration.** This type of question is straightforward. It asks you to calculate the sample size and the duration of a test.

3. **Causal inference.** These questions can also appear in the interview because it's a method that can be considered when A/B testing is not feasible. This type of question is more common in companies that employ a two-sided or three-sided marketplace business model, such as Uber, Lyft, and DoorDash.

4. **Multiple testing.** These problems occur frequently in reality — when we run multiple tests in parallel or when we test multiple metrics in one A/B test.

5. **Launch decision.** Once we finish running the test and we have gathered the data, we need to make the launch decision: Should we launch this feature or the product or not?

   * Example: *Based on the data from this test, what is your suggestion — launch this feature or not?*

---

### The Design Question
Now you know the top five A/B testing interview questions, and in this video we are going to focus on the **design question**.

As we talked about earlier, the design question accounts for over 30% of A/B testing interview questions. That's one out of three, so it's a pretty big percentage, right?

A typical format of this type of question is:
*A company is considering designing a new feature or changing something in the product. How do you design a test to evaluate the effectiveness of this change?*

The goal is to make sure this change is beneficial for the users or the business.

* Example: Uber is considering launching a new referral program to get new riders. *How do you design a test to evaluate the effectiveness of this program?*
* Example: TikTok is thinking about launching this new repost feature that allows users to amplify their videos by sending those videos to their followers. *How do you design a test to evaluate the effectiveness of this new feature?*

This sounds to be a very broad question, isn’t it?

Designing a test involves multiple steps such as selecting the right metric, choosing the right randomization unit, calculating the sample size and test duration, determining the minimum detectable effect, et cetera.


### Refresher

Now, before we dive into the strategy to handle this type of question, if you need refreshers on different components that involve designing a test, I have this playlist that covers topics such as choosing randomization units, metric selection, sample size estimation, and common mistakes in A/B testing.

So feel free to watch those videos to refresh your memory on how to design a test.

---

### How to Answer Design Questions
Now, going back to our question on how to design a test to test a new change, my tip for you is to **provide a summary before you dive into the details** of designing the test.

What do I mean by that?

Well, my suggestion is you can provide a summary at the beginning before getting into the details. You can say something like:

> Designing an A/B test involves multiple steps such as choosing the right metrics, selecting the right randomization unit, calculating the sample size and test duration, as well as determining the minimum detectable effect. So which component would you like me to dive into?

So the idea is that you provide a high-level summary about different components that are involved in designing a test before talking about the details of each component.

---



## Randomization Units in A/B Testing: Easy Explanation for Data Science Interviews




### What are randomization units
First of all, what is a randomization unit? The term may sound a bit intimidating, but a randomization unit (AKA unit of diversion) is who or what we randomly assign to each variant or group of an A/B test.

Selecting the right randomization unit is critical because it impacts both user experience and what metrics we can use in an A/B test. You may think that randomization units are simply users because in many experiments we refer to the randomization process as assigning users to each group. But this description is not accurate. It's actually a bit more complicated than it initially appears, and it will become clearer when we talk about different options of randomization units.

---

### What are the commonly used randomization units
Let's start our discussion by defining the commonly used randomization units in practice.

User ID is one of the most common randomization units. It's fairly easy to see why, right? Using the user ID ensures a consistent user experience and allows for long-term measurements such as user retention and users’ learning effects.

But there are other randomization units suitable for different scenarios in a test. Here are some examples: a cookie, an event, a device ID, or even an IP address can also be used as a randomization unit. Each of these options has some pros and cons. Let's look at them one by one.

### User ID
As we mentioned earlier, user IDs are stable across time and platforms. Once the person registers on a website, there will be a unique user ID attached to their account, which is a huge pro of using a user ID as a randomization unit.

But one thing to note is that a user ID can be used to reveal a person's identity. Consequently, we need to be mindful of confidentiality and security issues when using user ID for identification purposes. Another issue to note is that the identification of registered users generally requires the user to log into their accounts, which is the limitation of user ID as the randomization unit.

### Cookies
Unlike user IDs, cookies are pseudonymous IDs that are specific to a browser and a device. If you don't know what web browser cookies are: they are small footprints that are automatically assigned and stored in a browser when you log onto a website.

Because cookies are anonymous, they are not identifiable. But the users can clear cookies, and cookies do not persist across browsers or platforms. For example, if you change your web browser from Chrome to Firefox, the cookie will change. Cookies can also expire, so you can think of them as temporary user IDs.

### Events
Another kind of randomization unit is an event. An event such as a page view or session represents a finer level of granularity than a user ID. In other words, one user can be connected to many page views or sessions.

Using page-level randomization, every page visit is considered a randomization unit. This randomization method is simple because it does not require users to log on nor does it distinguish between actions from the same or different users.

Session-level randomization can also be used. Usually, a session is defined as a continuous period of activities. For instance, you might log onto your Facebook account, check out your friends’ activities, and then close your browser. Typically, the session will expire after 30 minutes of inactivity—that's one session. The next day, if you log onto your Facebook account again, it's considered a new session.

Session-level randomization treats every session as an independent occurrence. Accordingly, one user can be assigned to different variants from time to time. Because there are typically more page views and sessions than users, using an event as a randomization unit provides more units and gives us more power to detect smaller changes. However, it may also lead to inconsistent user experience. So if the change is visible to users, it's better not to use an event as a randomization unit.




### Device ID
Another commonly used randomization unit is a device ID. It is an immutable ID associated with a specific device. Device IDs are only available for mobile devices, so they are most commonly used as randomization units for A/B testing of changes in mobile apps.

---

### Considerations when choosing randomization units
Now we know some commonly used randomization units, but how do we choose which one to use in a given scenario?

The first consideration is ensuring a consistent user experience. You don't want users to experience one design today and another design tomorrow because it will likely make users very frustrated. Therefore, for changes that are visible to users—like changes in color, button sizes, and the layout of a page or other major design changes on a website—we should use a user ID or cookie as a randomization unit.

Conversely, if the change is not visible or easily noticed by users, then the selection of a randomization unit depends on what we want to measure. Examples include website performance changes and changes in backend algorithms. For instance, if we want to compare the latency of loading web pages, then we might use a page view as a randomization unit.

We can also consider session-level or page-level randomization. Page-level randomization may be more powerful than user-level randomization because it can reduce the variance in page-level metrics. But if we want users to see what happens and track their experience over time, a user ID is also a good option.

Another related consideration is a comparison of the coarseness of the randomization unit and the unit of analysis. As we have mentioned earlier, the unit of analysis is the unit of your metric. The general recommendation is that the randomization unit should be at least as coarse as the unit of analysis.

Let's consider a few examples to make this recommendation clear. For example, let's say we want to analyze click-through rate, in which case the unit of analysis is a page view. If we want the randomization unit to be a user, let's ask ourselves whether a page view or user is a coarser or bigger unit. Because many page views can be connected to a user ID, the user ID is a coarser unit. So we have met the conditions of this recommendation.

Notice, however, that in this case we may have an issue with the previous consideration, which is variability. This example demonstrates the importance of all considerations in designing the randomization unit.

Now suppose our randomization unit is a page view and we analyze user-level metrics such as average number of clicks per user. Here, a page view is less coarse than the number of clicks per user, and the user's experience is likely to include a mix of variants—some pages in the control group and some pages in the treatment groups. That will make computing user-level metrics meaningless.

---

### Summary

In this series of videos, we talked about the top three A/B testing questions in data science interviews: sample size estimation, metric selection, and choosing a randomization unit. Understanding these questions and answers is important and is a good starting point, but you need more than that for data science interviews.


## Metric Selection in A/B Testing: Easy Explanation for Data Science Interviews


Metric selection is one of the most common A/B testing questions in data science interviews. Interviewers often ask questions that require interviewees to select metrics for the purpose of A/B testing. For example: *What metrics would you use in an experiment to understand a feature change?*

Given the importance of metric selection in the context of A/B testing, this video will discuss this topic in depth and focus on the following things:

* What are driver metrics?
* What are the attributes of driver metrics?
* How to develop and select driver metrics?

I've used an A/B test example from YouTube to talk about how to select metrics. Let's dive in.

---

### What are driver metrics?

Driver metrics, also known as surrogate metrics, are indirect or predictive metrics that are often used to measure short-term objectives. These metrics align with the goal of the company, are sensitive to short-term progress, and are actionable so that teams can be driven to work towards them.

In a nutshell, driver metrics are the major metrics used for A/B testing.

Let's consider a simple A/B test of an ad campaign. The difference between the ad in the control and treatment group is its design. The goal of the ad campaign is to increase total revenue from sales of items.

But given the goal of the campaign, how do we select which metrics are best for the situation? Conversely, which metrics are bad?

There are three overall criteria we can use to evaluate driver metrics.

---

### What are the attributes of driver metrics?

First and most importantly, a driver metric should be **sensitive and timely**.

In particular, a driver metric should be sensitive enough to reflect the change made in the product. The click rate is an ideal driver metric for this example because once we change the design of the ad, the change will be reflected in the click-through rate.

We can also look at the conversion rate. In this case, it's the percentage of users who take the desired action, such as making a purchase, which is also an indicator that one ad is better than the other.

On the other hand, the daily active user of a particular product being advertised might not be a good driver metric, because it can take time for people to purchase the product, start using it, and become a daily active user. Daily active user may be better suited as a **success metric** than a driver metric, because it can be impacted by multiple factors and may not be sensitive enough to the change in the ad.

This is not to say that bad driver metrics are unimportant for business — they are. But such metrics are not appropriate for running A/B tests because they are not sensitive enough to measure the treatment effect.

Driver metrics should also be **measurable**, meaning we should be able to calculate the metric with the data collected during the experiment period. Typically, most experiments are designed to run within a time frame of days or weeks, so the metric we select should be suitable for such time frames.

Using the example of the ad campaign, the click rate is very simple and easy to measure, because the counts can be easily obtained even in real time. The measure is simply the number of clicks divided by the number of impressions. Conversely, metrics like monthly active users or retention rates are bad ideas for driver metrics, because they cannot be calculated within the time frame of the test.

Finally, a driver metric should be **attributable**. In other words, we must be able to attribute the change in the metric to the experiment variant. This requires us to be able to measure the metrics in the control and the treatment groups separately.

Using the ad example, we can easily attribute the change in click rate to the design of the ads: good design results in higher click-through rate, while bad design results in lower click-through rate.

These are the three attributes of good driver metrics.

---




### How to select driver metrics?

Now let's talk about how to come up with driver metrics in practice.

There are many ways to come up with ideas for metrics and validate existing metrics. Metrics can also be developed by combining **qualitative** and **quantitative** methods.

Qualitative methods include techniques such as user experience research, focus groups, and surveys to understand users' needs. Quantitative methods include analysis of data such as logs to see what users do and to find patterns in the data.

But very often, we don't have time to leverage all those methods to come up with metrics for an A/B test — especially during interviews, where time is very limited.

So what can we do? We can try to:

1. **Understand the motivation of an A/B test and define metrics specifically for measuring the changes.**
2. **Analyze user experiences.**

---

#### Fully understand the goal of a test

One way that I found helpful to come up with metrics is to fully understand the goal of a test.

Is the goal about user growth? Is it to improve engagement? Is it to increase revenue? Is it about acquisition, activation, retention, referral, or revenue? We want to be as specific as possible to fully understand the goal. Doing this will help you come up with a metric or two.

Let's look at an example. In 2021, YouTube ran an experiment to test hiding dislike counts on videos.

According to the company, the goal was to better protect creators from harassment, help ensure small creators and those just getting started can thrive, and create an inclusive and respectful environment where creators have the opportunity to succeed and feel safe to express themselves.

So what metric can we think of based on that goal? Assuming this feature change will help YouTube achieve this goal, we would expect to see small creators become more engaged on the platform, post more videos, and spend more time on the platform.

Two possible driver metrics for the goal are:

* The average time spent on YouTube per creator.
* The average number of videos published per creator.

Ideally, both metrics will be larger in the treatment group than in the control group.

---

#### Analyze user experience

Other than focusing on the goal of the test, another way to come up with metrics is to analyze user experiences.

Consider the steps users in each group need to take to use a feature or product, and think about some metrics to measure the difference between user experiences. Typically, most products or features have a funnel that moves users towards taking key actions or desired outcomes that are meaningful to the business. Changing user experiences may positively influence more users to get the desired outcome.

For the YouTube example, what is the desired outcome? The desired outcome is fewer dislikes on videos on smaller channels.

In the experiment, YouTube assigned viewers to different groups. The control group could still see the dislike count for a video. In the treatment group, viewers could still dislike a video to share feedback with creators, but they were not able to see the counts.

Given the difference between the control and the treatment groups, as well as the desired outcome, what metrics should we use for the experiment?

One metric is the **average number of dislikes per viewer**, which will indicate if a user gives more or fewer dislikes.

Also, given that the goal of this experiment is to protect smaller creators, we can measure the **average number of dislikes for smaller creators**. Ideally, we will see a decrease in this number.

Here are some experimental findings shared by YouTube:

* Viewers were less likely to target a video's dislike button to drive up the count.
* Experiment data showed a reduction in dislike-attacking behavior.
* Smaller creators reported being unfairly targeted by dislike attacks, and data confirmed this behavior occurs at a higher proportion on smaller channels.

Although these findings do not mention the exact metric YouTube used, we can tell that it was closely related to the company's original goal of protecting smaller creators.

---

### Summary

To summarize, we have talked about the two main strategies for developing driver metrics:

1. By considering the overall business goal.
2. By analyzing the difference in user experiences.




## Sample Size Estimation in A/B Testing: Easy Explanation for Data Science Interviews




### How A/B Testing Questions Appear in Interviews

A/B testing is not a stand-alone interview. It's rare to see any company have an A/B testing interview round. But if the component appears in data science interviews, especially product case interviews, it means that A/B testing questions are often asked together with product case or product sense questions.

For example: *What are some pros and cons of YouTube removing dislikes of videos? And how do you design a test to evaluate the effectiveness of this change?*

As you can tell, it requires not only product knowledge but also A/B testing knowledge to answer this kind of question. It's hard to provide an insightful and in-depth answer if you don't know much about A/B testing.

Now that you know where A/B testing questions occur in data science interviews, let's go over one of the top A/B testing questions that I have seen in interviews. And that is: **to use power analysis to estimate the sample size needed for a test.**

---

### Rule of Thumb Formula
One last thing before diving into today’s topic: the questions in this series of videos require you to have some basic knowledge of A/B testing. If you are new to A/B testing, that's totally fine. I have a playlist for you that you can start from — I have shared the link in the video description.

Alright guys, let's get started.

Let's first review the general form of sample size estimation, which is:

$$
n = \frac{2\sigma^2 (z_{\alpha/2} + z_{\beta})^2}{\delta^2}
$$

Where:

* $\sigma^2$ is an estimate of variance,
* $\alpha$ is a significance level (also the same as Type I error or false positive rate),
* $z_{\alpha/2}$ is a z-score such that the area to the right of $z_{\alpha/2}$ is $\alpha/2$ under the standard normal curve,
* $\beta$ is a Type II error (false negative rate, the same as $1 - \text{power}$),
* $z_\beta$ is the z-score such that the area to the right of $z_\beta$ is $\beta$,
* $\delta$ is the difference between control and treatment.

This is a general form of sample size estimation. It's rare that you will get questions on how to derive this equation during an interview, but if you are interested in learning about the derivation, I have a link to a chapter of a book in the video description and it has details on the derivation.

Even though it's not required to know exactly how to derive the equation, it's helpful to understand how we obtain each component and how each component plays a part in estimating the sample size.

---

### Alpha (Significance Level)
When $\alpha = 0.05$ and $\beta = 0.2$, we can get the rule-of-thumb formula:

$$
n = 16 \cdot \frac{\sigma^2}{\delta^2}
$$

If we want to be more conservative and lower the significance level $\alpha$, we can set $\alpha$ to be a smaller value. For instance, when $\alpha = 0.01$, $z_{\alpha/2}$ becomes 2.23, which is larger than 1.96 when $\alpha = 0.05$.

Our coefficient increases from 16 to almost 19. So with decreasing $\alpha$, we need more samples.

Intuitively this makes sense: if we want to decrease Type I error $\alpha$, we increase our confidence level of the estimation, and we need more samples. As our sample size increases, the more information we have, our uncertainty decreases, and we have greater confidence in our estimation.

---

### Beta (Type II Error)
Now let's look at how $\beta$ impacts sample size.

$\beta$ is Type II error. It's equal to $1 - \text{power}$. So increasing $\beta$ means decreasing power, and vice versa. Power is often set at 0.8 in practice, which means $\beta = 0.2$.

Let's say we want to increase power to 0.9, then $\beta$ becomes 0.1. In this case, $z_\beta = 1.28$, which is larger than 0.84 when $\beta = 0.2$. And the coefficient increases from 16 to 21.

It means that increasing our sample size can give us greater power to detect differences. The smaller $\beta$, the greater power, and the more samples we need.

---

### Variance
Variance estimation should be done before running the experiment. It can be obtained from historical data.

Generally speaking, companies should have historical data such as system logs or user behavior  data for data scientists to query to estimate variance. For companies that have done some A/B tests before, we can estimate the variance from previous A/B tests.

If no such data is available, we can run an A/A test (a smaller experiment) to get an idea of the distribution of the data when there’s no treatment. And we should continue to improve estimations of variance with more data and experiments for future tests.

---

### Delta

Finally, let's take a look at $\Delta$, the difference between control and treatment.

How do we know $\delta$ before running the test? There's a reason to run the test to begin with — to know the difference between control and treatment.

The idea is to use the **Minimum Detectable Effect (MDE)**, also known as practical significance. That is the minimum change that makes sense for the business.

For instance: a \$10,000 increase in revenue, or 10,000 more button clicks.

Just because those values are noisy and can be impacted by many factors, we need a minimum practical difference in order to conclude there is a meaningful impact to the business.

As you can tell, when $\delta$ becomes smaller, the resulting sample size will become larger. It means when we want to detect smaller changes, we will need more samples.

---

### Summary
So to summarize:

* The lower the $\alpha$, the higher the confidence level, and the more samples we need.
* The lower the $\beta$, the greater the power, and the more samples we need.
* The larger the variance, the more samples needed to run the experiment.
* The smaller the variance, the fewer samples needed.
* When we want to detect smaller changes ($\delta$ smaller), we will need more samples.

---



## A/B Testing Mistakes to Avoid in Your Data Science Interview: Tips and Tricks!





I have categorized these mistakes into two groups. The first group of mistakes is about misunderstanding or lack of understanding of statistical concepts. The second group of mistakes is to assume the results are valid and reliable without considering factors that can make the results useless.

---

### Data Scientists' Roles

When working with other professionals in an organization, there are chances that others have some misunderstanding of certain stats concepts, or they misinterpret the results of online experiments. So as a data scientist, it's our responsibility to educate others and help them understand how to interpret the data correctly.

---

### Data Peeking

Let's look at a scenario. You are the data scientist leading an A/B test. After running the experiment for some time, a PM came to you and asked:

> The metric has shifted to the positive direction, and now the p-value is below 0.05. Could we stop the experiment and claim it a success?

What would you do? Well, this is a typical example of data peeking. Basically, we stop collecting data when the test comes out significant. While this is a tempting idea to stop the experiment earlier and draw a conclusion quicker, the result won't be reliable.

Let me elaborate on this. When we calculate the duration of the experiment in the experiment design phase, we have to consider statistical power, significance level, and a few other factors such as day-of-week effect, seasonality, etc.

If we stop an experiment earlier than designed, the data collected is not complete. There will be a high chance that the result is inaccurate — inaccurate in the sense that the estimated treatment effect is very different from the true treatment effect.

Also remember, the goal of A/B testing is to make sure the result is reproducible. Meaning, if we see a 0.1 percent increase in conversion rate in the experiment, we'd like to see the same amount of increase when launching the product.

But if we stop the data collection prematurely and use that incomplete data to evaluate treatment effect, the result won't be reproducible when we launch the change to production.

So instead of stopping the experiment earlier when we see the p-value is below the designed significance level, we should use the pre-determined experiment duration and run the experiment as long as designed.

Rather than picking p-value, there's a more generalized data peeking mistake people make from time to time.

---

### Multiple Testing Problem

Let's think about another scenario. Let's say you are the data scientist in charge of an experiment with five different target metrics. After finishing running the experiment, an engineer came to you and asked:

> Now the p-value of one metric is below 0.05. Could we ship the new feature to production?

We should not ship the feature based on the change of one metric only. And this is a typical multiple testing problem. It refers to any instance that involves the simultaneous testing of more than one hypothesis.

In this example, we have five hypotheses. Since we are testing five metrics, if decisions about the individual hypotheses are based on the unadjusted significance level, then there is a large probability that a type I error will occur. In other words, the null hypothesis is true, but we choose to reject it.

Multiple testing problems are also a kind of data peeking, and it's very common in practice. Remember that it happens when we test multiple hypotheses at the same time.

Here I have summarized a few scenarios it may occur:


* First, when we look at the multiple metrics in a single A/B test.
* Also, when testing one metric in an A/B test with multiple treatment groups (more than two variants).
* In addition, looking at a segment of the population (such as location, platform, or device type) can also lead to a multiple testing problem.
* Also, when we have multiple iterations of an A/B test, or when we run multiple A/B tests in parallel.

Back to the original question: when there are multiple metrics and we choose the one with the lowest p-value and then claim that metric has significantly shifted, our estimate is likely to be biased. In other words, it's very likely it's a false positive.

So it will be a bad decision to ship the new feature to production.

How do we deal with it then? A practical way, as recommended in the book *Trustworthy Online Controlled Experiments*, is a two-step rule of thumb to control type I and type II errors.

1. **Step one:** separate all metrics into three groups:

   * Metrics we expect to be impacted.
   * Metrics potentially impacted.
   * Metrics unlikely to be impacted.

2. **Step two:** apply different significance levels to each group.

Essentially, we want to set different thresholds for each group of metrics based on the change we expect to see.

#### Example:

* Metrics A (expected to change) → α = 0.05
* Metrics B, C (uncertain) → α = 0.01
* Metric D (should not change) → α = 0.001

If you observe metric A is significant while the others are not, then great. The changes are exactly what you expected.

But if metric A is not significant, while metrics B, C, or D are significant, then you need to debug why metrics that should not change have changed significantly.

---

### Lack of Statistical Power

Another mistake on misunderstanding stats concepts is to claim there's no treatment effect when a metric is not showing statistical significance.

An example would be: the change in conversion rate is not statistically significant, and we claim that there's no treatment effect at all.

This statement is only correct when the test has sufficient statistical power. In other words, there are enough randomization units to detect a change.

But it's possible that the test is underpowered to detect the effect size we are seeing. That is, there are not enough randomization units in the test.

For example: if we need 1000 users in each group to achieve 80% statistical power, but now we only have 900 users after running the experiment for the designed duration. This may happen because the real number of users is less than estimated (e.g., fewer checkouts than expected).

So if we draw a conclusion based on these 900 users, and the target metric is not significant, it would be wrong to claim there’s no treatment effect at all.

What we should do: either continue running the experiment if possible, or re-run the experiment with enough users to detect the expected change. Only then will the experiment have enough statistical power and the data will be reliable.

---


## A/B Testing Mistakes and Solutions: How to Excel in Your Data Science Interview!






In this video, we will dive into the other category of mistakes, which is to assume the results are valid and reliable without doing sanity checks, and make launch decisions based on invalid results. In reality, there are so many factors that can make the results unreliable.
In this video, we will cover the three most common issues. Let's get started.

---

### Sample Ratio Mismatch
One common issue that makes the results unreliable is sample ratio mismatch. It refers to the instance that the sample ratio between control and treatment is not as designed.

For example, if the experiment designed for ratio between the sample size of control and that of treatment is one to one, after running the experiment you observe the ratio is 1.01. In other words, the control group has more users than the treatment group.

Then you use a t-test to test if the number of users in the control group is different from that in the treatment group, and you obtain a p-value less than 0.05. You now realize that it's highly unlikely to observe such a ratio or a more extreme condition under the design ratio.

There are many causes of sample ratio mismatch. One is simply bugs or problems in assigning users to different groups. While randomly assigning users based on the user ID may sound simple and straightforward, achieving proper randomization can be very challenging in reality.

For example, many experiments have ramping-up plans to make sure there's no risk when exposing users to a new feature. The ramping plan typically starts with assigning only a small percentage of users in the treatment group, then gradually rolls out to more users. That could complicate the assignment of users.

Things also get complicated when running multiple experiments in parallel, and one user might be assigned to multiple experiments. Bugs and errors are more likely to occur than when running one experiment at a time.

Another potential issue is if we are looking at a particular segment of users, and if the segmentation is based on some attributes that can change over time. For example, location: people may move to a different geographic location, so it will result in bias in allocating users to different groups.

For example, you want to run experiments with the target population in the San Francisco Bay Area, so you assign users to the treatment group because his profile shows that he's in San Francisco. However, he has already moved to a different location, but he hasn't updated his profile yet. During the experiment, he updated his profile. So when you get the results and do the analysis, you need to filter the users who are not in the target region.

Besides all these things, it's also possible that the way the results are processed leads to sample ratio mismatch. For example, we have a data pipeline to filter out fraudulent users before analyzing the test results, and there is a bug in the pipeline that causes the false positive rate to be different in different groups.

False positive means that we wrongly flag legitimate users as fraudsters. Say the false positive rate in the treatment group is higher — this will cause a sample ratio mismatch.

---

### Debug Sample Ratio Mismatch
We have covered a few factors that may lead to sample ratio mismatch. Next, let's go over how to debug it if we observe such an issue.

I'm gonna summarize the recommendations for debugging from the book *Trustworthy Online Controlled Experiments*, because it's pretty comprehensive and practical.

We can start with checking if there's any discrepancy upstream of the randomization point. An example would be if our target users are those who enter checkout, we want to see if there's any difference between control and treatment for users upstream of entering checkout.

In other words, before entering checkout, we can check if there's a gap between groups for users landing on the home page, for users who put items in a shopping cart, and all these steps before users start the checkout process.

Another thing to check is if the variant assignment is done correctly. Are users allocated to different variants properly? If we use user ID to assign users, is the assignment truly random? Will any bias be introduced in this step?

For example, if we find one group has many females and the other group has many males, then the assignment is unbalanced and neither group represents the overall population. So the result is likely to be inaccurate.

We could also look into the data processing pipeline. A common source of sample ratio mismatch is bot detection and filtering. If there's any bug introduced in filtering bot traffic, it will potentially cause sample ratio mismatch.

To further debug the issue, we can check different segments of the population. For example, we could look into the ratios per day and see which days have anomalies. We can also segment by other dimensions. For example, are the ratios different for new users versus returning users?

Just to recap: the first factor that makes testing results unreliable is sample ratio mismatch. We have talked about different ways to debug it.

---


### Violation of SUTVA
Next, let's go over another common problem that invalidates testing results, and that is randomization units interfering with one another.

There's one important assumption of A/B testing, which is that randomization units are independent and there's no interaction between them. This is called the *Stable Unit Treatment Value Assumption* (SUTVA).

If there are interferences or spillovers between different groups, then the result is definitely not reliable. The estimated treatment effect could either underestimate or overestimate the true treatment effect.

In reality, it's common that this assumption is violated. For example, in social networks such as Facebook and LinkedIn, users' behaviors tend to be impacted by others in their social circles. If a user's close friends all use Facebook, then she tends to use it more.

Another violation of the Stable Unit Treatment Value Assumption often happens in two-sided markets such as eBay, Uber, and Lyft. In these markets, control and treatment groups compete for the same resources.

If a new feature increases demand in the treatment group, then the treatment group needs more supply to fulfill that demand, and this will impact the supply in the control group because the resources are shared.

Then what to do about it? Well, we could not really avoid the interference between groups, but we could predict where the interference will happen and take it into consideration in the experiment design phase.

If such a possibility exists, we can change our design to isolate control and treatment units. For example, considering a treatment group in a completely different geographic location to avoid potential interference.

If predicting the interference is very challenging, we should be able to monitor and detect it. And once the mechanism of interference is well understood, we could update the experiment design.

---

### Changes in Users' Behaviors
The last common cause of unreliable results lies in changes in users' behaviors.

Users react to new features or products differently. Some favor new things and tend to use more when they see a new feature, which is a *novelty effect*. But others would hate it, so they tend to use less, which is called a *primacy effect*.

Now let me share with you a personal example. I once used an online grocery shopping app, and there was a new feature that every time you share a photo of the delivery, you get a little cashback so that you can use it for your next purchase.

When I first saw this new feature, I thought it was so cool — I could get a little money back if I just took photos and shared them. So I used this feature for the very first few deliveries I got. I took the pictures and shared them on the app.

After doing it a few times, I realized the tiny cashback, if I remember correctly, was just a few cents. It was not worth my time at all, and I just stopped using it completely.

This is a typical example of a novelty effect: as a user who sees a new feature, I use it heavily at the beginning, and then I just stop using it at all.

Note that both novelty and primacy effects are not stable. As you can tell from the example I just gave you, they happen only during the initial period after users see a new feature or product.

While there's nothing we could do to prevent these effects, what we could do and should do is to monitor if such effects exist and quantify them, **so that we could filter them out when evaluating the real treatment effect.**

---



## A/B Testing Metrics: What You Need to Know About Success, Driver, and Guardrail Metrics!






---

In this video, we will start with business metrics including goal metrics, driver metrics, and guardrail metrics. Then we'll talk about how to format metrics for online experiments.

---

### Goal Metric
There are three kinds of operational metrics that companies use to measure success and progress, and to understand areas for improvements. The first kind of metric is a goal metric.

It's also known as a success metric, true north metric, north star metric, OKR metric, or the primary metric. This kind of metric reflects a company's long-term vision, and it always ties to a company's mission. Goal metrics are a small set of metrics that the company truly cares about.

I know it may sound abstract — how do we translate such a mission or vision to a set of metrics? Let me give you an example. Facebook's mission is to give people the power to build community and bring the world closer together. Its goal metrics include advertising revenue, daily active users, and monthly active users.

While the transformation from its mission to its goal metrics isn't perfect, the goal metrics do reflect what the company ultimately cares about. They are also simple enough to be easily communicated to different stakeholders, such as investors, customers, and employees.

The goal metric should also be stable over long periods of time to allow the whole organization to work towards improving it. While goal metrics are critical to measure the overall success of a company, they may not be suitable for online experiments because they can be difficult to measure or may not be sensitive enough to product changes.

For example, Facebook cares about ad revenue, but not every team could use it for A/B testing. There are teams focusing on improving user engagement, and also teams focusing on website or native app performance. For such teams, what they do definitely contributes to the company's overall success, but they don't use those company-level goal metrics to measure performance.

---

### Driver Metric
Compared with goal metrics, which are about the long-term vision, we also need metrics to reflect short-term progress. The driver metric, also known as surrogate metric, indirect metric, or predictive metric, is often used to measure short-term objectives.

They align with the goal metrics, but they are more sensitive and actionable to be able to measure short-term progress, and they drive teams to work on it. That's also why they are better than the goal metrics to be used for A/B testing.

Now let me give you a concrete example of a driver metric. A marketing team's goal is to acquire new users, and one of the driver metrics could be the number of new users registered per day.

The distinction between the goal metric and the driver metric is actually something new I learned from the book. Before reading it, I thought I knew what a success metric is — I had developed such metrics in practice and had used them to run online experiments. But after reading the book I realized that I was wrong.

What I thought of as success metrics were actually driver metrics. In fact, success metric is the same as the goal metric, and it's about the long-term vision. While the driver metric is used to measure short-term progress and is more suitable for online experiments.

Check out this blog written by my friend Rob and me — it covers several metric frameworks which can be very helpful for you to understand what metrics are used in different business domains.

---

### Guardrail Metric
The last category of metrics is a guardrail metric. As the name suggests, guardrail metrics guard us from harming the business and violating important experiment assumptions.

In this book, it categorizes guardrail metrics into two groups, which I think is very helpful to understand different roles of guardrail metrics.



The first one is the organizational guardrail metric. If this kind of metric shifts in a negative direction, the business will suffer significant loss. For example, if the loading time of a web page increases by a few million seconds, there can be a significant loss of customers and revenue.

In practice, page loading latency is often used as a guardrail metric when new features are developed and tested through A/B testing. A few other commonly used organizational guardrail metrics include errors per page and client crashes.

The other kind of guardrail metric is trustworthiness-related metrics. They are used to monitor the trustworthiness of an experiment — that is, to check if there is any violation of its assumptions.

One commonly used metric is to check if randomization units assigned to each variant are truly random. When the numbers in different groups are different, the authors refer to this as sample ratio mismatch. We then need to perform a t-test or a chi-square test to check if the assignment ratio matches with what was designed.

---




### Application of Metrics
Now you know the definition of goal metrics, driver metrics, as well as guardrail metrics. In practice, we need to be clear about the context when talking about a specific metric, because the same metric can be used differently for different teams.

One team's driver metric can be another team's guardrail metric. For example, one front-end team's goal is to improve web performance, so reducing latency is their goal, and Time To Interactive (TTI) can be one of their driver metrics. A product team may use the same metric as a guardrail metric to make sure any product changes don't increase latency.

---

### Attributes of a Good Metric
Next, let's look at what are the attributes of a good metric.

In the blog post I mentioned earlier, it has a few general rules to formulate metrics:

* A good metric should be simple: easy to understand and calculate, and people should be able to remember and discuss it easily. If you cannot use one sentence to explain a metric, it's not simple.
* The definition of a good metric is clear and there is no ambiguity in interpretation.
* A good metric should be actionable: the metric can be shifted by changes in products, and it offers insights on how you can improve.

It should not be easily gamed. Gaming means that a metric makes you feel like you are getting results, but it offers no insights into actual business health or growth. Short-term revenue is an example of such a metric. Increasing prices of products may increase short-term revenue; however, the business may lose customers in the long run.

We have talked about operational metrics, which are critical to measure accounting or team performance. However, not all of them are suitable for online controlled experiments.

---

### Metrics for Experiments
Next, let's go over what are the requirements of metrics that can be used directly for experimentation. In the book, the authors summarize three attributes for metrics that are suitable for experimentation:

* **Measurable**: We should be able to calculate those metrics with data collected during the experiment period.
* **Attributable**: We should be able to attribute metric values to the experiment variants. It means that the metrics will be able to be calculated separately for different variants in the experiment.
* **Sensitive and timely**: Experiment metrics should be sensitive enough to detect changes in a timely manner.

In online experiments, we typically select a few driver metrics as key metrics, as well as some guardrail metrics to monitor impacts on other aspects of the business.

---

### Trade-offs and Launch Decisions
Now I want to share one question I get constantly: since we have multiple metrics for an experiment, how do we make the launch decision when one metric goes up and one metric goes down?

It's a very reasonable question, and this scenario happens often in practice. Many organizations have a mental model of the trade-offs they are willing to accept when they see any particular results. For example, trade-offs between user acquisition and revenue.

How should the company strike the optimal balance between revenue maximization and user acquisition? Acquiring new users can always be done by expensive campaigns, such as providing large discounts or gifts, but it will degrade revenue.

This kind of trade-off is not something that can be determined by a single data scientist or a single data science team. It's something that is discussed and aligned among various stakeholders, such as product teams, marketing teams, engineering teams, and the leadership team.

---

### Practical Suggestions
Finally, I want to share with you two practical suggestions from the book on formulating metrics for experiments:

1. Combine a few target metrics into an overall evaluation criterion (OEC): a weighted combination of the most important driver metrics, and use it as the only criterion for an experiment.
2. If coming up with such an OEC is difficult, the authors recommend choosing no more than five metrics as target metrics.

There are two main disadvantages of having too many metrics:

* Too many metrics may confuse people, and may possibly lead to ignoring the key metrics.
* Including too many metrics may affect the decision-making process and increase the chances of having false discoveries.

---


## A/B Testing Made Easy: Real-Life Example and Step-by-Step Walkthrough for Data Scientists!





While those steps may seem to be straightforward, there are many factors to consider when running an epitest in reality. That is why I made this video to help you understand what are the decisions that need to be made in order to run a test correctly.  

---

This example is about a fictional e-commerce website that sells physical products. One team proposed a new feature to show similar products during the checkout process. The idea is that a customer might want to buy some similar products together with those already in the shopping cart, and this feature is projected to increase revenue.

However, there is a concern that showing similar products might distract customers from checking out, which may degrade revenue. The argument is that a similar product may have a cheaper price or more features or might be more appealing, then customers may spend extra time exploring those products and delay or even abandon the checkout process. To evaluate if this is a good idea or not, the company decided to run an epitest.

---

Now let's follow the diagram to see what are the needed ingredients to run this experiment. Before running the experiment, we need to be clear on the objective and key metrics, what are the variants and the randomization unit. 

### First of all, let's choose a metric to reflect our goal.

Since we are interested in measuring the difference of revenue due to the new similar products feature, one good metric is simply revenue. Note that it will only be a fair comparison between control and treatment when the number of users in the control group is the same as that in the treatment group. However, the actual number of users may be different due to pure chance. So considering possible differences between groups, we could normalize revenue by the number of users in each group, and the metric would be revenue per user.

### Next, we need to make changes to the product and come up with variants of the experiment.

The control group will stay the same, i.e. does not have a similar products feature and customers could checkout and make a payment directly.

The first treatment group is to add a similar products section on the checkout page. This section shows products that are similar to the ones that are already in the shopping cart, and those products are recommended by a backend algorithm. Here we won't dive into how the backend algorithm works, but from this example, we can assume it's able to provide reasonable recommendations.

The second treatment group has a pop-up window to show similar products. Users could click on a product to explore more or exit the page by clicking the X icon. The products showing on the pop-up window are the same as the ones in the first treatment group, meaning that we are using the same backend algorithm.

If we change the algorithm, it will not be a fair comparison. Once we have different variants for the experiment, the next thing is to consider if we have enough randomization units. For this example, we could simply use user as a randomization unit, and our example assumes that we have enough users for the experiment. Now we are clear on the goal of the experiment and what randomization unit we will use. 

### Next, we are ready to design the experiment.

The very first question is, do we want to target all the users or a specific segment of users? To answer this question, let's analyze the user journey, which can be thought of as a funnel. Users first land on the website, then they browse or search for products they are interested in buying. After they find products they want to buy, they add them to the shopping cart. 

Then they start the checkout process to make a payment. Note that users could go back and forth between these steps, for example, starting the checkout process then back to search other products. Now we have a few options for the target population, but which one would be the most reasonable for our goal?

Well, we could use the users on the top of the funnel as the target population, basically users who land on the homepage. We will get the most users if we choose to do so. However, a large proportion of them will not start the checkout, so they will not be impacted by the change at all.

A more reasonable option is to consider only those users who have the intention to make a purchase and initiate the checkout process, because they will see the new similar products feature during the checkout. OK, we have selected the target population.



### Next, let's determine the sample size of the experiment.


Basically, how many users should be allocated randomly in each group? There are a few things to consider.

#### First, what is the practical significance boundary?

How much change is the change that matters from a business perspective, given all the costs associated with making a change? For our example, how much increase in revenue per user would be considered to outweigh the costs?

In this example, assuming different stakeholders have agreed on an increase of \$2 in revenue per user is practically significant. In other words, if revenue increased by \$2 on average, we could launch the change to production. Then, we need to choose the power of the test, as well as the significance level, alpha, in order to calculate the sample size. We will use the industry standard 80% power and 5% significance level.

#### With power and alpha

we can apply the formula 16 multiplied by sigma squared divided by delta squared, where sigma is the standard deviation of the population, and delta is the difference between treatment and control. If you need a refresher on how to derive this formula, check out this video for the details. But we still don't know the value of sigma yet.

#### Let's assume the sigma is 20 for this example.

So the sample size is 1600 users. It means that we need 1600 unique users in each variant.

So it will be 4800 users in total. Note that if we want to detect a smaller change, such as \$1 instead of \$2, we'd need more samples. Also, if we set a lower alpha, such as 2.5%, we would also need to increase the sample size.

---

### The last thing remaining before running the experiment is to decide how long to run it.

And we'll need to consider at least the following four factors.

#### First of all, it's common to have a ramp-up plan for an experiment.

especially for changes like this, which involves using a new backend algorithm. The goal is to make sure there's no bugs and the traffic can be handled without introducing too much latency. Typically, it exposes only a small percentage of users to the change, then gradually increase the percentage. While there's no fixed number of the starting point, the rule of thumb is to start with thousands of users.

For this example, we could start with 5% of the target population in each treatment group on the first day, and increase to 10% on the second day. Eventually, it would be a 34-33-33% split among control, treatment 1 and treatment 2. On average, there are 2,000 users per day entering the checkout process. This translates into running the experiment for a minimum of 4 days.

Note that 2,000 users is the total number of users, and some of them may checkout on multiple days. So if we have 2,000 users on the first day, the second day will have less than 4,000 distinct users in total.

#### The second factor is the day-of-week effect.

People behave differently on particular days of a week than other days. For example, people may make more purchases during weekends. So it is recommended to run at least a whole week to capture this effect.

#### The third factor is seasonality.

It refers to the holiday seasons. For example, e-commerce might experience a surge in sales during Black Friday, and the users' behavior during the time cannot be generalized to non-holiday time. So if an experiment runs during a national holiday, the data gathered during the holiday should not be used for analysis, then we need to run the experiment longer.

#### The last factor is primacy and novelty effects.

Users respond to changes differently. Some favor new things and some don't like change. We can only tell if there is such an effect after running the experiment and analyzing the data.

So to summarize, we want to run the experiment for a minimum of a full week, starting with 5% in both treatment groups, and eventually it will be a 34-33-33% split among control, treatment 1, and treatment 2. It will be potentially longer if we detect novelty or primacy effects.

Now we have 3600 users in total. As I mentioned earlier, the number of unique users is less, because some users may check out on multiple days. Assuming the number of unique users is 3000, it's still much larger than 1600 users we wanted initially.

Well, it's okay to get more users in each group. Actually, it's recommended to be overpowered rather than underpowered. However, if we run the experiment too long, we'll hit a diminishing return. The precision of the results won't be improved any further. So to decide when exactly to stop the experiment, we'll need to consider all those factors.

---

### After designing the experiment, we could start running it and collecting log data.


Finally, we get all the results and we are very close to making a launch decision. Before dive into the analysis, we first need to do some sanity checks.



Why do we need them? These checks are important, because we have a few assumptions of habitats and the results are unreliable if those assumptions are violated. For this example, we need to check if the number of users are expected according to our design, if the number of users assigned to each group is truly random. We also need to check the latency when loading the webpage in treatment 1, and the latency of the pop-up window in treatment 2.

This is to make sure there's no significant latency in any group, and the user experience is consistent among different groups. For our example, suppose the results successfully pass the sanity checks. 


### We could then use hypothesis testing to make recommendations.

Let's look at the results together.

Here's a table showing the results of the experiment. Let's plot them with the statistical and practical significance boundaries. Typically, we recommend launching a change if the result is both practically and statistically significant, otherwise no launching the change.

But this result does not seem to be that straightforward. When comparing treatment 1 versus control, we see that the result is not statistically significant based on the p-value, but it's likely to be practically significant because the point estimate is larger than the practical significance boundary. So we can argue that there's no impact at all, or we could argue the impact can be significant enough to launch the change.

Because of such uncertainty, we don't recommend launching the change. Running a follow-up test with more power would be helpful in this case. The comparison between treatment 2 and control is also interesting.

The confidence interval overlaps with the practical significance boundary. While the result is statistically significant, it's possible that the change is not practically significant. We also recommend running a follow-up test to make the final decision.


## A/B Testing Analysis Made Easy: How to Use Hypothesis Testing for Data Science Interviews!





### Intro

Hi guys, welcome back to my channel. In today’s video, we will dive into how to use hypothesis testing to solve real problems, specifically how to use hypothesis to analyze results of A/B testing. I will give you two examples and show you how to solve them step by step. This video is part two of cracking hypothesis testing problems in data science interviews. In part one of the video, we went through a few commonly used hypothesis tests, when to use them, and what are the differences between them. If you need a refresher, feel free to check out the video. Okay, let’s get started.



### Two-sample test of proportions

The first question is: we run an experiment where we test the color of a button. The metric we’re looking at is the click-through probability. It is calculated as the number of users who click the button over the total number of users. There are a thousand users in both control and treatment groups.



Here are the results of the experiment: the control group has a 1.1% click-through probability, while the treatment group has 2.3% click-through probability. Can we conclude there’s a significant difference between these two groups? Would you recommend launching the experiment? The practical significance boundary is 0.01, and we choose an alpha of 0.05.

Let’s start with outlining the steps to take to analyze the results. 

1. Which hypothesis test to use?
2. What is the null hypothesis?
3. Is the result statistically significant?
4. Is the result practically significant?
5. Make decisions

First of all, we want to decide which hypothesis test to use—the diagram we went through in the previous video could serve as a reference. 
Next, we should be clear on what the null hypothesis of the test is. 
Then we could evaluate if the test result is statistically significant by comparing the test statistic with the critical value. 
We also need to check if the result is practically significant by comparing the confidence interval of the estimation with the practical significance boundary. 
Finally, we can make decisions based on the result.



#### Which hypothesis test to use?

- Bernoulli population: either clicks or doesn’t click  
- Control group: \( n \cdot \hat{p} = 1000 \cdot 1.1\% = 11 \)
- Treatment group: \( n \cdot \hat{p} = 1000 \cdot 2.3\% = 23 \)
- Test statistic follows Z-distribution  


Now let’s go back to the question and analyze the experiment result. Each user is a click or not clicks a button, so it’s a Bernoulli population. In this case, *n × p̂* is 11 in the control group and 23 in the treatment group. Both can be considered as large samples, so we choose a z-test. It means that the test statistic *TS* follows a z-distribution, or a standard normal distribution.

Measurements:

- Users clicked: \( x_{ct}, x_{tr} \)  
- Total number of users: \( n_{ct}, n_{tr} \)  

\[
\hat{p}_{ct} = \frac{X_{ct}}{n_{ct}} = \frac{11}{1000}
\]

$$
\hat{p}_{tr} = \frac{X_{tr}}{n_{tr}} = \frac{23}{1000}
$$

Now we’ll measure the users who click in each group, which we will call *x\_control* and *x\_treatment*, as well as the total number of users in each group, which we will call *n\_control* and *n\_treatment*. The estimated probability *p* of the control group, *p̂\_control*, in this example is 11 over a thousand, which is 1.1%. Similarly, we can get *p̂\_treatment* is 2.3%.


#### Which hypothesis test to use?

- Bernoulli population: either clicks or doesn’t click  
- Control group: \( n \cdot \hat{p} = 1000 \cdot 1.1\% = 11 \)  
- Treatment group: \( n \cdot \hat{p} = 1000 \cdot 2.3\% = 23 \)  
- Test statistic follows Z-distribution  



**What is the null hypothesis?**


\(
d = \hat{p}_{tr} - \hat{p}_{ct}
\)

* Null hypothesis

\(H_0 : p_{ct} = p_{tr}, \quad d = 0\)


\(
\hat{d} \sim N(0, SE^2)
\)


Remember we want to estimate the difference between *p\_control* and *p\_treatment*, and I’ll call this difference *d*. Under the null hypothesis, *p\_control* and *p\_treatment* is the same. In other words, *d*, the true difference, is equal to zero, and we would expect our estimation *d* to be normally distributed with a mean of zero. We don’t know its standard deviation yet, and we need to estimate it. The test statistic is shown here.

Now I estimate *d* by subtracting the *p̂\_control* from the *p̂\_treatment*, and this comes out to 0.01. To calculate the standard error of *d*, since we have two samples, we need to choose a standard error that can give us a good comparison of both. We could calculate what is called pooled standard error.

\[
\hat{p} = \frac{X_{ct} + X_{tr}}{n_{ct} + n_{tr}} 
= \frac{11 + 23}{1000 + 1000} 
= 0.017
\]

Compute "pooled" SE

\[
S_{pool} = \sqrt{ \hat{p} \cdot (1 - \hat{p}) \cdot \left( \frac{1}{n_{ct}} + \frac{1}{n_{tr}} \right) }
\]

\[
= \sqrt{ 0.017 \cdot (1 - 0.017) \cdot \left( \frac{1}{1000} + \frac{1}{1000} \right) }
\]

Test statistics

\[
TS = \frac{\hat{p}_{tr} - \hat{p}_{ct}}{SE} 
= \frac{0.012}{0.00578} 
= 2.076
\]


To obtain the pooled standard error, the first thing we’ll calculate is the pooled probability of a click, *p̂* (and I’m using a hat here because this is an estimated probability), and the probability is the total probability of a click across two groups—that is, the total number of users who click the button divided by the total number of users. Then we can calculate the pooled standard error, which is given by this formula. 


So the pooled standard error for our experiment comes out to 0.00578. Now we can get the value of the test statistic, which is 2.076.






#### Is result *statistically* significant?

- critical z-score \((\alpha = 0.05)\) = **1.96**  
- If ( TS > 1.96 ) or ( TS < -1.96 ), reject null hypothesis  



In this example

TS = 2.076 > 1.96

- Test is **statistically significant**


Next we can compare it with the critical z-score values of the alpha equals to 0.05, or 95 percent confidence level, which is 1.96. If the test statistic is greater than 1.96, or less than the negative of this cutoff, then we can reject the null hypothesis and conclude that the difference represents a statistically significant difference. In this example, it is larger than 1.96, so the test is statistically significant.

---

#### Is the result practically significant?


We also want to know if the result is practically significant to help us make the decision. To do it, we need to calculate the confidence interval of the estimation. We already know the center of the confidence interval, which is 0.012. Let’s now calculate the width of the confidence interval, which is also called the margin of error.

**Confidence interval of *d***

- Center of C.I. = 0.012  
- Width of C.I. (margin of error)  

\[
m = Z \times S_{pool} = 1.96 \times 0.00578 = 0.0113
\]

\[
CI \text{ of } d: \; 0.012 \pm 0.0113 = 0.0007 \sim 0.0233
\]


For the normal distribution, the margin of error *m* is equal to the z-score of the confidence level times the standard error, which comes to 0.0113. So the confidence interval is from 0.0007 to 0.0233. We can draw a diagram to compare the confidence interval and the practical significance boundary. Here I’ve drawn the practical significance boundary as two dashed lines and zero as this solid line. A point estimate, which is shown as a solid red circle, is greater than the practical significance boundary, but the left end of the confidence interval is less than the practical significance boundary.


#### Make decisions

This is a tricky case. It means that our best guess—the point estimate—there is a practically significant change, but it’s also possible the change is not practically significant. So we are not confident the true change is large enough to be worth launching, so I would not recommend launching the feature.

Just to mention, we could also use the confidence interval to check statistical significance. We could check if it overlaps with zero. If it does, it’s not statistically significant. The result is the same as comparing test statistic with a critical value.

We have just talked about using the z-test to compare two Bernoulli populations and how to determine if the difference is statistically and practically significant. Let’s now move forward to the next example.

---

### Two-sample test of means

We run an experiment to test if adding a new feature will change the average number of posts created per user. Both control and treatment groups have 30 users. The first array represents the number of posts created by each user in the control group, and the second array has the number of posts created by each user in the treatment group. The control group has a mean 1.4, and the treatment group has a mean 2.

Assume variances are similar in the two groups. What conclusion can you draw from this experiment? Shall we launch the feature to all users? The practical significance boundary is 0.05, and we choose an alpha of 0.05.

Let’s start analyzing. Clearly, we’re not dealing with a Bernoulli population, and the variances are unknown. Based on the diagram we explained in part one, we will choose a two-sample t-test to compare the differences between control and treatment. We are told that the population variances in the two groups are similar, so we could calculate the so-called pooled variance. If the variances in the two groups are different, we will need to obtain the unpooled unequal variance—we’ll cover it shortly.

Remember our goal here is to measure the difference *D* between the average number of posts in control *μ\_c* and treatment *μ\_t*. I call the estimate of the *d* for difference. Under the null hypothesis, *d*, the true difference, is equal to zero. The test statistic of a two-sample t-test with pooled variance is given by this formula. As for the pooled standard error, it can be calculated using a formula like this.

Here we introduce two more parameters: sum of squares (*SS*) and degree of freedom (*df*). I will not go through in detail how to get the value of the pooled standard error, but all the numbers are shown here. Feel free to pause the video to derive it and verify your calculation. Now we have the value of the pooled standard error; we can compare the value of the test statistic and the critical t-score value of a 95 percent confidence level for degree of freedom 58, which is 2.002. The test statistic is larger than 2.002; it means the result is statistically significant.

Next we would construct the confidence interval of *d*. Similar to the previous example, we could draw a diagram to compare the confidence interval with the practical significance boundary and zero. In this case, both ends of the confidence interval were greater than the practical significance boundary, so it’s highly possible that the difference of the two means in fact changed by more than the practical significance level. So we would recommend launching the experiment.

We have just covered using t-test to compare two samples with similar variances and sample sizes. Let’s now look into how to deal with the case that two samples have very different variances or sample sizes.

---


### Welch’s t-test

Welch’s t-test is used to deal with this scenario. It is an adaptation of Student’s t-test. It’s specific to the case that when the two standard deviations are not similar—specifically, when one is more than twice of the other—then the unpooled standard error is used. We’ll calculate the unpooled standard error instead of the pooled standard error, and it follows this formula. *s\_c* and *s\_t* are the sample standard deviation of the control group and the treatment group, respectively. The confidence interval of the estimation can then be obtained using this formula.

If we compare this scenario with the previous example, where two samples have similar variances, two things are different: one is the standard error, and the other one is the degree of freedom. The rest are the same. The form of the degree of freedom is a bit complicated and you don’t need to remember it—you only need to know that Welch’s t-test is used to deal with such cases, and you could always look up the formula for the calculation.

I have just walked you through two examples using hypothesis testing in reality. Hopefully they are helpful to deepen your understanding of the subject. As always, guys, I appreciate you for taking the time to watch this video. Let me know if you have any questions or feedback. I will see you soon.




## Crack A/B Testing Problems for Data Science Interviews  Product Sense Interviews


Since we have lots of things to cover, here's an outline of this video. Feel free to choose the section you want to learn more about and skip the ones you are familiar with.

The topics we are going to cover are:

* What is A/B testing
* How long to run an A/B test
* Multiple testing problem
* Novelty and primacy effect
* Interference between variants
* Dealing with interference

### What is A/B testing

First and foremost, let me briefly explain what A/B testing is. Habitats, also known as controlled experiments, are used widely in industry to make product launch decisions. In the simplest form, there are two variants: control A and treatment B. Typically, control group uses the existing feature, while the treatment group uses the new feature.

A/B testing allows tech companies to evaluate a feature with a subset of users to infer how it may be received by all users. A/B testing is one of data scientists’ core competences, so A/B testing questions appear frequently in data science interviews. They are typically asked together with metric questions, and the questions can appear in any component of A/B testing, including developing new hypotheses, designing A/B test, evaluating test results, and making ship or no-ship decisions.

---

### Designing an A/B test

The second topic is about designing an A/B test, specifically how long to run an A/B test. This is one commonly asked question during interviews. To decide the duration of the test, we need to obtain the sample size, and three parameters are needed to get it. These parameters are type 2 error or power (because power equals to 1 minus type 2 error; you know one of them, you know the other), the significance level, and the minimum detectable effect.

The rule of thumb is: sample size approximately equals to 16 multiplied by sample variance divided by delta squared, whereas delta is the difference between treatment and control. I know some of you may be interested in learning how we come up with the rule of thumb formula, so I have another video to explain it step by step. Feel free to check out the link in the description. During the interview, it is not required to derive the formula, but you want to talk about how each parameter influences the sample size.

For example, we need more samples if the sample variance is larger, and we need less samples if the delta is larger. Sample variance can be obtained from the data, but how do we estimate the difference between treatment and the control? Actually, we don't know that before we run the experiment, and this is where we use the third parameter: the minimum detectable effect. It is the smallest difference that would matter in practice. For example, we may consider a 0.1 percent increase of revenue as a minimum detectable effect. In reality, this value is decided by multiple stakeholders.

Once we know the sample size, we could obtain the number of days to run the experiment by dividing the sample size by the number of users in each group. If we have the number less than 14 days, we typically would run for 14 days to capture the weekly pattern.

---

### Multiple testing problem

Sometimes we run tests with multiple variants to see which one is the best amongst all the features. It can happen when we want to test the multiple colors of a button, or test different home pages. Then we'll have more than one treatment group.

A sample interview question is: we are running 10 tests at the same time, trying different versions of our landing page. In one case, the test wins and the p-value is less than 0.05. Would you make the change?

The answer is no. In this case, we should not simply use the same significance level 0.05 to decide whether the test is significant, because we are dealing with more than two variants, and in such a scenario, the probability of false discoveries increases.

For example, if we have three groups to compare, what is the chance of observing at least one false positive, assuming the significance level is 0.05? Well, we could get the probability that there's no false positives, and it would be 0.95 to the power of 3. Then we can obtain the probability that there's at least one false positive. With only three groups, the probability of false positive or type 1 error is over 14 percent.

This is called the multiple testing problem.

There are several ways to deal with the multiple testing problem. One commonly used method is Bonferroni correction. It divides the significance level by the number of tests. For the interview question, since we are measuring 10 tests, then the significance level for the test should be 0.05 divided by 10, which is 0.005. Basically, only if a test shows a p-value less than 0.005, we claim it's significant. The drawback of this method is it tends to be too conservative.

Another method is to control false discovery rate (FDR). FDR is the expected value of number of false positives divided by number of rejections. It measures, out of all the rejections of the null hypothesis (that is, all the metrics that you declare to have a statistically significant difference), how many of them have a real difference, as opposed to how many were false positives.

This only makes sense if you have a huge number of metrics, say hundreds. Suppose you have 200 metrics and kept FDR at 0.05. This means you're okay with seeing false positives 5% of the time. You will observe at least one false positive in those 200 metrics every time.

---

### Novelty and primacy effect

When there is change in the product, people react to it differently. Some are used to the way it works and are reluctant to change. This is called primacy effect or change aversion. Others may welcome changes, and the new feature attracts them to use more. This is called the novelty effect.

But both effects will not last long. People's behavior will stabilize after a certain amount of time. So if an A/B test has a larger or smaller initial effect, it's probably due to novelty or primacy effect. It's a common problem in practice, and many interview questions are about this topic.

A simple question is: we ran an A/B test on a new feature and the test won, so we launched the change to all users. However, after launching the feature for a week, we found the treatment effect quickly declined. What was happening?

The answer is: the novelty effect. Over time, as the novelty wears off, repeat usage will be small, so we observe a declining treatment effect.

Now you understand both novelty and primacy effects. How do we deal with them? One way to deal with such effects is to completely rule out the possibility of those effects. We could run tests only on first-time users, because the novelty effect and the primacy effect obviously don't affect such users.

But if we already have a test running and we want to analyze if there's novelty effect, we could compare first-time users vs old users’ results in the treatment group, to get an actual estimate of the impact of novelty effect. Same for the primacy effect.

---

### Interference between groups

We have just covered two effects that make the test results unreliable. Interference between control and treatment groups can also lead to unreliable results.

Typically, we split control and treatment groups by randomly selecting users. In the ideal scenario, each user is independent and we expect no interference between control and treatment groups. However, sometimes this does not work. This may happen for testing social networks such as Facebook, or two-sided markets such as Uber, Lyft.

Let's look at a sample interview question. Company X has tested a new feature with a goal to increase the number of posts created per user. They assign each user randomly in either control or treatment group. The test won by 1% in terms of the number of posts. What do you expect to happen after new feature is launched to all users? Would it be the same as 1%? If not, would it be more or less? Assume there's no novelty effect.

The answer is that we will see a value different from 1%. Let me explain why.

In social networks such as Facebook, LinkedIn, and Twitter, users' behavior is likely impacted by other people in their social circles. A user tends to use a feature or product more often if their friends use it. This is called a network effect. So, if we use user as a randomization unit, and the treatment has an impact on users, the effect may spill over to the control group. That is, people in the control group are influenced by those in the treatment group. In that case, the difference between control and treatment groups underestimates the real benefit of the treatment effect. So, back to the question: there will be more than 1%. That's how network effect influences social networks.

For two-sided markets such as Uber, Lyft, and Airbnb, interference between control and treatment groups can also lead to biased estimates of treatment effect. It is mainly because resources are shared among control and treatment groups, meaning control and treatment groups will compete for the same resources. For example, if we have a new product that attracts more drivers in the treatment group, fewer drivers will be available in the control group, so we will not be able to estimate the treatment effect accurately.

But different from social networks, where the treatment in fact underestimates the real benefit of a new product, in two-sided markets the actual effect will be less than the treatment effect.

Now you understand why interference between control and treatment can cause the post-launch effect to be different from the treatment effect. It leads us to the next question: how do we design the test to prevent the spillover between control and treatment?

---

### Dealing with interference

A sample interview question is: we are launching a new feature that provides coupons to our riders. The goal is to increase the number of rides by decreasing the price for each ride. Outline a testing strategy to evaluate the effect of the new feature.

There are many ways to deal with the spillover between groups. The main idea is to isolate users in the control and treatment group. Here I will just list out a few commonly used solutions.

For two-sided markets, we could use geo-based randomization instead of splitting by users. We could split by geo locations. For example, we could have the New York metropolitan area in the control group and the San Francisco Bay area in the treatment group. This will allow us to isolate users in each group, but it will have big variance since each market is unique in certain ways.

The other method, though used less commonly, is time-based randomization. Basically, we select a random time, for example a day of a week, and assign all users to each treatment or control group. It works when the treatment effect only lasts for a short amount of time. For example, if a new surge price algorithm works better, it does not work when the treatment effect takes a long time to be effective. For example, a referral program: it can take some time for a user to refer his or her friend.

For social networks, one way is to create network clusters to represent groups of users who are more likely to interact with people within the group than people outside of the group. Once we have those clusters, we could split them into control and treatment groups.

Another way is called ego network randomization. The idea was originated from LinkedIn. A cluster is composed of an ego (a focal individual) and her alters (the individuals she's immediately connected to). It focuses on measuring the one-out network effect, meaning the effect of my immediate connections’ treatment on me. So each user either has a feature or does not. There's no complicated interactions between users needed. This approach is simpler and more scalable than the previous one.

To summarize, the methods we just mentioned apply in different scenarios, and all of them have limitations. In reality, we want to evaluate which methods work better in a certain scenario, and we could even combine more than one method to get reliable results.

So those are the six topics that I have promised to share with you.


## A/B Testing Fundamentals: What Every Data Scientist Needs to Know!


### What are A/B tests

An A/B test is an experiment in which all elements are held constant except for one variable.

Typically, it compares a control group against a treatment group. All variables are identical between the two groups except for one factor that’s being tested. Different versions of a product or user experience are formally referred to as the variants. Variants can be as simple as colors of a button or as complicated as different back-end algorithms to display search results.

In cases where there are two variants, one control and one treatment group, it is called an A/B test. If there are more than two variants, it’s called an A/B/n test. But in reality, A/B tests could also be used to refer to experiments with multiple variants.

I sometimes get this question: what are the differences between A/B tests and controlled experiments? Well, they are the same thing. A/B tests are sometimes called online experiments, controlled experiments, randomized controlled experiments, or split tests, but they all refer to the same thing.

Now let me give you an example of an A/B test. In the book, it mentions an interesting example: Google tested 41 gradations of blue on Google search result pages. In each treatment group, the color is different. Even though the tests frustrated the visual design lead at that time, the result showed that color schemes significantly changed user engagement.

A/B tests are widely adopted in the industry while evaluating new product ideas. In fact, when you are browsing a website or using a mobile app, you might be part of an experiment that is running behind the scene.

---

### Why A/B tests

Why do we need to run experiments?

Why do companies run experiments instead of simply rolling out a new feature? The goal of running A/B tests is to make data-driven decisions. Only when the results are reliable and repeatable can we make the right decision.

To make the result reproducible, an important requirement is that the factor we are testing is the cause of the change in the metric, so that when launching the feature to all the traffic, the impact can be predicted from the treatment effect measured in the experiment.

For example, changes of colors could cause changes in user engagement, assuming other things stay the same. Running A/B tests is the scientific way to do it.

In the book, the authors claim that randomized controlled experiments are the gold standard for establishing causality. We believe online controlled experiments are the best scientific way to establish causality, with a high probability, able to detect small changes that are hard to detect with other techniques such as change-over-time, and able to detect unexpected changes.

Often unappreciated, but many experiments uncover surprising impacts on other metrics.

Now you know what is an A/B test, as well as the importance of running experiments. Let’s dive into the steps to run A/B tests.

---

### Steps to run A/B tests

In general, there are five major steps involved in running a test correctly. I have drawn this diagram to help you understand it clearly. Let’s go through each step one by one.

---

### Experiment prerequisites

Before running experiments, a few things need to be ready.

First of all, we need to define key metrics to measure the goal of an experiment. The key metric is formally known as the Overall Evaluation Criteria (OEC). It should be agreed upon by different stakeholders and should be practically measurable.

For example, if we want to test if changing the color of the checkout button could impact revenue, the key metric of the OEC could be revenue per user per month.

The second requirement is that changes are easy to make. This should be obvious, because we need to compare different variants and find the one that has the highest positive impact on the OEC. If changes are very hard to make, it will introduce complexities to generate variants. For example, it would be very difficult to redesign the whole website and consider that redesign as a variant.

The last requirement is to have enough randomization units to be assigned to different variants. But what is a randomization unit? It’s simply the “who” or “what” that is randomly allocated to different groups. The most commonly used randomization unit is the user.

So how much is enough? The recommendation in the book is to have thousands of randomization units, because the larger the number, the smaller the effects that can be detected.

---

### Experiment design

After these requirements are fulfilled, we could move forward to designing the experiment. The book touches on a few things that need to be considered:

* What population of randomization units do we want to select? Basically, do we want to target a specific population or all the users? Sometimes it’s helpful to run experiments for a specific segment, because the change only affects that segment. For example, a new feature that is only available for users in a particular geographic region.

* Another factor to consider is the size of the experiment. We need to compute the sample size of the experiment in order to achieve the required statistical power. Detecting a small change will need more users. If you are interested in learning how to get the sample size, I have a video to derive the formula step by step.

* The last important consideration is how long to run an experiment. To determine the duration, we will need to consider seasonality, the day-of-week effect, as well as primacy and novelty effects. All of them will influence the decision on how long we should run an experiment.

---

### Running experiments

After all those decisions are made, we could run experiments and collect the data.

In this process, typically data scientists work with engineers to instrument logging to get logged data. For companies that have built their own experimentation platform, this is done automatically.

---

### Result to decision

After running the experiment for the required amount of time, we need to check and interpret the results and use them to make a decision.

In reality, this is where data scientists spend most time and energy on. Once we obtain the data, the very first step is to do sanity checks to make sure the data are reliable. We could only continue the analysis once the sanity checks are passed. If not, we need to discard the results and look into the root cause, and we may need to re-run the experiment.

Here we will not dive into those checks, but I will explain them in detail in an upcoming video.

Once those sanity checks are passed, we could use the results to make a launch decision. And there are many factors to consider.

In the book, it recommends examining at least these factors:

1. **Trade-offs between different metrics**: This refers to the scenario that different metrics move in opposite directions. For example, user engagement goes up but the revenue goes down. How to make the decision?

2. **Cost of launching a change**: For example, cost for engineering maintenance after launch. Since new code may introduce complexity and bugs to the code base, the maintenance efforts can be costly.

Also, there are opportunity costs. The time and effort we spend launching a change might not be as much as the opportunity cost of giving up a different idea.

If those costs are high, we need to ensure that the expected benefits can outweigh the costs. In fact, that’s why we typically set a practical significance boundary to reflect those costs, and we only launch a product if the result is practically significant.

On the contrary, if the cost is low, we will choose to launch any change that is positive. In other words, as long as the result is statistically significant, we can launch the change.

If you’re not familiar with the concept of practical significance and boundary, I highly recommend checking out this video which covers an analysis using both statistical and practical significance boundaries to make a launch decision.

At this point, you might think we’re done with experiments because we have made a decision. Well, we’re getting close, but we’re not done yet.

---

### Post-launch monitoring

If we decide to launch a new product based on the results of an experiment, we need to monitor the long-term effect after launch, because the short-term effect can be different from the long-term effect due to various reasons.

Also, measuring long-term effects has a few benefits, such as insights on long-term impacts that could help improve future iterations.





