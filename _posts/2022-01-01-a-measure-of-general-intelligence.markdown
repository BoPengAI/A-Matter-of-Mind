## Special Intelligence vs General Intelligence

Tremendous strides have been made in AI in recent years. Vast majority of the advances are in what I’d call Special Intelligence (SI), the type of model with a pre-defined set of goals, an optimization problem expressed in general as argmax. The set of goals is defined in either the code or the training data, or often a combination of both.

Some very interesting results have been achieved in expanding the goal set without rewriting the code. DeepMind’s [MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules]) has learned several games without prior knowledge of the rules or even representation of the game in focus. Transfer learning has made many exciting progress. However, the goal set is still likely limited by the model design, e.g., open (all information are out in the open) vs closed games, open-ended vs close-ended, fixed vs flexible rule sets.

A notable exception to the SI dominance is Marcus Hutter’s [AIXI](http://hutter1.net/ai/uai.htm). It establishes a theoretical foundation for the most intelligent general-purpose agent. Its role to AI is analogous to what universal Turing machine is to computing. AIXI implies that GI is the ability to adapt to change in the environment and input.

A few notes:

1. “Life as we know it” is obviously a GI agent, with evolution as its algorithm and the goal of surviving the current environment as well as the next environmental scenario change. The “next change” part is important; evolution is an open-ended arms race between the environment and a lifeform’s arsenal for predicting the change, sensing the ongoing change, and adapting to it. For example, [by going on-shore and sharpening vision sensors](https://www.preposterousuniverse.com/podcast/2019/03/25/episode-39-malcolm-maciver-on-sensing-consciousness-and-imagination/) (remote sensing compared to underwater world), early land life gained much longer early-warning time. This means more value in the accuracy and time-horizon of prediction, hence higher intelligence.
2. A GI agent doesn't have to be an individual physical, or even logical, entity. In evolution, the entire collection of life can be considered a GI agent. Or, as Yuval Harari [puts it](https://www.ynharari.com/book/homo-deus/), and I paraphrase, evolution could be viewed as a means to achieve greater and greater information processing efficiency. This begs the question of "what for?" but that's too interesting for this post.
3. There’s probably wide consensus that GI is not consciousness, even without much consensus on the latter's definition. But since we are here talking about this, consciousness must be either equivalent to GI, co-emergent with it, or emergent from it. I'm leaving out the possibility of consciousness preceding GI, or even [matter](http://cogsci.uci.edu/~ddhoff/Chapter17Hoffman.pdf) to avoid entanglement in meta-physics.
4. The term "general intelligence" is heavily overloaded and rarely with even an attempt at definition. The <i>g factor</i> in psychometrics, for example, is often described as "a type of genreal intelligence." We need to clean up our language if we were to get serious about it. I think AIXI's definition of GI should be the common foundation. Well-designed IQ tests measures what I'd argue is a very specific type of SI -- pattern recognition, in spatial/numerical/linguistic contexts. No wonder AI beat human in IQ tests [years ago](https://www.technologyreview.com/2015/06/12/167735/deep-learning-machine-beats-humans-in-iq-test/).
5. Abstraction is necessary for GI, but is hardly itself a form of GI. It may be a common algorithm used by GI systems, that's about it.

I’m proposing here an implementable quantity that measures GI. But first,

### What Do You Mean By Measuring GI?

When we say "measure" nowadays, we usually mean a disembodied observer with a tool that pokes, one way or another, at the object in question. Even for SI, we have all kinds of objective benchmarks.

When it comes to GI, though, we have to allow the very likely possibility of the measurer of GI is not merely a passive, invisible, omnipresent God observer, but rather a ball-park equally intelligent agent interacting with the subject agent. An Interrogator. By interacting with the subject, the Interrogator can perform systematic tests to reach a reliable conclusion.

Let's unpack this a little.
1. GI is an exception in our measurement because it is right at our level of intelligence, the limit of our cognition by definition. Any measure that can be delegated to a disembodied, pre-defined set of queries that we design can be defeated by something designed by someone else of equal intelligence. Measuring GI has to be open-ended and interactive. An open-ended, interactive game between two agents. Turing test may not be a good test per se but it captures the very essence of what it means to measure GI. We just need to come up with a better test design.
2. GI therefore is intrinsically relative. This should not be controversial. My Roomba appeared to be intelligent to my dog for about half an hour. Then she figured it's not worthy of much attention, however annoying it may be. But Boston Dynamics' [robot dogs](https://www.youtube.com/watch?v=RYzn_gmFs5w) could appear as intelligent as her fellow dogs.
3. This is not contradictory to AIXI. AIXI requires knowledge "over all possible future perceptions created by all possible environments q that are consistent with past perceptions," i.e., a God. Here I'm focusing on what a mere mortal could achieve, in the here and now.

Let's try to be a little more precise.

### Interrogator Suprise model

Here's the setup.

The measurer of agent $$\mathbb{A}$$ is the interrogator $$\mathbb{I}$$. It controls, or at least has read access to, the E(nvironment) and I(nput) received by the subject agent, and observes the O(utput) produced by it. Its goal is to measure the GI of the agent. To do this, it makes a set of all behaviors it considers inanimate, compares with O, and notes any positive/negative surprises, and keeps a running sum, possibly weighted, for each predictive model. At any point in time, it can look at the minimum absolute value of running sums from the set: if it differs from 0 in a statistically significant way, then the agent is likely in the range that the interrogator <b>considers</b> to be intelligent. The deviation from 0 of the minimum running sum is a measure of GI of $$\mathbb{A}$$ relative to $$\mathbb{I}$$ at time t.





A few observations that need more rigorous proof or refutation:

1.       For Markov agents, H=0, the expected value of the term in E1 is zero, therefore G=0.

2.       For deterministic, differentiable agents, the behavior is path independent in the fully specified state space, i.e., E1=0, therefore G=0.

3.       For deterministic, non-differentiable (such as chaotic) agents, the summation in E2 should canceled out across the input space as long as the contextual state space and the output space are bounded. This is speculative at this point.

4.       This leaves only one possibility for achieving G>0, non-Markovia, non-deterministic agents. Bayesian inference is one such example. Open-ended, adaptive evolution is another. But this is a necessary condition, not sufficient.

5.       I deliberately avoided “stochastic” in the above. Once again using evolution as an example. The formation of the genesis cell might have to rely on random trial and error. Once it’s formed, and especially after the emergence of sexual reproduction, there is an increasing element of self-directed adaptation. This is not random. GI seems to require this no-man’s land between stochastic and deterministic models, which I’d call adaptive in contrast.

6.       Survival of the agent is implicitly imperative. However, an agent needs not to be an individual lifeform as we are familiar with, as is survival not limited to physical survival. Indeed, definition of human being in biological terms is impossible considering the vast number of microbes all human lives depend on, as well as the persistent genetic flow from various microbes. The emergence of collective culture provided a scalable mechanism for the informational survival to transcend individual physical lives. It’s easy to see how agents could be defined at the collective and/or informational sense; civilizational survival is a thing.

7.       It may be worth singling out the fact that physical survival is not of particular value in general. An AGI agent needs not to pay particular attention to sustaining any particular physical medium. Unless the design allows it, intentionally or not, robots have no a priori incentives for self-preservation, not to mention colonization or enslavement of anything. These instincts are merely vestiges of human’s particular evolutionary history.

8.       One possible implementation to achieve G>0 is to predict the change in environment and future inputs in such a way that prolongs the agent’s survival, so as to increase the number of terms (life experiences) in E2. This is a quantitative way of stating the evolutionary advantage of intelligence.

GI Is Not An Optimization Problem

As stated near the beginning, SI are all optimization problems with some kind of well-defined objective functions. In contrast, GI are not because there is no reason to believe E2 is bounded in general. This underscores the open-endedness of adaptive models. GI strives to be better, but not necessarily the best.
