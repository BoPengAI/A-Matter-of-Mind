## Special Intelligence vs General Intelligence
Tremendous strides have been made in AI in recent years. Vast majority of the advances are in what I’d call Special Intelligence (SI), the type of model with a pre-defined set of goals, an optimization problem expressed in general as argmax. The set of goals is defined in either the code or the training data, or often a combination of both.

Some very interesting results have been achieved in expanding the goal set without rewriting the code. DeepMind’s [MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules]) has learned several games without prior knowledge of the rules or even representation of the game in focus. Transfer learning has made many exciting progress. However, the goal set is still likely limited by the model design, e.g., open (all information are out in the open) vs closed games, open-ended vs close-ended, fixed vs flexible rule sets.

A notable exception to the SI dominance is Marcus Hutter’s [AIXI](http://hutter1.net/ai/uai.htm). It establishes a theoretical foundation for the most intelligent general-purpose agent. Its role to AI is analogous to what universal Turing machine is to computing. AIXI implies that GI is the ability to adapt to change in the environment and input.

A few notes:

1. “Life as we know it” is obviously a GI agent, with evolution as its algorithm and the goal of surviving the current environment as well as the next environmental scenario change. The “next change” part is important; evolution is an open-ended arms race between the environment and a lifeform’s arsenal for predicting the change, sensing the ongoing change, and adapting to it. For example, [by going on-shore and sharpening vision sensors](https://www.preposterousuniverse.com/podcast/2019/03/25/episode-39-malcolm-maciver-on-sensing-consciousness-and-imagination/) (remote sensing compared to underwater world), early land life gained much longer early-warning time. This means more value in the accuracy and time-horizon of prediction, hence higher intelligence.

2. There’s probably wide consensus that GI is not consciousness, even without much consensus on the latter's definition. But since we are here talking about this, consciousness must be either equivalent to GI, co-emergent with it, or emergent from it. I'm leaving out the possibility of consciousness preceding GI, or even [matter](http://cogsci.uci.edu/~ddhoff/Chapter17Hoffman.pdf) to avoid entanglement in meta-physics.

I’m proposing here an implementable quantity that measures GI.

## A Measure of GI

GI <i>should</i> mean the ability for an agent to modify its own model when given the same input, as the context changes. The context includes the environment and the history -- of the environment, the agent's input, and its output (predictions/decisions/actions).

This excludes all of what we colloquially call “inanimate” entities, even those with very complicated, including chaotic, behaviors. I don’t want to go into the philosophical discussion of whether there is anything but inanimate entities. Let’s stop at such behavior that can be modeled, either deterministically or statistically, with “reasonable” precision under a “reasonably” diverse set of environmental scenarios.

The highlighted ambiguity above is a key difference where the current approach retreats from AIXI. The current goal is to provide an implementable framework, not theoretical completeness.

OK, on to some symbols. Consider the following setup:

An agent, at discrete time step $$i$$, receives input $$I_i$$ from environment $$E_i$$, and produces output $$O_i$$. Let's define context $$C$$ as the join of environment $$E$$ and history $$H$$, $$C = E + H$$, given identical input I-sub-i, can be expressed as

                                                                                                              (E1)

Summing this over all known inputs gives a measure of GI:

                                                                                                                                                    (E2)

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
