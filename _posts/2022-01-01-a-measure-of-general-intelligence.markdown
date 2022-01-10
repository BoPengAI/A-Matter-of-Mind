## Special Intelligence vs General Intelligence

Tremendous strides have been made in AI in recent years. Vast majority of the advances are in what I’d call Special Intelligence (SI), the type of model with a pre-defined set of goals, an optimization problem expressed in general as argmax. The set of goals is defined in either the code or the training data, or often a combination of both.

Some very interesting results have been achieved in expanding the goal set without rewriting the code. DeepMind’s [MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules]) has learned several games without prior knowledge of the rules or even representation of the game in focus. Transfer learning has made many exciting progress. However, the goal set is still likely limited by the model design, e.g., open (all information are out in the open) vs closed games, open-ended vs close-ended, fixed vs flexible rule sets.

A notable exception to the SI dominance is Marcus Hutter’s [AIXI](http://hutter1.net/ai/uai.htm). It establishes a theoretical foundation for the most intelligent general-purpose agent. Its role to AI is analogous to what universal Turing machine is to computing. AIXI implies that GI is the ability to adapt to change in the environment and input.

A few notes:

1. “Life as we know it” is obviously a GI agent, with [evolution as its algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm) and the goal of surviving the current environment as well as the next environmental scenario change. The “next change” part is important; evolution is an open-ended arms race between the environment and a lifeform’s arsenal for predicting the change, sensing the ongoing change, and adapting to it. For example, [by going on-shore and sharpening vision sensors](https://www.preposterousuniverse.com/podcast/2019/03/25/episode-39-malcolm-maciver-on-sensing-consciousness-and-imagination/) (remote sensing compared to underwater world), early land life gained much longer early-warning time. This means more value in the accuracy and time-horizon of prediction, hence higher intelligence, although many recent social developments testify to contrary.
2. A GI agent doesn't have to be an individual physical, or even logical, entity. In evolution, the entire collection of life can be considered a GI agent. Or, as Yuval Harari [puts it](https://www.ynharari.com/book/homo-deus/) (paraphrase), evolution could be viewed as a means to achieve greater and greater information processing efficiency. This begs the question of "what for?" but that's too interesting a question for this post.
3. There’s probably wide consensus that GI is not consciousness, even without much consensus on the latter's definition. But assuming at least I am conscious, consciousness must be either equivalent to GI, co-emergent with it, or emergent from it. I'm leaving out the possibility of consciousness preceding GI, or even [matter](http://cogsci.uci.edu/~ddhoff/Chapter17Hoffman.pdf) to avoid entanglement in meta-physics. In any case, it's legitimate to question the ill-defined consensus -- separation of SI and consciousness seems self-evident unless you develop an emotional attachment to Alexa; but a strong case can be made that GI and consciousness may be two sides of the same coin, which I'll cover in another post.
4. The term "general intelligence" is heavily overloaded and often without even an attempt at definition. The <i>g factor</i> in psychometrics, for example, is often described as "a type of genreal intelligence." AIXI's definition of GI should be the common foundation but it's not nearly as widely adopted as it deserves. Well-designed IQ tests measures what I'd argue is a very specific type of SI -- pattern recognition, in spatial/numerical/linguistic contexts. No wonder AI beat human in IQ tests [years ago](https://www.technologyreview.com/2015/06/12/167735/deep-learning-machine-beats-humans-in-iq-test/).
5. Abstraction is necessary for GI, but is hardly itself a form of GI. It may be a common algorithm used by GI systems, a basis for symbolic manipulation. That's about it.

I’m proposing here an implementable framework for measuring GI. But first,

### What Do You Mean By Measuring GI?

When we say "measure" nowadays, we usually mean a disembodied observer with a tool that pokes, one way or another, at the object in question. Even for SI, we have all kinds of objective benchmarks.

When it comes to GI, though, we have to allow the very likely possibility of the measurer of GI is not merely a passive, invisible, omnipresent God observer, but rather a ball-park equally intelligent agent interacting with the subject agent. An Interrogator. By interacting with the subject, the Interrogator can perform systematic tests to reach a reliable conclusion.

Let's unpack this a little.
1. GI is an exception in our measurement because it is right at our level of intelligence, the limit of our cognition by definition. Any measure that can be delegated to a disembodied, pre-defined set of queries that we design can be defeated by something designed by someone else of equal intelligence. Measuring GI has to be open-ended and interactive. An open-ended, interactive game between two agents. Turing test may not be a well-defined test per se but it captures the very essence of what it means to measure GI. We just need to come up with a better test design.
2. GI therefore is intrinsically relative. This should not be controversial. My Roomba appeared to be intelligent to my dog for about half an hour. Then she figured it's not worthy of much attention, however annoying it may be. But Boston Dynamics' [robot dogs](https://www.youtube.com/watch?v=RYzn_gmFs5w) could appear as intelligent as her peers.
3. This is not contradictory to AIXI. AIXI requires knowledge "over all possible future perceptions created by all possible environments q that are consistent with past perceptions," i.e., a God. Here I'm focusing on what a mere mortal could achieve, in the here and now.

Let's try to be a little more precise.

### Interrogator Cognitive Dissonance Model

Here's the setup.

The measurer of agent $$\mathbb{A}$$ is the interrogator $$\mathbb{I}$$. It controls, or at least has read access to, the $$E$$(nvironment) and $$I$$(nput) received by the subject agent, and observes the $$O$$(utput) produced by it. Its goal is to measure the GI of the agent. To do this, it constructs a set of predictive models, $$\{M_0, M_1, ...M_n\}$$, for all behaviors it considers inanimate, compares with $$O$$, and notes any prediction error $$P(M_i)-O_j$$ for model $$M_i$$ and output $$O_j$$ at time j. At any point in time, it can look at the minimum absolute value of running sums, possibly weighted for different models, from the set: if it differs from 0 in a statistically significant way, then the agent is likely in the range that the interrogator <i>considers</i> to be intelligent. The deviation from 0 of the minimum running sum is a measure of GI of $$\mathbb{A}$$ relative to $$\mathbb{I}$$ at time t.

<div style="text-align: right">$$RGI_{\mathbb{A}|\mathbb{I}} = \min_{M_i}\frac{\begin{vmatrix}\sum_{j}(P_j(M_i)-O_j)\end{vmatrix}}{\sigma((P_j(M_i)-O_j))}$$ (E1)</div>

where $$\sigma$$ is the stdev (leaving out all details about the distribution), to eliminate cheating by increasing variance.

By now the word "inanimate" has probably set off alarms among many GI agents. It's reasonable to [question](https://www.theatlantic.com/notes/2016/06/free-will-exists-and-is-measurable/486551/) whether there's anything but inanimate entities in the universe. Once again I'll just wimp out of the very interesting phisolophical discussion of free will here and stay willfully, resolutely heuristic; anything you can predict with "reasonable" accuracy is inanimate. 

Does this imply that GI is equivalent to unpredictability or complexity? I'd say yes in some sense, but the meaning of unpredictability needs some clarification beyond deterministic vs stochastic. I'll present that in another post.

Another immediate implication is that there seems no reason to think RGI is bounded. This is consistent with the intuition that GI is not an optimization problem (unless in the AIXI "complete knowledge" sense). Also, intuitively, if the subject agent's intelligence is MUCH higher than that of the interrogator, it would appear simply unknowable to the latter. The existence of an upper bound is unknowable. For any GI agent, the best it can hope for is to do better, but never the best.

Some boundary cases:
1. Any stochastic process with stable expectation, or in addition any predictable drift, has $$RGI=0$$ because there is at least one model, by definition, that reduces prediction error, i.e. cognitive dissonance, to 0.
2. Any non-pathologically (yes, just one of many cheap get-aways employed here) determinstic process has $$RGI=0$$.
3. What if you always include a model with uniform distribution over infinite interval ($$\sigma\to\infty$$)? This will always have the minimum normalized running sum of 0. Let's just call it pathological, biased interrogator, discrimination, banned.

So it passes some quick boundary sanity checks. But how do you achieve $$RGI>0$$?

You'll have to be smarter than the interrogator (<i>you're welcome</i>) in the sense that you can defy all of $$\mathbb{I}$$'s predictive models at least sometimes. The more often you defy predictions, the more intelligent you appear until in the limit you defy all predictions all the time. You are unknowable at this point. Welcome to Deityhood.

An important class of predictive models is pattern recognition. To the extend that $$\mathbb{I}$$ can recognize patterns in $$\mathbb{A}$$'s behavior, the latter is unintelligent, uninteresting. The more converage of behavior by the recognized patterns, the less intelligent $$\mathbb{A}$$ is relative to $$\mathbb{I}$$. This is directly reflected in the prediction error term in (E1).

OK, now we're ready to implement.

###Some Thoughts on Implementation

The word implementable is a bit of exaggeration. It's not exactly implementable for any GI agent to exhaust all of its own predictive models, simply by the logical fallacy of complete self-awareness. We can get some help from examining how we think.

Let's say you see a flying object, which is always UFO until identified. First you focus on predictive models in the visual domain, and further the flying subset. This sounds pretty vague but is a huge reduction from the set of all models. Then we use Occam's Razor to quickly evaluate a subset of minimal models, and then reduce it in the [Solomonff's indeuction](https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference) sense -- flying straight? Constant speed? Constant shape/color? By the time you quickly eliminate a dozen or so such minimal models, you're freaking out a little.
