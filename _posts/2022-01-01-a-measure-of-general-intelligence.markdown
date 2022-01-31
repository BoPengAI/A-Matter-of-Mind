### Specific Intelligence vs General Intelligence

Tremendous strides have been made in AI in recent years. But vast majority of the advances are in what can be called Specific Intelligence (SI), the type of model with a pre-defined set of goals, an optimization problem expressed in general as argmax. The set of goals is defined in either the code or the training data, or often a combination of both.

Some very interesting results have been achieved in expanding the goal set without rewriting the code. DeepMind’s [MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules]) has learned several games without prior knowledge of the rules or even representation of the game in focus. Transfer learning has made many exciting progress. However, the goal set is still likely limited by the model design, e.g., open (all information are out in the open) vs closed games, open-ended vs close-ended, fixed vs flexible rule sets.

A notable exception to the SI dominance is Marcus Hutter’s [AIXI](http://hutter1.net/ai/uai.htm). It establishes a theoretical foundation for the most intelligent general-purpose agent. Its role to AI is analogous to what universal Turing machine is to computing. AIXI implies that GI is the ability to adapt to change in the environment and input.

A few notes:

1. “Life as we know it” is obviously a GI agent, with [evolution as its algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm) and the goal of surviving the current environment as well as the next environmental scenario change. The “next change” part is important; evolution is an open-ended arms race between the environment and a lifeform’s arsenal for predicting the change, sensing the ongoing change, and adapting to it. For example, [by going on-shore and sharpening vision sensors](https://www.preposterousuniverse.com/podcast/2019/03/25/episode-39-malcolm-maciver-on-sensing-consciousness-and-imagination/) (remote sensing compared to underwater world), early land life gained much longer early-warning time. This means more value in the accuracy and time-horizon of prediction, hence higher intelligence, although some may argue that many recent social developments testify to the contrary.
2. Evolution is not about optimization. It's not selection of the fittest, just the surviving. This is an important distinction. Lowering the threshold from the "fittest" to the surviving allows much greater diversity of the surviving, therefore preserving those who may be more optimal under future circumstances. Diversity is our way of cheating when we don't know how to define a problem as optimization, or solve an optimization problem, or have insufficient confidence in our models.
3. A GI agent doesn't have to be an individual physical, or even logical, entity. In evolution, the entire collection of life can be considered a GI agent. Or, as Yuval Harari [puts it](https://www.ynharari.com/book/homo-deus/) (paraphrase), evolution could be viewed as a means to achieve greater and greater information processing efficiency. This begs the question of "what for"? Improving information processing efficiency may be a general means for survival and spread but not a goal in and of itself, unless some higher level agents are invoked, e.g., the hyper-intelligent, higher-dimensional beings building Earth to compute the question of Life, Universe, and Everything. Nevertheless, it may be productive to consider human societies or cultural groups as GI agents.
4. There’s probably wide consensus that GI is not consciousness, even without much consensus on the latter's definition. But assuming at least I am conscious, consciousness must be either equivalent to GI, co-emergent with it, or emergent from it. I'm leaving out the possibility of consciousness preceding GI, or even [matter](http://cogsci.uci.edu/~ddhoff/Chapter17Hoffman.pdf) to avoid entanglement in meta-physics. In any case, it's legitimate to question the ill-defined consensus -- separation of SI and consciousness seems self-evident unless you develop an emotional attachment to Alexa; but a strong case can be made that GI and consciousness may be two sides of the same coin, which I'll cover in another post.
5. The term "general intelligence" is heavily overloaded and often without even an attempt at definition. The <i>g factor</i> in psychometrics, for example, is often described as "a type of genreal intelligence." AIXI's definition of GI should be the common foundation but it's not nearly as widely adopted as it deserves. Well-designed IQ tests measure what I'd argue is a type of SI -- pattern recognition, in spatial/numerical/linguistic contexts. No wonder AI beat human in IQ tests [years ago](https://www.technologyreview.com/2015/06/12/167735/deep-learning-machine-beats-humans-in-iq-test/).
6. Abstraction is necessary for GI, but is hardly itself a form of GI. It may be a common algorithm used by GI systems, a basis for symbolic manipulation. That's about it.

I’m proposing here an implementable framework for measuring GI. But first,

### What Do You Mean By Measuring GI?

When we say "measure" nowadays, we usually mean a disembodied observer with a tool that pokes, one way or another, at the object in question. Even for SI, we have all kinds of objective benchmarks.

When it comes to GI, though, we have to allow the very likely possibility of the measurer of GI is not merely a passive, invisible, omnipresent God observer, but rather a ball-park equally intelligent agent interacting with the subject agent. An Interrogator. By interacting with the subject, the Interrogator can perform systematic tests to reach a reliable conclusion.

Let's unpack this a little.
1. GI is an exception in our measurement because it is right at our level of intelligence, the limit of our cognition by definition. Any measure that can be delegated to a disembodied, pre-defined set of queries that we design can be defeated by something designed by someone else of equal intelligence. Measuring GI has to be open-ended and interactive. An open-ended, interactive game between two agents. Turing test may not be a well-defined test per se but it captures the very essence of what it means to measure GI. We just need to come up with a better test design.
2. GI therefore is intrinsically relative. This should not be controversial. My Roomba appeared to be intelligent to my dog for about half an hour. Then she figured it's not worthy of much attention, however annoying it may be. But Boston Dynamics' [robot dogs](https://www.youtube.com/watch?v=RYzn_gmFs5w) could appear as intelligent as her peers.
3. This is not contradictory to AIXI. AIXI requires knowledge "over all possible future perceptions created by all possible environments that are consistent with past perceptions," i.e., a God. Here I'm focusing on what a mere mortal could achieve, in the here and now.

Let's try to be a little more precise.

### Agent-Interrogator Measure

Here's the setup.

A GI agent, the interrogator $$\mathbb{I}$$, tries to measure the GI of agent $$\mathbb{A}$$. It controls, or at least has read access to, the $$E$$(nvironment) and $$I$$(nput) received by $$\mathbb{A}$$, and observes the $$O$$(utput) produced by it. Its goal is to measure $$\mathbb{A}$$'s GI. To do this, it constructs a set of predictive models, $$\{M_0, M_1, ...M_n\}$$, compares with $$O$$, and notes any prediction error $$P(M_i)-O_j$$ for model $$M_i$$ and output $$O_j$$ at time j. At any point in time, it can look at the minimum absolute value of running sums, possibly weighted for different models, from the set: if it differs from 0 in a statistically significant way, then $$\mathbb{A}$$ is likely in the range that $$\mathbb{I}$$ <i>considers</i> to be intelligent. The deviation from 0 of the minimum running sum is a measure of GI of $$\mathbb{A}$$ relative to $$\mathbb{I}$$ at time t.

<div style="text-align: right">$$RGI_{\mathbb{A}|\mathbb{I}} = \underset{M_i}{argmin}\begin{bmatrix}w_i*\frac{\begin{vmatrix}\sum_{j}(P_j(M_i)-O_j)\end{vmatrix}}{\sigma((P_j(M_i)-O_j))}\end{bmatrix}$$(E1)  (E1)</div>

where 
  $$\sigma$$ is the stdev of prediction error (leaving out all details about the distribution), to eliminate cheating by increasing variance, and
  $$w_i$$ is the weight for model $$M_i$$, and
  The models (one or multiple) that give argmin are the best models $$\mathbb{I}$$ has of $$\mathbb{A}$$, and can be used to by $$\mathbb{I}$$ to contruct a Theory of Mind (ToM) of $$\mathbb{A}$$. I'll go into more details here in the next post.

Let's first check some boundary cases:
1. Any stochastic process with stable expectation, or in addition any predictable drift, has $$RGI=0$$ because there is at least one model, by definition, that reduces minimum prediction error to 0 as long as $$\mathbb{I}$$ can come up with it.
2. Any determinstic process has $$RGI=0$$ if $$\mathbb{I}$$ is smart enough.
3. What if you always include a model with uniform distribution over infinite interval ($$\sigma\to\infty$$)? This will always have the minimum normalized running sum of 0. Let's just call it pathological, no fair, biased interrogator, banned.
4. IF $$\mathbb{I}$$ is much smarter than $$\mathbb{A}$$, or written as $$\mathbb{I}>>\mathbb{A}$$, then $$\mathbb{I}$$ should have little trouble predicting $$\mathbb{A}$$'s behavior/response, therefore small $$RGI_{\mathbb{A}|\mathbb{I}}.
5. If $$\mathbb{I}<<\mathbb{A}$$, then $$\mathbb{A}$$ would have a good idea of $$\mathbb{I}$$'s model set, therefore "freedom of choice" (free-will discussions ensue...) whether/when to comply with or defy $$\mathbb{I}$$'s predictions. $$RGI_{\mathbb{A}|\mathbb{I}} can be any value $$\mathbb{A}$$ wants to register. Specifically, $$\mathbb{A}$$ can play dumb and show a low $$RGI_{\mathbb{A}|\mathbb{I}}. It's an interesting question what is required for an agent to make such choices.
6. If $$\mathbb{I}\approx\mathbb{A}$$, as when normal humans evaluate each other, none of the models would work perfectly but some would work well. $$RGI_{\mathbb{A}|\mathbb{I}} could be from meaningfully greater than 0 to some large but finite value. An interesting question is whether it's possible for a GI agent to have a model for predicting itself nearly perfectly. Maybe another post.

So it passes some quick boundary sanity checks. Now some observations:
1. An important class of predictive models is pattern recognition. To the extent that $$\mathbb{I}$$ can recognize patterns in $$\mathbb{A}$$'s behavior, the latter is unintelligent, uninteresting. The more converage of behavior by the recognized patterns, the less intelligent $$\mathbb{A}$$ is relative to $$\mathbb{I}$$. This is intuitive, and directly reflected in the prediction error term in (E1).
2. The models can be learning, including SI and GI.
3. Does this imply that GI is equivalent to unpredictability or complexity? I'd say yes in some sense, but the meaning of unpredictability needs some clarification beyond deterministic vs stochastic. I'll present that in another post.
4. One possible choice for $$w_i$$ is $$2^{-k(M_i)}$$, where $$k(M_i)$$ is the Kolmogorov Complexity of model $$M_i$$. This would be [Solomonoff's Lightsaber](https://www.alignmentforum.org/tag/solomonoff-induction), a quantitative version of Occam's Razor.
5. There seems no reason to think $$RGI$$ is bounded. This is consistent with the intuition that GI is not an optimization problem (unless in the AIXI "complete knowledge" sense). Also, intuitively, if $$\mathbb{A}$$'s intelligence is much higher than that of $$\mathbb{I}$$, it would appear simply unknowable to the latter. The existence of an upper bound is unknowable. For any GI agent, the best it can hope for is to do better, but never the best. Note that, although (E1) is written in argmin, it is about measuring GI rather than the learning process of either $$\mathbb{A}$$ or $$\mathbb{I}$$.
6. Cognitive dissonance can be quantified and measured as the minimum prediction error. If reducing the minimum is the objective function of GI, this could explain the deeply rooted and universal motivation, at least among humans and some animals (e.g., my dog) to reduce cognitive disonnance. GI agents are programmed to come up with models to reduce it, even if the model involves the Great Pumpkin in the Sky. However, it should be obvious, by examining human behavior, that it's one of multiple objective functions that a GI agent may <i>choose</i> (free will discussions ensue...) to use for particular circumstances. 
7. How to define what is predicted, therefore prediction error, is up to $$\mathbb{I}. For example, for evaluating image recognition of cats, it could be binary "yes cat"/"no cat", to shapes/positions of all cats, to name/status. Since we assume $$\mathbb{I} has no agenda w.r.t $$\mathbb{A}, this is fair. Importantly, this gives $$\mathbb{I} a lot of flexibility in quickly evaluating simpler models. Also note that coarse graining is a common tool, including in science. We can't predict very well individual molecule movements in a cup of water. So we try some macroscopic measurement. Evidently this enables us to cheat around our ignorance/incompetence.

But how do you achieve $$RGI>0$$?

You'll have to be smarter than $$\mathbb{I}$$ (<i>you're welcome</i>) in the sense that you can defy all of $$\mathbb{I}$$'s predictive models at least sometimes. The more often you do, the more intelligent you appear until in the limit you defy all predictions all the time. You are unknowable at this point. Welcome to Deityhood.

This is very much an interactive game where $$\mathbb{I}$$ tries to predict the agent while the latter tries to defy it. It's a well-defined Turing test.

### Solomonoff's Lightsaber

The word implementable above is a bit of exaggeration. It's not exactly implementable for any GI agent to exhaust all of its own predictive models, if only for the fact that its arsenal of models can change as it learns. Even if using a frame grab, the model set is likely to be too large to exhaust for any reasonably sophisticated intelligence. But perhaps we can get some help from examining how humans think.

Let's say you see a flying object, which is always UFO until identified. First you focus on predictive models in the visual domain, and further the flying subset. This sounds pretty vague but is a huge reduction from the set of all models. Then we use Occam's Razor to quickly evaluate a subset of minimal models, and then reduce them in the Solomonoff's induction sense -- flying straight? Constant speed? Constant shape/color? By the time you quickly eliminate a dozen or so such minimal models, you're freaking out a little.

Assume we have a library of all predictive models known to human. We divide them into a large but finite number of categories as a directed acyclic graph and asign each a complexity score. When faced with a subject, we choose a set of relevant categories and start evaluating the least complex ones. The minimum running sum of prediction error should generally descrease as we evaluate more complex models, and then increase again as the models get too complex for the subject. The local minimum problem could in principle be dealt with using common techniques such as random jump.

This is likely to be a gross over-simplification. The model space may be so high-dimensional that local gradient may be 0 for many models. Or maybe some variations of Kolmogorov's Complexity work better. But Solomonoff's Lightsaber is likely an essential tool.

Building "a library of all predictive models known to human" will probably never finish. But it can be massively parallelized and incremented, by category and complexity among other dimensions. For example, a decent library for image recognition could be built with known SI models.

As AI becomes more sohpisticated and approaches AGI, it's increasingly important to have a precise definition of GI and ways to quantify it. I hope the Agent-Interrogator Measurre presented here can be a starting point.
