# I Built an AI That Lands Rockets — Here's How You Can Too (From Zero to Hero in Reinforcement Learning)

*A weekend side project that taught me more about AI than any textbook ever did.*

![](https://raw.githubusercontent.com/fakemonk1/Reinforcement-Learning-Lunar_Lander/master/images/3.gif)
---

Last month, a buddy of mine sent me a link to a SpaceX landing compilation on YouTube. You know the ones — a 20-story rocket falling from the sky, firing its engines at the last second, and gently touching down on a tiny platform in the middle of the ocean. "Bet you can't make an AI do that," he texted.

Challenge accepted.

What started as a weekend dare turned into one of the most satisfying side projects I've ever built. I taught an AI agent to land a rocket on the moon — well, a simulated one — using **Reinforcement Learning**. And the best part? I went from knowing absolutely nothing about RL to having a working, optimized agent with an API and a dashboard in about two weekends.

This article is the tutorial I wish I had when I started. We'll go from zero to a fully trained lunar lander agent, step by step, with theory demystified along the way. By the end, you'll have:

- A deep understanding of core RL concepts
- Hands-on experience with Q-Learning, DQN, and PPO
- A trained agent that consistently lands a spacecraft
- A FastAPI backend serving your model
- A Streamlit dashboard tracking performance
- A recorded video of your agent in action

Let's go.

---

## Table of Contents

1. [What Even Is Reinforcement Learning?](#part-1-what-even-is-reinforcement-learning)
2. [Setting Up — Your First RL Environment](#part-2-setting-up--your-first-rl-environment)
3. [Q-Learning: Teaching AI with a Cheat Sheet](#part-3-q-learning-teaching-ai-with-a-cheat-sheet)
4. [Deep Q-Networks: When the Cheat Sheet Gets Too Big](#part-4-deep-q-networks-when-the-cheat-sheet-gets-too-big)
5. [The Mission: Landing on the Moon](#part-5-the-mission-landing-on-the-moon)
6. [Hyperparameter Tuning: The Art of Tweaking](#part-6-hyperparameter-tuning-the-art-of-tweaking)
7. [Building the API: Serving Your Agent](#part-7-building-the-api-serving-your-agent)
8. [The Dashboard: Telling the Story of Your Agent](#part-8-the-dashboard-telling-the-story-of-your-agent)
9. [Recording a Victory Lap](#part-9-recording-a-victory-lap)
10. [Wrapping Up](#wrapping-up)

---

## Part 1: What Even Is Reinforcement Learning?

Before writing a single line of code, let's build some real intuition. I'm going to spend time here because if you understand these concepts deeply, the rest of the tutorial will feel like a natural progression. If you skip this, the code will feel like magic — and magic is hard to debug.

### Where RL Fits in the AI Landscape

![](https://fr.mathworks.com/discovery/reinforcement-learning/_jcr_content/thumbnail.adapt.1200.medium.png/1601891958466.png)
You've probably heard of **supervised learning**: you give the algorithm a bunch of labeled examples ("this image is a cat", "this email is spam") and it learns to predict labels for new data. And maybe you've heard of **unsupervised learning**: you give it unlabeled data and it finds patterns on its own (like clustering customers into groups).

Reinforcement Learning is the **third pillar** of machine learning, and it's fundamentally different from both. Here's why:

- In supervised learning, a teacher gives you the correct answer for each input. "This is a cat." Period.
- In unsupervised learning, there's no feedback at all. You just look for structure in the data.
- In RL, there's **no correct answer** — but there's **feedback**. You try something, and the world tells you whether it was a good idea or a bad one. Then you try again.

Think about how a child learns to walk. Nobody gives the child a manual with the exact angle of each joint at each millisecond. Instead, the child tries to stand, falls, feels the pain (negative feedback), tries a slightly different approach, takes one step, feels the excitement (positive feedback), and repeats thousands of times. That's RL.

Now imagine you're training a puppy. You don't hand it a manual titled "How to Sit: A Comprehensive Guide." Instead, you let it try stuff. When it does something good, you give it a treat. When it does something bad... well, no treat. Over time, the puppy figures out which behaviors lead to treats and which don't.

That's Reinforcement Learning in a nutshell: **learning optimal behavior through trial and error, guided by rewards.**

> **Key insight for later:** The fact that there's no "right answer dataset" is what makes RL both powerful and tricky. The agent has to generate its own training data by interacting with the environment. This is fundamentally different from training a classifier on a CSV file.

### The Five Core Concepts — Your RL Vocabulary

There are five concepts you absolutely need to internalize. I'll explain each one in depth, because these are the building blocks for *everything* that follows.

![](https://www.artiba.org/Content/Images/components-of-reinforcement-learning.jpg)
---

**1. Agent — The Decision-Maker**

The agent is the entity that learns and makes decisions. It's the "brain" of your system.

![](https://miro.medium.com/v2/resize:fit:800/0*PRDfQwGlwD4JvrmA.gif)

In our project, the agent will be a lunar lander trying to land safely on the moon. But agents can be anything: a chess-playing program, a self-driving car's navigation system, a robot arm learning to pick up objects, or an algorithm deciding what ad to show you.

The key thing about the agent: **it doesn't start smart**. On day one, the agent is essentially a coin-flipper — making random decisions with no understanding of consequences. Through experience, it gradually builds a strategy (called a **policy**) that tells it what to do in each situation.

A **policy** is just a fancy word for "decision-making strategy." It can be as simple as a lookup table ("if in state X, do action Y") or as complex as a deep neural network. When we talk about "training an agent," we really mean "improving its policy."

---

**2. Environment — The World**

The environment is everything the agent interacts with. It's the "world" that the agent lives in.

![](https://miro.medium.com/v2/resize:fit:1400/1*jBbLJZYVlJlVAjs-c7z0Zw.gif)

In a video game, the environment is the game itself — the physics, the rules, the other characters. In robotics, the environment is the physical world. In our case, the environment is a physics simulation of the moon's surface, complete with gravity, friction, and fuel consumption.

Here's what makes the environment special from the agent's perspective: **the agent can't see the environment's internal code.** It can only observe what the environment shows it (the state) and see how the environment reacts to its actions (the next state and reward). It's like playing a game where you don't know the rules — you have to figure them out by playing.

The environment is the only part of an RL system that **the agent doesn't control**. Everything else — how it makes decisions, how it learns — is up to the agent's design.

---

**3. State (or Observation) — What the Agent Sees Right Now**

A state is a snapshot of the environment at a given moment. It's the information the agent has available to make its next decision.

Let's make this concrete with our lunar lander:

```
State = [x_position, y_position, x_velocity, y_velocity, 
         angle, angular_velocity, left_leg_touching, right_leg_touching]
```

That's 8 numbers. At any given moment, these 8 numbers tell the agent everything it needs to know: "I'm at position (0.3, 0.8), moving right at 0.1 m/s, tilted 5 degrees clockwise, and neither leg is touching the ground."

The quality and completeness of the state is crucial. If the state doesn't include enough information for the agent to make a good decision, the agent will struggle. Imagine trying to land a spacecraft but you can only see your altitude — not your horizontal position or speed. That would be a terrible state representation.

**State vs. Observation:** You'll see these terms used interchangeably in most tutorials. Technically, "state" means the complete internal description of the environment, while "observation" is what the agent actually gets to see (which might be partial). For our purposes, they mean the same thing.

---

**4. Action — What the Agent Can Do**

Actions are the choices available to the agent at each moment. In our lunar lander:

```
Action 0 = Do nothing (let gravity do its thing)
Action 1 = Fire left orientation engine
Action 2 = Fire main (bottom) engine
Action 3 = Fire right orientation engine
```

A critical distinction you'll encounter throughout this tutorial is between **discrete** and **continuous** action spaces:

- **Discrete actions** = a finite set of choices. Like a light switch — on or off. Our lander has 4 discrete choices. A chess agent chooses from a finite set of legal moves. A simple robot can go left, right, forward, or backward.

- **Continuous actions** = a value from a range. Like a volume knob — you can set it anywhere from 0 to 100. A self-driving car sets the steering angle to any value between -30° and +30°. A robotic arm sets each joint to a specific angle.

**Why does this matter?** Because the type of action space determines which algorithms you can use. Some algorithms (like DQN) only work with discrete actions. Others (like PPO) can handle both. We'll see this play out when we start coding.

---

**5. Reward — The Feedback Signal**

The reward is a number the environment gives the agent after each action. It tells the agent "how good or bad was that decision?"

This is deceptively simple but profoundly important. The reward is the **only teaching signal** the agent ever receives. It never gets told "you should have fired the left engine." It only gets told "+3" or "-10." The agent has to figure out from those numbers which actions were good.

Let's look at concrete reward examples for our lunar lander:

```
Landing safely on the pad        = +100 to +140 (depending on precision)
Landing outside the pad           = varies (less reward)
Crashing                          = -100
Each leg touching the ground      = +10 per leg  
Firing the main engine            = -0.3 per frame (fuel costs money!)
Firing side engines               = -0.03 per frame
Moving toward the landing pad     = small positive reward
Moving away from the landing pad  = small negative reward
```

See the subtlety? The reward signal encodes multiple objectives simultaneously: land safely (big positive), don't crash (big negative), conserve fuel (small negatives for engine use), and be accurate (bonus for landing on the pad, not next to it).

**The Reward Hypothesis** — This is a foundational idea in RL, articulated by [Richard Sutton](http://www.incompleteideas.net/ebook/the-book.html): *all goals can be described by the maximization of expected cumulative reward.* In other words, no matter what you want the agent to do, you can (in theory) express it as a reward function. Want it to land softly? Give high rewards for soft landings. Want it to conserve fuel? Penalize fuel use. The reward is how you *communicate your goals* to the agent.

> **A word of warning:** Designing good reward functions is one of the hardest problems in RL. If the reward is poorly designed, the agent will find loopholes you never imagined. There's a famous example where an agent learned to crash a racing game into a wall repeatedly because it collected more reward from minor pickups along the wall than from actually finishing the race. This is sometimes called "reward hacking."

---

### The Interaction Loop — How It All Connects

Now that we understand each piece, let's see how they work together. Every RL problem follows the same fundamental loop:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   ┌─────────┐    action     ┌──────────────┐   │
│   │  AGENT  │ ────────────► │ ENVIRONMENT  │   │
│   │         │               │              │   │
│   │ (brain) │ ◄──────────── │ (the world)  │   │
│   └─────────┘  state +      └──────────────┘   │
│                reward                           │
│                                                 │
│   This loop repeats at every TIMESTEP           │
│   until the EPISODE ends.                       │
└─────────────────────────────────────────────────┘
```

Step by step:

1. The environment presents a **state** to the agent (e.g., "you are at position X, moving at speed Y")
2. The agent looks at this state and chooses an **action** (e.g., "fire the main engine")
3. The environment takes that action, computes the physics, and returns two things: a **new state** and a **reward**
4. The agent uses this reward to update its internal strategy (its policy)
5. Go back to step 1 with the new state

A **timestep** (or "step") is one iteration of this loop — one decision, one action, one reward.

An **episode** is one complete run from start to finish. For our lander, one episode is one landing attempt. The episode ends either when the lander touches the ground (success or crash), when it flies off-screen, or when a maximum number of timesteps is reached.

During training, the agent plays thousands of episodes. Each episode is like one practice attempt — the agent tries to land, gets feedback, and hopefully does a little better next time.

### Cumulative Reward — Why the Long Game Matters

![](https://towardsdatascience.com/wp-content/uploads/2020/06/17TDcDII6iH9tKpMBH3enwA.png)
Here's a subtlety that trips up a lot of beginners: the agent doesn't just try to maximize the **immediate** reward. It tries to maximize the **cumulative reward** over the entire episode.

Why does this matter? Consider this scenario: the lander is hovering high above the pad. Firing the main engine costs fuel (-0.3 reward per frame). Doing nothing costs nothing (0 reward). If the agent only cared about the immediate reward, it would never fire its engines — because every engine use has a negative immediate reward!

But of course, doing nothing means eventually crashing (-100 reward). A smart agent understands that a small sacrifice now (spending fuel) leads to a much bigger reward later (safe landing). This is the concept of **delayed reward** — sometimes the best immediate action looks bad, but it pays off in the long run.

The mathematical way we handle this is through a **discount factor** (γ, gamma), which we'll explore in detail when we get to Q-Learning. The idea is: a reward now is worth more than the same reward in the future, but future rewards still matter.

Think of it like money: $100 today is worth more than $100 in a year (because of inflation, opportunity cost, etc.). But $100 in a year is still definitely worth something. The discount factor is like an "interest rate" that determines how much future rewards are worth relative to immediate ones.

### The Markov Property and MDP (Don't Panic — It's Simpler Than It Sounds)

You'll encounter the term **Markov Decision Process (MDP)** in every RL textbook. Let me demystify it.

The **Markov Property** says one simple thing: **the current state contains all the information the agent needs to make a good decision.** The agent doesn't need to remember what happened 10 steps ago, or how it got to its current state. The present state is sufficient.

Think about chess: if someone shows you a chessboard mid-game, you can figure out the best move without knowing the history of the game. The current position of all the pieces is all the information you need. That's the Markov Property.

Now, an **MDP** is just a formal mathematical framework that packages together all the components of an RL problem. Specifically, it defines:

- **S** — the set of all possible states (e.g., all possible lander positions and velocities)
- **A** — the set of all possible actions (e.g., fire left, fire right, fire main, do nothing)
- **P(s'|s,a)** — the transition function: if I'm in state s and take action a, what's the probability of ending up in state s'? (This captures the environment's physics. The agent usually doesn't know this function — it has to learn through experience.)
- **R(s,a,s')** — the reward function: how much reward do I get for transitioning from s to s' via action a?
- **γ** — the discount factor (how much future rewards matter)

When all five of these are defined, you have a well-specified RL problem. You don't need to derive equations from this — just know that when someone says "we model this as an MDP," they mean "we've clearly defined the states, actions, transitions, rewards, and discount factor."

> **Further reading:** For a brilliant visual introduction to MDPs, check out [this interactive article by Lilian Weng](https://lilianweng.github.io/posts/2018-02-19-rl-overview/). For a more academic deep dive, the [DeepMind x UCL RL Lecture Series](https://www.youtube.com/watch?v=2pWv7GOvuf0) is fantastic (and free on YouTube).

### Exploration vs. Exploitation — The Fundamental Dilemma

This is one of the most important concepts in RL, and it shows up everywhere — not just in AI.

Imagine you've just moved to a new city and you're looking for a good restaurant. You found one decent place on your first try. Now what?

- **Exploit**: Go back to that decent restaurant every night. You know it's okay. Safe bet.
- **Explore**: Try a new restaurant tonight. Maybe it's terrible. Maybe it's the best food you've ever had. You won't know until you try.

![](https://huggingface.co/blog/assets/63_deep_rl_intro/expexpltradeoff.jpg)

If you *only* exploit, you'll never find anything better. If you *only* explore, you'll waste time eating at terrible places even though you already know a good one.

RL agents face this exact same dilemma at every single timestep. Should the agent:

- **Explore** — try a new, possibly random action to discover something better?
- **Exploit** — use its current knowledge to take the best-known action?

The classic solution is called **epsilon-greedy** (ε-greedy):

```
Generate a random number between 0 and 1.
If it's less than ε → take a RANDOM action (explore)
If it's greater than ε → take the BEST KNOWN action (exploit)
```

The trick is how ε changes over time:

- **Start of training** (ε = 1.0): The agent explores 100% of the time. It knows nothing, so random exploration is the best strategy.
- **During training** (ε gradually decreases): The agent explores less and less as it learns. Why? Because it's building up knowledge about what works, so it should increasingly trust that knowledge.
- **End of training** (ε ≈ 0.01): The agent almost always exploits, but retains a tiny bit of exploration "just in case" there's something better it hasn't found yet.

This "start random, gradually become strategic" pattern is called **epsilon decay**, and you'll see it in every RL algorithm we implement.

> **Real-world analogy:** Think about how you learned your commute to work. The first week, you probably tried several different routes (exploration). Over time, you settled on the fastest one and stopped experimenting (exploitation). But occasionally, you might try a new route if there's construction — that's your "residual epsilon" keeping you from being too rigid.

### Types of RL Agents — Value, Policy, and Actor-Critic

There are three main philosophies for how an agent can make decisions. Understanding these now will help you see why we use different algorithms throughout this tutorial.

---

**Value-Based Agents** (what we'll start with)

The idea: learn the *value* of every state-action combination, then always pick the highest-value action.

Analogy: Imagine you're playing a board game and you have a cheat sheet that scores every possible move. "Moving to square A is worth +5 points. Moving to square B is worth +2 points." You'd always pick square A.

In RL, this "cheat sheet" is called a **Q-function** (Q for "Quality"). Q(state, action) tells you: "if I'm in this state and take this action, how much total reward can I expect to earn from here until the end of the episode?"

Algorithms: **Q-Learning** (the cheat sheet is a table), **DQN** (the cheat sheet is a neural network).

Limitation: Only works with **discrete** actions (finite choices). You can't score every possible value of a continuous action space.

---

**Policy-Based Agents**

The idea: instead of learning values, directly learn the **best action** for each state. The agent's brain is a function that takes a state and outputs an action (or a probability distribution over actions).

Analogy: Instead of a cheat sheet with scores, imagine a coach who just tells you "in this situation, do *that*." No scores, no reasoning — just direct instructions.

Algorithms: **REINFORCE**, **Policy Gradient methods**.

Advantage: Can handle continuous action spaces (because the output is a value, not a choice from a list). More flexible but harder to train.

---

**Actor-Critic Agents** (the best of both worlds)

The idea: use *two* components working together:
- The **Actor** — a policy network that decides what action to take
- The **Critic** — a value network that evaluates how good the actor's decisions are

Analogy: Imagine a musician (actor) performing on stage, with a music teacher (critic) in the audience. The musician plays, the teacher gives feedback ("that was great" or "you were flat in the chorus"), and the musician adjusts. Over time, the musician improves based on the teacher's feedback, and the teacher gets better at evaluating performances.

Algorithms: **PPO** (Proximal Policy Optimization), **A2C** (Advantage Actor-Critic), **SAC** (Soft Actor-Critic).

**PPO** is currently one of the most popular and robust RL algorithms. It's what OpenAI used to train the bots that beat pro Dota 2 players, and a variant was used in training ChatGPT through RLHF (Reinforcement Learning from Human Feedback). We'll use it in this tutorial as an alternative to DQN.

---

**Which one should you use?** Here's a practical rule of thumb for beginners:

| Situation | Recommended approach |
|-----------|---------------------|
| Discrete actions, simple environment | Q-Learning (table-based) |
| Discrete actions, complex environment | DQN (value-based with neural network) |
| Continuous actions | PPO or SAC (actor-critic) |
| Not sure / want a safe default | PPO (works well on almost everything) |

Don't worry about memorizing this. You'll develop intuition for it as we code through the examples.

### How RL Differs from Other Types of Machine Learning — A Summary

Let me crystallize the key differences, because understanding *what makes RL unique* will help you avoid common mistakes:

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|-------------------|----------------------|
| Training data | Labeled dataset (given upfront) | Generated by the agent through interaction |
| Feedback | Correct answer for each input | Reward signal (good/bad, not "correct answer") |
| Timing of feedback | Immediate (each example has a label) | Often delayed (good landing happens many steps after good decisions) |
| Goal | Minimize prediction error | Maximize cumulative reward |
| Data distribution | Fixed (you train on a static dataset) | Changing (as the agent improves, it visits different states) |
| Exploration | Not applicable | Critical — agent must discover good strategies |

The last row — about the data distribution changing — is what makes RL fundamentally harder than supervised learning. In supervised learning, your training data is fixed. In RL, the training data depends on the agent's current strategy, which changes as it learns. This creates all sorts of stability problems, which is why tricks like Experience Replay and Target Networks (more on those later) were invented.

> **Key insight:** If you're coming from a supervised learning background, the biggest mental shift is accepting that there's no dataset to download. The agent generates its own training data. The quality of that data depends on how the agent explores, which depends on its current policy, which depends on the training data... it's circular. Managing this circularity is what makes RL algorithm design so interesting (and sometimes so frustrating).

---

## Part 2: Setting Up — Your First RL Environment

Time to get our hands dirty. Fire up a Google Colab notebook (or your local Jupyter) and let's go.

### Installation

```python
# ============================================================
# STEP 0: Install all the libraries we'll need for this project
# ============================================================
!pip install gymnasium[box2d] stable-baselines3[extra] tensorboard matplotlib numpy torch
```

### Meeting CartPole: Your First Playground

Before we tackle the moon, let's start with something simpler. CartPole is the "Hello World" of RL — a pole balanced on a cart that you need to keep upright.

```python
# ============================================================
# STEP 1: Create and explore a Gymnasium environment
# We start with CartPole-v1 to understand the basics
# ============================================================
import gymnasium as gym

# Create the environment
env = gym.make("CartPole-v1")

# ----------------------------------------------------------
# Let's explore what our agent can SEE (observation space)
# and what it can DO (action space)
# ----------------------------------------------------------
print("=== Observation Space ===")
print(f"Type: {env.observation_space}")
print(f"Shape: {env.observation_space.shape}")
print(f"Sample observation: {env.observation_space.sample()}")
# CartPole observations are 4 continuous values:
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

print("\n=== Action Space ===")
print(f"Type: {env.action_space}")
print(f"Number of actions: {env.action_space.n}")
print(f"Sample action: {env.action_space.sample()}")
# CartPole has 2 discrete actions:
# 0 = push cart left, 1 = push cart right
```

**Key insight:** Notice the types. The observation space is `Box` (continuous values — infinite possibilities) while the action space is `Discrete` (a finite set of choices). This distinction will determine which algorithms we can use later.

### Running a Random Agent

Let's see how a completely clueless agent performs — one that just takes random actions.

```python
# ============================================================
# STEP 2: Run a random agent for 10 episodes
# This establishes a BASELINE — the worst possible performance
# ============================================================

NUM_EPISODES = 10

for episode in range(NUM_EPISODES):
    # Reset the environment at the start of each episode
    # This gives us the initial observation
    observation, info = env.reset()
    
    total_reward = 0
    terminated = False
    truncated = False
    step_count = 0
    
    # Run until the episode ends
    while not terminated and not truncated:
        # Choose a random action (our agent is clueless)
        action = env.action_space.sample()
        
        # Take the action and observe the result
        # env.step() returns 5 values:
        #   observation: the new state
        #   reward: how good/bad that action was
        #   terminated: did the episode end naturally? (pole fell)
        #   truncated: did we hit a time limit?
        #   info: extra debug information
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
    
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.1f} "
          f"({step_count} steps)")

# Always clean up
env.close()
```

**Important distinction:** `terminated` means the episode ended because of the **environment's own rules** — the pole fell over, the lander crashed, or the lander landed successfully. It's a "natural" ending. `truncated` means an **external limit** was hit — typically a maximum number of timesteps. It's an "artificial" ending imposed to prevent episodes from running forever.

Why does this matter? Because they mean different things for learning. When an episode is *terminated*, the agent is in a genuine end state — there's no future. When it's *truncated*, the agent was in the middle of doing something; the future was cut short artificially. Some algorithms handle these differently (for instance, when computing the target Q-value for the last step, a terminated state has no future value, but a truncated state technically does — it was just cut off). Always track both.

You should see rewards around 20-40. That's terrible. A good agent can score 500 (the maximum). Let's fix that.

---

## Part 3: Q-Learning: Teaching AI with a Cheat Sheet

### The Concept — A Mental Model

Before we look at any formula, let's build an intuition with a simple analogy.

Imagine you're new in a city, and every morning you need to get to work. You have several routes, and each route has unpredictable traffic. On day 1, you pick a random route. It takes 45 minutes. On day 2, you try another — 30 minutes. On day 3, you try a third — 55 minutes.

After a few weeks, you've built up a mental model: "Route B is usually the best, Route A is second-best, Route C is terrible." You didn't need anyone to tell you this — you learned it from experience. And that mental model is essentially a **Q-table**.

A Q-table is a giant spreadsheet where:
- Each **row** is a situation you might find yourself in (a state)
- Each **column** is an action you could take
- Each **cell** contains a number that estimates how good it is to take that action in that situation

Here's what a simple Q-table might look like:

```
                    Action 0    Action 1    Action 2    Action 3
                    (Left)      (Down)      (Right)     (Up)
State 0  (start)    0.12        0.05        0.31        0.08
State 1             0.00        0.42        0.15        0.00
State 2             0.18        0.00        0.00        0.55
...                 ...         ...         ...         ...
State 15 (goal)     0.00        0.00        0.00        0.00
```

The "Q" stands for **Quality**. Q(state, action) answers the question: "If I'm in this state and take this action, then play optimally from here on, how much total reward can I expect?"

If you had a perfect Q-table, the optimal strategy would be trivially simple: in every state, just pick the action with the highest Q-value. The challenge is **building** that table from scratch, through trial and error, with no teacher to tell you the right answers.

### The Bellman Equation — Your New Best Friend

The Q-table is updated using the **Bellman equation**. This is the single most important equation in RL, so let's take our time with it.

Here's the formula:

```
New Q(s,a) = Old Q(s,a) + α * [reward + γ * max(Q(s',a')) - Old Q(s,a)]
```

I know that looks intimidating. Let me break it down piece by piece using a concrete scenario.

Imagine our agent is in state 5 and takes action "go right." It receives a reward of +1 and lands in state 6.

**`Old Q(s,a)`** = the current value in our table for (state 5, go right). Let's say it's **0.3**. This is our current "guess" for how good this action is.

**`reward`** = the immediate reward we just got: **+1**. This is new information the environment gave us.

**`max(Q(s',a'))`** = the highest Q-value in the next state (state 6). We look at all four actions in state 6 and find the maximum. Let's say it's **0.8**. This represents "the best we expect to do from state 6 onward."

**`γ` (gamma, the discount factor)** = let's say **0.95**. This is a number between 0 and 1 that determines how much we care about future rewards.
- γ = 0: "I only care about immediate rewards" (very short-sighted)
- γ = 0.5: "Future rewards are worth half of immediate rewards"
- γ = 0.99: "Future rewards are almost as important as immediate ones" (very far-sighted)
- γ = 1: "Future rewards are equally important" (can be unstable)

So `γ * max(Q(s',a'))` = 0.95 × 0.8 = **0.76**. This is the discounted value of the future.

**`reward + γ * max(Q(s',a'))`** = 1 + 0.76 = **1.76**. This is the **actual experienced return** — "what really happened + what we expect from here."

**`Old Q(s,a)`** = **0.3**. This was our **old prediction** — "what we thought would happen."

**The difference** = 1.76 - 0.3 = **1.46**. This is the **TD error** (Temporal Difference error) — it measures how surprised we are. A big positive error means "things turned out better than expected." A negative error means "things were worse than expected."

**`α` (alpha, the learning rate)** = let's say **0.8**. This controls how much we adjust our estimate. Think of it as "how quickly we change our mind."

**Final update:** New Q = 0.3 + 0.8 × 1.46 = 0.3 + 1.168 = **1.468**

So we update our table: Q(state 5, go right) goes from 0.3 to 1.468. We're more confident now that "going right from state 5" is a good idea.

**In plain English, the Bellman equation says:** *"Adjust your current guess a little bit toward what actually happened (the reward you got plus the best you expect to do from here on)."*

Over thousands of episodes, these small adjustments converge. The Q-table gradually fills up with accurate values, and the agent's strategy improves.

> **Visual explanation:** [This YouTube video](https://www.youtube.com/watch?v=qhRNvCVVJaA) walks through the Bellman equation with animations. Highly recommended if the math still feels abstract.

> **Deeper dive:** If you want to understand the mathematical foundations of why this converges, the [Sutton & Barto textbook (Chapter 6)](http://www.incompleteideas.net/ebook/the-book.html) is the gold standard — and it's free online.

### Why FrozenLake and Not Something Cooler?

We'll use **FrozenLake-v1** for our Q-Learning implementation. It's a 4×4 grid where an agent must navigate from the start tile to the goal tile without falling into holes. It has exactly **16 states** and **4 actions**, which means our Q-table is a tiny 16×4 grid — small enough to visualize and understand completely.

Why not jump straight to something flashier like our lunar lander? Because the lander has a **continuous** observation space — its state values (position, velocity, angle) can be any decimal number, meaning there are essentially infinite possible states. You can't build a table with infinite rows. We need a finite-state environment to understand Q-Learning properly, and then we'll graduate to neural networks (DQN) for continuous spaces.

### Implementation: FrozenLake

```python
# ============================================================
# STEP 3: Implement Q-Learning from scratch on FrozenLake-v1
# ============================================================
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 3.1 Create the environment and initialize the Q-table
# FrozenLake has 16 states (4x4 grid) and 4 actions (left, down, right, up)
# ----------------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=True)

# Get dimensions dynamically — never hardcode these!
n_states = env.observation_space.n    # 16
n_actions = env.action_space.n        # 4

print(f"States: {n_states}, Actions: {n_actions}")

# Initialize Q-table with zeros
# Shape: (16 states, 4 actions) — one Q-value per state-action pair
q_table = np.zeros((n_states, n_actions))
print(f"Q-table shape: {q_table.shape}")
print(f"Initial Q-table (all zeros):\n{q_table}")
```

```python
# ----------------------------------------------------------
# 3.2 Define hyperparameters
# These control HOW the agent learns
# ----------------------------------------------------------
LEARNING_RATE = 0.8        # α — how quickly we update Q-values
DISCOUNT_FACTOR = 0.95     # γ — how much we value future rewards
EPSILON = 1.0              # Starting exploration rate (100% random)
EPSILON_DECAY = 0.001      # How much epsilon decreases per episode
MIN_EPSILON = 0.01         # Never go below this (always explore a little)
NUM_EPISODES = 10000       # Number of training episodes
```

```python
# ----------------------------------------------------------
# 3.3 Training loop — the heart of Q-Learning
# ----------------------------------------------------------
rewards_per_episode = []

for episode in range(NUM_EPISODES):
    # Reset environment and get initial state
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        # ----- ACTION SELECTION: Epsilon-Greedy Strategy -----
        # Generate a random number between 0 and 1
        if np.random.random() < EPSILON:
            # EXPLORE: take a random action
            action = env.action_space.sample()
        else:
            # EXPLOIT: pick the best known action for this state
            action = np.argmax(q_table[state, :])
        
        # ----- TAKE THE ACTION -----
        new_state, reward, terminated, truncated, info = env.step(action)
        
        # ----- UPDATE THE Q-TABLE (Bellman Equation) -----
        # Best Q-value achievable from the next state
        max_future_q = np.max(q_table[new_state, :])
        
        # Current Q-value for the (state, action) pair we just used
        current_q = q_table[state, action]
        
        # The Bellman update
        q_table[state, action] = current_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * max_future_q - current_q
        )
        
        # Move to the next state
        state = new_state
        total_reward += reward
    
    # ----- DECAY EPSILON -----
    # Gradually shift from exploration to exploitation
    EPSILON = max(MIN_EPSILON, EPSILON - EPSILON_DECAY)
    
    rewards_per_episode.append(total_reward)

print("Training complete!")
print(f"Final epsilon: {EPSILON:.4f}")
```

```python
# ----------------------------------------------------------
# 3.4 Visualize the training progress
# ----------------------------------------------------------
# Calculate rolling average (window of 100 episodes)
window = 100
rolling_avg = [np.mean(rewards_per_episode[max(0, i-window):i+1]) 
               for i in range(len(rewards_per_episode))]

plt.figure(figsize=(12, 5))
plt.plot(rolling_avg, color='#2196F3', linewidth=2)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Average Reward (last 100 episodes)', fontsize=12)
plt.title('Q-Learning Training Progress on FrozenLake', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

```python
# ----------------------------------------------------------
# 3.5 Evaluate the trained agent
# IMPORTANT: During evaluation, we NEVER explore (no epsilon)
# We only exploit the learned Q-table
# ----------------------------------------------------------
NUM_EVAL_EPISODES = 100
total_wins = 0

for episode in range(NUM_EVAL_EPISODES):
    state, info = env.reset()
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        # Always pick the best action — no randomness!
        action = np.argmax(q_table[state, :])
        state, reward, terminated, truncated, info = env.step(action)
    
    # In FrozenLake, reward is 1.0 only if you reached the goal
    if reward == 1.0:
        total_wins += 1

win_rate = total_wins / NUM_EVAL_EPISODES * 100
print(f"Win rate over {NUM_EVAL_EPISODES} episodes: {win_rate:.1f}%")

env.close()
```

**A note on evaluation:** We keep the evaluation loop completely separate from training. The Q-table is frozen — no more updates. This gives us an honest measure of what the agent actually learned.

You should see a win rate around 70-75%. Not bad for a simple table! But what happens when the environment is more complex?

---

## Part 4: Deep Q-Networks: When the Cheat Sheet Gets Too Big

### The Problem with Q-Tables — Why We Need Something Better

FrozenLake has 16 states. That's a 16×4 table — 64 cells. Easy to store, easy to look up, easy to understand.

But CartPole has **continuous** states. The cart position can be 0.01, 0.011, 0.0115, or any number in between. The velocity can be any real number. The pole angle can be any value. Multiply all these possibilities together and you get... infinity. You can't build a table with infinite rows.

Even if you tried to discretize the space (say, round position to the nearest 0.1), you'd end up with a table so large it would take forever to fill and most cells would never be visited. This is called the **curse of dimensionality** — as the number of dimensions in your state space increases, the number of possible states explodes exponentially.

And our lunar lander? It has 8 continuous state variables. That's even worse.

We need a way to *generalize* — to make good predictions about states we've never seen before, based on states that are similar. And that's exactly what neural networks do.

### The Core Idea: Replace the Table with a Neural Network

Instead of looking up Q-values in a table, we **train a neural network** to predict them. Here's the shift in thinking:

```
Q-Table approach:      Q-table[state_5][action_right] → look up → 1.468
Neural Network approach: neural_net(state_vector) → predict → [0.3, 1.468, 0.1, 0.7]
                                                                 ↑ Q-value for each action
```

The neural network takes a state as input (a vector of numbers, like [cart_position, cart_velocity, pole_angle, angular_velocity]) and outputs one Q-value per possible action.

**Why does this solve the infinity problem?** Because neural networks are great at interpolation. If the network has seen state [0.1, 0.5, 0.02, -0.1] and learned that firing right is good, it can make a reasonable prediction for the very similar state [0.11, 0.49, 0.02, -0.1] — even if it's never seen that exact state before. A Q-table couldn't do this; if it hadn't visited that exact state, the corresponding row would still be all zeros.

This combination of Q-Learning + neural networks is the **Deep Q-Network (DQN)**, and it's one of the most important breakthroughs in AI history. In 2015, DeepMind used DQN to play Atari games (Breakout, Space Invaders, Pong) at superhuman level — the same algorithm, with no game-specific modifications, just raw pixels as input. The paper ([Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236), Mnih et al., 2015) is a landmark in the field.

### But Wait — Two Big Problems

If you just naively replace the Q-table with a neural network and train it the same way, it doesn't work. The training is wildly unstable and often diverges (the agent gets worse over time, not better). There are two reasons for this, and DQN introduces two clever solutions:

---

#### Problem 1: Correlated Training Data

In normal supervised learning (like training an image classifier), your training data is a fixed, shuffled dataset. Each training batch contains a random mix of cats, dogs, birds, etc. This randomness is important for stable training.

But in naive RL, the agent's training data is a sequence of consecutive experiences: step 1, step 2, step 3, step 4... These are highly correlated — step 2 is very similar to step 1 (the cart only moved a tiny bit). It's like training an image classifier where the first 1,000 images are all cats, the next 1,000 are all dogs, then 1,000 birds. The network would keep forgetting what it learned about cats by the time it's done with dogs.

**Solution: Experience Replay**

Instead of learning from experiences one-by-one in order, the agent stores all its experiences in a large **replay buffer** (think of it as a memory bank). When it's time to learn, it randomly samples a batch of experiences from this buffer.

```
REPLAY BUFFER (stores recent experiences):
┌──────────────────────────────────────────────┐
│ Experience 847: (state, action, reward, next_state) │
│ Experience 231: (state, action, reward, next_state) │
│ Experience 1502: (state, action, reward, next_state) │
│ ...thousands more...                                 │
│                                                      │
│ → Randomly sample a batch of 64 for training         │
└──────────────────────────────────────────────┘
```

This random sampling breaks the temporal correlations. The batch might contain an experience from 500 steps ago mixed with one from 10 steps ago and one from 2,000 steps ago. Just like shuffling a dataset in supervised learning, this makes the training much more stable.

**Bonus:** it also makes the agent more data-efficient. Each experience can be sampled multiple times, so the agent gets more learning out of each interaction with the environment.

> **Analogy:** Think about studying for an exam. If you study chapter 1 on Monday, chapter 2 on Tuesday, chapter 3 on Wednesday, you'll probably forget chapter 1 by Thursday. But if every day you randomly review flashcards from *all* chapters, you'll retain everything much better. That's Experience Replay.

---

#### Problem 2: Moving Targets

In Q-Learning with a table, the update formula uses the same table for both sides of the equation: we're updating Q(s,a) toward a target that includes max Q(s', a'). When the table is updated for one state, it doesn't affect the values of other states.

But with a neural network, updating the weights to improve the prediction for one state **changes the predictions for all states** (because the weights are shared). So when you update the network to make Q(s,a) more accurate, you're simultaneously changing the target max Q(s', a') — because the same network computes both!

This is like trying to hit a target that moves every time you take a shot. It's extremely unstable.

**Solution: Target Network**

The solution is beautifully simple: maintain **two copies** of the neural network.

1. **Policy Network** (the "student"): This is the network we're actively training. It makes decisions and gets updated every step.
2. **Target Network** (the "reference"): This is a frozen copy that provides stable targets for training. It's only updated periodically (e.g., every 1,000 steps) by copying the weights from the policy network.

```
TRAINING STEP:
1. Policy network predicts Q(s,a) for the batch        ← this gets updated
2. Target network predicts max Q(s',a') for the batch  ← this stays fixed
3. Compute loss: (policy prediction - target)²
4. Update ONLY the policy network's weights
5. Every N steps: copy policy weights → target network
```

Because the target network is frozen most of the time, the targets don't shift with every update. It's like having a stable reference point to aim at. Then, every N steps, the target network "catches up" to the policy network, and we get a new (better) reference point.

> **Analogy:** Imagine you're learning to throw darts at a bullseye. With a single network, the bullseye moves after every throw — impossible to improve. With a target network, the bullseye stays fixed for 100 throws, giving you time to adjust your aim. Then it moves to a new position, and you adjust again.

---

### Quick Primer: Just Enough PyTorch

If you've never used PyTorch, here's all you need to know for this section:

- **Tensors** are PyTorch's equivalent of NumPy arrays, but they can run on GPUs and track gradients for backpropagation.
- **`nn.Module`** is the base class for all neural network layers and models.
- **`nn.Linear(in, out)`** is a fully connected layer that maps `in` inputs to `out` outputs.
- **`F.relu(x)`** is an activation function that replaces negative values with zero (introduces non-linearity).
- **`optim.Adam`** is an optimizer that adjusts the network's weights based on the loss.
- **`loss.backward()`** computes gradients, and **`optimizer.step()`** applies them.

If you want to go deeper, the [PyTorch 60-Minute Blitz tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) is an excellent resource. But for now, the code comments should be enough to follow along.

### Manual DQN Implementation with PyTorch

Let's build a DQN from scratch to understand every moving piece. This is the most code-heavy section, but each line is commented.

```python
# ============================================================
# STEP 4: Implement DQN manually with PyTorch on CartPole-v1
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 4.1 Define the DQN network architecture
# This neural network replaces our Q-table.
# Input: state (4 values for CartPole)
# Output: Q-value for each action (2 for CartPole)
# ----------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Three fully-connected layers
        self.fc1 = nn.Linear(state_size, 128)   # Input → 128 neurons
        self.fc2 = nn.Linear(128, 128)           # 128 → 128 neurons
        self.fc3 = nn.Linear(128, action_size)   # 128 → Output (one Q-value per action)
    
    def forward(self, x):
        # ReLU activation between layers (introduces non-linearity)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation on output (Q-values can be any number)


# ----------------------------------------------------------
# 4.2 Define the Experience Replay Buffer
# Stores transitions and allows random sampling
# ----------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        # deque automatically discards oldest items when full
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a single transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of transitions"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
```

```python
# ----------------------------------------------------------
# 4.3 DQN Training Loop
# This is the most complex part — read the comments carefully!
# ----------------------------------------------------------

# Environment setup
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]  # 4
action_size = env.action_space.n              # 2

# Create two networks: policy (for decisions) and target (for stable targets)
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())  # Start with same weights
target_net.eval()  # Target network is never trained directly

# Optimizer and memory
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayBuffer(capacity=10000)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99            # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995   # Multiplicative decay per episode
TARGET_UPDATE = 10       # Update target network every N episodes
NUM_EPISODES = 300

epsilon = EPSILON_START
episode_rewards = []

for episode in range(NUM_EPISODES):
    state, info = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        # ----- ACTION SELECTION (Epsilon-Greedy) -----
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():  # Don't compute gradients for action selection
                q_values = policy_net(state)
                action = q_values.argmax().item()
        
        # ----- EXECUTE ACTION -----
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = torch.FloatTensor(next_state)
        done = terminated or truncated
        
        # ----- STORE TRANSITION IN REPLAY BUFFER -----
        memory.push(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        # ----- LEARN FROM A RANDOM BATCH -----
        if len(memory) >= BATCH_SIZE:
            # Sample a random batch from memory (Experience Replay!)
            batch = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.stack(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.stack(next_states)
            dones = torch.FloatTensor(dones)
            
            # Current Q-values: what does our policy network predict
            # for the actions we actually took?
            # .gather(1, actions) selects the Q-value for the chosen action
            current_q = policy_net(states).gather(1, actions).squeeze()
            
            # Target Q-values: what does the TARGET network say
            # is the best Q-value in the next state?
            # (This is the "stable target" trick!)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                # If the episode is done, there's no future reward
                target_q = rewards + GAMMA * max_next_q * (1 - dones)
            
            # Compute loss and update
            loss = F.mse_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # ----- UPDATE TARGET NETWORK periodically -----
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    # ----- DECAY EPSILON -----
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    episode_rewards.append(total_reward)
    
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode+1}/{NUM_EPISODES} | "
              f"Avg Reward (last 50): {avg_reward:.1f} | "
              f"Epsilon: {epsilon:.3f}")

env.close()
```

```python
# ----------------------------------------------------------
# 4.4 Visualize DQN training
# ----------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(episode_rewards, alpha=0.3, color='gray', label='Episode Reward')
rolling = [np.mean(episode_rewards[max(0,i-50):i+1]) for i in range(len(episode_rewards))]
plt.plot(rolling, color='#E91E63', linewidth=2, label='Rolling Average (50)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress on CartPole-v1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### The Lazy Way: Stable-Baselines3

Now, here's the kicker. Everything we just built in ~100 lines? Stable-Baselines3 does it in **5 lines**. That's the power of a production-ready RL library.

```python
# ============================================================
# STEP 5: Do the exact same thing with Stable-Baselines3
# ============================================================
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Create, train, evaluate, save — that's it!
model = DQN(
    "MlpPolicy",            # Multi-layer perceptron policy
    env,
    verbose=1,
    tensorboard_log="./logs/"  # Enable TensorBoard logging
)

# Train for 25,000 timesteps (not episodes!)
# total_timesteps = total number of interactions with the environment
model.learn(total_timesteps=25000)

# Evaluate on 100 episodes for a reliable metric
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"\nEvaluation: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# Save the model (creates a .zip file)
model.save("dqn_cartpole")
print("Model saved as dqn_cartpole.zip")

env.close()
```

Five lines of real logic. That manual implementation we sweated over? It was worth it — you now understand what's happening under the hood. But from here on, we'll use SB3 for everything.

> **Why both?** Understanding the internals makes you a better practitioner. When SB3 does something unexpected, you'll know why. But for production work, always use battle-tested libraries.

---

## Part 5: The Mission: Landing on the Moon

Alright, warm-up's over. Time for the real deal.

### Why LunarLander Is Harder Than CartPole

CartPole has 4 state variables and 2 actions. The physics are simple — a cart sliding left and right on a track. The success criterion is easy: just keep the pole from falling for 500 steps.

LunarLander is a significant step up in complexity:

- **8 state variables** instead of 4 — there's more information to process, meaning the agent needs to learn more complex relationships between its observations and its actions.
- **4 actions** instead of 2 — more choices at each timestep means a larger search space. With 2 actions, there's a 50% chance of guessing right. With 4, it's 25%.
- **Multi-objective rewards** — the agent has to balance multiple competing goals simultaneously (land accurately, land softly, don't crash, conserve fuel). This is much harder than CartPole's single objective (stay upright).
- **Sparse reward structure** — the big rewards (successful landing or crash) only happen at the very end of the episode, after hundreds of small decisions. The agent has to figure out which of those hundreds of decisions actually mattered. This is the **credit assignment problem** — one of the hardest challenges in RL.
- **More realistic physics** — gravity, momentum, and rotational inertia make the control problem genuinely difficult. A slight miscorrection early can cascade into an unrecoverable spin later.

This is where our foundations pay off. We understand environments, states, actions, rewards, and the training loop. Now we apply that knowledge to a harder problem.

### Meet LunarLander-v3

This is where my weekend dare really started. The environment simulates a spacecraft trying to land between two flags on the moon's surface. The physics are realistic — gravity, inertia, fuel consumption, the works.

```python
# ============================================================
# STEP 6: Explore the LunarLander-v3 environment
# Understanding your environment BEFORE training is crucial
# ============================================================
import gymnasium as gym

env = gym.make("LunarLander-v3")

# ----------------------------------------------------------
# What can the agent SEE?
# ----------------------------------------------------------
print("=== Observation Space ===")
print(f"Type: {env.observation_space}")
print(f"Shape: {env.observation_space.shape}")
print(f"Low bounds:  {env.observation_space.low}")
print(f"High bounds: {env.observation_space.high}")
# 8 continuous values:
# [x_position, y_position, x_velocity, y_velocity,
#  angle, angular_velocity, left_leg_contact, right_leg_contact]

print("\n=== Action Space ===")
print(f"Type: {env.action_space}")
print(f"Number of actions: {env.action_space.n}")
# 4 discrete actions:
# 0 = do nothing
# 1 = fire left engine
# 2 = fire main engine  
# 3 = fire right engine

env.close()
```

```python
# ----------------------------------------------------------
# Run a random agent to establish a BASELINE
# This tells us "how bad is doing nothing intelligent?"
# ----------------------------------------------------------
env = gym.make("LunarLander-v3")

baseline_rewards = []
for episode in range(50):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    baseline_rewards.append(total_reward)

print(f"Random Agent Baseline:")
print(f"  Mean Reward: {np.mean(baseline_rewards):.2f}")
print(f"  Std Reward:  {np.std(baseline_rewards):.2f}")
print(f"  Min:  {np.min(baseline_rewards):.2f}")
print(f"  Max:  {np.max(baseline_rewards):.2f}")

env.close()
```

You'll see the random agent scoring around **-200 to -100**. Our target? **+200 consistently** — the threshold for a successful landing.

### Training a First Agent (Default Parameters)

Let's start with a baseline model — default hyperparameters, no optimization. The goal is to get a working pipeline first.

```python
# ============================================================
# STEP 7: Train a baseline DQN agent on LunarLander-v3
# We use DEFAULT parameters to establish a starting point
# ============================================================
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

env = gym.make("LunarLander-v3")

# ----------------------------------------------------------
# Train with defaults — we want to see where we start
# The action space is DISCRETE, so DQN is a good fit
# ----------------------------------------------------------
baseline_model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

print("Training baseline model (this may take a few minutes)...")
baseline_model.learn(total_timesteps=100000)

# ----------------------------------------------------------
# Evaluate on 100 episodes for a reliable metric
# IMPORTANT: A single episode tells us nothing — we need
# the AVERAGE over many episodes to be meaningful
# ----------------------------------------------------------
mean_reward, std_reward = evaluate_policy(
    baseline_model, env, n_eval_episodes=100
)

print(f"\n{'='*50}")
print(f"Baseline DQN Performance:")
print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"{'='*50}")

# Save it — always save your models!
baseline_model.save("lunar_lander_baseline")
print("Baseline model saved.")

env.close()
```

This baseline will probably score somewhere around 50-150. Decent, but not reliable for a successful landing. Time to optimize.

---

## Part 6: Hyperparameter Tuning: The Art of Tweaking

This is where the real engineering happens. Hyperparameter tuning is less about magic and more about **methodical experimentation**. If the training part felt like science, this part feels more like cooking — you know the ingredients, but the exact proportions make the difference between a good dish and a great one.

### What Are Hyperparameters (And Why Can't the Algorithm Set Them Itself)?

In machine learning, **parameters** are what the model learns automatically (like the weights in a neural network). **Hyperparameters** are settings that *you* the human must choose *before* training starts. The algorithm can't figure these out on its own — they control *how* the learning process works.

Think of it this way: if the algorithm is a student, the hyperparameters are the study habits. The student's knowledge (parameters) improves through studying, but the study habits (hyperparameters) — how long to study, when to take breaks, how many flashcards per session — are set by someone else.

### The Golden Rule

**Change one thing at a time.** If you change three parameters simultaneously and performance improves, you don't know which change helped. Maybe one helped a lot, one did nothing, and one actually hurt. Be scientific: isolate variables, run controlled experiments, document everything.

### Key Hyperparameters — What Each One Actually Does

Here's each important hyperparameter explained in depth:

**`learning_rate` (α)** — How much the network's weights change with each update. Too high = the agent overreacts to each new experience (jumps around wildly, never converges). Too low = the agent barely changes its mind (learns extremely slowly, might get stuck). Think of it like the step size when walking toward a target: too large and you overshoot, too small and you never arrive.

**`gamma` (γ, discount factor)** — How much the agent cares about future rewards vs. immediate ones. At 0.99, a reward 100 steps in the future is worth 0.99^100 ≈ 0.37 of its face value. At 0.95, that same reward is worth 0.95^100 ≈ 0.006 — almost nothing. For LunarLander, where the big reward comes at the end, a high gamma (0.99-0.999) is usually better because you want the agent to plan ahead.

**`exploration_fraction`** — What fraction of total training time is spent decreasing epsilon from 1.0 to its final value. At 0.1, the agent finishes exploring after 10% of training. At 0.3, it takes longer to transition from exploration to exploitation. Longer exploration can help discover better strategies but delays convergence.

**`exploration_final_eps`** — The minimum epsilon value after exploration is done. At 0.05, the agent still takes random actions 5% of the time. Lower values (0.01-0.02) mean more exploitation, which is usually better once the agent has learned a lot.

**`batch_size`** — How many experiences are sampled from the replay buffer for each training step. Larger batches give more stable gradient estimates (less noise in the updates) but require more memory and computation. Typical values: 32, 64, 128.

**`buffer_size`** — How many experiences the replay buffer stores. A larger buffer means more diverse training data (the agent can learn from experiences far in the past). But too large and you dilute recent, more relevant experiences with old, possibly irrelevant ones.

**`target_update_interval`** — How many steps between copying the policy network's weights to the target network. More frequent updates = the target network tracks the policy more closely (less stable but more current). Less frequent = more stable targets (but might be outdated).

| Parameter | What it does | Default | Try |
|-----------|-------------|---------|-----|
| `learning_rate` | Speed of weight updates | 1e-4 | 1e-3, 5e-4, 1e-4, 5e-5 |
| `gamma` | Discount factor for future rewards | 0.99 | 0.99, 0.999, 0.95 |
| `exploration_fraction` | % of training spent exploring | 0.1 | 0.1, 0.2, 0.3 |
| `exploration_final_eps` | Minimum exploration rate | 0.05 | 0.01, 0.05, 0.1 |
| `batch_size` | Samples per training step | 32 | 32, 64, 128 |
| `buffer_size` | Replay buffer capacity | 1M | 50K, 100K, 500K |
| `target_update_interval` | Steps between target net updates | 10K | 5K, 10K, 20K |

### Experiment Tracking with TensorBoard

Professional ML engineers don't just train models — they **track experiments**. TensorBoard lets you compare different runs visually.

```python
# ============================================================
# STEP 8: Systematic hyperparameter experiments
# We document each experiment and compare results
# ============================================================
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import json

# ----------------------------------------------------------
# Define experiments: each modifies ONE parameter from baseline
# ----------------------------------------------------------
experiments = {
    "exp1_lr_high": {
        "learning_rate": 1e-3,
        "description": "Higher learning rate"
    },
    "exp2_lr_low": {
        "learning_rate": 5e-5,
        "description": "Lower learning rate"
    },
    "exp3_gamma_high": {
        "gamma": 0.999,
        "description": "Higher discount factor (care more about future)"
    },
    "exp4_explore_more": {
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.02,
        "description": "More exploration time, lower final epsilon"
    },
    "exp5_bigger_batch": {
        "batch_size": 128,
        "buffer_size": 100000,
        "description": "Larger batch and buffer"
    },
}

results = {}
best_reward = -float('inf')
best_experiment = None
best_model = None

for exp_name, params in experiments.items():
    print(f"\n{'='*60}")
    print(f"Running: {exp_name} — {params.pop('description')}")
    print(f"Parameters: {params}")
    print(f"{'='*60}")
    
    env = gym.make("LunarLander-v3")
    
    # Create model with custom parameters
    # All unspecified parameters use defaults
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./tensorboard_logs/",
        **params
    )
    
    # Train
    model.learn(total_timesteps=200000, tb_log_name=exp_name)
    
    # Evaluate rigorously: 100 episodes
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=100
    )
    
    print(f"Result: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    results[exp_name] = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "params": params
    }
    
    # Track the best model
    if mean_reward > best_reward:
        best_reward = mean_reward
        best_experiment = exp_name
        best_model = model
        # Save the best model so far — not just the last one!
        model.save("lunar_lander_best")
    
    env.close()

# ----------------------------------------------------------
# Summary of all experiments
# ----------------------------------------------------------
print(f"\n{'='*60}")
print("EXPERIMENT RESULTS SUMMARY")
print(f"{'='*60}")
for name, res in sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True):
    marker = " *** BEST ***" if name == best_experiment else ""
    print(f"  {name}: {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}{marker}")

print(f"\nBest model saved from: {best_experiment}")
```

> **Pro tip:** Launch TensorBoard with `%load_ext tensorboard` and `%tensorboard --logdir ./tensorboard_logs/` in Colab to see beautiful training curves in real-time.

### Fine-tuning the Winner

Once you identify the best hyperparameter configuration, train a final model with more timesteps.

```python
# ============================================================
# STEP 9: Train the final optimized model
# Use the best hyperparameters found, with more training time
# ============================================================

env = gym.make("LunarLander-v3")

# These are example "best" params — yours may differ!
# Replace with whatever worked best in YOUR experiments
final_model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    gamma=0.99,
    batch_size=128,
    buffer_size=100000,
    exploration_fraction=0.2,
    exploration_final_eps=0.02,
    target_update_interval=10000,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

print("Training final model with optimized hyperparameters...")
final_model.learn(total_timesteps=500000, tb_log_name="final_optimized")

# ----------------------------------------------------------
# Final rigorous evaluation
# Target: mean reward > 200 with LOW standard deviation
# ----------------------------------------------------------
mean_reward, std_reward = evaluate_policy(
    final_model, env, n_eval_episodes=100
)

print(f"\n{'='*60}")
print(f"FINAL MODEL PERFORMANCE")
print(f"  Mean Reward: {mean_reward:.2f}")
print(f"  Std Reward:  {std_reward:.2f}")
print(f"  Target:      > 200 with low std")
print(f"  Status:      {'PASSED' if mean_reward > 200 else 'NEEDS MORE WORK'}")
print(f"{'='*60}")

# Save the final model
final_model.save("lunar_lander_final")
print("Final model saved as lunar_lander_final.zip")

env.close()
```

If you're not hitting 200, try **PPO** instead. PPO (Proximal Policy Optimization) is fundamentally different from DQN:

- **DQN is value-based**: it learns the value of each action and picks the best one. It's like having a scoring sheet for every possible move.
- **PPO is an actor-critic method**: it has two neural networks working together. The **actor** directly decides which action to take (no scoring sheet needed). The **critic** evaluates how good the actor's decisions are and provides feedback.

PPO is called "proximal" because it makes careful, small policy updates — it never changes the strategy too drastically in one step. Think of it as a cautious optimizer that says "let me improve a little bit, check the results, then improve a little more." This makes training remarkably stable, which is why PPO is one of the most widely used RL algorithms in practice — it powered OpenAI's Dota 2 bots and is a core component of RLHF (Reinforcement Learning from Human Feedback) used to fine-tune large language models.

For LunarLander specifically, PPO often converges faster and more reliably than DQN because the actor-critic architecture handles the multi-objective reward structure (landing precision + fuel conservation + safety) more naturally.

```python
# ============================================================
# ALTERNATIVE: Try PPO if DQN struggles
# PPO (Proximal Policy Optimization) is a policy-based method
# that often performs well on environments like LunarLander
# ============================================================
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

ppo_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

ppo_model.learn(total_timesteps=500000, tb_log_name="ppo_lunar")

mean_reward, std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=100)
print(f"PPO Performance: {mean_reward:.2f} +/- {std_reward:.2f}")

if mean_reward > 200:
    ppo_model.save("lunar_lander_final")
    print("PPO model saved as final model!")

env.close()
```

---

## Part 7: Building the API: Serving Your Agent

A model sitting in a notebook is useless to anyone else. Let's expose it through a **REST API** so that any application can ask our agent "what should I do in this state?"

### Architecture Decision

The RL logic (loading the model, predicting actions) belongs on the **backend** (API), not the frontend. The GUI should only display results — it should never import the RL model directly. This separation makes everything more maintainable and testable.

### FastAPI Implementation

```python
# ============================================================
# FILE: api.py
# A FastAPI backend that serves the trained RL model
# The API accepts a state and returns the predicted action
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from stable_baselines3 import DQN
import numpy as np
import gymnasium as gym
import json
from typing import List, Optional

# ----------------------------------------------------------
# Initialize FastAPI app
# ----------------------------------------------------------
app = FastAPI(
    title="Lunar Lander RL Agent API",
    description="API to interact with a trained DQN agent for LunarLander-v3",
    version="1.0.0"
)

# ----------------------------------------------------------
# Load the trained model at startup
# ----------------------------------------------------------
MODEL_PATH = "lunar_lander_final"

try:
    model = DQN.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# ----------------------------------------------------------
# Data models — define the format for requests/responses
# These ensure proper formatting of RL data (state → action)
# ----------------------------------------------------------
class StateInput(BaseModel):
    """
    Input: the 8-dimensional observation from LunarLander-v3
    """
    state: List[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="8-dimensional state vector: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]"
    )

class ActionOutput(BaseModel):
    """
    Output: the action chosen by the agent
    """
    action: int = Field(description="Action index (0=noop, 1=left, 2=main, 3=right)")
    action_name: str = Field(description="Human-readable action name")
    q_values: Optional[List[float]] = Field(description="Q-values for all actions")

class EpisodeResult(BaseModel):
    """
    Result of a complete episode played by the agent
    """
    total_reward: float
    steps: int
    success: bool
    actions_taken: List[int]
    rewards_per_step: List[float]
    states_visited: List[List[float]]


# ----------------------------------------------------------
# Action name mapping
# ----------------------------------------------------------
ACTION_NAMES = {
    0: "Do Nothing",
    1: "Fire Left Engine",
    2: "Fire Main Engine",
    3: "Fire Right Engine"
}


# ----------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------

@app.get("/health")
def health_check():
    """Check if the API and model are ready"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=ActionOutput)
def predict_action(input_data: StateInput):
    """
    Given a state, predict the best action.
    This is the core endpoint: state → action
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert the input state to the format the model expects
    state_array = np.array(input_data.state, dtype=np.float32)
    
    # Get the model's prediction (deterministic = always best action)
    action, _ = model.predict(state_array, deterministic=True)
    action = int(action)
    
    # Also compute Q-values for transparency
    import torch
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
        q_vals = model.q_net(state_tensor).squeeze().tolist()
    
    return ActionOutput(
        action=action,
        action_name=ACTION_NAMES.get(action, "Unknown"),
        q_values=[round(q, 4) for q in q_vals]
    )


@app.post("/play", response_model=EpisodeResult)
def play_episode():
    """
    Play a complete episode and return detailed results.
    This endpoint lets you see the agent in action, step by step.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    env = gym.make("LunarLander-v3")
    state, info = env.reset()
    
    total_reward = 0
    actions_taken = []
    rewards_per_step = []
    states_visited = [state.tolist()]
    terminated = False
    truncated = False
    steps = 0
    
    while not terminated and not truncated:
        action, _ = model.predict(state, deterministic=True)
        action = int(action)
        
        state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        actions_taken.append(action)
        rewards_per_step.append(float(reward))
        states_visited.append(state.tolist())
        steps += 1
    
    env.close()
    
    return EpisodeResult(
        total_reward=round(total_reward, 2),
        steps=steps,
        success=total_reward > 200,
        actions_taken=actions_taken,
        rewards_per_step=rewards_per_step,
        states_visited=states_visited
    )


@app.get("/evaluate")
def evaluate_agent(n_episodes: int = 100):
    """
    Run multiple episodes and return aggregate statistics.
    Use this to verify the agent's overall performance.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    env = gym.make("LunarLander-v3")
    rewards = []
    wins = 0
    steps_list = []
    
    for _ in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        while not terminated and not truncated:
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        steps_list.append(steps)
        if total_reward > 200:
            wins += 1
    
    env.close()
    
    return {
        "n_episodes": n_episodes,
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "min_reward": round(float(np.min(rewards)), 2),
        "max_reward": round(float(np.max(rewards)), 2),
        "win_rate": round(wins / n_episodes * 100, 1),
        "avg_steps": round(float(np.mean(steps_list)), 1)
    }


# ----------------------------------------------------------
# Run with: uvicorn api:app --reload --port 8000
# Then visit: http://localhost:8000/docs for interactive docs
# ----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Testing the API

```python
# ============================================================
# Quick test script for the API
# Run this after starting the API server
# ============================================================
import requests

BASE_URL = "http://localhost:8000"

# Test 1: Health check
print("--- Health Check ---")
r = requests.get(f"{BASE_URL}/health")
print(r.json())

# Test 2: Predict action for a given state
print("\n--- Predict Action ---")
sample_state = [0.1, 0.5, -0.1, -0.3, 0.02, 0.0, 0.0, 0.0]
r = requests.post(f"{BASE_URL}/predict", json={"state": sample_state})
print(r.json())

# Test 3: Play a full episode
print("\n--- Play Episode ---")
r = requests.post(f"{BASE_URL}/play")
result = r.json()
print(f"Total Reward: {result['total_reward']}")
print(f"Steps: {result['steps']}")
print(f"Success: {result['success']}")

# Test 4: Evaluate agent
print("\n--- Evaluate (20 episodes) ---")
r = requests.get(f"{BASE_URL}/evaluate?n_episodes=20")
print(r.json())
```

---

## Part 8: The Dashboard: Telling the Story of Your Agent

A good dashboard doesn't just show numbers — it tells a story. Who is this agent? How did it learn? How reliable is it now?

```python
# ============================================================
# FILE: dashboard.py
# Interactive Streamlit dashboard for monitoring agent performance
# Run with: streamlit run dashboard.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# ----------------------------------------------------------
# Page configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Lunar Lander RL Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("Lunar Lander RL Agent — Performance Dashboard")
st.markdown("*Interactive monitoring of a DQN agent trained to land on the moon.*")

# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    return DQN.load("lunar_lander_final")

model = load_model()

# ----------------------------------------------------------
# Sidebar controls — interactive filters
# ----------------------------------------------------------
st.sidebar.header("Configuration")
n_episodes = st.sidebar.slider(
    "Number of evaluation episodes", 
    min_value=10, max_value=200, value=100, step=10
)
show_details = st.sidebar.checkbox("Show per-episode details", value=True)

# ----------------------------------------------------------
# Run evaluation and collect detailed data
# ----------------------------------------------------------
@st.cache_data
def run_evaluation(_model, n_eps):
    env = gym.make("LunarLander-v3")
    episode_data = []
    
    for i in range(n_eps):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        actions = {0: 0, 1: 0, 2: 0, 3: 0}
        
        while not terminated and not truncated:
            action, _ = _model.predict(state, deterministic=True)
            action = int(action)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            actions[action] += 1
        
        episode_data.append({
            "episode": i + 1,
            "reward": round(total_reward, 2),
            "steps": steps,
            "success": total_reward > 200,
            "noop_pct": round(actions[0] / steps * 100, 1),
            "left_pct": round(actions[1] / steps * 100, 1),
            "main_pct": round(actions[2] / steps * 100, 1),
            "right_pct": round(actions[3] / steps * 100, 1),
        })
    
    env.close()
    return pd.DataFrame(episode_data)

if st.sidebar.button("Run Evaluation", type="primary"):
    with st.spinner(f"Running {n_episodes} episodes..."):
        df = run_evaluation(model, n_episodes)
        st.session_state["results"] = df

# ----------------------------------------------------------
# Display results
# ----------------------------------------------------------
if "results" in st.session_state:
    df = st.session_state["results"]
    
    # ---- KPI Row ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Reward", f"{df['reward'].mean():.1f}")
    with col2:
        st.metric("Std Reward", f"{df['reward'].std():.1f}")
    with col3:
        win_rate = df['success'].mean() * 100
        st.metric("Win Rate", f"{win_rate:.0f}%")
    with col4:
        st.metric("Avg Steps", f"{df['steps'].mean():.0f}")
    
    st.markdown("---")
    
    # ---- Charts Row 1 ----
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Reward Distribution")
        fig = px.histogram(
            df, x="reward", nbins=30,
            color_discrete_sequence=["#2196F3"],
            title="Distribution of Episode Rewards"
        )
        fig.add_vline(x=200, line_dash="dash", line_color="green",
                      annotation_text="Success Threshold (200)")
        fig.add_vline(x=df['reward'].mean(), line_dash="dot", line_color="red",
                      annotation_text=f"Mean ({df['reward'].mean():.1f})")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Reward Per Episode")
        fig = px.scatter(
            df, x="episode", y="reward",
            color="success",
            color_discrete_map={True: "#4CAF50", False: "#F44336"},
            title="Reward Over Episodes (Green = Success)"
        )
        fig.add_hline(y=200, line_dash="dash", line_color="green")
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- Charts Row 2 ----
    col_left2, col_right2 = st.columns(2)
    
    with col_left2:
        st.subheader("Win vs Loss")
        win_loss = df['success'].value_counts().reset_index()
        win_loss.columns = ['success', 'count']
        win_loss['label'] = win_loss['success'].map(
            {True: 'Successful Landing', False: 'Failed Landing'}
        )
        fig = px.pie(
            win_loss, values='count', names='label',
            color_discrete_sequence=["#4CAF50", "#F44336"],
            title="Wins vs Losses"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right2:
        st.subheader("Steps to Landing")
        fig = px.histogram(
            df, x="steps", nbins=20,
            color_discrete_sequence=["#FF9800"],
            title="Number of Steps Before Episode Ends"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- Action Distribution ----
    st.subheader("Agent Decision Patterns")
    st.markdown("How does the agent distribute its actions across all episodes?")
    
    action_df = pd.DataFrame({
        "Action": ["Do Nothing", "Fire Left", "Fire Main", "Fire Right"],
        "Average %": [
            df['noop_pct'].mean(),
            df['left_pct'].mean(),
            df['main_pct'].mean(),
            df['right_pct'].mean()
        ]
    })
    fig = px.bar(
        action_df, x="Action", y="Average %",
        color="Action",
        title="Average Action Distribution Across Episodes"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ---- Detailed Table ----
    if show_details:
        st.subheader("Per-Episode Details")
        st.dataframe(
            df.style.applymap(
                lambda x: 'background-color: #C8E6C9' if x else 'background-color: #FFCDD2',
                subset=['success']
            ),
            use_container_width=True,
            height=400
        )

else:
    st.info("Click 'Run Evaluation' in the sidebar to start the analysis.")
```

---

## Part 9: Recording a Victory Lap

The final piece — a video of your agent in action. This is both satisfying and useful for showing stakeholders what your agent actually does.

```python
# ============================================================
# STEP 10: Record a video of the trained agent
# ============================================================
from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os

# Create a directory for videos
os.makedirs("videos", exist_ok=True)

# Load the trained model
model = DQN.load("lunar_lander_final")

# Wrap the environment with RecordVideo
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="videos",
    name_prefix="lunar_landing",
    episode_trigger=lambda e: True  # Record every episode
)

# ----------------------------------------------------------
# Play episodes until we get a good one (reward > 200)
# ----------------------------------------------------------
best_reward = -float('inf')

for attempt in range(10):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    print(f"Attempt {attempt + 1}: Reward = {total_reward:.2f}")
    
    if total_reward > 200:
        print(f"Great landing! Video saved.")
        break

env.close()
print(f"\nVideos saved in: {os.path.abspath('videos')}")
```

---

## Wrapping Up

Let's recap what we built:

1. **Explored RL fundamentals** — agent, environment, state, action, reward
2. **Implemented Q-Learning from scratch** — understood Bellman equation, epsilon-greedy, training vs evaluation
3. **Built a DQN manually with PyTorch** — learned Experience Replay, Target Networks, tensor flow
4. **Mastered Stable-Baselines3** — went from 100 lines of code to 5
5. **Trained and optimized a lunar lander agent** — systematic hyperparameter tuning, TensorBoard tracking
6. **Built a REST API with FastAPI** — proper data flow (state → action), documentation, testing
7. **Created an interactive Streamlit dashboard** — reward curves, win/loss ratios, action distributions, filterable data
8. **Recorded a video** of a successful landing

The complete code is available in the accompanying notebook. Clone it, run it, break it, improve it.

### What's Next?

If you want to go deeper:
- Try **A2C** or **SAC** algorithms on LunarLander
- Experiment with **custom reward shaping**
- Deploy the API on a cloud service (Render, Railway, or AWS Lambda)
- Try harder environments: `BipedalWalker-v3`, Atari games, or MuJoCo

### Resources That Helped Me

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [DeepMind x UCL RL Lecture Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- [PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Lilian Weng's RL Blog Series](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)

---

*My buddy? He still hasn't acknowledged the landing video I sent him. But every time I see a SpaceX rocket touch down, I think: "I know how that works." And honestly, that's cooler than any text response.*

---

**If you found this useful, give it a clap and share it with someone who's curious about RL. And if you build something cool with this, I'd love to hear about it — find me on [Twitter/X].**
