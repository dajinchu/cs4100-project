
# Lightning Talk

* Who are you?

We are Matthew Gries and Da-Jin Chu

* Problem and Tech?

We are trying to beat the minimax othello player from HW2 using Q-Learner with a
neural network backend

* Data?

We used OpenAI gym to generate environments in the game of Othello.

* Problem Encountered and how we solved it?

Minimax is deterministic (it used the same players), so sometimes we would win against the
minimax player 100% of the time or loses 100% of the time. We are still working on this issue,
which could be resolved by starting the game at every possible state that can be achieved after 4
moves

* Current state of development?

We can beat minimax 100% of the time starting from a normal starting board. Depending on hyperparameters,
we can win against a board with a random setup about 70% of the time.

* Fun fact (SAY THIS EXPLICITLY)

Linux graphic drivers are bad.
