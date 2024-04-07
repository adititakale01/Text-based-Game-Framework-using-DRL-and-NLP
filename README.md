Text-based Games using Deep Reinforcement Learning
===================================================

Tensorflow implementation of [Language Understanding for Text-based Games using Deep Reinforcement Learning](http://arxiv.org/abs/1506.08941). The original code of author can be found [here](https://github.com/karthikncode/text-world-player).

![model](./assets/model.png)


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)


Usage
-----

The code you provided implements a framework for interacting with a text-based game environment using Deep Reinforcement Learning (DRL) and Natural Language Processing (NLP). Here's a breakdown of the key functionalities:

Core Functionalities:

Interaction with Text-based Game:
Connects to the game environment using sockets (zmq).
Sends commands (actions) to the game and receives responses (text descriptions, rewards).
Parses the game output to extract relevant information like room descriptions, quest details, and rewards.
State Representation:
Processes the textual information from the game using NLP techniques.
Converts the processed text into a numerical representation (vector) suitable for DRL algorithms.
Offers different conversion methods like bag-of-words (BOW), bigrams, or ordered lists depending on the chosen approach (controlled by RECURRENT variable).
Action Space:
Defines a set of possible actions the agent can take in the game (e.g., move, look, interact).
Reward System:
Assigns rewards based on the agent's actions and the game's response.
Higher rewards for completing quests or achieving goals.
Lower rewards for neutral actions or penalties for invalid actions.
DEFAULT_REWARD, JUNK_CMD_REWARD, and quest-specific rewards are defined.
Quest Management:
Generates random quests for the agent to complete.
Tracks completed quests using checklists.
Creates misleading quests for a more complex learning environment.
Additional Features:

Symbol Mapping:
Creates a mapping between words encountered in the game and unique numerical IDs (symbols).
This allows efficient representation of textual information as vectors.
Logging and Debugging:
Provides functions for logging game interactions and rewards.
Enables debugging with DEBUG flag for printing additional information.
Customization:
Allows customizing the number of actions, objects, and maximum steps per episode.
Supports different state representation methods through vector_function.
Overall, this code serves as a foundation for building an agent that can learn to play text-based games through Deep Reinforcement Learning.


References
----------

- [EMNLP 2015 slide](http://people.csail.mit.edu/karthikn/pdfs/mud-play15-slides.pdf)

