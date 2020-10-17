import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd

from pong_agent_ddqn import PongAgent, make_env

master_parser = argparse.ArgumentParser().add_subparsers()

game_args_parser = master_parser.add_parser("Game Settings")
game_args_parser.add_argument('--render', '-r', action='store_true', default=False, help="Set to render environment")
game_args_parser.add_argument('--episodes', '--ep', type=int, default=500,
                              help="Sets number of episodes to play frequency")
game_args_parser.add_argument('--plot_frequency', '--pf', type=int, default=20,
                              help="Sets plotting frequency, set to 0 to disable. Will plot progress every n episodes")
game_args_parser.add_argument('--log_frequency', '--lf', type=int, default=1,
                              help="Sets logging frequency, set to 0 to disable. Will log progress every n episodes")

agent_args_parser = master_parser.add_parser("Agent settings")
agent_args_parser.add_argument('--training', '-t', action='store_true', default=False, help="Set to train agent")
agent_args_parser.add_argument('--epsilon', '--eps', '-e', type=float, default=1,
                               help="Set initial epsilon value (number of random steps) used in epsilon-greedy policy")
agent_args_parser.add_argument('--epsilon_min', '--eps_min', '--em', type=float, default=0.01,
                               help="Set minimum epsilon as decimal fraction - agent will always perform random "
                                    "action this often")
agent_args_parser.add_argument('--epsilon_decay', '--eps_dec', '--ed', type=float, default=0.00003,
                               help="Set epsilon decay - how fast epsilon approaches epsilon min")
agent_args_parser.add_argument('--gamma', '-g', type=float, default=0.99,
                               help="Set gamma also known as discount factor")
agent_args_parser.add_argument('--learning_rate', '--lr', type=float, default=0.0002,
                               help="Set learning rate")
agent_args_parser.add_argument('--batch', '-b', type=int, default=64, dest="batch_size",
                               help="Batch size used in training phase")
agent_args_parser.add_argument('--frames', '--fr', type=int, default=4,
                               help="Number of frames used to capture motion")
agent_args_parser.add_argument('--save_freq', '--sf', type=int, default=10000,
                               help="How often agent is saved - every n steps")

game_args = game_args_parser.parse_known_args()[0]
agent_args = agent_args_parser.parse_known_args()[0]

PLOT_FILE_PATH = "./output/pong-plot.png"
env = make_env('PongNoFrameskip-v4')
state_space = env.observation_space.shape
action_space = env.action_space.n

training = True


def init_parameters():
    pass


def plot(episodes, scores):
    x_label = 'Episode'
    y_label = 'Score'
    df = pd.DataFrame({
        x_label: episodes,
        y_label: scores
    })
    ax = sns.regplot(x=x_label, y=y_label, data=df)
    ax.set_title("DDQN Agent Score by Number of Episodes")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(PLOT_FILE_PATH)
    plt.close()


def play(agent: PongAgent, render=False, episodes=100000, plot_frequency=50, log_frequency=1):
    scores = []
    total_steps = 0
    for episode in range(1, episodes):

        print("Episode: " + str(episode))

        done = False

        state = env.reset()
        score = 0
        steps = 1
        avg_loss = 0
        while not done:
            agent.before()

            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            loss = agent.learn(state=state, action=action, reward=reward, next_state=next_state, done=done)

            state = next_state

            if render:
                env.render()

            score += reward

            if loss:
                avg_loss = (avg_loss + loss) / steps
            steps += 1
            total_steps += 1

            agent.after()

        if (episode + 1) % log_frequency == 0:
            print(
                "SCORE: {} LOSS: {} EPSILON: {} STEPS: {} TOTAL_STEPS: {}".format(score, avg_loss, agent.get_epsilon(),
                                                                                  steps, total_steps))
        scores.append(score)

        if (episode + 1) % plot_frequency == 0:
            plot([i for i in range(1, episode + 1)], scores)
    env.close()


if __name__ == "__main__":
    play(PongAgent(action_space, 80, 80, **vars(agent_args)), **vars(game_args))
