import argparse
from time import sleep

from pong_agent_ddqn import PongAgent, make_env

master_parser = argparse.ArgumentParser().add_subparsers()

PLOT_FILE_PATH = "./output/pong-plot.png"
env = make_env('PongNoFrameskip-v4')
action_space = env.action_space.n


def play(agent: PongAgent, log_frequency=1):
    scores = []
    total_steps = 0
    for episode in range(1, 5):

        print("Episode: " + str(episode))

        done = False

        state = env.reset()
        score = 0
        steps = 1
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            state = next_state

            env.render()
            sleep(0.025)

            score += reward

            steps += 1
            total_steps += 1

            agent.after()

        if (episode + 1) % log_frequency == 0:
            print("SCORE: {} STEPS: {} TOTAL_STEPS: {}".format(score, steps, total_steps))
        scores.append(score)

    env.close()


if __name__ == "__main__":
    play(PongAgent(action_space, 80, 80, training=False, load_agent=True, eval_agent_path="test-agent/eval_net-episode-590000",
                   targ_agent_path="test-agent/targ_net-episode-590000"))
