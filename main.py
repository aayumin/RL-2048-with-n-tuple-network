import numpy as np
from game import Board
from game import IllegalAction, GameOver
from agent import nTupleNewrok
import torch
import matplotlib.pyplot as plt


from tqdm import tqdm

import pickle
import time

from collections import namedtuple

"""
Vocabulary
--------------

Transition: A Transition shows how a board transfromed from a state to the next state. It contains the board state (s), the action performed (a), 
the reward received by performing the action (r), the board's "after state" after applying the action (s_after), and the board's "next state" (s_next) after adding a random tile to the "after state".

Gameplay: A series of transitions on the board (transition_history). Also reports the total reward of playing the game (game_reward) and the maximum tile reached (max_tile).
"""
Transition = namedtuple("Transition", "s, a, r, s_after, s_next")
Gameplay = namedtuple("Gameplay", "transition_history game_reward max_tile")



keymap = {0:"↑", 1: "→", 2: "↓" , 3:"←"}

def play(agent, board, ep_idx, total_steps, spawn_random_tile=True):
    "Return a gameplay of playing the given (board) until terminal states."
    
    
    b = Board(board)
    r_game = 0
    transition_history = []
    while True:
        #time.sleep(0.2)
        a_best = agent.best_action(b.board, ep_idx)
        s = b.copyboard()
        #if (total_steps+1) % 1000 == 0: print(f"epsilon : {epsilon}")
        try:
            total_steps += 1
            r = b.act(a_best)
            #print(f"Pressed Key to Act :  {keymap[a_best]}")
            r_game += r
            #print(f"r: {r}")
            #r_game = max(r_game, r)
            s_after = b.copyboard()
            status = b.spawn_tile(random_tile=spawn_random_tile)
            s_next = b.copyboard()
            if status == False: 
                r = -1 * 2 ** max(b.board)
                break
        except (IllegalAction) as e:
            # game ends when agent makes illegal moves or board is full
            r = None
            s_after = None
            s_next = None
            break
        except (GameOver) as e:
            r = torch.tensor( -1 * 2 ** max(b.board))
            s_after = None
            s_next = None
            break
        finally:
            transition_history.append(
                Transition(s=s, a=a_best, r=r, s_after=s_after, s_next=s_next)
            )
    gp = Gameplay(
        transition_history=transition_history,
        game_reward=r_game,
        max_tile=2 ** max(b.board),
    )
    
    learn_from_gameplay(agent, gp, ep_idx, total_steps)
    return gp, total_steps


def learn_from_gameplay(agent, gp, ep_idx, total_steps, alpha=0.1):
    "Learn transitions in reverse order except the terminal transition"
    REWARD_BACK_STEPS = 5
    past_transitions = []
    #total_steps = 0
    
    
    for i_step, tr in enumerate(gp.transition_history[:-1][::-1]):
        total_steps += 1
        agent.learn(ep_idx, total_steps, tr.s, tr.a, tr.r, tr.s_after, tr.s_next, alpha=alpha)
        
        ## TODO
        past_transitions.append(tr)
        if len(past_transitions) > REWARD_BACK_STEPS:
            del past_transitions[0]
        
        ## TODO
        GAMMA = 0.95
        if i_step > 0:
            for i in range(min(len(past_transitions), REWARD_BACK_STEPS)):
                ptr = past_transitions[-1-i]
                agent.learn(ep_idx, 0, ptr.s, ptr.a, ptr.r * GAMMA ** (i+1), ptr.s_after, ptr.s_next, alpha=alpha)


def load_agent(path):
    return pickle.load(path.open("rb"))


def save_csv(csv_data):
    with open("result.csv", "w") as f:
        for epi_data in csv_data:
            line = ','.join(epi_data)
            f.write(line + "\n")

# map board state to LUT
TUPLES = [
    # horizontal 4-tuples
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    # vertical 4-tuples
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    # all 4-tile squares
    [0, 1, 4, 5],
    [4, 5, 8, 9],
    [8, 9, 12, 13],
    [1, 2, 5, 6],
    [5, 6, 9, 10],
    [9, 10, 13, 14],
    [2, 3, 6, 7],
    [6, 7, 10, 11],
    [10, 11, 14, 15],
    
    # additional 4-tuples
    # [0,1,5,6],
    # [3,2,6,5],
    # [12,13,9,10],
    # [15,14,10,9],
    
    # additional 7-tuples
    # [0,4,8,12,13,14,15],
    # [12,13,14,15,11,7,3],
    # [15,11,7,3,2,1,0],
    # [3,2,1,0,4,8,12],
    
    # # additional 16-tuples
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    # [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
]




if __name__ == "__main__":
    import numpy as np

    agent = None
    # prompt to load saved agents
    csv_data = [["idx", "max_tile", "total_merge_rewards"]]
    from pathlib import Path

    path = Path("tmp")
    saves = list(path.glob("*.pkl"))
    if len(saves) > 0:
        print("Found %d saved agents:" % len(saves))
        for i, f in enumerate(saves):
            print("{:2d} - {}".format(i, str(f)))
        k = input(
            "input the id to load an agent, input nothing to create a fresh agent:"
        )
        if k.strip() != "":
            k = int(k)
            n_games, agent = load_agent(saves[k])
            print("load agent {}, {} games played".format(saves[k].stem, n_games))
    if agent is None:
        print("initialize agent")
        n_games = 0
        agent = nTupleNewrok(TUPLES)



    #n_session = 5000
    n_session = 5000
    #n_episode = 100
    n_episode = 100
    plot_mean_rewards = []
    plot_max_tile = []
    print("training")
    try:
        total_steps = 0
        for i_se in range(n_session):
            if n_games > 100000: break
            gameplays = []
            for i_ep in range(n_episode):
                if n_games > 100000: break
                gp, total_steps = play(agent, None, ep_idx = i_se, total_steps=total_steps, spawn_random_tile=True)
                gameplays.append(gp)
                n_games += 1
                
                epi_max_tile = gp.max_tile
                epi_total_merge_rewards = gp.game_reward
                csv_data.append([str(i_ep), str(epi_max_tile), str(epi_total_merge_rewards)])
            n2048 = sum([1 for gp in gameplays if gp.max_tile == 2048])
            mean_maxtile = np.mean([gp.max_tile for gp in gameplays])
            maxtile = max([gp.max_tile for gp in gameplays])
            mean_gamerewards = np.mean([gp.game_reward for gp in gameplays])
            print(
                "Game# %d, tot. %dk games, " % (n_games, n_games / 1000)
                + "mean game rewards {:.0f}, mean max tile {:.0f}, 2048 rate {:.0%}, maxtile {}".format(
                    mean_gamerewards, mean_maxtile, n2048 / len(gameplays), maxtile
                ),
            )
            plot_mean_rewards.append(mean_gamerewards)
            plot_max_tile.append(maxtile)
            
            if n_games > 10000:
                print("{} games played by the agent. Terminate learning.  save plot and model".format(n_games))
                plt.figure()
                plt.plot(plot_mean_rewards, "r-")
                plt.title("mean rewards")
                plt.savefig("mean rewards.png")
                
                plt.figure()
                plt.plot(plot_max_tile, "r-")
                plt.title("max-tile")
                plt.savefig("max-tile.png")
                
                fout = "tmp/{}_{}games.pkl".format(agent.__class__.__name__, n_games)
                pickle.dump((n_games, agent), open(fout, "wb"))
                print("agent saved to", fout)
                
                save_csv(csv_data)
                
    except KeyboardInterrupt:
        print("training interrupted")
        print("{} games played by the agent".format(n_games))
        if input("save the result data? (y/n)") == "y":
            save_csv(csv_data)
            
            plt.figure()
            plt.plot(plot_mean_rewards, "r-")
            plt.title("mean rewards")
            plt.savefig("mean rewards.png")
            
            plt.figure()
            plt.plot(plot_max_tile, "r-")
            plt.title("max-tile")
            plt.savefig("max-tile.png")
            
        if input("save the model weight? (y/n)") == "y":
            fout = "tmp/{}_{}games.pkl".format(agent.__class__.__name__, n_games)
            pickle.dump((n_games, agent), open(fout, "wb"))
            print("agent saved to", fout)
    except Exception as e:
        print(e)
    
