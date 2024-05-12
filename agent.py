import numpy as np
from game import Board, UP, RIGHT, DOWN, LEFT, action_name
from game import IllegalAction, GameOver
import random

from model import Model2048
import torch

DEVICE = "cuda:0"

class nTupleNewrok:
    def __init__(self, tuples):
        self.TUPLES = tuples
        self.TARGET_PO2 = 15
        #self.LUTS = self.initialize_LUTS(self.TUPLES)
        self.model = Model2048(_total_tuple_len = 4 * 17, _num_tuples = 17).to(DEVICE)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.criterion = torch.nn.MSELoss()
        self.LOSSES = []
    

    # def initialize_LUTS(self, tuples):
    #     LUTS = []
    #     for tp in tuples:
    #         LUTS.append(np.zeros((self.TARGET_PO2 + 1) ** len(tp)))
    #     return LUTS

    def tuple_id(self, values):
        values = values[::-1]
        k = 1
        n = 0
        for v in values:
            if v >= self.TARGET_PO2:
                raise ValueError(
                    "digit %d should be smaller than the base %d" % (v, self.TARGET_PO2)
                )
            n += v * k
            
            k *= self.TARGET_PO2
        return n

    def V(self, board, delta=None, debug=False, n_tuples = 17, total_tuple_length = 17*4):
        """Return the expected total future rewards of the board.
        Updates the LUTs if a delta is given and return the updated value.
        """
        
        #print(type(board), len(board)) ## 16
        #print(board) ## [1,2,0,3,1,0,5,7,...]
        
        if debug:
            print(f"V({board})")
        vals = []
        LUTS = []
        for tp in self.TUPLES:
            LUTS.append(np.zeros((self.TARGET_PO2 + 1) ** len(tp)))
            #LUTS.append(np.zeros((self.TARGET_PO2 + 1) * tuple_length))
        
        #print(len(LUTS), len(LUTS[0]), len(LUTS[1]))
        for i, (tp, LUT) in enumerate(zip(self.TUPLES, LUTS)):
            tiles = [board[ii] for ii in tp]
            #tpid = self.tuple_id(tiles)
            #print(tpid, end=", ")
            #LUT[tpid] = 1
            LUT = tiles
        
        state = torch.cat([torch.tensor(lut) for lut in LUTS], dim=0)
        
        logits = self.model(state.to(DEVICE))
        
        
        return logits

    def evaluate(self, s, a):
        "Return expected total rewards of performing action (a) on the given board state (s)"
        b = Board(s)
        try:
            r = b.act(a)
            s_after = b.copyboard()
        except IllegalAction:
            return 0
        
        #print(self.V(s_after).shape, self.V(s_after))
        return r + torch.max(self.V(s_after))

    def best_action(self, s, steps=None):
        "returns the action with the highest expected total rewards on the state (s)"
        a_best = None
        r_best = -1
        if steps == None: epsilon = 0.0
        else: epsilon = max(0.00001, 0.05 - 0.01 * steps)
        
        p = random.random()
        if p < epsilon:
            return random.randint(0,3), epsilon
        #return torch.argmax(self.V(s)), epsilon
    
        for a in range(4):
           r = self.evaluate(s, a)
           if r > r_best:
               r_best = r
               a_best = a
        return a_best, epsilon
        

    def learn(self, ep_idx, total_steps, s, a, r, s_after, s_next, alpha=0.01, debug=False):
        """Learn from a transition experience by updating the belief
        on the after state (s_after) towards the sum of the next transition rewards (r_next) and
        the belief on the next after state (s_after_next).

        """
        a_next, _ = self.best_action(s_next, total_steps)
        b = Board(s_next)
        try:
            r_next = b.act(a_next)
            s_after_next = b.copyboard()
            v_after_next = torch.max(self.V(s_after_next))
            #v_next = torch.max(self.V(s_next))
        except IllegalAction:
            return
        except Exception as e:
            print(e)

        #delta = r_next + v_after_next - self.V(s_after)
        VALUE_MAX = 10.0
        logits = torch.max(self.V(s_after))
        target = torch.tensor(r + v_after_next)
        #target = torch.sigmoid(target)
        #target = torch.sigmoid(target) * VALUE_MAX
        target = 1 / (1 + torch.exp(0.05 * -1 * target)) * VALUE_MAX
        
        
        #print(f"pred: {logits}, target : {target}")
        
        loss = self.criterion(logits.float(), target.float().to(DEVICE))
        #loss *= 100.0
        #loss += torch.mean(logits**2) + torch.mean(target**2)
        loss += torch.mean(target**2) * (1 / VALUE_MAX)
        #loss += (-1) * torch.std(target)
        
        #loss = self.criterion(logits.float(), (r_next + v_next).float().to(DEVICE))
        self.LOSSES.append(loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        if (total_steps+1) % 1000 == 0:
            print(f"mean loss : {torch.mean(torch.tensor(self.LOSSES))}")
            self.LOSSES = []

        #if debug:
        #    print("s_next")
        #    Board(s_next).display()
        #    print("a_next", action_name(a_next), "r_next", r_next)
        #    print("s_after_next")
        #    Board(s_after_next).display()
            #self.V(s_after_next, debug=True)
            #print(
            #    f"delta ({delta:.2f}) = r_next ({r_next:.2f}) + v_after_next ({v_after_next:.2f}) - V(s_after) ({V(s_after):.2f})"
            #)
            #print(
            #    f"V(s_after) <- V(s_after) ({V(s_after):.2f}) + alpha * delta ({alpha} * {delta:.1f})"
            #)
        #self.V(s_after, alpha * delta)
