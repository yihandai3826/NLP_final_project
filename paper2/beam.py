import numpy as np
import random
import torch
import copy
from config import device

class Hypothesis(object):
    '''
    a hypothesis which holds the generated tokens,
        current state and beam score
    '''
    def __init__(self, tokens, states, score):
        self.score = score
        self.states = states
        self.candidate = copy.deepcopy(tokens)


class PoetryBeam(object):
    def __init__(self, beam_size, length, B_ID, E_ID, UNK_ID,
         level_char_ids, oblique_char_ids):
        """Initialize params."""
        self.__beam_size = beam_size
        self.__length = length

        self.__B_ID = B_ID
        self.__E_ID = E_ID
        self.__UNK_ID = UNK_ID

        self.__level_cids = level_char_ids
        self.__oblique_cids = oblique_char_ids


    def reset(self, init_state, rhythms, rhyme, rhyme_char_ids, repetitive_ids):
        # reset before generating each line
        self.__hypotheses \
            = [Hypothesis([self.__B_ID], [init_state.clone().detach()], 0.0)
            for _ in range(0, self.__beam_size)]
        self.__completed_hypotheses = []

        self.__rhythms = rhythms # rhythm pattern of each chars in a line
        self.__rhyme = rhyme
        self.__rhyme_cids = rhyme_char_ids # char ids in the required rhyme category
        self.__repetitive_ids = repetitive_ids


    def get_candidates(self, completed=False, with_states=False):
        if completed:
            hypotheses = self.__completed_hypotheses
        else:
            hypotheses = self.__hypotheses

        candidates = [hypo.candidate for hypo in hypotheses]
        scores = [hypo.score for hypo in hypotheses]

        if with_states:
            # (L, H) * B
            all_states = [hypo.states for hypo in hypotheses]
            return candidates, scores, all_states

        else:
            return candidates, scores


    def get_search_results(self, only_completed=True, sort=True):
        candidates, scores, states = self.get_candidates(True, True)

        if not only_completed:
            add_candis, add_scores, add_states = self.get_candidates(True, True)
            candidates = candidates + add_candis
            scores = scores + add_scores
            states = states + add_states

        scores = [score/(len(candi)-1) for score, candi in zip(scores, candidates)]
        # sort with costs
        if sort:
            sort_indices = list(np.argsort(scores))
            candidates = [candidates[i] for i in sort_indices]
            scores = [scores[i] for i in sort_indices]
            states = [states[i] for i in sort_indices]

        return candidates, scores, states


    def get_beam_tails(self):
        # get the last token and state of each hypothesis
        tokens = [hypo.candidate[-1] for hypo in self.__hypotheses]
        # [B, 1]
        tail_tokens \
            = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)

        tail_states = [hypo.states[-1] for hypo in self.__hypotheses]
        # [1, H] * B -> [B, H]
        tail_states = torch.cat(tail_states, dim=0)

        return tail_tokens, tail_states


    def uncompleted_num(self):
        return len(self.__hypotheses)


    def advance(self, logits, states, position):
        # outs: (B, V)
        # states: (B, H)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().data.numpy()

        beam_ids, word_ids, scores = self.__beam_select(log_probs, position)

        # update beams
        updated_hypotheses = []
        for beam_id, word_id, score in zip(beam_ids, word_ids, scores):
            #print (beam_idx, word_idx, score)
            state = states[beam_id, :].unsqueeze(0) # (1, H)
            new_mutable_states = self.__hypotheses[beam_id].mutable_states + [state]

            new_candidate = self.__hypotheses[beam_id].candidate + [word_id]

            hypo = Hypothesis(new_candidate, new_mutable_states, score)

            if word_id == self.__E_ID:
                self.__completed_hypotheses.append(hypo)
            else:
                updated_hypotheses.append(hypo)

        self.__hypotheses = updated_hypotheses


    def __beam_select(self, log_probs, position):
        B, V = log_probs.shape[0], log_probs.shape[1]

        current_scores = np.array([hypo.score for hypo in self.__hypotheses]).reshape(B, 1)

        if position == 0:
            costs = -log_probs[0, :].reshape(1, V)  # (1, V)
        else:
            costs = current_scores - log_probs  # (B, V)

        filter_v = 1e5
        costs[:, self.__UNK_ID] = filter_v
        if position < self.__length:
            costs[:, self.__E_ID] = filter_v

        costs[:, self.__repetitive_ids] = filter_v
        inline_filter_ids = self.inline_filter(position)
        for i in range(B):
            costs[i, inline_filter_ids[i]] = filter_v

        if (self.__rhyme != -1) and (position == self.__length - 1):
            filter_ids = list(set(range(0, V)) - set(self.__rhyme_cids))
            costs[:, filter_ids] = filter_v

        if position < self.__length and len(self.__rhythms) > 0:
            pos_rhythm = self.__rhythms[position]
            if pos_rhythm == 0:  # level tone
                costs[:, self.__oblique_cids] = filter_v
            elif pos_rhythm == 1:  # oblique
                costs[:, self.__level_cids] = filter_v

        flat_costs = costs.flatten()  # (B*V)
        best_indices = np.argpartition(flat_costs, B)[0:B].copy()
        scores = flat_costs[best_indices]

        beam_ids = [int(idx // V) for idx in best_indices]
        word_ids = [int(idx % V) for idx in best_indices]

        if position == 0:
            beam_ids = list(range(B))

        return beam_ids, word_ids, scores


    def inline_filter(self, pos):
        candidates, _ = self.get_candidates()
        # candidates: (L_i) * B
        B = len(candidates)
        forbidden_list = [[] for _ in range(0, B)]

        limit_pos = pos - 1 if pos % 2 != 0 else pos
        preidx = range(0, limit_pos)

        for i in range(0, B):  # iter ever batch
            forbidden_list[i] = [candidates[i][c] for c in preidx]

        return forbidden_list
