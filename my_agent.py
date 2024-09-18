__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

# Import the random number generation library
import numpy as np
import random
from cosc343TicTacToe import maxs_possible_moves, mins_possible_moves, terminal, evaluate, state_change_to_action, remove_symmetries

class TicTacToeAgent:
    """
    A class that encapsulates the code dictating the
    behaviour of the TicTacToe playing agent.

    Methods
    -------
    AgentFunction(percepts)
        Returns the move made by the agent given state of the game in percepts.
    """

    def __init__(self, h):
        """Initialises the agent.

        :param h: Handle to the figures showing state of the board -- only used
                  for human_agent.py to enable selecting the next move by clicking
                  on the matplotlib figure.
        """
        pass

    def AgentFunction(self, percepts):
        """The agent function of the TicTacToe agent -- returns action
        relating to the row and column of where to make the next move.

        :param percepts: the state of the board, a list of rows, each
        containing a value of three columns, where 0 identifies the empty
        square, 1 is a square with this agent's mark and -1 is a square with
        the opponent's mark.
        :return: tuple (r, c) where r is the row and c is the column index
                 where this agent wants to place its mark.
        """
        if terminal(percepts):
            return None  # No move if the game is over

        depth = 6  # Depth of Minimax search
        best_value = -np.inf  # Initialize to negative infinity
        best_move = None

        # Get all possible moves for the agent
        possible_moves = maxs_possible_moves(percepts)
        possible_moves = remove_symmetries(possible_moves)

        for state in possible_moves:
            value = self.minimax(state, depth - 1, False)  # Call minimax for each possible move
            if value > best_value:
                best_value = value
                best_move = state

        if best_move:
            r, c = state_change_to_action(percepts, best_move)
            return r, c
        
        return None

    def minimax(self, state, depth, maximizing_player):
        """Minimax algorithm to evaluate the best move.

        :param state: Current state of the board.
        :param depth: Depth of the minimax search.
        :param maximizing_player: Boolean indicating if the current player is the maximizing player.
        :return: The value of the state after applying minimax.
        """
        if terminal(state) or depth == 0:
            return evaluate(state)

        if maximizing_player:  # Agent's move
            max_value = -np.inf
            max_states = maxs_possible_moves(state)
            for new_state in max_states:
                value = self.minimax(new_state, depth - 1, False)
                max_value = max(max_value, value)
            return max_value
        else:  # Opponent's move
            min_value = np.inf
            min_states = mins_possible_moves(state)
            for new_state in min_states:
                value = self.minimax(new_state, depth - 1, True)
                min_value = min(min_value, value)
            return min_value
