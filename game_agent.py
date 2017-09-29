"""#zip isolation-674.zip game_agent.py heuristic_analysis.pdf research_review.pdf 

Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # The heuristic here should use look ahead on the optimal value function to 
    # determine success or failure for the given board state and options.   

    if game.is_winner(player):
        return float("inf")
    if game.is_loser(player):
        return float("-inf")

    player_y, player_x = game.get_player_location(player)

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
 
    # sum of blank spaces and total moves, used as lambda to mix board preference into the heuristic 
    theta_d = player_moves + opponent_moves + len(game.get_blank_spaces())
    middle_out_score = (game.width - player_x)**2 +  (game.height - player_y)**2

    return ((player_moves+middle_out_score/theta_d)**2 - opponent_moves**2)/theta_d

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.  This 
        heuristic simply attempts to place value on the delta between player and opponent moves.
    """

    if game.is_winner(player):
        return float("inf")
    if game.is_loser(player):
        return float("-inf")

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(player_moves**2 - opponent_moves**2)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.  This 
        heuristic naively attempts to place value a large move advantage early in the game.
    """

    if game.is_winner(player):
        return float("inf")
    if game.is_loser(player):
        return float("-inf")

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    josé_jiménez_heuristic = game.move_count*(player_moves-opponent_moves) / (opponent_moves*opponent_moves+1)

    return josé_jiménez_heuristic



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """


        legal_moves = game.get_legal_moves()

        # min the move score for incremental improvement
        score_best_move = float("-inf")
        best_move = (-1, -1)

        # if we are out of time, set up a valid move so we don't default the match due to bad time management
        if (len(legal_moves) > 0):
            best_move = legal_moves[0]
        else: # no 
            return best_move

        for move in legal_moves:
            move_score = self.min_value(game.forecast_move(move), depth-1)
            # beak out here to update both if better that previous best
            if move_score > score_best_move:
                score_best_move = move_score
                best_move = move

            # out of time, return best move so far
            if self.time_left() < self.TIMER_THRESHOLD:
                return best_move

        return best_move

    def max_value(self, game, depth):
        # is leaf node?
        if depth == 0:
            return self.score(game, self)

        # min the score
        best_score = float("-inf")
        for move in game.get_legal_moves():
            best_score = max(best_score, self.min_value( game.forecast_move(move), depth - 1) )
            # always return when needed
            if self.time_left() < self.TIMER_THRESHOLD:
                return best_score
        return best_score

    def min_value(self, game, depth):
        # is leaf node?
        if depth == 0:
            return self.score(game, self)

        # max the score
        best_score = float("inf")
        for move in game.get_legal_moves():
            best_score = min(best_score, self.max_value( game.forecast_move(move), depth - 1) )

            # always return when needed
            if self.time_left() < self.TIMER_THRESHOLD:
                return best_score
        return best_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left


        legal_moves = game.get_legal_moves()

        score_best_move = float("-inf")
        best_move = (-1, -1)

        best_move = (-1, -1)
        if best_move != (-1,-1):
            return best_move
                  
        try:
            iterative_depth = 1
            max_depth = 25
    
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while iterative_depth <= max_depth:
                # use adversarial search that stops evaluation when if is proved that the current move
                # is worse than the previous, since these will not impact the final decision
                best_move = self.alphabeta(game, iterative_depth, float("-inf"), float("inf"))
                iterative_depth += 1

        except SearchTimeout:
            return best_move

        return best_move

    # https://en.wikipedia.org/wiki/Alpha–beta_pruning
    def max_value(self, game, current_depth, alpha, beta):

        # game state checks
        legal_moves = game.get_legal_moves()
        # if depth = 0 or node is a terminal node
        if (current_depth <= 0) or (not legal_moves):
            return self.score(game, self)
        if game.utility(self) != 0.0:
            return game.utility(self)
        
        maxing_player_v = float("-inf")
        current_alpha = alpha
        for move in legal_moves:
            result_game = game.forecast_move(move)

            # v := max(v, alphabeta(child, depth – 1, α, β, FALSE))
            maxing_player_v = max(maxing_player_v,self.min_value(result_game, current_depth - 1, current_alpha, beta))

            #  break (* β cut-off *)
            if maxing_player_v >= beta:
                break
                #maxing_player_v

            # α := max(α, v)
            current_alpha = max(current_alpha, maxing_player_v)

            # always return when needed
            if self.time_left() < self.TIMER_THRESHOLD:
                return maxing_player_v
                
        return maxing_player_v


    def min_value(self, game, current_depth, alpha, beta):

        # game state checks
        legal_moves = game.get_legal_moves()
        # if depth = 0 or node is a terminal node
        if (current_depth <= 0) or (not legal_moves):
            return self.score(game, self)
        if game.utility(self) != 0.0:
            return game.utility(self)
        
        mining_player_v = float("+inf")
        current_beta = beta

        for move in legal_moves:
            result_game = game.forecast_move(move)

            # v := min(v, alphabeta(child, depth – 1, α, β, TRUE))
            mining_player_v = min(mining_player_v,self.max_value(result_game, current_depth - 1, alpha, current_beta))

            # break (* α cut-off *)
            if mining_player_v <= alpha:
                break
                #return mining_player_v

            # β := min(β, v)
            current_beta = min(current_beta, mining_player_v)
            
            # always return when needed
            if self.time_left() < self.TIMER_THRESHOLD:
                return mining_player_v

        return mining_player_v
  

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # game state checks
        legal_moves = game.get_legal_moves()
        
        # if depth = 0 or node is a terminal node
        if (not legal_moves):
            return (-1, -1)        
        if (depth < 1):
            return legal_moves[0]
       
        # As we have legal moves and might timeout anytime, it is better to select
        # an arbitrary first move than an invalid move.
        max_move = legal_moves[0]
        mining_player_v = float("-inf")

        current_alpha = alpha
        for move in legal_moves:
            result_game = game.forecast_move(move)

            # start with minimizing player
            current_score = self.min_value(result_game, depth - 1, current_alpha, beta)

            # leave verbose, update move and score
            # # v := min(v, alphabeta(child, depth – 1, α, β, TRUE))
            if (current_score > mining_player_v):
                max_move = move
                mining_player_v = current_score

            # break (* α cut-off *)
            if (mining_player_v >= beta):
                break
                #return max_move

            # α := max(α, v)
            current_alpha = max(current_alpha, mining_player_v)

            # always return the best move when needed
            if self.time_left() < self.TIMER_THRESHOLD:
                return max_move

        return max_move