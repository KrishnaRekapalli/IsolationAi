"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
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
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")


    opponentPlayer = game.get_opponent(player)
    activeMovesList = game.get_legal_moves(player=player)
    opponentMovesList = game.get_legal_moves(player=opponentPlayer)
    # finding the common moves for both players
    commonMovesList = [i for i in activeMovesList if i in opponentMovesList]

    activeMoves = len(activeMovesList)
    opponentMoves = len(opponentMovesList)
    commonMoves = len(commonMovesList)



    custom_score1 = float((activeMoves)/(opponentMoves+1.0))

    custom_score1 = float(np.log(activeMoves+1) - np.log(opponentMoves+1.0))

    return custom_score1



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
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")


    opponentPlayer = game.get_opponent(player)
    activeMoves = len(game.get_legal_moves(player=player))
    opponentMoves = len(game.get_legal_moves(player=opponentPlayer))

    #custom_score3 = float((activeMoves)/(opponentMoves+1))


    # center of board
    w, h = game.width / 2., game.height / 2.

    # coordinates of opponent player
    y_o, x_o = game.get_player_location(opponentPlayer)

    # coordinates of active player
    y, x = game.get_player_location(player)

    # geometric center of line joing center and opponentPlayer

    x_c,y_c = (x_o + w)/2.0 , (y_o + h)/2.0

    return -1*float(((x - x_o)**2 + (y - y_o)**2)**0.5)




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
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")


    blankSpaces = game.get_blank_spaces()

    """
    The strategy is to pursue the opponent initially i.e. in the first few steps go close to the opponent
    and then the strategy will be to act in a way to reduce oppone's moves

    """
    opponentPlayer = game.get_opponent(player)
    activeMoves = len(game.get_legal_moves(player=player))
    opponentMoves = len(game.get_legal_moves(player=opponentPlayer))

    #custom_score3 = float((activeMoves)/(opponentMoves+1))


    # center of board
    w, h = game.width / 2., game.height / 2.

    # coordinates of opponent player
    y_o, x_o = game.get_player_location(opponentPlayer)

    # coordinates of active player
    y, x = game.get_player_location(player)

    # geometric center of line joing center and opponentPlayer

    x_c,y_c = (x_o + w)/2.0 , (y_o + h)/2.0

    if len(blankSpaces) > 35:

        return -1*float(((x - x_o)**2 + (y - y_o)**2)**0.5)

    else:
        return float(activeMoves/(opponentMoves+1.0))





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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!

        """
        Given a game and depth minimax needs to be implemented

        """

        # We create a copy of the board and then do the minimax

        boardImage = game.copy()

        # Get the legal moves for the player
        legalMoves = game.get_legal_moves()

        if legalMoves == []:
            return (-1,-1)

        else:




            # A list of scores will be in the order of the moves
            scoreLedger = []

            for move in legalMoves:

                # for every move in legal moves we have to span a tree and
                depthRem = depth
                depthRem = depthRem-1

                # return the game board with the move applied
                newBoard = game.forecast_move(move)

                #we have to return a move whose score is maximum

                scoreLedger.append(self.minValue(newBoard,depthRem))

            # once all moves are evaluated we return the move with highest score

            bestMoveLoc = scoreLedger.index(max(scoreLedger))

            return legalMoves[bestMoveLoc]




    def minValue(self,game,depthRem):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depthRem == 0:
                # if there is no more allowable depth to explore then it should return the value of evaluation function
                # at the given state
            return self.score(game,self)

        else:
            depthRem = depthRem-1
            # for each possible action now do a maxValue search

            v = float('Inf')

            # Get the list of legal moves of the active player (now the opponent)
            legalMovesMin = game.get_legal_moves()

            if legalMovesMin == []:
                v = float('Inf')

            else:
                for move in legalMovesMin:


                    newBoardMin = game.forecast_move(move)

                    v = min(v,self.maxValue(newBoardMin,depthRem))

            return v


    def maxValue(self,game,depthRem):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depthRem == 0:
            # if there is no more allowable depth to explore then it should return the value of evaluation function
            # at the given state
            return self.score(game,self)

        else:
            depthRem = depthRem-1
            # for each possible action now do a maxValue search

            v = float('-Inf')

            # Get the list of legal moves of the active player (now our player)
            legalMovesMax = game.get_legal_moves()

            if legalMovesMax == []:
                v = float('-Inf')

            else:
                for move in legalMovesMax:

                    depthRemMax = depthRem
                    depthRemMax = depthRemMax-1

                    newBoardMax = game.forecast_move(move)

                    v = max(v,self.minValue(newBoardMax,depthRem))

            return v







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

        # TODO: finish this function!
        #raise NotImplementedError

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            self.search_depth = 1

            while self.time_left() > 1:

                currentBest = self.alphabeta(game, self.search_depth)

                self.search_depth = self.search_depth+1





            #return self.alphabeta(game, self.search_depth)

        except SearchTimeout:

            return currentBest
            #pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

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



        # Get the legal moves for the player
        legalMoves = game.get_legal_moves()

        if legalMoves == []:
            return (-1,-1)

        else:




            # A list of scores will be in the order of the moves
            scoreLedger = []

            v = float('-Inf')

            for move in legalMoves:

                # for every move in legal moves we have to span a tree and
                depthRem = depth
                depthRem = depthRem-1

                # return the game board with the move applied
                newBoard = game.forecast_move(move)

                #we have to return a move whose score is maximum

                v = max(v,self.minValue(newBoard,depthRem,alpha,beta))

                alpha = max(alpha,v)

                scoreLedger.append(v)




            # once all moves are evaluated we return the move with highest score


            bestMoveLoc = scoreLedger.index(max(scoreLedger))



            return legalMoves[bestMoveLoc]





    def minValue(self,game,depthRem,alpha,beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depthRem == 0:
                # if there is no more allowable depth to explore then it should return the value of evaluation function
                # at the given state
            return self.score(game,self)

        else:
            depthRem = depthRem-1
            # for each possible action now do a maxValue search

            v = float('Inf')

            # Get the list of legal moves of the active player (now the opponent)
            legalMovesMin = game.get_legal_moves()

            if legalMovesMin == []:
                v = float('Inf')

            else:
                for move in legalMovesMin:




                    newBoardMin = game.forecast_move(move)

                    v = min(v,self.maxValue(newBoardMin,depthRem,alpha,beta))

                    beta = min(beta,v)

                    if v <= alpha:
                        return v



            return v


    def maxValue(self,game,depthRem,alpha,beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depthRem == 0:
            # if there is no more allowable depth to explore then it should return the value of evaluation function
            # at the given state
            return self.score(game,self)

        else:
            depthRem = depthRem-1
            # for each possible action now do a maxValue search

            v = float('-Inf')

            # Get the list of legal moves of the active player (now our player)
            legalMovesMax = game.get_legal_moves()

            if legalMovesMax == []:
                v = float('-Inf')

            else:
                for move in legalMovesMax:

                    depthRemMax = depthRem
                    depthRemMax = depthRemMax-1

                    newBoardMax = game.forecast_move(move)

                    v = max(v,self.minValue(newBoardMax,depthRem,alpha,beta))

                    alpha = max(alpha,v)

                    if v >= beta:
                        return v


            return v
