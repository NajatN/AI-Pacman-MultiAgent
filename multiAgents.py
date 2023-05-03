# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import heapq
from game import Agent, Actions
from pacman import GameState
from random import choice
from sys import maxsize
from pacman import Directions
from util import PriorityQueue


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self, index=0):
        self.index = index
        self.numActionsTaken = 0
    
    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        self.numActionsTaken += 1

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Compute the distance to the nearest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(foodDistances) > 0:
            foodDistance = min(foodDistances)
        else:
            foodDistance = 0

        # Compute the distance to the nearest ghost
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        if len(ghostDistances) > 0:
            ghostDistance = min(ghostDistances)
            nearestGhostScaredTime = newScaredTimes[ghostDistances.index(ghostDistance)]
        else:
            ghostDistance = float('inf')
            nearestGhostScaredTime = 0

        # Compute the remaining food pellets and incentivize eating them quickly
        remainingFood = successorGameState.getNumFood()
        foodEaten = currentGameState.getNumFood() - remainingFood
        foodBonus = 1000 if foodEaten > 0 else 0
        foodScore = remainingFood * -10 + foodBonus

        # Incentivize eating vulnerable ghosts
        vulnerableGhosts = [ghost for ghost, scaredTime in zip(newGhostStates, newScaredTimes) if scaredTime > 0]
        if len(vulnerableGhosts) > 0:
            vulnerableGhostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in vulnerableGhosts]
            if len(vulnerableGhostDistances) > 0:
                nearestVulnerableGhostDistance = min(vulnerableGhostDistances)
            else:
                nearestVulnerableGhostDistance = float('inf')
            vulnerableGhostScore = 500 / (nearestVulnerableGhostDistance + 1)
        else:
            vulnerableGhostScore = 0

        # Compute the number of actions taken
        numActionsTaken = self.numActionsTaken

        # Add capsule consideration
        newCapsules = successorGameState.getCapsules()
        capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in newCapsules]
        if len(capsuleDistances) > 0:
            capsuleDistance = min(capsuleDistances)
            capsuleScore = 200 / (capsuleDistance + 1)
        else:
            capsuleScore = 0

        if ghostDistance <= 1 and nearestGhostScaredTime == 0:
            score = -float('inf')
        else:
            score = (foodScore - 1.5 * foodDistance) + vulnerableGhostScore + capsuleScore - numActionsTaken

            safetyBuffer = sum(newScaredTimes) / len(newScaredTimes) if len(newScaredTimes) > 0 else 0
            if ghostDistance <= safetyBuffer and nearestGhostScaredTime == 0:
                # Penalize based on proximity to non-vulnerable ghosts
                score -= 1000 / (ghostDistance + 1)
                # Increase the priority of chasing vulnerable ghosts over collecting food when ghosts are scared
            if nearestGhostScaredTime > 0:
                score += vulnerableGhostScore * 2

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    if currentGameState.isWin():
        return float('inf')

    if currentGameState.isLose():
        return -float('inf')

    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    # Encourage getting food by adding the reciprocal of the closest food distance to the score
    closestFoodDist = min([util.manhattanDistance(currentGameState.getPacmanPosition(), food) for food in foodList])
    score += 1 / closestFoodDist

    # Encourage getting away from ghosts by subtracting the reciprocal of the closest ghost distance from the score
    closestGhostDist = min([util.manhattanDistance(currentGameState.getPacmanPosition(), ghost.getPosition()) for ghost in ghostStates])
    score -= 1 / closestGhostDist

    return score

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, agentIndex, depth):
            stack = [(state, agentIndex, depth)]
            maxEval = -float('inf')
            while stack:
                state, agentIndex, depth = stack.pop()
                if depth == 0 or state.isWin() or state.isLose():
                    eval = self.evaluationFunction(state)
                    maxEval = max(maxEval, eval)
                else:
                    actions = state.getLegalActions(agentIndex)
                    nextAgent = (agentIndex + 1) % state.getNumAgents()
                    if agentIndex == 0:  # Pacman's turn (maximizing agent)
                        for action in actions:
                            successor = state.generateSuccessor(agentIndex, action)
                            stack.append((successor, nextAgent, depth))  # add successor to stack
                    else:  # Ghost's turn (minimizing agent)
                        minEval = float('inf')
                        for action in actions:
                            successor = state.generateSuccessor(agentIndex, action)
                            eval = minimax(successor, nextAgent, depth - 1)  # recursive call
                            minEval = min(minEval, eval)
                        maxEval = max(maxEval, minEval)
            return maxEval

        # Call minimax for Pacman to get the best action
        actions = gameState.getLegalActions(0)
        scores = [minimax(gameState.generateSuccessor(0, action), 1, self.depth) for action in actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        return actions[random.choice(bestIndices)]
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        with alpha-beta pruning (cuts off branch if alpha >= beta).

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        actions = gameState.getLegalActions(0)
        if not actions:
            return None
        def alpha_beta_prune(gameState, agentIndex, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents

            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                value = -float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    nextState = gameState.generateSuccessor(agentIndex, action)
                    value = max(value, alpha_beta_prune(nextState, nextAgent, depth, alpha, beta))
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value
            # Ghost's turn (minimizing agent)
            else:
                value = float('inf')
                if nextAgent == 0:
                    depth -= 1
                for action in gameState.getLegalActions(agentIndex):
                    nextState = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, alpha_beta_prune(nextState, nextAgent, depth, alpha, beta))
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value

        alpha = -float('inf')
        beta = float('inf')
        bestValue = -float('inf')
        bestAction = actions[0]
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            currentValue = alpha_beta_prune(nextState, 1, self.depth, alpha, beta)
            if currentValue > bestValue:
                bestValue = currentValue
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        def expectimax(state, agentIndex, depth, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return betterEvaluationFunction(state), None

            actions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()

            if agentIndex == 0:  # Pacman's turn (maximizing agent)
                bestScore = -float('inf')
                bestAction = None
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = expectimax(successor, nextAgent, depth, alpha, beta)
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                    alpha = max(alpha, bestScore)
                    if alpha >= beta:
                        break  # prune remaining nodes
                return bestScore, bestAction
            else:  # Ghost's turn (expectimax agent)
                expectedScore = 0
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score, _ = expectimax(successor, nextAgent, depth - 1, alpha, beta)
                    expectedScore += score
                expectedScore /= len(actions)
                return expectedScore, None

        _, bestAction = expectimax(gameState, 0, self.depth, -float('inf'), float('inf'))  # Use alpha-beta pruning

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    score = currentGameState.getScore()
    pacman_position = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    remaining_food = len(food_list)
    ghosts = currentGameState.getGhostStates()
    scared_ghosts = sum([1 for ghost in ghosts if ghost.scaredTimer > 0])

    food_distances = [manhattanDistance(pacman_position, food) for food in food_list]
    closest_food_distance = min(food_distances) if food_distances else 0

    power_pellet_list = currentGameState.getCapsules()
    power_pellet_distances = [manhattanDistance(pacman_position, pellet) for pellet in power_pellet_list]
    closest_power_pellet_distance = min(power_pellet_distances) if power_pellet_distances else 0

    scared_ghost_states = [ghost for ghost in ghosts if ghost.scaredTimer > 0]
    non_scared_ghost_states = [ghost for ghost in ghosts if ghost.scaredTimer == 0]

    non_scared_ghost_distances = [manhattanDistance(pacman_position, ghost.getPosition()) for ghost in non_scared_ghost_states]
    closest_non_scared_ghost_distance = min(non_scared_ghost_distances) if non_scared_ghost_distances else 0

    # Updated weights for each factor
    score_weight = 1.0
    closest_food_weight = 2.0
    remaining_food_weight = -50.0
    closest_power_pellet_weight = 15.0
    scared_ghosts_weight = 150.0
    closest_non_scared_ghost_weight = -100.0

    evaluation = (score_weight * score +
                  closest_food_weight / (closest_food_distance + 1) +
                  remaining_food_weight * remaining_food +
                  closest_power_pellet_weight / (closest_power_pellet_distance + 1) +
                  scared_ghosts_weight * scared_ghosts +
                  closest_non_scared_ghost_weight / (closest_non_scared_ghost_distance + 1))      
    
    return evaluation

# Abbreviation
better = betterEvaluationFunction