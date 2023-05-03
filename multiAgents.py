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

