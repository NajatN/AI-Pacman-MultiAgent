# AI Pacman MultiAgent Project

## Question 1: Evaluation Function
An action and the game's current state are passed to the evaluation function, which then provides a score indicating how effective the action was. The score is determined by considering a number of variables, such as the distance to the nearest food, the distance to the nearest ghost, the quantity of food pellets still in the bag, the number of actions taken thus far, and whether any weak ghosts are around.

There are various sections that make up the evaluation function. The function begins by extracting valuable data from the current state of the game, such as the new location of Pac-Man, the quantity of food pellets left, the locations of the ghosts, and how long each ghost will be vulnerable. Then it determines the distances to the closest food pellet, ghost, and vulnerable ghost (if any) as well as the distance to the nearest location.

Pac-Man is encouraged to consume the remaining food pellets rapidly since the function determines a score for each one. The number of food pellets Pac-Man still has, the number of food pellets he has previously consumed, and a bonus if Pac-Man ate any food during the current move all factor into the final score. In order to encourage Pac-Man to eat vulnerable ghosts, the function additionally computes a score based on the location of the closest vulnerable ghost.

The function then adds the scores from the earlier sections and deducts the total number of actions taken to arrive at the final score. The score is reduced if Pac-Man is too near to a ghost that is not vulnerable. The score is raised to favor eating weak ghosts over gathering food if there are any around.

## Question 2: Minimax
In the Pac-Man game, adversarial search agents use the scoreEvaluationFunction as an evaluation function. It gives back the score for the current game state, which corresponds to the score shown in the Pac-Man GUI.

The MinimaxAgent class implements the minimax algorithm for Pac-Man and is a subclass of MultiAgentSearchAgent. The minimax function evaluates the utility of each potential action in the game tree iteratively and returns the action with the highest value. The evaluation function is called to assess the utility if the depth is 0, the state is terminal (win/lose), or it returns infinity or negative infinity.

The action with the highest score is returned after calling the minimax function for each potential action in the getAction method. If more than one action has the same highest score, a random action is picked.

## Question 3: Alpha Beta
The MultiAgentSearchAgent abstract class is extended by the AlphaBetaAgent class, which also implements the getAction function. The AlphaBetaAgent class's goal is to employ the Alpha-Beta pruning optimization strategy to enhance the Minimax algorithm's performance while determining the best move to make in a game.

The alpha_beta_prune function returns the value of the best move after recursively computing the optimal action for each agent by considering all potential moves up to a given depth. The function returns the maximum value of the succeeding agents if the present agent is a maximizing agent. The function returns the minimal value of the succeeding agents if the present agent is a minimizing agent. The game tree is pruned using the Alpha-Beta pruning optimization technique to remove any branches that can produce no better results than the best solution so far.

The alpha_beta_prune function is implemented in the getAction method to help the Pac-Man agent make the best choice of action. Using the alpha_beta_prune function, it iterates through all of the lawful moves that the Pac-Man agent is permitted to make. The move with the highest computed value is then returned.

The two variables alpha and beta, which reflect the greatest results so far for maximizing and minimizing agents, respectively, are likewise maintained by the AlphaBetaAgent class. The function prunes the remaining branches of the tree in question whenever it discovers a move that results in a value larger than or equal to beta for a maximizing agent or less than or equal to alpha for a minimizing agent.

In  general, the Alpha-Beta pruning technique  allows for a significant reduction in the number of nodes that need to be explored in the game  tree, resulting in much faster search times and improved performance for adversarial search agents in Pac-Man game. 
 
## Question 4: Expectimax
Another adversarial search agent that tries to play the game by replicating the activities of all players, including the ghosts, is the ExpectimaxAgent class. Contrary to Minimax and Alpha-Beta pruning, the ExpectimaxAgent instead assigns probabilities to each ghost's potential moves rather than assuming that they will always make the best moves.

The optimum move to make given the current game state is returned by the ExpectimaxAgent class's getAction function, which accepts a gameState. It does this by invoking the expectimax function, which determines the best move for each agent in the game via a recursive algorithm.

The function accepts as inputs the current state, the agent index (which is 0 for Pac-Man and >0 for ghosts), the depth of the search at the moment, and the alpha and beta values for alpha-beta pruning. The function returns the score of the state with no action if the depth is 0 or the game state is a win or lose condition. On the other hand, if it is Pac-Man's turn (the maximizing agent), it calculates the best score for each of Pac-Man's potential actions and returns it together with the matching optimal action.

When it is a ghost's turn (expectimax agent), it calculates the expected score for each of the ghost's potential actions, averages them, and returns it along with no action. Additionally, the function employs alpha-beta pruning to omit nodes that are unable to produce a score higher than a previously investigated node.

The ExpectimaxAgent is typically utilized to play more tactically than the ReflexAgent, which is more straightforward. It accomplishes this by modeling all potential game plays and selecting the move that has the highest probability of success given the probabilities of the ghost's actions. Pac-Man is encouraged to prioritize particular objectives, such as consuming food pellets, power pellets, and staying away from unscared ghosts, by using the betterEvaluationFunction to assess the score of each state.

Pac-Man can make better decisions and score more by utilizing Expectimax search with alpha-beta pruning and the betterEvaluationFunction. The closest food pellet and power pellet to Pac-Man should be prioritized. Other winning tactics include avoiding non-scared ghosts by moving farther away from the closest one, eating scared ghosts if possible, and balancing the quantity of scared ghosts and remaining food pellets. Better game performance may result from these adjustments to the weights of the betterEvaluationFunction.

## Question 5: Better Evaluation Function
The scoreEvaluationFunction from the Pac-Man game has been replaced by the betterEvaluationFunction. It prioritizes particular aims by considering more elements and giving them different weights. This feature aims to improve Pac-Man's gameplay and help him rack up more points.

The function determines the distance to the nearest food pellet and gives it a heavier weight. In order to motivate Pac-Man to stay away from ghosts, it also determines the distance to the nearest one and gives it a penalty. By giving power pellets a heavier weight than regular pellets, the feature further encourages Pac-Man to consume them.

Additionally, the function gives a higher weight to the number of ghosts that are currently scared in order to encourage Pac-Man to eat them when they are weak. In order to motivate Pac-Man to consume them as quickly as possible, the function also determines the distance to the closest scared ghost and awards it a bonus.

Overall, the betterEvaluationFunction prioritizes the following goals for Pac-Man:
•	Eating the nearest food pellet.
•	Avoiding ghosts.
•	Eating power pellets.
•	Eating scared ghosts.
•	Reducing the distance to the nearest scared ghost.
By adjusting the weights of these factors, the betterEvaluationFunction can be fine-tuned to achieve better game performance.

## Question 6: Custom Pacman Agent
The CustomPacmanAgent class provides a unique Pac-Man gameplay strategy as a subclass of the Agent class. To choose the best course of action in the game, the agent combines heuristics with the A* search method.
The evaluationFunction method uses the current game state to determine a score based on a number of variables, such as the number of food pellets remaining in the game, the distance to the closest food pellet, the distance to the closest ghost, the distance to the closest vulnerable ghost, and the distance to the nearest capsule. The evaluation function additionally takes into account how many actions have been taken thus far and deducts points if Pac-Man comes too close to a ghost that is not vulnerable.

The minimum distance to a food pellet, the total distance to all unscared ghosts, and the distance to the closest capsule are used by the newHeuristic technique to determine a heuristic score. The A* search algorithm use this technique to identify the best route to the nearby food pellet or capsule.

The aStarSearch method determines the best route to a food pellet or capsule using the A* search algorithm. Starting from Pac-Man's current location, the approach generates successor states for each potential course of action. The cost and heuristic values are then calculated for each successor state, and they are added to a priority queue. Once the goal state is attained, the method chooses the state with the lowest f value and repeats the process.

To determine Pac-Man's optimal choice of action, the getAction method employs the evaluationFunction and A* search algorithm. Pac-Man uses the aStarSearch method to try to eat any capsules that may be on the board first. Otherwise, Pac-Man uses the newHeuristic approach and aStarSearch method to find the nearest food pellet to consume. Pac-Man employs the evaluationFunction approach to choose a secure spot to move if it becomes trapped or is unable to locate any food pellets or capsules. In order to promote exploration, the agent also records visited locations.
