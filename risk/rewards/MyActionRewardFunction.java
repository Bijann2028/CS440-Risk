package pas.risk.rewards;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.action.NoAction;
import edu.bu.pas.risk.agent.rewards.RewardFunction;
import edu.bu.pas.risk.agent.rewards.RewardType;
import edu.bu.pas.risk.territory.Territory;

public class MyActionRewardFunction
    extends RewardFunction<Action>
{
    public MyActionRewardFunction(final int agentId)
    {
        super(RewardType.FULL_TRANSITION, agentId);
    }

    public double getLowerBound() { return -50.0; }
    public double getUpperBound() { return  150.0; }

    public double getStateReward(final GameView state) { return 0.0; }

    public double getHalfTransitionReward(final GameView state, final Action action)
    {
        return 0.0;
    }

    public double getFullTransitionReward(final GameView state,
                                          final Action action,
                                          final GameView nextState)
    {
        int myId = this.getAgentId();
        double reward = 0.0;

        int totalTerritories = state.getBoard().territories().size();

        // Territory progress: reward gaining territories, penalize losing them
        int myBefore = state.getTerritoriesOwnedBy(myId).size();
        int myAfter  = nextState.getTerritoriesOwnedBy(myId).size();
        int delta = myAfter - myBefore;
        reward += delta * 10.0;

        // Continent progress: big reward for capturing continents
        int contBefore = state.getContinentsOwnedBy(myId).size();
        int contAfter  = nextState.getContinentsOwnedBy(myId).size();
        reward += (contAfter - contBefore) * 30.0;

        // Enemy elimination bonus
        int aliveBefore = 0, aliveAfter = 0;
        for (int i = 0; i < state.getNumAgents(); i++) {
            if (i == myId) continue;
            if (!state.getTerritoriesOwnedBy(i).isEmpty()) aliveBefore++;
            if (!nextState.getTerritoriesOwnedBy(i).isEmpty()) aliveAfter++; 
        }
        reward += (aliveBefore - aliveAfter) * 50.0;

        // Win bonus
        if (nextState.isOver() && myAfter == totalTerritories) {
            reward += 100.0;
        }

        // Heavy penalty for doing nothing — NoAction should never score higher than attacking
        if (action instanceof NoAction) {
            reward -= 10.0;
        }

        // Small penalty per turn to encourage ending games quickly
        reward -= 0.1;

        return Math.max(getLowerBound(), Math.min(getUpperBound(), reward));
    }
}