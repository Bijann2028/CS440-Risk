package pas.risk.rewards;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
import edu.bu.pas.risk.agent.rewards.RewardFunction;
import edu.bu.pas.risk.agent.rewards.RewardType;
import edu.bu.pas.risk.territory.Territory;

/**
 * R(s, t, s') reward for placement: reward territory and continent progress.
 */
public class MyPlacementRewardFunction
    extends RewardFunction<Territory>
{
    public MyPlacementRewardFunction(final int agentId)
    {
        super(RewardType.FULL_TRANSITION, agentId);
    }

    public double getLowerBound() { return -50.0; }
    public double getUpperBound() { return 150.0; }

    public double getStateReward(final GameView state) { return 0.0; }

    public double getHalfTransitionReward(final GameView state, final Territory territory)
    {
        return 0.0;
    }

    public double getFullTransitionReward(final GameView state,
                                          final Territory territory,
                                          final GameView nextState)
    {
        int myId = this.getAgentId();
        double reward = 0.0;

        // Territory progress
        int myBefore = state.getTerritoriesOwnedBy(myId).size();
        int myAfter  = nextState.getTerritoriesOwnedBy(myId).size();
        reward += (myAfter - myBefore) * 10.0;

        // Continent progress
        int contBefore = state.getContinentsOwnedBy(myId).size();
        int contAfter  = nextState.getContinentsOwnedBy(myId).size();
        reward += (contAfter - contBefore) * 30.0;

        // Bonus for placing on a border territory facing enemies
        for (Territory neighbor : territory.adjacentTerritories()) {
            TerritoryOwnerView ntov = nextState.getTerritoryOwners().getById(neighbor.id());
            if (ntov != null && !ntov.isUnclaimed() && ntov.getOwner() != myId) {
                reward += 1.0;
                break;
            }
        }

        return Math.max(getLowerBound(), Math.min(getUpperBound(), reward));
    }
}