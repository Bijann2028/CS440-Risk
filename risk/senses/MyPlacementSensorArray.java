package pas.risk.senses;


// SYSTEM IMPORTS
import edu.bu.jmat.Matrix;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
import edu.bu.pas.risk.agent.senses.PlacementSensorArray;
import edu.bu.pas.risk.territory.Territory;


// JAVA PROJECT IMPORTS


/**
 * A suite of sensors to convert a {@link Territory} into a feature vector (must be a row-vector).
 *
 * Features (5 total):
 *  0: current armies at this territory (normalized by 20)
 *  1: number of adjacent enemy territories (normalized by 6)
 *  2: number of adjacent friendly territories (normalized by 6)
 *  3: continent bonus armies if completing it (normalized by 7), else 0
 *  4: fraction of continent territories I already own
 */
public class MyPlacementSensorArray
    extends PlacementSensorArray
{

    public static final int NUM_FEATURES = 5;

    public MyPlacementSensorArray(final int agentId)
    {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state,
                                  final int numRemainingArmies,
                                  final Territory territory)
    {
        double[] features = new double[NUM_FEATURES];
        int myId = this.getAgentId();

        TerritoryOwnerView tov = state.getTerritoryOwners().getById(territory.id());
        int armies = (tov != null) ? tov.getArmies() : 0;
        features[0] = Math.min(1.0, armies / 20.0);

        int adjEnemy    = 0;
        int adjFriendly = 0;
        for (Territory neighbor : territory.adjacentTerritories()) {
            TerritoryOwnerView ntov = state.getTerritoryOwners().getById(neighbor.id());
            if (ntov == null || ntov.isUnclaimed()) continue;
            if (ntov.getOwner() == myId) adjFriendly++;
            else                         adjEnemy++;
        }
        features[1] = Math.min(1.0, adjEnemy    / 6.0);
        features[2] = Math.min(1.0, adjFriendly / 6.0);

        // Continent completion and progress
        int contSize    = territory.continent().territories().size();
        int myContCount = 0;
        for (Territory t : territory.continent().territories()) {
            TerritoryOwnerView ctov = state.getTerritoryOwners().getById(t.id());
            if (ctov != null && ctov.getOwner() == myId) myContCount++;
        }
        boolean alreadyMine = (tov != null && tov.getOwner() == myId);
        int wouldOwn = alreadyMine ? myContCount : myContCount + 1;
        if (wouldOwn == contSize) {
            features[3] = Math.min(1.0, territory.continent().armiesPerTurn() / 7.0);
        }
        features[4] = (double) myContCount / Math.max(1, contSize);

        Matrix result = Matrix.zeros(1, NUM_FEATURES);
        for (int i = 0; i < NUM_FEATURES; i++) {
            result.set(0, i, features[i]);
        }
        return result;
    }

}