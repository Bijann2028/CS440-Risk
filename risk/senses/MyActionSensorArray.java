package pas.risk.senses;


// SYSTEM IMPORTS
import edu.bu.jmat.Matrix;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.action.AttackAction;
import edu.bu.pas.risk.action.FortifyAction;
import edu.bu.pas.risk.action.NoAction;
import edu.bu.pas.risk.action.RedeemCardsAction;
import edu.bu.pas.risk.agent.senses.ActionSensorArray;
import edu.bu.pas.risk.territory.Territory;


// JAVA PROJECT IMPORTS


/**
 * A suite of sensors to convert an {@link Action} into a feature vector (must be a row-vector).
 *
 * Features (10 total):
 *  0: is AttackAction?
 *  1: is FortifyAction?
 *  2: is NoAction/terminal?
 *  3: is RedeemCardsAction?
 *  4: attacking armies (normalized by 10)
 *  5: armies at "from" territory (normalized by 20)
 *  6: armies at "to/dest" territory (normalized by 20)
 *  7: army advantage ratio (attacking / defending), normalized to [0,1]
 *  8: number of enemy neighbors of "to" territory (normalized by 6)
 *  9: deltaArmies for fortify (normalized by 20)
 */
public class MyActionSensorArray
    extends ActionSensorArray
{

    public static final int NUM_FEATURES = 10;

    public MyActionSensorArray(final int agentId)
    {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state,
                                  final int actionCounter,
                                  final Action action)
    {
        double[] features = new double[NUM_FEATURES];
        int myId = this.getAgentId();

        if (action instanceof AttackAction) {
            AttackAction aa = (AttackAction) action;
            features[0] = 1.0;
            features[4] = Math.min(1.0, aa.attackingArmies() / 10.0);

            TerritoryOwnerView fromTov = state.getTerritoryOwners().getById(aa.from().id());
            TerritoryOwnerView toTov   = state.getTerritoryOwners().getById(aa.to().id());
            int fromArmies = (fromTov != null) ? fromTov.getArmies() : 0;
            int toArmies   = (toTov   != null) ? toTov.getArmies()   : 0;

            features[5] = Math.min(1.0, fromArmies / 20.0);
            features[6] = Math.min(1.0, toArmies   / 20.0);

            double ratio = (toArmies == 0) ? 3.0 : (double) aa.attackingArmies() / toArmies;
            features[7] = Math.min(1.0, ratio / 3.0);

            int enemyNeighbors = 0;
            for (Territory neighbor : aa.to().adjacentTerritories()) {
                TerritoryOwnerView ntov = state.getTerritoryOwners().getById(neighbor.id());
                if (ntov != null && ntov.getOwner() != myId) enemyNeighbors++;
            }
            features[8] = Math.min(1.0, enemyNeighbors / 6.0);

        } else if (action instanceof FortifyAction) {
            FortifyAction fa = (FortifyAction) action;
            features[1] = 1.0;

            TerritoryOwnerView fromTov = state.getTerritoryOwners().getById(fa.from().id());
            TerritoryOwnerView toTov   = state.getTerritoryOwners().getById(fa.to().id());
            int fromArmies = (fromTov != null) ? fromTov.getArmies() : 0;
            int toArmies   = (toTov   != null) ? toTov.getArmies()   : 0;

            features[5] = Math.min(1.0, fromArmies   / 20.0);
            features[6] = Math.min(1.0, toArmies     / 20.0);
            features[9] = Math.min(1.0, fa.deltaArmies() / 20.0);

        } else if (action instanceof RedeemCardsAction) {
            features[3] = 1.0;
        } else {
            features[2] = 1.0; // NoAction or terminal
        }

        Matrix result = Matrix.zeros(1, NUM_FEATURES);
        for (int i = 0; i < NUM_FEATURES; i++) {
            result.set(0, i, features[i]);
        }
        return result;
    }

}