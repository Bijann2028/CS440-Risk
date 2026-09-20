package pas.risk.senses;


// SYSTEM IMPORTS
import edu.bu.jmat.Matrix;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
import edu.bu.pas.risk.agent.senses.StateSensorArray;
import edu.bu.pas.risk.territory.Territory;
import edu.bu.pas.risk.territory.Continent;

import java.util.List;


// JAVA PROJECT IMPORTS


/**
 * A suite of sensors to convert a {@link GameView} into a feature vector (must be a row-vector).
 *
 * Features (15 total):
 *  0:  fraction of territories owned by me
 *  1:  fraction of territories owned by enemies (aggregate)
 *  2:  fraction of total armies that are mine
 *  3:  fraction of continents fully owned by me
 *  4:  fraction of continents fully owned by enemies
 *  5:  my bonus armies per turn (normalized by total territories)
 *  6:  fraction of my territories that border enemy territories
 *  7:  ratio of my armies on border territories to my total armies
 *  8:  average armies per territory I own (normalized by 10)
 *  9:  average armies per territory enemies own (normalized by 10)
 * 10:  number of cards in my hand (normalized by 5)
 * 11:  turns elapsed (normalized by 200)
 * 12:  fraction of total armies that are on my border territories
 * 13:  fraction of enemy players still alive
 * 14:  fraction of remaining unclaimed territories
 */
public class MyStateSensorArray
    extends StateSensorArray
{
    public static final int NUM_FEATURES = 15;

    public MyStateSensorArray(final int agentId)
    {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state)
    {
        double[] features = new double[NUM_FEATURES];

        int myId = this.getAgentId();
        int totalTerritories = state.getBoard().territories().size();
        int numAgents = state.getNumAgents();

        List<Territory> myTerritories = state.getTerritoriesOwnedBy(myId);
        int myCount = myTerritories.size();

        List<Territory> unclaimed = state.getUnownedTerritories();
        int unclaimedCount = unclaimed.size();
        int enemyCount = totalTerritories - myCount - unclaimedCount;

        features[0]  = (double) myCount       / Math.max(1, totalTerritories);
        features[1]  = (double) enemyCount     / Math.max(1, totalTerritories);
        features[14] = (double) unclaimedCount / Math.max(1, totalTerritories);

        // Army counts
        int myArmies    = 0;
        int totalArmies = 0;
        for (Territory t : state.getBoard().territories()) {
            TerritoryOwnerView tov = state.getTerritoryOwners().getById(t.id());
            if (tov == null) continue;
            int a = tov.getArmies();
            totalArmies += a;
            if (tov.getOwner() == myId) myArmies += a;
        }
        int enemyArmies = totalArmies - myArmies;
        features[2] = (double) myArmies / Math.max(1, totalArmies);

        // Continent ownership
        int myContinents   = state.getContinentsOwnedBy(myId).size();
        int totalContinents = state.getBoard().continents().size();
        int enemyContinents = 0;
        for (int i = 0; i < numAgents; i++) {
            if (i != myId) enemyContinents += state.getContinentsOwnedBy(i).size();
        }
        features[3] = (double) myContinents   / Math.max(1, totalContinents);
        features[4] = (double) enemyContinents / Math.max(1, totalContinents);

        // Bonus armies
        features[5] = (double) state.getBonusArmiesFor(myId) / Math.max(1, totalTerritories);

        // Border territory analysis
        int borderCount  = 0;
        int borderArmies = 0;
        for (Territory t : myTerritories) {
            boolean isBorder = false;
            for (Territory neighbor : t.adjacentTerritories()) {
                TerritoryOwnerView ntov = state.getTerritoryOwners().getById(neighbor.id());
                if (ntov != null && !ntov.isUnclaimed() && ntov.getOwner() != myId) {
                    isBorder = true;
                    break;
                }
            }
            if (isBorder) {
                borderCount++;
                TerritoryOwnerView tov = state.getTerritoryOwners().getById(t.id());
                if (tov != null) borderArmies += tov.getArmies();
            }
        }
        features[6]  = (double) borderCount  / Math.max(1, myCount);
        features[7]  = (double) borderArmies / Math.max(1, myArmies);
        features[8]  = myCount   > 0 ? (double) myArmies    / myCount    / 10.0 : 0.0;
        features[9]  = enemyCount > 0 ? (double) enemyArmies / enemyCount / 10.0 : 0.0;
        features[10] = (double) state.getAgentInventory(myId).size() / 5.0;
        features[11] = Math.min(1.0, state.getNumTurns() / 200.0);
        features[12] = (double) borderArmies / Math.max(1, totalArmies);

        // Alive enemies
        int aliveEnemies = 0;
        for (int i = 0; i < numAgents; i++) {
            if (i != myId && !state.getTerritoriesOwnedBy(i).isEmpty()) aliveEnemies++;
        }
        features[13] = (double) aliveEnemies / Math.max(1, numAgents - 1);

        Matrix result = Matrix.zeros(1, NUM_FEATURES);
        for (int i = 0; i < NUM_FEATURES; i++) {
            result.set(0, i, features[i]);
        }
        return result;
    }

}