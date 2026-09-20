package pas.risk.agent;

import edu.bu.jnn.layers.*;
import edu.bu.jnn.models.Sequential;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.action.AttackAction;
import edu.bu.pas.risk.action.FortifyAction;
import edu.bu.pas.risk.action.NoAction;
import edu.bu.pas.risk.agent.NeuralQAgent;
import edu.bu.pas.risk.agent.rewards.RewardFunction;
import edu.bu.pas.risk.agent.senses.*;
import edu.bu.pas.risk.model.DualDecoderModel;
import edu.bu.pas.risk.territory.Territory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import pas.risk.rewards.MyActionRewardFunction;
import pas.risk.rewards.MyPlacementRewardFunction;
import pas.risk.senses.MyActionSensorArray;
import pas.risk.senses.MyPlacementSensorArray;
import pas.risk.senses.MyStateSensorArray;

public class RiskQAgent extends NeuralQAgent
{
    private static final double EPSILON_START = 1.0;
    private static final double EPSILON_END   = 0.05;
    private static final double EPSILON_DECAY = 5000000.0;
    private int totalDecisions = 0;

    private int attacksThisTurn = 0;
    private static final int MAX_ATTACKS_PER_TURN = 10;
    private static final double ATTACK_BIAS = 0.95;
    private static final double MIN_ATTACK_RATIO = 0.8;

    private int gameCount = 0;

    private final Random rng = new Random();

    public RiskQAgent(int agentId) { super(agentId); }

    private boolean shouldExplore()
    {
        if (!this.isTraining()) return false;
        totalDecisions++;
        double epsilon = EPSILON_END + (EPSILON_START - EPSILON_END)
                         * Math.exp(-totalDecisions / EPSILON_DECAY);
        return rng.nextDouble() < epsilon;
    }

    private static <T> T chooseRandom(final List<T> list, final Random r)
    {
        if (list.isEmpty()) return null;
        return list.get(r.nextInt(list.size()));
    }

    private static Action findTerminal(final List<Action> actions)
    {
        for (Action a : actions) { if (a.isTerminal()) return a; }
        return null;
    }

    @Override
    public DualDecoderModel initModel()
    {
        final int stateFeatures     = MyStateSensorArray.NUM_FEATURES;
        final int actionFeatures    = MyActionSensorArray.NUM_FEATURES;
        final int placementFeatures = MyPlacementSensorArray.NUM_FEATURES;
        final int encodingDim = 32;

        Sequential encoder = new Sequential();
        encoder.add(new Dense(stateFeatures, 64));
        encoder.add(new Tanh());
        encoder.add(new Dense(64, encodingDim));
        encoder.add(new Tanh());

        Sequential actionDecoder = new Sequential();
        actionDecoder.add(new Dense(encodingDim + actionFeatures, 32));
        actionDecoder.add(new Tanh());
        actionDecoder.add(new Dense(32, 1));

        Sequential placementDecoder = new Sequential();
        placementDecoder.add(new Dense(encodingDim + placementFeatures, 16));
        placementDecoder.add(new Tanh());
        placementDecoder.add(new Dense(16, 1));

        return new DualDecoderModel(encoder, actionDecoder, placementDecoder);
    }

    @Override public StateSensorArray createStateSensors()
    { return new MyStateSensorArray(this.agentId()); }

    @Override public ActionSensorArray createActionSensors()
    { return new MyActionSensorArray(this.agentId()); }

    @Override public PlacementSensorArray createPlacementSensors()
    { return new MyPlacementSensorArray(this.agentId()); }

    @Override public RewardFunction<Action> createActionReward()
    { return new MyActionRewardFunction(this.agentId()); }

    @Override public RewardFunction<Territory> createPlacementReward()
    { return new MyPlacementRewardFunction(this.agentId()); }

    @Override
    public void onTurnStart(final GameView game, final int agentIdx)
    {
        super.onTurnStart(game, agentIdx);
        if (agentIdx == this.agentId()) attacksThisTurn = 0;
    }

    @Override
    public void onGameEnd(final GameView game)
    {
        gameCount++;
        System.out.println("[GAME " + gameCount + "] turns=" + game.getNumTurns()
                           + " territories=" + game.getTerritoriesOwnedBy(this.agentId()).size()
                           + "/" + game.getBoard().territories().size()
                           + " won=" + (game.getTerritoriesOwnedBy(this.agentId()).size()
                                        == game.getBoard().territories().size())
                           + " training=" + this.isTraining()
                           + " agents=" + game.getNumAgents());
    }

    public void onDisqualify(final GameView game, final int agentIdx,
                             final java.lang.Throwable thrownException)
    {
        if (agentIdx == this.agentId()) {
            gameCount++;
            System.out.println("[DISQUALIFIED game=" + gameCount
                               + "] turns=" + game.getNumTurns()
                               + " territories=" + game.getTerritoriesOwnedBy(this.agentId()).size()
                               + "/" + game.getBoard().territories().size()
                               + " training=" + this.isTraining());
        }
    }

    @Override
    public void onActionApplied(final GameView game, final int agentIdx,
                                final Action action, final boolean success)
    {
        if (agentIdx == this.agentId() && 
            !game.getTerritoriesOwnedBy(this.agentId()).isEmpty()) {
            System.out.println("[ACTION] game=" + gameCount
                              + " territories=" + game.getTerritoriesOwnedBy(this.agentId()).size()
                              + "/" + game.getBoard().territories().size()
                              + " action=" + action.getClass().getSimpleName()
                              + " success=" + success
                              + " training=" + this.isTraining());
        }
    }

    // =========================================================================
    // REDEEM PHASE
    // =========================================================================

    @Override
    public Action getExplorationRedeemAction(final GameView game,
                                             final int actionCounter,
                                             final boolean canRedeemCards)
    {
        List<Action> options = this.getRedeemActions(game, actionCounter, canRedeemCards, false);
        if (!options.isEmpty()) {
            Action a = chooseRandom(options, rng);
            if (a != null) return a;
        }
        List<Action> fallback = this.getRedeemActions(game, actionCounter, canRedeemCards, true);
        if (!fallback.isEmpty()) {
            Action a = chooseRandom(fallback, rng);
            if (a != null) return a;
        }
        return new NoAction(this.agentId());
    }

    @Override
    public boolean shouldExploreRedeemMovePhase(final GameView game,
                                                final int actionCounter,
                                                final boolean canRedeemCards)
    { return true; }

    // =========================================================================
    // ATTACK PHASE
    // =========================================================================

    @Override
    public Action getExplorationAttackActionRedeemIfForced(final GameView game,
                                                           final int actionCounter,
                                                           final boolean canRedeemCards)
    {
        List<Action> options = this.getAttackRedeemActions(game, actionCounter, canRedeemCards);

        if (options.isEmpty()) {
            List<Action> fortify = this.getFortifyActions(game, actionCounter, canRedeemCards);
            Action t = findTerminal(fortify);
            if (t != null) return t;
            Action r = chooseRandom(fortify, rng);
            if (r != null) return r;
            return new NoAction(this.agentId());
        }

        if (attacksThisTurn >= MAX_ATTACKS_PER_TURN) {
            Action t = findTerminal(options);
            if (t != null) return t;
        }

        List<Action> attackOptions = new ArrayList<>();
        Action terminal = null;
        for (Action a : options) {
            if (a.isTerminal()) terminal = a;
            else attackOptions.add(a);
        }

        List<Action> goodAttacks = new ArrayList<>();
        for (Action a : attackOptions) {
            if (!(a instanceof AttackAction)) continue;
            AttackAction aa = (AttackAction) a;
            TerritoryOwnerView fromTov = game.getTerritoryOwners().getById(aa.from().id());
            TerritoryOwnerView toTov   = game.getTerritoryOwners().getById(aa.to().id());
            int attacker = (fromTov != null) ? fromTov.getArmies() : 1;
            int defender = (toTov   != null) ? toTov.getArmies()   : 1;
            if ((double) attacker / defender >= MIN_ATTACK_RATIO) goodAttacks.add(a);
        }

        List<Action> candidates = goodAttacks.isEmpty() ? attackOptions : goodAttacks;

        if (!candidates.isEmpty() && rng.nextDouble() < ATTACK_BIAS) {
            Action best = heuristicBestAttack(game, candidates);
            if (best == null) best = chooseRandom(candidates, rng);
            if (best != null) {
                attacksThisTurn++;
                return best;
            }
        }

        if (terminal != null) return terminal;
        Action fallback = chooseRandom(options, rng);
        if (fallback != null) return fallback;
        return new NoAction(this.agentId());
    }

    private Action heuristicBestAttack(final GameView game, final List<Action> attackOptions)
    {
        if (attackOptions.isEmpty()) return null;

        Action best = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (Action a : attackOptions) {
            if (!(a instanceof AttackAction)) continue;
            AttackAction aa = (AttackAction) a;

            TerritoryOwnerView toTov   = game.getTerritoryOwners().getById(aa.to().id());
            TerritoryOwnerView fromTov = game.getTerritoryOwners().getById(aa.from().id());

            int defenderArmies = (toTov   != null) ? toTov.getArmies()   : 1;
            int attackerArmies = (fromTov != null)  ? fromTov.getArmies() : 1;

            double score = (double) attackerArmies / Math.max(1, defenderArmies)
                           + rng.nextGaussian() * 0.3;

            int contSize    = aa.to().continent().territories().size();
            int myContCount = 0;
            for (Territory t : aa.to().continent().territories()) {
                TerritoryOwnerView tov = game.getTerritoryOwners().getById(t.id());
                if (tov != null && tov.getOwner() == this.agentId()) myContCount++;
            }
            if (myContCount == contSize - 1) score += 10.0;

            TerritoryOwnerView toOwner = game.getTerritoryOwners().getById(aa.to().id());
            if (toOwner != null && !toOwner.isUnclaimed()) {
                int enemyTerritories = game.getTerritoriesOwnedBy(toOwner.getOwner()).size();
                if (enemyTerritories <= 3) score += 5.0;
            }

            if (score > bestScore) {
                bestScore = score;
                best = a;
            }
        }

        if (best == null && !attackOptions.isEmpty()) best = attackOptions.get(0);
        return best;
    }

    @Override
    public boolean shouldExploreAttackRedeemIfForcedMovePhase(final GameView game,
                                                              final int actionCounter,
                                                              final boolean canRedeemCards)
    { return shouldExplore(); }

    // =========================================================================
    // FORTIFY PHASE
    // =========================================================================

    @Override
    public Action getExplorationFortifySkipAction(final GameView game,
                                                  final int actionCounter,
                                                  final boolean canRedeemCards)
    {
        List<Action> options = this.getFortifyActions(game, actionCounter, canRedeemCards);

        if (options.isEmpty()) return new NoAction(this.agentId());

        List<Action> fortifyOptions = new ArrayList<>();
        Action terminal = null;
        for (Action a : options) {
            if (a.isTerminal()) terminal = a;
            else fortifyOptions.add(a);
        }

        if (!fortifyOptions.isEmpty() && rng.nextDouble() < 0.6) {
            Action best = heuristicBestFortify(game, fortifyOptions);
            if (best != null) return best;
        }

        if (terminal != null) return terminal;
        Action fallback = chooseRandom(options, rng);
        if (fallback != null) return fallback;
        return new NoAction(this.agentId());
    }

    private Action heuristicBestFortify(final GameView game, final List<Action> fortifyOptions)
    {
        if (fortifyOptions.isEmpty()) return null;

        Action best = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (Action a : fortifyOptions) {
            if (!(a instanceof FortifyAction)) continue;
            FortifyAction fa = (FortifyAction) a;

            int enemyNeighbors = 0;
            for (Territory neighbor : fa.to().adjacentTerritories()) {
                TerritoryOwnerView ntov = game.getTerritoryOwners().getById(neighbor.id());
                if (ntov != null && !ntov.isUnclaimed() && ntov.getOwner() != this.agentId()) {
                    enemyNeighbors++;
                }
            }

            TerritoryOwnerView toTov = game.getTerritoryOwners().getById(fa.to().id());
            int toArmies = (toTov != null) ? toTov.getArmies() : 0;

            double score = enemyNeighbors * 2.0 - toArmies * 0.5 + rng.nextGaussian() * 0.3;

            if (score > bestScore) {
                bestScore = score;
                best = a;
            }
        }

        if (best == null && !fortifyOptions.isEmpty()) best = fortifyOptions.get(0);
        return best;
    }

    @Override
    public boolean shouldExploreFortifySkipMovePhase(final GameView game,
                                                     final int actionCounter,
                                                     final boolean canRedeemCards)
    { return shouldExplore(); }

    // =========================================================================
    // PLACEMENT PHASE
    // =========================================================================

    @Override
    public Territory getExplorationPlacement(final GameView game,
                                             final boolean isDuringSetup,
                                             final int remainingArmies)
    {
        List<Territory> options = this.getPotentialPlacements(game, isDuringSetup, remainingArmies);

        if (options.isEmpty()) {
            List<Territory> owned = game.getTerritoriesOwnedBy(this.agentId());
            if (!owned.isEmpty()) return owned.get(0);
            return game.getBoard().territories().iterator().next();
        }

        Territory best = heuristicBestPlacement(game, options, isDuringSetup);
        if (best != null) return best;
        Territory fallback = chooseRandom(options, rng);
        if (fallback != null) return fallback;
        return options.get(0);
    }

    private Territory heuristicBestPlacement(final GameView game,
                                              final List<Territory> options,
                                              final boolean isDuringSetup)
    {
        if (options.isEmpty()) return null;

        Territory best = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (Territory t : options) {
            double score = 0.0;

            if (isDuringSetup) {
                int contSize = t.continent().territories().size();
                score += 12.0 / Math.max(1, contSize);
            }

            int enemyNeighbors = 0;
            for (Territory neighbor : t.adjacentTerritories()) {
                TerritoryOwnerView ntov = game.getTerritoryOwners().getById(neighbor.id());
                if (ntov != null && !ntov.isUnclaimed() && ntov.getOwner() != this.agentId()) {
                    enemyNeighbors++;
                }
            }
            score += enemyNeighbors * 1.5;

            int contSize    = t.continent().territories().size();
            int myContCount = 0;
            for (Territory ct : t.continent().territories()) {
                TerritoryOwnerView ctov = game.getTerritoryOwners().getById(ct.id());
                if (ctov != null && ctov.getOwner() == this.agentId()) myContCount++;
            }
            double contProgress = (double) myContCount / Math.max(1, contSize);
            score += contProgress * 8.0;
            score += t.continent().armiesPerTurn() * 1.5;
            score += rng.nextGaussian() * 0.4;

            if (score > bestScore) {
                bestScore = score;
                best = t;
            }
        }

        if (best == null && !options.isEmpty()) best = options.get(0);
        return best;
    }

    @Override
    public boolean shouldExplorePlacementPhase(final GameView game,
                                               final boolean isDuringSetup,
                                               final int remainingArmies)
    { return shouldExplore(); }

    @Override
    public void onTurnEnd(final GameView game, final int agentIdx)
    {
        // Do not call super - prevents FSMAgent error spam after elimination
        // Professor confirmed onTurnEnd in FSMAgent/NeuralQAgent does nothing important
    }
}