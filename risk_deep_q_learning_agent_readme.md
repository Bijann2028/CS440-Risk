# Deep Q-Learning Agent for Risk (Java)

An implementation of a Deep Q-Learning (DQN) agent designed to play 2–6 player Risk against automated and human opponents. This project utilizes a custom dual-head feed-forward neural network to manage multi-phase decision-making, combined with custom sensor arrays and shaped reward functions.

## Project Architecture

* **Dual-Decoder Neural Network (`RiskQAgent`):** Employs a shared state encoder paired with two distinct decoders:
  * **Placement Decoder:** Evaluates territory quality and determines army deployment.
  * **Action Decoder:** Handles combat and tactical unit movement during turn phases.
* **Feature Extraction Sensor Arrays (`src/pas/risk/senses/`):**
  * `MyStateSensorArray`: Vectorizes game configurations and territory ownership.
  * `MyPlacementSensorArray`: Encodes placement-phase-specific actions.
  * `MyActionSensorArray`: Processes combat and movement-phase transitions.
* **Reward Engineering (`src/pas/risk/rewards/`):**
  * `MyPlacementRewardFunction` & `MyActionRewardFunction`: Implement shaped rewards ($R(s)$, $R(s,a)$, $R(s,a,s')$) combined with piecewise Bellman/SARSA target formulations to preserve turn-ending agency rules.

## Setup & Execution

### Compilation
```bash
# Mac / Linux
javac -cp "./lib/*:." @risk.srcs

# Windows
javac -cp "./lib/*;." @risk.srcs
```

### Running Evaluation Games
```bash
# Run a single evaluation game with UI rendering
java -cp "./lib/*:." edu.bu.pas.risk.SingleGameEval
```

### Training the Agent
To launch training using the replay buffer and $\epsilon$-greedy exploration:
```bash
java -cp "./lib/*:." edu.bu.pas.risk.SequentialTrain | tee training_log.log
```

---

## Submission & Artifacts
The optimal network parameters generated from training runs are exported and serialized to `params.model` for autograder evaluation.