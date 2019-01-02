
package smc.runmodes;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;

import mapek.AdaptationOption;
import mapek.Goal;
import mapek.Goals;

public class MachineLearning extends SMCConnector {
   
	@Override
	public void startVerification() {
		if (cycles <= TRAINING_CYCLE) {
			training();
		} else {
			testing();
		}
	}

	private void training() {
		// Formally verify all the adaptation options, and send them to the learners for training
		for (AdaptationOption adaptationOption : adaptationOptions) {
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
					adaptationOption.verificationResults);
		}
		send(adaptationOptions, taskType, Mode.TRAINING);
	}


	private void testing() {
		if (taskType == TaskType.CLASSIFICATION || taskType == TaskType.REGRESSION) {
			testing1Goal();
		} else if (taskType == TaskType.PLLAMULTICLASS || taskType == TaskType.PLLAMULTIREGR) {
			testing2Goals();
		} else {
			throw new RuntimeException(String.format("Testing unsupported for mode: %s", taskType.val));
		}
	}

	private void testing1Goal() {
		// Send the adaptation options to the learner with mode testing, returns the predictions of the learner
		JSONObject response = send(adaptationOptions, taskType, Mode.TESTING);

		// Retrieve the amount of options that were predicted to meet the goal by the learner
		int adaptationSpace = Integer.parseInt(response.get("adaptation_space").toString());
		System.out.print(";" + adaptationSpace);

		ArrayList<Float> predictions = new ArrayList<>();
		List<AdaptationOption> qosEstimates = new LinkedList<>();

		JSONArray arr = response.getJSONArray("predictions");
		for (int i = 0; i < arr.length(); i++) {
			predictions.add(Float.parseFloat(arr.get(i).toString()));
		}
		
		Goal pl = goals.getPacketLossGoal();

		int i = 0;
		for (AdaptationOption option : adaptationOptions) {
			if (adaptationSpace != 0) {
				boolean prediction = taskType == TaskType.CLASSIFICATION ? predictions.get(i) == 1.0 :
					pl.evaluate(predictions.get(i));
					
				if (prediction) {
					smcChecker.checkCAO(option.toModelString(), environment.toModelString(),
						option.verificationResults);
					qosEstimates.add(option);
				} else {
					option.verificationResults.packetLoss = 100;
				}
			} else {
				// In case no options were predicted to meet the goals, verify all of them
				smcChecker.checkCAO(option.toModelString(), environment.toModelString(),
					option.verificationResults);
				qosEstimates.add(option);
			}
			i++;
		}

		// Perform online learning on the samples that were predicted to meet the user goal
		// Note: if no samples were predicted to meet the goal, all the options are sent back for online learning
		send(qosEstimates, taskType, Mode.TRAINING);
	}

	private void testing2Goals() {
		// TODO maybe unify both functions
		if (taskType == TaskType.PLLAMULTICLASS) {
			testing2GoalsClassification();
		} else {
			throw new UnsupportedOperationException("Regression with multiple goals is currently not supported.");
			// testing2GoalsRegression();
		}
	}

	private void testing2GoalsClassification() {
		// Send the adaptation options to the learner with mode testing, returns the predictions of the learner
		JSONObject response = send(adaptationOptions, taskType, Mode.TESTING);
		
		// Retrieve the amount of options that were predicted to meet the goal by the learner
		int adaptationSpace = Integer.parseInt(response.get("adaptation_space").toString());
		System.out.print(";" + adaptationSpace);

		ArrayList<Float> predictions = new ArrayList<>();
		List<AdaptationOption> qosEstimates = new LinkedList<>();

		// The different prediction classes in case of 2 goals (latency & packet loss)
		int[] amtPredClass = {0, 0, 0, 0};

		JSONArray arr = response.getJSONArray("predictions");
		for (int i = 0; i < arr.length(); i++) {

			int predictedClass = Integer.parseInt(arr.get(i).toString());
			amtPredClass[predictedClass]++;
			predictions.add(Float.parseFloat(arr.get(i).toString()));
		}

		int nbCorrect = 0;
		// Here I set nbCorrect to the highest ammount of 
		// goals predicted correct for every option in the adaption space
		if (amtPredClass[3] > 0) {
			// There is at least one option which satisfies both goals
			nbCorrect = 2;
		} else if (amtPredClass[2] + amtPredClass[1] > 0) {
			// There is at least one option which satisfies one of the goals
			nbCorrect = 1;
		}

		
		int i = 0;

		for (AdaptationOption adaptationOption : adaptationOptions) {
			if (adaptationSpace != 0) {
				boolean isPredictedCorrect = false;
				double pred = predictions.get(i);

				if (nbCorrect == 2) {
					if (pred == 3.0) isPredictedCorrect = true;
				} else if (nbCorrect == 1) {
					if (pred == 2.0 || pred == 1.0) isPredictedCorrect = true;
				}


				if(isPredictedCorrect) {
					smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
						adaptationOption.verificationResults);

					// Add this option to the list of options that should be sent back for online learning
					qosEstimates.add(adaptationOption);
				} else {
					// The packet loss is manually set to 100 here to make sure this option is never considered.
					adaptationOption.verificationResults.packetLoss = 100;
				}
			} else {
				// In case no options were predicted to meet the goals, verify all of them
				smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
					adaptationOption.verificationResults);
				qosEstimates.add(adaptationOption);
			}
			i++;
		}

		// Perform online learning on the samples that were predicted to meet the user goal
		// Note: if no samples were predicted to meet the goal, all the options are sent back for online learning
		send(qosEstimates, taskType, Mode.TRAINING);
	}

	private void testing2GoalsRegression() {
		// Send the adaptation options to the learner with mode testing, returns the predictions of the learner
		JSONObject response = send(adaptationOptions, taskType, Mode.TESTING);
		
		// Retrieve the amount of options that were predicted to meet the goal by the learner
		int adaptationSpace = Integer.parseInt(response.get("adaptation_space").toString());
		System.out.print(";" + adaptationSpace);

		ArrayList<Float> predictionsPL = new ArrayList<>();
		ArrayList<Float> predictionsLA = new ArrayList<>();
		List<AdaptationOption> qosEstimates = new LinkedList<>();

		// The different prediction classes in case of 2 goals (latency & packet loss)
		int[] amtPredClass = {0, 0, 0, 0};

		Goal pl = Goals.getInstance().getPacketLossGoal();
		Goal la = Goals.getInstance().getLatencyGoal();

		JSONArray arrPL = response.getJSONArray("predictions_pl");
		JSONArray arrLA = response.getJSONArray("predictions_la");
		for (int i = 0; i < arrPL.length(); i++) {
			float predictedPL = Float.parseFloat(arrPL.get(i).toString());
			float predictedLA = Float.parseFloat(arrLA.get(i).toString());

			int predictedClass = (pl.evaluate(predictedPL) ? 1 : 0) + 2 * (la.evaluate(predictedLA) ? 1 : 0);
			amtPredClass[predictedClass]++;

			predictionsPL.add(predictedPL);
			predictionsLA.add(predictedLA);
		}

		int nbCorrect = 0;
		if (amtPredClass[3] > 0) {
			// There is at least one option which satisfies both goals
			nbCorrect = 2;
		} else if (amtPredClass[2] + amtPredClass[1] > 0) {
			// There is at least one option which satisfies one of the goals
			nbCorrect = 1;
		}

		
		int i = 0;

		for (AdaptationOption adaptationOption : adaptationOptions) {
			if (adaptationSpace != 0) {
				boolean isPredictedCorrect = false;
				int predictedClass = (pl.evaluate(predictionsPL.get(i)) ? 1 : 0) + 
					2 * (la.evaluate(predictionsLA.get(i)) ? 1 : 0);

				if (nbCorrect == 2) {
					if (predictedClass == 3) isPredictedCorrect = true;
				} else if (nbCorrect == 1) {
					if (predictedClass == 2 || predictedClass == 1) isPredictedCorrect = true;
				}


				if(isPredictedCorrect) {
					smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
						adaptationOption.verificationResults);

					// Add this option to the list of options that should be sent back for online learning
					qosEstimates.add(adaptationOption);
				} else {
					// The packet loss is manually set to 100 here to make sure this option is never considered.
					adaptationOption.verificationResults.packetLoss = 100;
				}
			} else {
				// In case no options were predicted to meet the goals, verify all of them
				smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
					adaptationOption.verificationResults);
				qosEstimates.add(adaptationOption);
			}
			i++;
		}

		// Perform online learning on the samples that were predicted to meet the user goal
		// Note: if no samples were predicted to meet the goal, all the options are sent back for online learning
		send(qosEstimates, taskType, Mode.TRAINING);
	}

}
