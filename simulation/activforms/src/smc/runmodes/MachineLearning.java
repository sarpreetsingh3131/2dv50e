
package smc.runmodes;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONObject;

import mapek.AdaptationOption;
import mapek.Goal;
import mapek.Goals;
import util.ConfigLoader;

public class MachineLearning extends SMCConnector {
   
	private int lastLearningIndex = 0;

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
		Long startTime = System.currentTimeMillis();
		int timeCap = ConfigLoader.getInstance().getTimeCap();
		List<Integer> verifiedOptions = new ArrayList<>();

		AdaptationOption adaptationOption;
		int amtOptions = adaptationOptions.size();
		for (int i = 0; i < amtOptions; i++) {
			int actualIndex = (i + lastLearningIndex) % amtOptions;

			if ((System.currentTimeMillis() - startTime) / 1000 > timeCap) {
				lastLearningIndex = actualIndex;
				break;
			}

			adaptationOption = adaptationOptions.get(actualIndex);
			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				adaptationOption.verificationResults);

			verifiedOptions.add(actualIndex);
		}

		
		send(verifiedOptions.stream().map(i -> adaptationOptions.get(i)).collect(Collectors.toList()), taskType, Mode.TRAINING);
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
		Long startTime = System.currentTimeMillis();

		// Send the adaptation options to the learner with mode testing, returns the predictions of the learner
		JSONObject response = send(adaptationOptions, taskType, Mode.TESTING);
		
		// Retrieve the amount of options that were predicted to meet the goal by the learner
		int adaptationSpace = Integer.parseInt(response.get("adaptation_space").toString());
		System.out.print(";" + adaptationSpace);

		ArrayList<Integer> predictions = new ArrayList<>();
		List<AdaptationOption> qosEstimates = new LinkedList<>();

		// The different prediction classes in case of 2 goals (latency & packet loss)
		int[] amtPredClass = {0, 0, 0, 0};

		switch (taskType) {
			case PLLAMULTICLASS:
				JSONArray pred = response.getJSONArray("predictions");

				for (int i = 0; i < pred.length(); i++) {
					int predictedClass = Integer.parseInt(pred.get(i).toString());
					amtPredClass[predictedClass]++;
					predictions.add(predictedClass);
				}
				break;

			case PLLAMULTIREGR:
				JSONArray pred_pl = response.getJSONArray("predictions_pl");
				JSONArray pred_la = response.getJSONArray("predictions_la");

				Goal pl = Goals.getInstance().getPacketLossGoal();
				Goal la = Goals.getInstance().getLatencyGoal();

				for (int i = 0; i < pred_la.length(); i++) {
					int predictedClass = 
							(pl.evaluate(Double.parseDouble(pred_pl.get(i).toString())) ? 1 : 0) +
							(la.evaluate(Double.parseDouble(pred_la.get(i).toString())) ? 2 : 0);
					amtPredClass[predictedClass]++;
					predictions.add(predictedClass);
				}
				break;

			default:
				throw new RuntimeException("Trying to do testing for 2 goals with incompatible task type.");
		}
		
		// The indices for the options of the best class predicted
		List<Integer> indicesMain = new ArrayList<>();
		// The indices for the options which are considered for exploration
		List<Integer> indicesSub = new ArrayList<>();
		// The remaining indices which should not be considered
		List<Integer> remainingIndices = new ArrayList<>();

		if (amtPredClass[3] > 0) {
			// There is at least one option which satisfies both goals
			for (int i = 0; i < predictions.size(); i++) {
				int prediction = predictions.get(i);
				if (prediction == 3) {
					indicesMain.add(i);
				} else if (prediction == 2 || prediction == 1) {
					indicesSub.add(i);
				} else {
					remainingIndices.add(i);
				}
			}
		} else if (amtPredClass[2] + amtPredClass[1] > 0) {
			// There is at least one option which satisfies one of the goals
			for (int i = 0; i < predictions.size(); i++) {
				int prediction = predictions.get(i);
				if (prediction == 0) {
					indicesSub.add(i);
				} else {
					indicesMain.add(i);
				}
			}
		} else {
			for (int i = 0; i < predictions.size(); i++) {
				indicesMain.add(i);
			}
		}
		
		double explorationPercentage = ConfigLoader.getInstance().getExplorationPercentage();

		// Shuffle the main indices first (to ensure all options are reached after some time in case not all can be verified each cycle)
		Collections.shuffle(indicesMain);
		// Similar reasoning for the exploration indices
		Collections.shuffle(indicesSub);
		
		// Only select a percentage of the predictions of the other classes
		int subIndex = (int) Math.floor(indicesSub.size() * explorationPercentage);
		remainingIndices.addAll(indicesSub.subList(subIndex, indicesSub.size()));
		indicesSub = indicesSub.subList(0, subIndex);
		
		List<Integer> overallIndices = new ArrayList<>();
		overallIndices.addAll(indicesMain);
		overallIndices.addAll(indicesSub);

		
		int timeCap = ConfigLoader.getInstance().getTimeCap();
		AdaptationOption adaptationOption;
		int lastIndex = 0;

		for (Integer index : overallIndices) {
			adaptationOption = adaptationOptions.get(index);

			smcChecker.checkCAO(adaptationOption.toModelString(), environment.toModelString(),
				adaptationOption.verificationResults);

			// Add this option to the list of options that should be sent back for online learning
			qosEstimates.add(adaptationOption);

			lastIndex++;

			if ((System.currentTimeMillis() - startTime) / 1000 > timeCap) {
				break;
			}
		}

		// TODO maybe add a boolean in the adaptation option which indicates this
		// If not all the (relevant) options were verified
		for (int i = lastIndex; i < overallIndices.size(); i++) {
			adaptationOptions.get(overallIndices.get(i)).verificationResults.packetLoss = 100;
		}

		for (Integer i : remainingIndices) {
			adaptationOptions.get(i).verificationResults.packetLoss = 100;
		}


		// Perform online learning on the samples that were predicted to meet the user goal
		// Note: if no samples were predicted to meet the goal, all the verified options are sent back for online learning
		send(qosEstimates, taskType, Mode.TRAINING);
	}

}
